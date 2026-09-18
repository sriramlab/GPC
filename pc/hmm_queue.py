"""Queues HMM training jobs across chunks and datasets."""

import os
import sys
import math
import time
import shutil
import argparse
import subprocess

from hmm_config import CONFIGS, model_paths

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
QLOGS = os.path.join(ROOT, "logs", "queue")
CACHE = "/scratch2/prateek/tmp/triton_hmm"
PYTHON = sys.executable


def log_path(cfg, chunk):
    return os.path.join(
        cfg["model_dir"],
        f"{cfg['tag']}_{chunk}_{cfg['split']}_hmm_{cfg['n_train']}_128_"
        f"{cfg['epochs']}epochs_ps0.005.log")


def last_epoch(log):
    """Epoch number on the final [Epoch n/N] line, or 0.

    Read from the line rather than counting lines: a log can legitimately hold
    more [Epoch lines than epochs run if a previous partial attempt was spliced
    in front of it, and counting would then over-report progress and could call
    a chunk finished before it is.
    """
    if not os.path.exists(log):
        return 0
    last = 0
    with open(log) as fh:
        for l in fh:
            if l.startswith("[Epoch"):
                last = l
    if not last:
        return 0
    try:
        return int(last[len("[Epoch"):].split("/", 1)[0])
    except (ValueError, IndexError):
        return 0


def epochs_done(cfg, chunk):
    """Epochs actually backed by a checkpoint (checkpoints land every 500)."""
    lp, cp = log_path(cfg, chunk), model_paths(cfg)[chunk]
    n = last_epoch(lp)
    if n >= cfg["epochs"]:
        return cfg["epochs"]
    return (n // 500) * 500 if os.path.exists(cp) else 0


# Critical-path order. 1KG first (the main-text figures), and within each
# dataset the AG-route models before the direct-only ones, because only the
# AG-route models unblock Impute5 sweeps.
# The Impute5 sweeps run on the cluster (see impute5/CLUSTER_HANDOFF.md), so the
# thing to optimise is how soon all nine reference panels can be shipped. Every
# AG-route config comes first for that reason; the eur_and_* configs are
# direct-imputation only, stay entirely on this machine, and gate nothing.
PRIORITY = [
    "1KG:afr",                    # done; panels built
    "1KG:noneur",                 # -> 10K_hmm_noneur{,_combined} panels
    "UKBB:afr",                   # -> UKBB_hmm_afr{,_combined} panels
    "UKBB:noneur",                # -> UKBB_hmm_noneur{,_combined} panels
    "1KG:eur_and_afr_train",      # direct-only from here down
    "1KG:eur_and_noneur_train",
    "UKBB:eur_and_afr_train",
    "UKBB:eur_and_noneur_train",
]


def build_queue(batch_size, only=None, order=None):
    order = order or PRIORITY
    rank = {k: i for i, k in enumerate(order)}
    jobs = []
    for key in sorted(CONFIGS):
        if only and key not in only:
            continue
        cfg = CONFIGS[key]
        nb = math.ceil(cfg["n_train"] / batch_size)
        for chunk in range(4):
            done = epochs_done(cfg, chunk)
            left = cfg["epochs"] - done
            if left > 0:
                jobs.append(dict(key=key, chunk=chunk, done=done, left=left,
                                 nb=nb, work=left * nb))
    # config priority first, then longest chunk within a config
    jobs.sort(key=lambda j: (rank.get(j["key"], len(rank)), -j["work"]))
    return jobs


def running_trainers():
    try:
        out = subprocess.run(["pgrep", "-af", "hmm.py"], capture_output=True,
                             text=True).stdout
    except FileNotFoundError:
        return []
    return [l for l in out.splitlines() if "hmm.py" in l and "hmm_queue" not in l]


def launch(job, gpu, batch_size, ps_mode):
    os.makedirs(QLOGS, exist_ok=True)
    slot_cache = os.path.join(CACHE, f"gpu{gpu}")
    os.makedirs(slot_cache, exist_ok=True)
    env = dict(os.environ,
               CUDA_VISIBLE_DEVICES=str(gpu),
               TRITON_CACHE_DIR=slot_cache)
    out = os.path.join(QLOGS, f"{job['key'].replace(':', '_')}_c{job['chunk']}.log")
    cmd = [PYTHON, "-u", os.path.join(HERE, "hmm.py"), job["key"],
           "--chunks", str(job["chunk"]),
           "--batch-size", str(batch_size),
           "--pseudocount-mode", ps_mode]
    fh = open(out, "a")
    fh.write(f"\n=== launched {time.strftime('%Y-%m-%d %H:%M:%S')} on GPU {gpu} "
             f"(resume from {job['done']}) ===\n")
    fh.flush()
    p = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return p, fh, out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", type=int, nargs="+", default=[3],
                    help="GPU ids to use, one training job per GPU")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--pseudocount-mode", choices=("scaled", "fixed"), default="scaled")
    ap.add_argument("--run", action="store_true", help="actually launch (default: dry run)")
    ap.add_argument("--force", action="store_true",
                    help="launch even if other hmm.py processes are running")
    ap.add_argument("--only", nargs="+", default=None,
                    help="restrict to these config keys")
    ap.add_argument("--per-gpu", type=int, default=1,
                    help="jobs per GPU (default 1; >1 measured to lose throughput)")
    ap.add_argument("--order", nargs="+", default=None,
                    help="config keys in the order to run them (default: critical path)")
    a = ap.parse_args()

    jobs = build_queue(a.batch_size, a.only, a.order)
    if not jobs:
        print("nothing left to train")
        return

    total = sum(j["work"] for j in jobs)
    slots = [g for g in a.gpus for _ in range(a.per_gpu)]
    print(f"{len(jobs)} chunks remaining, {total:,} batch-epochs at bs={a.batch_size}")
    print(f"{len(slots)} slots across GPUs {a.gpus}\n")
    print(f"{'#':>3s} {'config':28s} {'ch':>2s} {'resume':>7s} {'epochs left':>11s} {'batch-epochs':>13s}")
    print("-" * 72)
    for i, j in enumerate(jobs, 1):
        print(f"{i:3d} {j['key']:28s} {j['chunk']:2d} {j['done']:7d} "
              f"{j['left']:11d} {j['work']:13,}")
    print("-" * 72)

    live = running_trainers()
    if live:
        print(f"\n{len(live)} hmm.py process(es) already running:")
        for l in live:
            print("   ", l[:120])
        if not a.force:
            print("\nRefusing to launch -- two processes training the same chunk would")
            print("interleave writes to the same .log and .jpc. Stop them first, or pass --force.")
            if a.run:
                sys.exit(1)

    if not a.run:
        print("\n(dry run -- pass --run to launch)")
        return

    print(f"\nlaunching; per-job logs under {QLOGS}\n", flush=True)
    pending = list(jobs)
    active = {}          # slot index -> (proc, filehandle, job, logpath, start)
    t_start = time.time()
    try:
        while pending or active:
            for si, gpu in enumerate(slots):
                if si in active or not pending:
                    continue
                job = pending.pop(0)
                p, fh, out = launch(job, gpu, a.batch_size, a.pseudocount_mode)
                active[si] = (p, fh, job, out, time.time())
                print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu} <- {job['key']} c{job['chunk']} "
                      f"({job['left']} epochs, resume {job['done']}) -> {os.path.basename(out)}",
                      flush=True)
                time.sleep(20)      # stagger so compiles do not land together

            for si in list(active):
                p, fh, job, out, t0 = active[si]
                if p.poll() is None:
                    continue
                fh.close()
                mins = (time.time() - t0) / 60
                status = "ok" if p.returncode == 0 else f"FAILED rc={p.returncode}"
                print(f"[{time.strftime('%H:%M:%S')}] GPU {slots[si]} done {job['key']} "
                      f"c{job['chunk']} in {mins:.1f} min  [{status}]", flush=True)
                if p.returncode != 0:
                    print(f"    see {out}", flush=True)
                del active[si]
            time.sleep(5)
    except KeyboardInterrupt:
        print("\ninterrupted -- terminating active jobs "
              "(they resume from their last checkpoint)")
        for si, (p, fh, job, out, t0) in active.items():
            p.terminate()
        for si, (p, fh, job, out, t0) in active.items():
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
            fh.close()
        sys.exit(130)

    print(f"\nall chunks complete in {(time.time()-t_start)/3600:.1f} h")


if __name__ == "__main__":
    main()
