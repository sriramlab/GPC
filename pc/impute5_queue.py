"""Queues Impute5 jobs over artificial-genome reference panels."""

import os
import sys
import time
import math
import signal
import argparse
import subprocess

from hmm_config import CONFIGS, resolve, model_paths
from hmm_queue import last_epoch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
QLOGS = os.path.join(ROOT, "logs", "sweeps")
PYTHON = sys.executable
BOOTSTRAP = os.path.join(ROOT, "impute5", "results", "bootstrap")

MAPS = {
    "1KG": "/scratch2/prateek/b37_recombination_maps/chr15.b37.gmap.gz",
    "UKBB": "/scratch2/prateek/b37_recombination_maps/chr22.b37.gmap.gz",
}

# (config key, panel variant). 1KG first, baseline before combined.
SWEEPS = [
    ("1KG:afr", ""), ("1KG:afr", "_combined"),
    ("1KG:noneur", ""), ("1KG:noneur", "_combined"),
    ("UKBB:8020", ""),
    ("UKBB:afr", ""), ("UKBB:afr", "_combined"),
    ("UKBB:noneur", ""), ("UKBB:noneur", "_combined"),
]


def n_snps(cfg):
    return cfg["num_snps"]


def models_ready(cfg):
    """All four chunks trained to the final epoch."""
    E = cfg["epochs"]
    for i, ckpt in enumerate(model_paths(cfg)):
        lg = os.path.join(cfg["model_dir"],
                          f"{cfg['tag']}_{i}_{cfg['split']}_hmm_{cfg['n_train']}_128_"
                          f"{E}epochs_ps0.005.log")
        if not (os.path.exists(ckpt) and os.path.exists(lg)):
            return False
        # Read the last [Epoch n/N] rather than counting lines -- a spliced-in
        # partial run would inflate the count and could green-light sampling
        # from an undertrained model.
        if last_epoch(lg) < E:
            return False
    return True


def method_name(cfg, variant):
    return os.path.basename(cfg["sample_prefix"])[: -len("_samples")] + variant


def panel_path(cfg, variant):
    return f"{cfg['sample_prefix']}{variant}.vcf.gz"


def dosage_dir(cfg, variant):
    return os.path.join(BOOTSTRAP, f"{method_name(cfg, variant)}_dosages")


def done_count(cfg, variant):
    d = dosage_dir(cfg, variant)
    return len(os.listdir(d)) if os.path.isdir(d) else 0


def sweep_complete(cfg, variant):
    return done_count(cfg, variant) >= n_snps(cfg)


def test_vcf(cfg):
    for ext in (".vcf", ".vcf.gz"):
        p = cfg["test_prefix"] + ext
        if os.path.exists(p):
            return p
    return None


def ensure_inputs(cfg, variant, sample_gpu, log):
    """Sample AGs and build panels if they are not on disk yet."""
    if os.path.exists(panel_path(cfg, variant)):
        return True
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(sample_gpu),
               TRITON_CACHE_DIR=f"/scratch2/prateek/tmp/triton_hmm/sample")
    if not os.path.exists(cfg["sample_prefix"] + ".txt"):
        log(f"  sampling AGs for {cfg['key']} (GPU {sample_gpu}) ...")
        r = subprocess.run([PYTHON, "-u", os.path.join(HERE, "generate_hmm.py"), cfg["key"]],
                           cwd=HERE, env=env)
        if r.returncode != 0:
            log(f"  !! sampling failed for {cfg['key']}")
            return False
    log(f"  building panels for {cfg['key']} ...")
    cmd = [PYTHON, "-u", os.path.join(ROOT, "aux", "scripts", "make_hmm_panels.py"), cfg["key"]]
    if cfg["split"] != "8020":
        cmd.append("--combined")
    r = subprocess.run(cmd, cwd=HERE, env=dict(os.environ,
                       BCFTOOLS_PLUGINS=os.environ.get("BCFTOOLS_PLUGINS",
                                                       "/scratch2/prateek/bcftools/plugins")))
    if r.returncode != 0:
        log(f"  !! panel build failed for {cfg['key']}")
        return False
    return os.path.exists(panel_path(cfg, variant))


def start_sweep(cfg, variant, workers, log):
    os.makedirs(QLOGS, exist_ok=True)
    method = method_name(cfg, variant)
    out = os.path.join(QLOGS, f"{method}.log")
    fh = open(out, "a")
    fh.write(f"\n=== started {time.strftime('%Y-%m-%d %H:%M:%S')} "
             f"with {workers} workers ===\n")
    fh.flush()
    # Stable per-sweep workdir: the panel BCF (~200 MB for UKBB) is built once
    # and reused, which matters because preemption restarts a sweep repeatedly.
    workdir = f"/scratch2/prateek/tmp/impute5_panel_{method}"
    os.makedirs(workdir, exist_ok=True)
    cmd = [PYTHON, "-u", os.path.join(ROOT, "impute5", "impute5_loo_local.py"),
           "--panel", panel_path(cfg, variant),
           "--test", test_vcf(cfg),
           "--chr", str(cfg["chrom"]),
           "--map", MAPS[cfg["dataset"]],
           "--method", method,
           "--outdir", BOOTSTRAP,
           "--workdir", workdir,
           "--workers", str(workers)]
    env = dict(os.environ, BCFTOOLS_PLUGINS=os.environ.get(
        "BCFTOOLS_PLUGINS", "/scratch2/prateek/bcftools/plugins"))
    p = subprocess.Popen(cmd, cwd=os.path.join(ROOT, "impute5"), env=env,
                         stdout=fh, stderr=subprocess.STDOUT,
                         start_new_session=True)
    log(f"[{time.strftime('%H:%M:%S')}] START {method} "
        f"({done_count(cfg, variant)}/{n_snps(cfg)} done) -> {os.path.basename(out)}")
    return p, fh, out


def stop_sweep(p, fh, why, log):
    log(f"[{time.strftime('%H:%M:%S')}] {why} -- stopping sweep (resumable)")
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception:
        p.terminate()
    try:
        p.wait(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            p.kill()
    fh.close()
    # any impute5 binaries orphaned by the kill
    subprocess.run(["pkill", "-9", "-u", str(os.getuid()), "-f", "impute5_v1.2.0_static"],
                   stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--sample-gpu", type=int, default=3,
                    help="GPU for the AG sampling step (short; keep off the training GPUs)")
    ap.add_argument("--check-every", type=int, default=300,
                    help="seconds between readiness/preemption checks")
    ap.add_argument("--run", action="store_true", help="actually run (default: dry run)")
    ap.add_argument("--prepare-only", action="store_true",
                    help="sample AGs and build reference panels as each config finishes "
                         "training, then stop. Use when the leave-one-SNP-out sweeps are "
                         "being run elsewhere (e.g. the cluster job arrays).")
    a = ap.parse_args()

    def log(m):
        print(m, flush=True)

    print(f"{'#':>2s} {'sweep':28s} {'SNPs':>12s}  status")
    print("-" * 74)
    for i, (key, var) in enumerate(SWEEPS, 1):
        cfg = resolve(key)
        m = method_name(cfg, var)
        n, tot = done_count(cfg, var), n_snps(cfg)
        if sweep_complete(cfg, var):
            st = "COMPLETE"
        elif not models_ready(cfg):
            st = "waiting on training"
        elif not os.path.exists(panel_path(cfg, var)):
            st = "ready (needs AGs/panel)"
        else:
            st = "ready"
        print(f"{i:2d} {m:28s} {f'{n}/{tot}':>12s}  {st}")
    print("-" * 74)
    if not a.run:
        print("(dry run -- pass --run to start)")
        return

    if a.prepare_only:
        # Sample AGs + build panels for every AG-route config, waiting on training.
        # The sweeps themselves run on the cluster; see impute5/CLUSTER_HANDOFF.md.
        pending = []
        for key, var in SWEEPS:
            if var:                      # panels are built once per config, not per variant
                continue
            if key not in [p for p in pending]:
                pending.append(key)
        while pending:
            progressed = False
            for key in list(pending):
                cfg = resolve(key)
                if not models_ready(cfg):
                    continue
                if os.path.exists(panel_path(cfg, "")) and (
                        cfg["split"] == "8020" or os.path.exists(panel_path(cfg, "_combined"))):
                    log(f"[{time.strftime('%H:%M:%S')}] {key}: panels already built")
                    pending.remove(key); progressed = True; continue
                log(f"[{time.strftime('%H:%M:%S')}] {key}: models ready, preparing panels")
                if ensure_inputs(cfg, "", a.sample_gpu, log):
                    log(f"[{time.strftime('%H:%M:%S')}] {key}: panels ready")
                    pending.remove(key)
                else:
                    log(f"[{time.strftime('%H:%M:%S')}] {key}: panel prep FAILED, will retry")
                progressed = True
            if pending and not progressed:
                time.sleep(a.check_every)
        log("all panels built -- ready to transfer to the cluster")
        return

    cur = None          # (idx, cfg, variant, proc, fh)
    while True:
        # highest-priority sweep that is ready and unfinished
        want = None
        for idx, (key, var) in enumerate(SWEEPS):
            cfg = resolve(key)
            if sweep_complete(cfg, var) or not models_ready(cfg):
                continue
            want = (idx, cfg, var)
            break

        if cur and cur[3].poll() is not None:
            cur[4].close()
            log(f"[{time.strftime('%H:%M:%S')}] sweep {method_name(cur[1], cur[2])} exited "
                f"({done_count(cur[1], cur[2])}/{n_snps(cur[1])})")
            cur = None

        if want is None:
            if cur is None:
                if all(sweep_complete(resolve(k), v) for k, v in SWEEPS):
                    log("all sweeps complete")
                    return
                log(f"[{time.strftime('%H:%M:%S')}] nothing ready; waiting on training")
        elif cur is None:
            idx, cfg, var = want
            if ensure_inputs(cfg, var, a.sample_gpu, log):
                p, fh, _ = start_sweep(cfg, var, a.workers, log)
                cur = (idx, cfg, var, p, fh)
        elif want[0] < cur[0]:
            stop_sweep(cur[3], cur[4],
                       f"higher-priority sweep {method_name(want[1], want[2])} is ready", log)
            cur = None
            continue        # start it on the next pass

        time.sleep(a.check_every)


if __name__ == "__main__":
    main()
