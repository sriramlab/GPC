"""Queues direct (conditional) imputation jobs."""

import os
import sys
import time
import argparse
import subprocess

from hmm_config import CONFIGS, resolve, r2_csv
from impute5_queue import models_ready
from hmm_queue import running_trainers

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
QLOGS = os.path.join(ROOT, "logs", "direct")
PYTHON = sys.executable

# 1KG before UKBB (main-text figures first); within a dataset, the
# population-only models before the combined ones.
ORDER = [
    "1KG:afr", "1KG:noneur", "1KG:eur_and_afr_train", "1KG:eur_and_noneur_train",
    "UKBB:afr", "UKBB:noneur", "UKBB:eur_and_afr_train", "UKBB:eur_and_noneur_train",
    "1KG:8020", "UKBB:8020",
]


def dosage_path(cfg):
    return f"{cfg['model_dir']}/hmm_{cfg['split']}_direct_dosages.npy"


def state(key):
    cfg = resolve(key)
    if os.path.exists(r2_csv(cfg, "direct")):
        return "csv done"
    if os.path.exists(dosage_path(cfg)):
        return "dosages done, needs csv"
    if not models_ready(cfg):
        return "waiting on training"
    return "ready"


def run_direct(key, gpu, batch_size, log, keep_dosages=False):
    cfg = resolve(key)
    os.makedirs(QLOGS, exist_ok=True)
    out = os.path.join(QLOGS, f"{key.replace(':', '_')}.log")
    fh = open(out, "a")
    fh.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} on GPU {gpu} ===\n")
    fh.flush()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               TRITON_CACHE_DIR=f"/scratch2/prateek/tmp/triton_hmm/direct{gpu}")

    if not os.path.exists(dosage_path(cfg)):
        log(f"[{time.strftime('%H:%M:%S')}] {key}: direct imputation -> {os.path.basename(out)}")
        r = subprocess.run([PYTHON, "-u", os.path.join(HERE, "predict_hmm.py"), key,
                            "--batch-size", str(batch_size)],
                           cwd=HERE, env=env, stdout=fh, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            log(f"[{time.strftime('%H:%M:%S')}] {key}: predict_hmm FAILED (see {out})")
            fh.close()
            return False

    csv = r2_csv(cfg, "direct")
    log(f"[{time.strftime('%H:%M:%S')}] {key}: assembling {os.path.basename(csv)}")
    r = subprocess.run([PYTHON, "-u", os.path.join(HERE, "assemble_r2.py"), key, "--out", csv],
                       cwd=HERE, env=env, stdout=fh, stderr=subprocess.STDOUT)
    fh.close()
    if r.returncode != 0:
        log(f"[{time.strftime('%H:%M:%S')}] {key}: assemble_r2 FAILED (see {out})")
        return False
    log(f"[{time.strftime('%H:%M:%S')}] {key}: DONE -> {os.path.basename(csv)}")

    # The dosage matrices are an intermediate: assemble_r2.py has turned them into
    # the CSV the figures read, so drop them rather than leave ~1GB of .npy in the
    # results tree. Only after the CSV is confirmed written.
    # Caveat: regenerating them (to re-derive r^2 with a different bootstrap count
    # or MAF binning) costs ~2.5 h of GPU per config. Pass --keep-dosages to retain.
    if not keep_dosages:
        removed = 0
        for q in [dosage_path(cfg)] + [f"{dosage_path(cfg)}.chunk{c}.npy" for c in range(4)]:
            if os.path.exists(q):
                os.remove(q)
                removed += 1
        if removed:
            log(f"[{time.strftime('%H:%M:%S')}] {key}: removed {removed} intermediate .npy")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--check-every", type=int, default=600,
                    help="seconds to wait when the next config is still training")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--keep-dosages", action="store_true",
                    help="keep the intermediate .npy dosage matrices "
                         "(default: delete them once the CSV is written)")
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--ignore-trainers", action="store_true",
                    help="start even while hmm.py training jobs hold the GPUs "
                         "(they will not both fit in 24 GB; default is to wait)")
    a = ap.parse_args()

    keys = [k for k in ORDER if not a.only or k in a.only]

    print(f"{'config':30s} {'n_test':>7s}  status")
    print("-" * 62)
    for k in keys:
        cfg = resolve(k)
        n = sum(1 for _ in open(cfg["valid_path"]))
        print(f"{k:30s} {n:7d}  {state(k)}")
    print("-" * 62)
    if not a.run:
        print("(dry run -- pass --run to start)")
        return

    def log(m):
        print(m, flush=True)

    pending = [k for k in keys if state(k) != "csv done"]
    log(f"\n{len(pending)} config(s) to process on GPU {a.gpu}\n")
    while pending:
        progressed = False
        for k in list(pending):
            s = state(k)
            if s == "csv done":
                pending.remove(k); progressed = True; continue
            if s == "waiting on training":
                continue
            # A training job holds ~16 GB on its GPU and the direct arm needs a
            # similar amount, so the two do not fit together on one 24 GB card.
            # Wait for training to drain rather than racing it into an OOM.
            if not a.ignore_trainers and running_trainers():
                log(f"[{time.strftime('%H:%M:%S')}] {k} ready, but {len(running_trainers())} "
                    f"training job(s) still hold the GPUs -- waiting")
                continue
            if run_direct(k, a.gpu, a.batch_size, log, a.keep_dosages):
                pending.remove(k)
            progressed = True
        if pending and not progressed:
            log(f"[{time.strftime('%H:%M:%S')}] waiting on training for "
                f"{len(pending)} config(s)")
            time.sleep(a.check_every)
    log("all direct-arm CSVs complete")


if __name__ == "__main__":
    main()
