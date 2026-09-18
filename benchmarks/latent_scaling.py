"""Region size reachable as a function of the number of latent states."""

import argparse
import csv
import os
import sys
import time
import traceback

import numpy as np
import torch

sys.setrecursionlimit(1000000)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_DATA = f"{ROOT}/results/b38/8020/data/8020_train.txt"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "latents")

BATCH_SIZE = 128          # as in pc/train.py
PSEUDOCOUNT = 0.005
WARMUP, MEASURE = 2, 3


def load(n, m, _cache={}):
    if "d" not in _cache:
        print(f"loading {BASE_DATA} ...", flush=True)
        _cache["d"] = np.loadtxt(BASE_DATA, dtype=np.int8, delimiter=' ')
    return np.ascontiguousarray(_cache["d"][:n, :m])


def measure(n, m, latents, device):
    import pyjuice as juice
    import pyjuice.nodes.distributions as dists

    x = torch.tensor(load(n, m), dtype=torch.long)      # host, as in train.py

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    x_struct = x.float().to(device)
    ns = juice.structures.HCLT(x_struct, num_latents=latents,
                               input_dist=dists.Categorical(num_cats=2))
    del x_struct
    torch.cuda.empty_cache()
    pc = juice.compile(ns).to(device)
    build_sec = time.perf_counter() - t0

    nb = max(1, -(-n // BATCH_SIZE))
    splits = np.array_split(np.arange(n), nb)

    def one_epoch():
        pc.init_param_flows(flows_memory=0.0)
        for idx in splits:
            pc(x[idx].to(device)).mean().backward()
        pc.mini_batch_em(step_size=1.0, pseudocount=PSEUDOCOUNT)

    times = []
    for i in range(WARMUP + MEASURE):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        one_epoch()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    res = dict(sec_per_epoch=float(np.median(times[WARMUP:])),
               peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
               peak_gpu_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
               build_sec=build_sec, num_parameters=int(ns.num_parameters()),
               batches_per_epoch=nb)
    del pc, ns
    torch.cuda.empty_cache()
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latents", type=int, nargs="+", default=[32, 64, 128, 256])
    p.add_argument("--snps", type=int, nargs="+", default=[2047, 4095, 8191])
    p.add_argument("--n", type=int, default=4006)
    p.add_argument("--out", default=OUT)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "latent_scaling.csv")
    rows = list(csv.DictReader(open(csv_path))) if os.path.exists(csv_path) else []
    done = {(int(r["latents"]), int(r["M"])) for r in rows}

    device = torch.device("cuda")
    for L in sorted(args.latents):
        for m in sorted(args.snps):
            if (L, m) in done:
                print(f"[L={L} M={m}] already done")
                continue
            print(f"\n[L={L} M={m} N={args.n}]", flush=True)
            row = dict(latents=L, M=m, N=args.n)
            try:
                row.update(status="ok", **measure(args.n, m, L, device))
                print(f"  {row['sec_per_epoch']:.3f}s/epoch, peak {row['peak_gpu_gib']:.2f} GiB, "
                      f"build {row['build_sec']:.0f}s, {row['num_parameters']:,} params",
                      flush=True)
            except torch.cuda.OutOfMemoryError as e:
                row.update(status="OOM")
                print(f"  OOM: {str(e)[:90]}", flush=True)
                torch.cuda.empty_cache()
            except Exception:
                row.update(status="FAILED")
                traceback.print_exc()
                torch.cuda.empty_cache()
            rows.append(row)
            cols = ["latents", "M", "N", "status", "sec_per_epoch", "peak_gpu_gib",
                    "peak_gpu_reserved_gib", "build_sec", "num_parameters",
                    "batches_per_epoch"]
            for r in rows:
                for c in cols:
                    r.setdefault(c, "")
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                w.writeheader(); w.writerows(rows)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
