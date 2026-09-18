"""How large a region GPC can fit, and what lowering the latent count buys."""

import argparse
import csv
import gc
import os
import resource
import sys
import time
import traceback

import numpy as np
import torch

sys.setrecursionlimit(1000000)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_DATA = f"{ROOT}/results/b38/8020/data/8020_train.txt"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "frontier")
BATCH_SIZE = 128
PSEUDOCOUNT = 0.005


def host_gib():
    """Peak resident set size of this process, in GiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def build_matrix(n, m, _cache={}):
    """n x m haplotypes, tiling the real region when m exceeds it."""
    if "d" not in _cache:
        print(f"loading {BASE_DATA} ...", flush=True)
        _cache["d"] = np.loadtxt(BASE_DATA, dtype=np.int8, delimiter=' ')
    full = _cache["d"][:n]
    if m <= full.shape[1]:
        return np.ascontiguousarray(full[:, :m])
    rng = np.random.default_rng(0)
    tiles, width = [], full.shape[1]
    while sum(t.shape[1] for t in tiles) < m:
        # a fresh row permutation per tile: LD is preserved within a tile and
        # absent between tiles, as between separate genomic regions
        tiles.append(full[rng.permutation(full.shape[0])])
    out = np.concatenate(tiles, axis=1)[:, :m]
    return np.ascontiguousarray(out)


def measure(n, m, latents, device, epochs=3):
    import pyjuice as juice
    import pyjuice.nodes.distributions as dists

    x = torch.tensor(build_matrix(n, m), dtype=torch.long)
    gc.collect()
    host_before = host_gib()

    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    x_struct = x.float().to(device)
    ns = juice.structures.HCLT(x_struct, num_latents=latents,
                               input_dist=dists.Categorical(num_cats=2))
    struct_sec = time.perf_counter() - t0
    struct_host_gib = host_gib()
    del x_struct
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    pc = juice.compile(ns).to(device)
    compile_sec = time.perf_counter() - t0

    nb = max(1, -(-n // BATCH_SIZE))
    splits = np.array_split(np.arange(n), nb)

    def one_epoch():
        pc.init_param_flows(flows_memory=0.0)
        for idx in splits:
            pc(x[idx].to(device)).mean().backward()
        pc.mini_batch_em(step_size=1.0, pseudocount=PSEUDOCOUNT)

    times = []
    for _ in range(epochs):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        one_epoch()
        torch.cuda.synchronize(); times.append(time.perf_counter() - t0)

    res = dict(struct_sec=struct_sec, compile_sec=compile_sec,
               sec_per_epoch=float(np.median(times[1:] or times)),
               peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
               peak_host_gib=host_gib(), struct_host_gib=struct_host_gib,
               host_before_gib=host_before, num_parameters=int(ns.num_parameters()))
    del pc, ns, x
    gc.collect(); torch.cuda.empty_cache()
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latents", type=int, nargs="+", default=[32])
    p.add_argument("--snps", type=int, nargs="+", default=[8191, 12287, 14670, 18000])
    p.add_argument("--n", type=int, default=4006)
    p.add_argument("--out", default=OUT)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "gpc_frontier.csv")
    rows = list(csv.DictReader(open(csv_path))) if os.path.exists(csv_path) else []
    done = {(int(r["latents"]), int(r["M"])) for r in rows}
    cols = ["latents", "M", "N", "status", "struct_sec", "compile_sec", "sec_per_epoch",
            "peak_gpu_gib", "peak_host_gib", "struct_host_gib", "host_before_gib",
            "num_parameters"]

    device = torch.device("cuda")
    for L in sorted(args.latents):
        for m in sorted(args.snps):
            if (L, m) in done:
                print(f"[L={L} M={m}] already done"); continue
            print(f"\n[L={L} M={m} N={args.n}]", flush=True)
            row = dict(latents=L, M=m, N=args.n)
            try:
                row.update(status="ok", **measure(args.n, m, L, device))
                print(f"  structure {row['struct_sec']/60:.1f} min (host peak "
                      f"{row['struct_host_gib']:.1f} GiB), compile "
                      f"{row['compile_sec']/60:.1f} min, {row['sec_per_epoch']:.2f}s/epoch, "
                      f"GPU peak {row['peak_gpu_gib']:.2f} GiB", flush=True)
            except (torch.cuda.OutOfMemoryError, MemoryError) as e:
                row.update(status=f"OOM: {type(e).__name__}")
                print(f"  OOM at L={L} M={m}: {str(e)[:110]}", flush=True)
                gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                row.update(status=f"FAILED: {type(e).__name__}")
                traceback.print_exc()
                gc.collect(); torch.cuda.empty_cache()
            rows.append(row)
            for r in rows:
                for c in cols:
                    r.setdefault(c, "")
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                w.writeheader(); w.writerows(rows)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
