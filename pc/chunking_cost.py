"""Cost of training the HMM baseline in contiguous chunks rather than as one
circuit."""

import argparse
import math
import sys
import time

import numpy as np
import torch
import pyjuice as juice

sys.setrecursionlimit(1000000)

REF_BATCH = 256
PS = 0.005


def train_block(train, valid, epochs, latents, batch_size, seed, device, label):
    """Train one HMM over a contiguous block; return its held-out LL."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n, seq_length = train.shape
    nb = max(1, math.ceil(n / batch_size))
    eff = n / nb
    ps = PS * (REF_BATCH / eff)

    ns = juice.structures.HMM(seq_length=seq_length, num_latents=latents,
                              homogeneous=False, num_emits=2)
    pc = juice.compile(ns).to(device)

    tr = train.to(device=device, dtype=torch.uint8)
    va = valid.to(device=device, dtype=torch.uint8)
    bounds = [(i * n) // nb for i in range(nb + 1)]
    nvb = (va.shape[0] + REF_BATCH - 1) // REF_BATCH

    t0 = time.time()
    for ep in range(1, epochs + 1):
        pc.init_param_flows(flows_memory=0.0)
        perm = torch.randperm(n, device=device)
        for i in range(nb):
            pc(tr[perm[bounds[i]:bounds[i + 1]]].long()).mean().backward()
        pc.mini_batch_em(step_size=1.0, pseudocount=ps)
        if ep % 100 == 0 or ep == epochs:
            with torch.no_grad():
                vacc = torch.zeros((), device=device)
                for s in range(0, va.shape[0], REF_BATCH):
                    vacc += pc(va[s:s + REF_BATCH].long()).mean()
            print(f"    {label} epoch {ep}: valid LL {(vacc / nvb).item():.3f} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    with torch.no_grad():
        vacc = torch.zeros((), device=device)
        for s in range(0, va.shape[0], REF_BATCH):
            vacc += pc(va[s:s + REF_BATCH].long()).mean()
        ll = (vacc / nvb).item()

    del pc
    torch.cuda.empty_cache()
    return ll


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default="../results/1KG/8020/data/8020_train.txt")
    p.add_argument("--valid", default="../results/1KG/8020/data/8020_test.txt")
    p.add_argument("--snps", type=int, default=5000)
    p.add_argument("--chunks", type=int, nargs="+", default=[1, 2],
                   help="chunk counts to compare; the region is split evenly")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--latents", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda")
    tr = torch.from_numpy(np.loadtxt(args.train, dtype=np.int8, delimiter=' ')[:, :args.snps])
    va = torch.from_numpy(np.loadtxt(args.valid, dtype=np.int8, delimiter=' ')[:, :args.snps])
    print(f"train {tuple(tr.shape)}  valid {tuple(va.shape)}  "
          f"{args.epochs} epochs, {args.latents} latents\n", flush=True)

    results = {}
    for k in args.chunks:
        assert args.snps % k == 0, f"{args.snps} not divisible by {k}"
        width = args.snps // k
        print(f"{k} chunk(s) of {width} SNPs", flush=True)
        # Chunk log-likelihoods add: the chunked model is the product of the
        # per-chunk distributions, so its LL for a haplotype is the sum.
        total = 0.0
        for c in range(k):
            lo, hi = c * width, (c + 1) * width
            total += train_block(tr[:, lo:hi], va[:, lo:hi], args.epochs, args.latents,
                                 args.batch_size, args.seed, device, f"chunk {c}")
        results[k] = total
        print(f"  -> {k} chunk(s): total valid LL {total:.3f}\n", flush=True)

    base = results[min(results)]
    print(f"{'chunks':>8} {'valid LL':>12} {'vs 1 chunk':>12} {'per SNP':>10}")
    for k in sorted(results):
        d = results[k] - base
        print(f"{k:>8} {results[k]:12.3f} {d:12.3f} {d / args.snps:10.5f}")


if __name__ == "__main__":
    main()
