"""Direct imputation: reads conditional probabilities straight from the circuit."""

import argparse
import os
import sys
import time

import numpy as np
import pyjuice as juice
import torch
from tqdm import tqdm

sys.setrecursionlimit(100000)


def main():
    p = argparse.ArgumentParser(description="Direct LOO imputation from one circuit.")
    p.add_argument("--model", required=True, help="compiled .jpc checkpoint")
    p.add_argument("--test", required=True, help="target haplotype matrix (.txt)")
    p.add_argument("--out", required=True, help="output dosage matrix (.npy)")
    p.add_argument("--batch-size", type=int, default=512)
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"Loading targets {args.test}")
    test = np.loadtxt(args.test, dtype=np.int8, delimiter=' ')
    n, v = test.shape
    print(f"Target shape: {test.shape}")

    print(f"Loading model {args.model}")
    pc = juice.compile(juice.load(args.model)).to(device)
    assert pc.num_vars == v, f"model has {pc.num_vars} vars, targets have {v}"

    out = np.zeros((n, v), dtype=np.float32)
    mask = torch.zeros(v, dtype=torch.bool, device=device)

    t0 = time.time()
    for start in range(0, n, args.batch_size):
        x = torch.tensor(test[start:start + args.batch_size], dtype=torch.long, device=device)
        end = start + x.shape[0]
        for pos in tqdm(range(v), desc=f"[{start}:{end}]", unit="snp", leave=False):
            mask[pos] = True
            with torch.no_grad():
                probs = juice.queries.conditional(pc, data=x, missing_mask=mask)
            out[start:end, pos] = probs[:, pos, 1].detach().cpu().numpy()
            mask[pos] = False

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, out)
    print(f"wrote {args.out}  {out.shape}  in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
