"""Samples artificial genomes from the trained HMM chunks."""

import os
import sys
import argparse
import numpy as np
import torch
import pyjuice as juice
from tqdm import tqdm

from hmm_config import resolve, model_paths

sys.setrecursionlimit(100000)


def sample_chunk(model_path, num_samples, batch_size, device):
    pc = juice.compile(juice.load(model_path)).to(device)
    out = []
    remaining = num_samples
    pbar = tqdm(total=num_samples, desc=os.path.basename(model_path), unit="hap")
    while remaining > 0:
        b = min(batch_size, remaining)
        out.append(juice.queries.sample(pc, num_samples=b).cpu().numpy().astype(np.int8))
        remaining -= b
        pbar.update(b)
    pbar.close()
    del pc
    torch.cuda.empty_cache()
    return np.vstack(out)


def main():
    p = argparse.ArgumentParser(description="Sample AGs from the chunked HMM.")
    p.add_argument("config", help="dataset/split key, e.g. 1KG:afr")
    p.add_argument("--num-samples", type=int, default=None,
                   help="number of artificial haplotypes (default: match the other methods)")
    p.add_argument("--chunks", type=int, default=None,
                   help="default: the dataset's chunk count from hmm_config "
                        "(4 for 1KG/UKBB, 6 for b38)")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    cfg = resolve(args.config)
    n = args.num_samples if args.num_samples is not None else cfg["num_ags"]
    device = torch.device("cuda")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    num_chunks = args.chunks or cfg["num_chunks"]
    paths = model_paths(cfg, num_chunks=num_chunks)
    missing = [q for q in paths if not os.path.exists(q)]
    if missing:
        raise SystemExit("missing chunk checkpoints:\n  " + "\n  ".join(missing))

    blocks = [sample_chunk(q, n, args.batch_size, device) for q in paths]
    d = np.hstack(blocks)
    assert d.shape == (n, cfg["num_snps"]), f"got {d.shape}, expected {(n, cfg['num_snps'])}"

    txt = f"{cfg['sample_prefix']}.txt"
    np.savetxt(txt, d, fmt="%d")
    print(f"wrote {txt}  {d.shape}")

    # .hapt: same layout as aux/scripts/create_hap.py
    meta = np.column_stack([
        np.array(["Test"] * n),
        np.array([f"Sample{i+1}" for i in range(n)]),
    ])
    hapt = f"{cfg['sample_prefix']}.hapt"
    np.savetxt(hapt, np.hstack((meta, d.astype(str))), fmt="%s", delimiter=" ")
    print(f"wrote {hapt}")


if __name__ == "__main__":
    main()
