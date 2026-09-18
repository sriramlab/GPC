"""Leave-one-SNP-out imputation from a chunked HMM."""

import os
import sys
import time
import argparse
import numpy as np
import torch
import pyjuice as juice
from tqdm import tqdm

from hmm_config import resolve, model_paths

sys.setrecursionlimit(100000)


def chunk_dosages(model_path, test_block, batch_size, device):
    """P(allele=1) for every (haplotype, SNP) in this chunk, leave-one-out."""
    pc = juice.compile(juice.load(model_path)).to(device)
    n, v = test_block.shape
    assert v == pc.num_vars, f"{model_path}: model has {pc.num_vars} vars, data has {v}"

    out = np.zeros((n, v), dtype=np.float32)
    mask = torch.zeros(v, dtype=torch.bool, device=device)

    for start in range(0, n, batch_size):
        x = torch.tensor(test_block[start:start + batch_size], dtype=torch.long, device=device)
        end = start + x.shape[0]
        for pos in tqdm(range(v), desc=f"{os.path.basename(model_path)} [{start}:{end}]",
                        unit="snp", leave=False):
            mask[pos] = True
            with torch.no_grad():
                probs = juice.queries.conditional(pc, data=x, missing_mask=mask)
            out[start:end, pos] = probs[:, pos, 1].detach().cpu().numpy()
            mask[pos] = False

    del pc
    torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser(description="Direct imputation with the chunked HMM.")
    p.add_argument("config", help="dataset/split key, e.g. 1KG:afr")
    p.add_argument("--chunks", type=int, default=None,
                   help="default: the dataset's chunk count from hmm_config "
                        "(4 for 1KG/UKBB, 6 for b38)")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--out", default=None, help="output .npy (default: alongside the models)")
    p.add_argument("--test", default=None,
                   help="override the target haplotype matrix (e.g. the admixed subset)")
    p.add_argument("--only-chunk", type=int, default=None,
                   help="compute a single chunk and store it as a partial file")
    args = p.parse_args()

    cfg = resolve(args.config)
    device = torch.device("cuda")

    out_path = args.out or f"{cfg['model_dir']}/hmm_{cfg['split']}_direct_dosages.npy"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    test_path = args.test or cfg["valid_path"]
    print(f"Loading test data {test_path}")
    test = np.loadtxt(test_path, dtype=np.int8, delimiter=' ')
    n, v = test.shape
    assert v == cfg["num_snps"], f"expected {cfg['num_snps']} SNPs, got {v}"
    print(f"Test shape: {test.shape}")

    num_chunks = args.chunks or cfg["num_chunks"]
    paths = model_paths(cfg, num_chunks=num_chunks)
    missing = [q for q in paths if not os.path.exists(q)]
    if missing:
        raise SystemExit("missing chunk checkpoints:\n  " + "\n  ".join(missing))

    width = v // num_chunks
    todo = [args.only_chunk] if args.only_chunk is not None else list(range(num_chunks))

    for c in todo:
        part = f"{out_path}.chunk{c}.npy"
        if os.path.exists(part):
            print(f"chunk {c}: already done ({part})")
            continue
        t0 = time.time()
        block = test[:, c * width:(c + 1) * width]
        dos = chunk_dosages(paths[c], block, args.batch_size, device)
        np.save(part, dos)
        print(f"chunk {c}: {(time.time()-t0)/60:.1f} min -> {part}")

    parts = [f"{out_path}.chunk{c}.npy" for c in range(num_chunks)]
    if all(os.path.exists(q) for q in parts):
        full = np.hstack([np.load(q) for q in parts])
        assert full.shape == (n, v), f"got {full.shape}, expected {(n, v)}"
        np.save(out_path, full)
        print(f"wrote {out_path}  {full.shape}")
    else:
        print("partial run: rerun without --only-chunk to concatenate")


if __name__ == "__main__":
    main()
