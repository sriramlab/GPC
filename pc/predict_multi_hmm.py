"""Array-based imputation from a chunked HMM."""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import pyjuice as juice
import torch

from hmm_config import resolve, model_paths, r2_csv, ROOT

sys.setrecursionlimit(100000)


def vcf_to_haplotype_array(vcf_file):
    """Rows = samples, columns = SNPs. Same parser as predict_multi.py."""
    haplotypes = []
    with open(vcf_file, 'r') as file:
        for line in file:
            if line.startswith('##') or line.startswith('#CHROM'):
                continue
            fields = line.strip().split('\t')
            haplotypes.append([int(g) for g in fields[9:]])
    return np.array(haplotypes).T


def compute_r_squared_per_feature(true_values, predicted_values):
    """Per-column squared Pearson correlation, matching predict_multi.py."""
    num_features = true_values.shape[1]
    r2_scores = np.zeros(num_features)
    for i in range(num_features):
        cm = np.corrcoef(true_values[:, i], predicted_values[:, i])
        corr = cm[0, 1] if not np.isnan(cm[0, 1]) else 0
        r2_scores[i] = corr ** 2
    return r2_scores


def run_r2_on_data(valid_data, cfg, mask_indices, batch_size, device, single_model=None):
    """P(allele=1) for every masked SNP, one conditional query per chunk.

    Returns the dosage matrix restricted to `mask_indices`, in that order.
    """
    n, v = valid_data.shape
    width = v if single_model else v // cfg["num_chunks"]
    dosages = np.zeros((n, v), dtype=np.float32)

    ckpts = [single_model] if single_model else model_paths(cfg)
    for c, ckpt in enumerate(ckpts):
        lo, hi = (0, v) if single_model else (c * width, (c + 1) * width)
        local = [i - lo for i in mask_indices if lo <= i < hi]
        if not local:
            continue

        pc = juice.compile(juice.load(ckpt)).to(device)
        mask = torch.zeros(width, dtype=torch.bool, device=device)
        mask[torch.tensor(local, dtype=torch.long, device=device)] = True

        t0 = time.time()
        for s in range(0, n, batch_size):
            block = valid_data[s:s + batch_size, lo:hi]
            x = torch.tensor(block, dtype=torch.long, device=device)
            with torch.no_grad():
                probs = juice.queries.conditional(pc, data=x, missing_mask=mask)
            dosages[s:s + x.shape[0], lo:hi] = probs[:, :, 1].detach().cpu().numpy()
        print(f"  chunk {c}: {len(local)} masked SNPs in {time.time()-t0:.1f}s", flush=True)

        del pc
        torch.cuda.empty_cache()

    return dosages[:, mask_indices]


def main():
    p = argparse.ArgumentParser(description="Array-based direct imputation with the chunked HMM.")
    p.add_argument("config", help="dataset/split key, e.g. b38:8020")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--mask", default=f"{ROOT}/results/b38/missing_indices_hum5.txt")
    p.add_argument("--info", default=f"{ROOT}/aux/b38_legend.maf.txt")
    p.add_argument("--single-model", default=None,
                   help="use one unchunked circuit (GPC) instead of the per-chunk HMM models")
    p.add_argument("--test", default=None, help="override the target matrix")
    p.add_argument("--bootstrap-dir", default=None, help="override the bootstrap VCF directory")
    p.add_argument("--out", default=None)
    p.add_argument("--n-bootstraps", type=int, default=10)
    args = p.parse_args()

    cfg = resolve(args.config)
    assert cfg["dataset"] == "b38", "this script is for the array-based b38 experiments"
    device = torch.device("cuda")

    out_csv = args.out or r2_csv(cfg, "direct")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    missing = ([args.single_model] if args.single_model and not os.path.exists(args.single_model)
               else [] if args.single_model
               else [q for q in model_paths(cfg) if not os.path.exists(q)])
    if missing:
        raise SystemExit("missing chunk checkpoints:\n  " + "\n  ".join(missing))

    with open(args.mask) as f:
        mask_indices = [int(l.strip()) for l in f if l.strip()]
    print(f"Masking {len(mask_indices)} of {cfg['num_snps']} SNPs "
          f"across {cfg['num_chunks']} chunks")

    # Same two-column legend predict_multi.py reads (space separated).
    maf_df = pd.read_csv(args.info, sep=" ", header=None, names=["SNP", "MAF"])
    snp_list, maf_list = maf_df["SNP"].tolist(), maf_df["MAF"].tolist()

    print(f"Computing base R2 from {cfg['valid_path']}")
    valid = np.loadtxt(args.test or cfg["valid_path"], dtype=np.int8, delimiter=' ')
    r2_base = compute_r_squared_per_feature(
        valid[:, mask_indices].astype(float),
        run_r2_on_data(valid, cfg, mask_indices, args.batch_size, device, args.single_model))

    r2_boots = []
    for b in range(1, args.n_bootstraps + 1):
        f = f"{args.bootstrap_dir or cfg['bootstrap_dir']}/bootstrap_{b}.vcf"
        if not os.path.exists(f):
            print(f"  bootstrap {b}: {f} not found, skipping")
            continue
        print(f"Processing bootstrap_{b}.vcf")
        vb = vcf_to_haplotype_array(f)
        r2_boots.append(compute_r_squared_per_feature(
            vb[:, mask_indices].astype(float),
            run_r2_on_data(vb, cfg, mask_indices, args.batch_size, device, args.single_model)))

    rows = []
    for idx, mi in enumerate(mask_indices):
        row = {"SNP Set": snp_list[mi], "R2": r2_base[idx], "MAF": maf_list[mi]}
        for b, arr in enumerate(r2_boots, 1):
            row[f"R2_boot_{b}"] = arr[idx]
        rows.append(row)
    cols = (["SNP Set", "R2"] + [f"R2_boot_{b}" for b in range(1, len(r2_boots) + 1)] + ["MAF"])
    pd.DataFrame(rows, columns=cols).to_csv(out_csv, index=False)
    print(f"\nSaved per-SNP results to {out_csv}")

    if r2_boots:
        bm = [np.nanmean(a) for a in r2_boots]
        print(f"Base mean R2 = {np.nanmean(r2_base):.4f}")
        print(f"Bootstrap mean R2 = {np.mean(bm):.4f} +/- {1.96*np.std(bm, ddof=1):.4f} (95% CI)")


if __name__ == "__main__":
    main()
