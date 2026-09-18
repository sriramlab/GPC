"""Assembles per-chunk imputation results into one r2 table."""

import argparse
import os

import numpy as np
import pandas as pd

from hmm_config import resolve


def r2_per_snp(truth, pred):
    """Vectorised squared Pearson correlation per column (same as pc/predict.py)."""
    t = truth.astype(np.float64)
    p = pred.astype(np.float64)
    tc = t - t.mean(axis=0)
    pc_ = p - p.mean(axis=0)
    numer = (tc * pc_).sum(axis=0)
    denom = np.sqrt((tc ** 2).sum(axis=0) * (pc_ ** 2).sum(axis=0))
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = np.where(denom > 0, numer / denom, 0.0)
    return corr ** 2


def main():
    p = argparse.ArgumentParser(description="Build the R2/bootstrap CSV from a dosage matrix.")
    p.add_argument("config", help="dataset/split key, e.g. 1KG:afr")
    p.add_argument("--dosages", default=None, help="input .npy (default: predict_hmm.py output)")
    p.add_argument("--out", required=True, help="output CSV, e.g. .../results/r2/pc_hmm_afr_afr.csv")
    p.add_argument("--test", default=None,
                   help="override the truth matrix (e.g. the admixed subset)")
    p.add_argument("--bootstrap-dir", default=None,
                   help="override the directory of indices_N.txt")
    p.add_argument("--n-bootstraps", type=int, default=10)
    args = p.parse_args()

    cfg = resolve(args.config)
    dos_path = args.dosages or f"{cfg['model_dir']}/hmm_{cfg['split']}_direct_dosages.npy"

    pred = np.load(dos_path)
    truth = np.loadtxt(args.test or cfg["valid_path"], dtype=np.int8, delimiter=' ')
    assert pred.shape == truth.shape, f"{pred.shape} dosages vs {truth.shape} truth"

    legend = pd.read_csv(cfg["legend_maf"], sep="\t", header=None, names=["SNP Set", "MAF"])
    assert len(legend) == truth.shape[1], \
        f"{len(legend)} SNPs in {cfg['legend_maf']} vs {truth.shape[1]} in the data"

    out = legend[["SNP Set"]].copy()
    out["R2"] = r2_per_snp(truth, pred)

    for b in range(1, args.n_bootstraps + 1):
        idx_file = f"{args.bootstrap_dir or cfg['bootstrap_dir']}/indices_{b}.txt"
        if not os.path.exists(idx_file):
            raise SystemExit(f"missing {idx_file}; run aux/scripts/recover_bootstrap_indices.py first")
        idx = np.loadtxt(idx_file, dtype=int)
        out[f"R2_boot_{b}"] = r2_per_snp(truth[idx], pred[idx])

    out["MAF"] = legend["MAF"].values

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)

    boot_cols = [f"R2_boot_{b}" for b in range(1, args.n_bootstraps + 1)]
    keep = out[out["MAF"] > 0]
    means = keep[boot_cols].mean()
    print(f"wrote {args.out}  ({len(out)} SNPs)")
    print(f"  mean R2 (MAF>0)      {keep['R2'].mean():.4f}")
    print(f"  bootstrap mean       {means.mean():.4f} +/- {1.96*means.std(ddof=1):.4f}")
    print(f"  low-freq  (<1%)      {keep[keep['MAF'] < 0.01][boot_cols].mean().mean():.4f}")
    print(f"  rare      (<0.1%)    {keep[keep['MAF'] < 0.001][boot_cols].mean().mean():.4f}")


if __name__ == "__main__":
    main()
