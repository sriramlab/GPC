# %%
import argparse
import numpy as np
import pandas as pd
import subprocess
import os
import shutil
import tempfile

# ============================================================
# Argument parser
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run impute5 and per-SNP R² evaluation using VCF.GZ inputs."
    )

    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--train_vcf", type=str, required=True,
                        help="Training VCF (.vcf.gz)")
    parser.add_argument("--test_vcf", type=str, required=True,
                        help="Test VCF (.vcf.gz)")
    parser.add_argument("--chr", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--snp_index_file", type=str, required=True)
    parser.add_argument("--info_file", type=str, required=True)

    return parser.parse_args()

# ============================================================
# bcftools helpers
# ============================================================
BCFTOOLS = "/scratch2/prateek/bcftools/bcftools"
IMPUTE5 = "/scratch2/prateek/impute5_v1.2.0/impute5_v1.2.0_static"
GMAP = "/scratch2/prateek/b37_recombination_maps"

def run(cmd):
    subprocess.run(cmd, check=True)

def vcf_to_ac_bcf(vcf_path, out_prefix, threads):
    """Convert VCF/VCF.GZ → AC-tagged BCF"""
    bcf = f"{out_prefix}.bcf"
    ac_bcf = f"{out_prefix}_AC.bcf"

    if not os.path.exists(ac_bcf):
        run([
            BCFTOOLS, "view",
            "-Ob", "--threads", str(threads),
            "-o", bcf, vcf_path
        ])
        run([
            BCFTOOLS, "+fill-tags",
            bcf, "-Ob", "-o", ac_bcf,
            "--", "-t", "AN,AC"
        ])
        run([BCFTOOLS, "index", ac_bcf])

    return ac_bcf

def drop_snps_from_vcf(vcf_path, snp_ids, temp_dir, threads):
    """Remove SNPs by ID from VCF"""
    exclude = os.path.join(temp_dir, "exclude_snps.txt")
    with open(exclude, "w") as f:
        for s in snp_ids:
            f.write(s + "\n")

    bcf = os.path.join(temp_dir, "masked.bcf")
    ac_bcf = os.path.join(temp_dir, "masked_AC.bcf")

    run([
        BCFTOOLS, "view",
        "-Ob", "--threads", str(threads),
        "-e", f"ID=@{exclude}",
        "-o", bcf,
        vcf_path
    ])
    run([
        BCFTOOLS, "+fill-tags",
        bcf, "-Ob", "-o", ac_bcf,
        "--", "-t", "AN,AC"
    ])
    run([BCFTOOLS, "index", ac_bcf])

    return ac_bcf

def get_region(vcf_path):
    """Get min/max position from VCF"""
    p = subprocess.run(
        [BCFTOOLS, "query", "-f", "%POS\n", vcf_path],
        capture_output=True, text=True, check=True
    )
    pos = list(map(int, p.stdout.splitlines()))
    return min(pos), max(pos)

def extract_snp_ids(vcf_path):
    """Extract SNP IDs (column 3) from VCF - matches original behavior"""
    p = subprocess.run(
        [BCFTOOLS, "query", "-f", "%ID\n", vcf_path],
        capture_output=True, text=True, check=True
    )
    return p.stdout.splitlines()

# ============================================================
# Genotype extraction - aligned with original
# ============================================================
def extract_true_genotypes(vcf_path, snp_set, debug_snp=None):
    """
    Extract hard genotypes (0/1) for masked SNPs.
    Uses %ID for SNP matching (like original) and handles haploid genotypes.
    """
    data = {}
    p = subprocess.Popen(
        [BCFTOOLS, "query",
         "-f", "%ID[\\t%GT]\\n",
         vcf_path],
        stdout=subprocess.PIPE,
        text=True
    )

    for line in p.stdout:
        fields = line.strip().split("\t")
        snp = fields[0]
        if snp not in snp_set:
            continue
        
        if snp == debug_snp:
            print(f"DEBUG true GT for {snp}: first 10 raw values = {fields[1:11]}")
        
        gts = np.empty(len(fields) - 1, dtype=np.int8)
        for i, g in enumerate(fields[1:]):
            # Handle both haploid (0, 1) and diploid (0/0, 0|0, 1/1, 1|1)
            if g in ("0", "0|0", "0/0"):
                gts[i] = 0
            elif g in ("1", "1|1", "1/1"):
                gts[i] = 1
            else:
                gts[i] = -1
        
        if snp == debug_snp:
            print(f"DEBUG true GT for {snp}: first 20 parsed = {gts[:20]}")
            unique, counts = np.unique(gts, return_counts=True)
            print(f"DEBUG true GT for {snp}: unique values = {dict(zip(unique, counts))}")
        
        data[snp] = gts
    
    p.wait()
    return data

def extract_imputed_probs(vcf_path, snp_set, debug_snp=None):
    """
    Extract imputed probabilities.
    For haploid (2-value GP): returns P(1)
    For diploid (3-value GP): returns P(het) which is what original code does
    """
    data = {}
    p = subprocess.Popen(
        [BCFTOOLS, "query",
         "-f", "%ID[\\t%GP]\\n",
         vcf_path],
        stdout=subprocess.PIPE,
        text=True
    )

    for line in p.stdout:
        fields = line.strip().split("\t")
        snp = fields[0]
        if snp not in snp_set:
            continue
        
        if snp == debug_snp:
            print(f"DEBUG imputed GP for {snp}: first 10 raw values = {fields[1:11]}")
        
        probs = np.empty(len(fields) - 1, dtype=np.float32)
        for i, gp in enumerate(fields[1:]):
            parts = gp.split(",")
            # Original code does: probs[1] - takes second value
            # 2-value GP (haploid): P(0), P(1) -> parts[1] = P(1) ✓
            # 3-value GP (diploid): P(00), P(01), P(11) -> parts[1] = P(01) ✗
            # For diploid, we want dosage = P(01) + 2*P(11)
            if len(parts) == 2:
                probs[i] = float(parts[1])
            elif len(parts) == 3:
                # Dosage for diploid
                probs[i] = float(parts[1]) + 2.0 * float(parts[2])
            else:
                probs[i] = np.nan
        
        if snp == debug_snp:
            print(f"DEBUG imputed probs for {snp}: first 20 = {probs[:20]}")
            print(f"DEBUG imputed probs for {snp}: min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")
            print(f"DEBUG GP field has {len(fields[1].split(','))} values (2=haploid, 3=diploid)")
        
        data[snp] = probs
    
    p.wait()
    return data

def compute_r2(a, b):
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    
    # Filter out missing values
    valid = (a >= 0) & ~np.isnan(b)
    a = a[valid]
    b = b[valid]
    
    if len(a) == 0 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1] ** 2)

# ============================================================
# Main imputation + evaluation
# ============================================================
def run_impute_eval(train_vcf, test_vcf, masked_snps, chrnum, temp_dir, threads):
    min_pos, max_pos = get_region(test_vcf)
    region = f"{chrnum}:{min_pos}-{max_pos}"

    print(f"Masking {len(masked_snps)} SNPs")

    train_ac = vcf_to_ac_bcf(train_vcf, os.path.join(temp_dir, "train"), threads)
    test_ac = drop_snps_from_vcf(test_vcf, masked_snps, temp_dir, threads)

    imputed_vcf = os.path.join(temp_dir, "imputed.bcf")

    run([
        IMPUTE5,
        "--h", train_ac,
        "--g", test_ac,
        "--m", f"{GMAP}/chr{chrnum}.b37.gmap.gz",
        "--r", region,
        "--buffer-region", region,
        "--o", imputed_vcf,
        "--threads", str(threads),
        "--haploid"
    ])

    print("IMPUTE5 finished successfully", flush=True)

    # Debug with first masked SNP
    debug_snp = masked_snps[0]
    print(f"\n{'='*60}")
    print(f"DEBUGGING SNP: {debug_snp}")
    print(f"{'='*60}")
    
    snp_set = set(masked_snps)
    true_gt = extract_true_genotypes(test_vcf, snp_set, debug_snp=debug_snp)
    imp_probs = extract_imputed_probs(imputed_vcf, snp_set, debug_snp=debug_snp)
    
    print(f"\nSNPs found in true_gt: {len(true_gt)}")
    print(f"SNPs found in imp_probs: {len(imp_probs)}")
    
    if debug_snp in true_gt and debug_snp in imp_probs:
        print(f"\nSamples in true_gt[{debug_snp}]: {len(true_gt[debug_snp])}")
        print(f"Samples in imp_probs[{debug_snp}]: {len(imp_probs[debug_snp])}")
        
        # Show correlation for debug SNP
        r2 = compute_r2(true_gt[debug_snp], imp_probs[debug_snp])
        print(f"R² for {debug_snp}: {r2:.6f}")
    else:
        print(f"WARNING: Debug SNP not found! In true_gt: {debug_snp in true_gt}, In imp_probs: {debug_snp in imp_probs}")
        print(f"First 5 SNPs in true_gt: {list(true_gt.keys())[:5]}")
        print(f"First 5 SNPs in imp_probs: {list(imp_probs.keys())[:5]}")
    
    print(f"{'='*60}\n")

    results = {}
    for snp in masked_snps:
        if snp in true_gt and snp in imp_probs:
            results[snp] = compute_r2(true_gt[snp], imp_probs[snp])
        else:
            results[snp] = np.nan
    
    return results

# ============================================================
# Entrypoint
# ============================================================
def main():
    args = parse_args()
    os.environ["BCFTOOLS_PLUGINS"] = "/scratch2/prateek/bcftools/plugins"

    temp_dir = tempfile.mkdtemp(prefix="impute5_", dir="/scratch2/prateek/tmp")

    # Get SNP IDs from VCF (like original code)
    all_snp_ids = extract_snp_ids(args.test_vcf)
    print(f"Total SNPs in test VCF: {len(all_snp_ids)}")
    
    # Load MAF info
    legend = pd.read_csv(args.info_file, sep=r"\s+", header=None, names=["SNP", "MAF"])
    maf_lookup = dict(zip(legend["SNP"], legend["MAF"]))

    # Get indices of SNPs to mask
    snp_indices = [int(x.strip()) for x in open(args.snp_index_file)]
    
    # Get masked SNP IDs from VCF (like original)
    masked_snps = [all_snp_ids[i] for i in snp_indices]
    
    print(f"First 5 masked SNPs: {masked_snps[:5]}")

    base_r2 = run_impute_eval(
        args.train_vcf,
        args.test_vcf,
        masked_snps,
        args.chr,
        temp_dir,
        args.threads
    )

    rows = []
    for snp in masked_snps:
        rows.append({
            "SNP": snp,
            "R2": base_r2.get(snp, np.nan),
            "MAF": maf_lookup.get(snp, np.nan)
        })

    df = pd.DataFrame(rows)
    out_csv = f"{args.out}_chr{args.chr}_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    # Summary statistics
    valid_r2 = df["R2"].dropna()
    print(f"\n{'='*40}")
    print(f"SUMMARY")
    print(f"{'='*40}")
    print(f"Total SNPs: {len(df)}")
    print(f"SNPs with valid R²: {len(valid_r2)}")
    print(f"Average R²: {valid_r2.mean():.6f}")
    print(f"Median R²: {valid_r2.median():.6f}")
    print(f"R² > 0.5: {(valid_r2 > 0.5).sum()} ({100*(valid_r2 > 0.5).mean():.1f}%)")
    print(f"R² > 0.8: {(valid_r2 > 0.8).sum()} ({100*(valid_r2 > 0.8).mean():.1f}%)")
    print(f"{'='*40}")

    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()