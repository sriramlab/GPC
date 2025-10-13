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
        description="Run imputation and per-SNP R² evaluation pipeline."
    )

    parser.add_argument(
        "--threads", type=int, default=1, required=False,
        help="Number of threads to use for bcftools/impute5."
    )
    parser.add_argument(
        "--train", type=str, required=True,
        help="Full path prefix for PLINK training dataset (no extension)."
    )
    parser.add_argument(
        "--test", type=str, required=True,
        help="Full path prefix for PLINK test dataset (no extension)."
    )
    parser.add_argument(
        "--chr", type=int, required=True,
        help="Chromosome number for the region being analyzed."
    )
    parser.add_argument(
        "--out", type=str, required=True,
        help="Full output prefix for saving results (e.g. /scratch2/prateek/output/run1)."
    )
    parser.add_argument(
        "--snp_index_file", type=str, required=True,
        help="Path to file containing indices of SNPs to mask/drop."
    )
    parser.add_argument(
        "--info_file", type=str, required=True,
        help="Path to file containing MAF or SNP info for annotation."
    )

    return parser.parse_args()

# ============================================================
# Core helper functions
# ============================================================
def process_plink_data(plink_prefix):
    """Convert PLINK .bed/.bim/.fam to .vcf and .bcf with allele count tags."""
    vcf_file = f"{plink_prefix}.vcf"
    bcf_file = f"{plink_prefix}.bcf"
    ac_bcf_file = f"{plink_prefix}_AC.bcf"

    if not os.path.exists(vcf_file):
        subprocess.run([
            "../plink2", "--bfile", plink_prefix,
            "--recode", "vcf", "--out", plink_prefix,
            "--silent"
        ], check=True)

    if not os.path.exists(bcf_file):
        subprocess.run([
            "/scratch2/prateek/bcftools/bcftools", "view",
            "-Ou", "-o", bcf_file, vcf_file
        ], check=True)
        subprocess.run([
            "/scratch2/prateek/bcftools/bcftools", "+fill-tags",
            bcf_file, "-Ou", "-o", ac_bcf_file, "--", "-t", "AN,AC"
        ], check=True)
        subprocess.run([
            "/scratch2/prateek/bcftools/bcftools", "index", ac_bcf_file
        ], check=True)

def process_plink_data_with_drop(plink_prefix, rs_ids, temp_dir, threads):
    """Remove given rsIDs from VCF and regenerate a valid .bcf file."""
    vcf_file = f"{plink_prefix}.vcf"
    modified_vcf = f"{temp_dir}/modified_test.vcf"

    with open(vcf_file, 'r') as f, open(modified_vcf, 'w') as out:
        for line in f:
            if line.startswith('#'):
                out.write(line)
                continue
            fields = line.strip().split('\t')
            if fields[2] not in rs_ids:
                out.write(line)

    subprocess.run([
        "/scratch2/prateek/bcftools/bcftools", "view",
        "-Ou", "--threads", str(threads),
        "-o", f"{temp_dir}/modified_test.bcf", modified_vcf
    ], check=True)
    subprocess.run([
        "/scratch2/prateek/bcftools/bcftools", "+fill-tags",
        f"{temp_dir}/modified_test.bcf", "-Ou", "-o",
        f"{temp_dir}/modified_test_AC.bcf", "--", "-t", "AN,AC"
    ], check=True)
    subprocess.run([
        "/scratch2/prateek/bcftools/bcftools", "index",
        f"{temp_dir}/modified_test_AC.bcf"
    ], check=True)

def extract_rs_ids_from_vcf(vcf_file):
    rs_ids = []
    with open(vcf_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            rs_ids.append(line.split('\t')[2])
    return rs_ids

def analyze_vcf(vcf_file):
    min_pos, max_pos, num_snps, num_samples = float('inf'), float('-inf'), 0, 0
    with open(vcf_file, 'r') as f:
        for line in f:
            if line.startswith('##'):
                continue
            if line.startswith('#CHROM'):
                num_samples = len(line.strip().split('\t')) - 9
                continue
            pos = int(line.split('\t')[1])
            min_pos, max_pos = min(min_pos, pos), max(max_pos, pos)
            num_snps += 1
    return min_pos, max_pos, num_snps, num_samples

def extract_test_genotype_array(vcf_file, snp_set):
    results = {}
    with open(vcf_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            snp_id = f"{fields[0]}:{fields[1]}"
            if snp_id in snp_set:
                genotype_data = fields[9:]
                genotype_array = [0 if g == "0" else 1 if g == "1" else -1 for g in genotype_data]
                results[snp_id] = genotype_array
    return results

def extract_imputed_genotype_array(vcf_file, snp_set, correct_genotype_array):
    results = {}
    with open(vcf_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            snp_id = f"{fields[0]}:{fields[1]}"
            if snp_id in snp_set:
                genotype_data = fields[9:]
                prob1 = []
                for g in genotype_data:
                    _, _, gp_info = g.split(':')
                    probs = list(map(float, gp_info.split(',')))
                    prob1.append(probs[1])
                results[snp_id] = prob1
    return results

def compute_r2(y_true, y_pred):
    a, b = np.array(y_true), np.array(y_pred)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1] ** 2)

# ============================================================
# Main pipeline
# ============================================================
def run_imputation_and_eval(test_vcf, snp_indices, rs_ids, chrnum,
                            train_prefix, test_prefix, temp_dir, threads):
    idx1, idx2, _, _ = analyze_vcf(test_vcf)
    region = f"{chrnum}:{idx1}-{idx2}"
    snp_set = [rs_ids[i] for i in snp_indices]

    print(f"Masking {len(snp_set)} SNPs")

    process_plink_data(train_prefix)
    process_plink_data_with_drop(test_prefix, snp_set, temp_dir, threads)

    subprocess.run([
        '/scratch2/prateek/impute5_v1.2.0/impute5_v1.2.0_static',
        '--h', f'{train_prefix}_AC.bcf',
        '--m', f'/scratch2/prateek/b37_recombination_maps/chr{chrnum}.b38.gmap.gz',
        '--g', f'{temp_dir}/modified_test_AC.bcf',
        '--r', region,
        '--buffer-region', region,
        '--o', f"{temp_dir}/imputed_custom.vcf",
        '--l', f"{temp_dir}/imputed_custom.log",
        '--haploid',
        '--threads', str(threads)
    ], check=True)

    test_arrays = extract_test_genotype_array(test_vcf, snp_set)
    imputed = extract_imputed_genotype_array(f"{temp_dir}/imputed_custom.vcf", snp_set, test_arrays)

    per_snp_r2 = {}
    for snp in snp_set:
        per_snp_r2[snp] = compute_r2(test_arrays[snp], imputed[snp])
    return per_snp_r2

# ============================================================
# Entrypoint
# ============================================================
def main():
    args = parse_args()
    os.environ['BCFTOOLS_PLUGINS'] = '/scratch2/prateek/bcftools/plugins'

    temp_dir = tempfile.mkdtemp(prefix="temp_impute_")

    rs_ids = extract_rs_ids_from_vcf(f"{args.test}.vcf")
    snp_indices = [int(line.strip()) for line in open(args.snp_index_file) if line.strip()]
    maf_list = pd.read_csv(args.info_file, sep=" ", header=None, usecols=[1]).squeeze("columns").tolist()

    print(f"Found {len(rs_ids)} SNPs, dropping {len(snp_indices)}")

    # Base
    base_r2s = run_imputation_and_eval(f"{args.test}.vcf", snp_indices, rs_ids, args.chr,
                                       args.train, args.test, temp_dir, args.threads)

    # Bootstraps
    bootstrap_r2s = []
    base_dir = os.path.dirname(args.test)
    for i in range(1, 11):
        boot_vcf = os.path.join(base_dir, f"test_bootstraps/bootstrap_{i}.vcf")
        boot_prefix = os.path.splitext(boot_vcf)[0]
        print(f"Running bootstrap {i}...")
        boot_r2s = run_imputation_and_eval(boot_vcf, snp_indices, rs_ids, args.chr,
                                           args.train, boot_prefix, temp_dir, args.threads)
        bootstrap_r2s.append(boot_r2s)

    # Combine results
    rows = []
    for idx, snp in enumerate(base_r2s.keys()):
        row = {"SNP": snp, "R2": base_r2s[snp]}
        for j, boot_dict in enumerate(bootstrap_r2s, 1):
            row[f"R2_boot_{j}"] = boot_dict.get(snp, np.nan)
        maf_idx = snp_indices[idx] if idx < len(snp_indices) else None
        row["MAF"] = maf_list[maf_idx] if maf_idx is not None else np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = f"{args.out}_chr{args.chr}_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")

    # Summary stats
    base_mean = np.nanmean(list(base_r2s.values()))
    boot_means = [np.nanmean(list(b.values())) for b in bootstrap_r2s]
    ci = 1.96 * np.std(boot_means, ddof=1)
    print(f"\nBase mean R²={base_mean:.4f}, Bootstrap mean={np.mean(boot_means):.4f} ± {ci:.4f}")

    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
# %%
'''
example: python3 impute5_multi.py --train /scratch2/prateek/genetic_pc_github/results/b38/8020/hclt/b38_hclt_8020_samples_2 --test /scratch2/prateek/genetic_pc_github/results/b38/8020/data/8020_test --chr 15 --out /scratch2/prateek/genetic_pc_github/plots/impute/results/multi/8020_multi_hclt_b38_2 --snp_index_file /scratch2/prateek/genetic_pc_github/results/b38/missing_indices.txt --info_file /scratch2/prateek/genetic_pc_github/aux/b38_legend.maf.txt
'''