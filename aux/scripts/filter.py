import pandas as pd

# -----------------------------
# Load MAC/MAF file
# -----------------------------
mac_file = "../b38_unrelated_mac_maf.txt"
hap_file = "../b38_unrelated_real.hapt"
output_file = "../b38_unrelated_filtered_real.hapt"

# -----------------------------
# Load MAC/MAF and determine SNPs to keep
# -----------------------------
mac_df = pd.read_csv(mac_file, sep="\t")
keep_snps = mac_df.loc[mac_df['mac'] > 20, 'snp'].tolist()

# -----------------------------
# Load haplotype file into DataFrame
# -----------------------------
# Assuming haplotype file has no header; first two columns are labels
hap_df = pd.read_csv(hap_file, sep="\s+", header=None)

# First two columns are sample identifiers
label_cols = hap_df.iloc[:, :2]

# SNP columns
snp_cols = hap_df.iloc[:, 2:]
snp_cols.columns = mac_df['snp']  # assign SNP names to columns

# Keep only SNPs with MAC > 20
filtered_snp_cols = snp_cols[keep_snps]

# Combine labels with filtered SNPs
filtered_hap_df = pd.concat([label_cols, filtered_snp_cols], axis=1)

# -----------------------------
# Save filtered haplotype file
# -----------------------------
filtered_hap_df.to_csv(output_file, sep=" ", header=False, index=False)

print(f"Filtered haplotype file saved to {output_file}")