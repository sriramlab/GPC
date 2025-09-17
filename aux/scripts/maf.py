import pandas as pd
import numpy as np

# --- Input/Output ---
data = "b38_unrelated"
infile = f"/scratch2/prateek/genetic_pc_github/aux/{data}_real.hapt"   # your haplotype file
outfile = f"../{data}_mac_maf.txt" # output file

# --- Load haplotypes ---
# First two columns are text (metadata), so skip them
df = pd.read_csv(infile, sep=r"\s+", header=None)
haplotypes = df.iloc[:, 2:].to_numpy()  # take all columns from the 3rd onward

# --- Calculate MAC and MAF ---
num_haplotypes = haplotypes.shape[0]
snp_ids = [f"SNP{i+1}" for i in range(haplotypes.shape[1])]

macs = []
mafs = []

for j in range(haplotypes.shape[1]):
    allele_counts = np.bincount(haplotypes[:, j], minlength=2)  # count 0s and 1s
    ref_count, alt_count = allele_counts
    mac = min(ref_count, alt_count)
    maf = mac / num_haplotypes
    macs.append(mac)
    mafs.append(maf)

# --- Save results ---
out_df = pd.DataFrame({
    "snp": snp_ids,
    "mac": macs,
    "maf": mafs
})
out_df.to_csv(outfile, sep="\t", index=False)

print(f"Saved {outfile} with {len(snp_ids)} SNPs")