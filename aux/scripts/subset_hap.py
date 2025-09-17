# Paths to your files
file_to_filter = "../b38_real.hapt"      # The haplotype file you want to subset
reference_file = "../10K_real.hapt" # The file with haplotypes to keep
output_file = "../b38_unrelated.hapt"       # Output file

# --- Load the reference haplotype names ---
reference_haplotypes = set()
with open(reference_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        reference_haplotypes.add(parts[1])  # Sample_Haplotype column

# --- Filter the full haplotype file ---
with open(file_to_filter, "r") as fin, open(output_file, "w") as fout:
    for line in fin:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        sample_hap = parts[1]
        if sample_hap in reference_haplotypes:
            fout.write(line)

print(f"Subset haplotype file saved as {output_file}")