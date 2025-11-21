# compare_positions.py

file1 = "/scratch2/prateek/genetic_pc_github/results/b38/array_subset_no_dups_hum5.txt"
file2 = "/scratch2/prateek/genetic_pc_github/aux/b38_legend.maf.txt"
output = "missing_indices_hum5.txt"

# Get all positions from file1 (column 3)
positions1 = set()
with open(file1) as f1:
    for line in f1:
        if line.strip():
            parts = line.split()
            if len(parts) >= 3:
                positions1.add(parts[2])

# Compare with positions from file2
missing_indices = []
with open(file2) as f2:
    for idx, line in enumerate(f2):
        if line.strip():
            pos = line.split()[0].split(":")[1]
            if pos not in positions1:
                missing_indices.append(idx)

# Write indices to output
with open(output, "w") as out:
    for i in missing_indices:
        out.write(f"{i}\n")

print(f"Found {len(missing_indices)} missing positions. Written to {output}.")