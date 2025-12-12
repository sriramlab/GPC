import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------
# File paths
# -----------------------------
hap_file = "../b38_real_afr_smaller.hapt"       # Your haplotype file
train_file = "../../results/b38/afr/data/afr_train_smaller.txt"
test_file = "../../results/b38/afr/data/afr_test_smaller.txt"
test_size = 0.2                    # fraction for test set
random_state = 42                  # reproducibility

# -----------------------------
# Load haplotype file
# -----------------------------
# First two columns are identifiers
hap_df = pd.read_csv(hap_file, sep="\s+", header=None)
labels = hap_df.iloc[:, :2]      # sample identifiers
data = hap_df.iloc[:, 2:]        # SNP data only

# -----------------------------
# Extract unique individuals
# -----------------------------
# Assumes identifiers look like "HG00096_A" and "HG00096_B"
individuals = labels[1].str.replace(r'_[AB]$', '', regex=True).unique()

# -----------------------------
# Train-test split at individual level
# -----------------------------
train_inds, test_inds = train_test_split(individuals, test_size=test_size, random_state=random_state)

# Select rows corresponding to train/test individuals
train_mask = labels[1].str.replace(r'_[AB]$', '', regex=True).isin(train_inds)
test_mask = labels[1].str.replace(r'_[AB]$', '', regex=True).isin(test_inds)

train_data = data[train_mask].reset_index(drop=True)
test_data = data[test_mask].reset_index(drop=True)

# -----------------------------
# Save to files (SNP data only)
# -----------------------------
train_data.to_csv(train_file, sep=" ", header=False, index=False)
test_data.to_csv(test_file, sep=" ", header=False, index=False)

print(f"Train and test haplotype files saved to:\n{train_file}\n{test_file}")