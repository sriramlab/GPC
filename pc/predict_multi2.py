import pyjuice as juice
import torch
# import torchvision
import time
from torch.utils.data import TensorDataset, DataLoader
import pyjuice.nodes.distributions as dists
import numpy as np
import pandas as pd

import seaborn as sns
import pandas as pd
import numpy as np
import importlib
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from bed_reader import open_bed
import gc
from joblib import Parallel, delayed

def vcf_to_haplotype_array(vcf_file):
    haplotypes = []

    with open(vcf_file, 'r') as file:
        for line in file:
            # Skip header lines starting with '##' or the CHROM/POS line
            if line.startswith('##'):
                continue
            elif line.startswith('#CHROM'):
                # The first line starting with '#CHROM' contains sample names
                header = line.strip().split('\t')
                continue

            # Extract genotype data for each SNP
            fields = line.strip().split('\t')
            genotypes = fields[9:]  # Genotype data starts from the 10th column
            
            # Convert genotypes to integers (0 or 1)
            genotype_row = [int(genotype) for genotype in genotypes]
            
            # Append the row for this SNP
            haplotypes.append(genotype_row)

    # Convert the list of haplotypes into a NumPy array
    haplotype_array = np.array(haplotypes)
    
    # Return as-is (rows = SNPs, columns = samples)
    return haplotype_array.T

print("Number of CUDA devices:", torch.cuda.device_count())
print(torch.version.cuda)

device = torch.device("cuda:0")
np.random.seed(1)

print(device)

def compute_r_squared_per_feature(true_values, predicted_values):
    true_values = true_values.cpu().numpy()
    predicted_values = predicted_values.cpu().numpy()
    num_features = true_values.shape[1]
    r2_scores = np.zeros(num_features)
    for i in range(num_features):
        correlation_matrix = np.corrcoef(true_values[:, i], predicted_values[:, i])
        correlation_xy = correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0
        r2_scores[i] = correlation_xy ** 2
    return r2_scores

def run_r2_on_data_with_mask(valid_data_tensor, pc_model, mask_indices_file):
    # === Load indices to mask ===
    with open(mask_indices_file, "r") as f:
        mask_indices = [int(line.strip()) for line in f if line.strip()]
    mask_indices = torch.tensor(mask_indices, dtype=torch.long)

    dataset = TensorDataset(valid_data_tensor)
    dataloader = DataLoader(dataset, batch_size=128)
    num_samples = valid_data_tensor.size(0)
    num_features = valid_data_tensor.size(1)

    all_expected_counts = torch.zeros(num_samples, num_features, device="cpu")
    all_originals = torch.zeros(num_samples, num_features, device="cpu")

    batch_start = 0
    for batch in dataloader:
        data = batch[0].to(device)
        batch_size = data.size(0)
        batch_end = batch_start + batch_size

        # Build missing mask (same for all samples in the batch)
        missing_mask = torch.zeros(num_features, dtype=torch.bool, device=device)
        missing_mask[mask_indices] = True

        # Query conditional distribution for all missing features simultaneously
        with torch.cuda.device("cuda:0"):
            lls = juice.queries.conditional(pc_model, data=data, missing_mask=missing_mask)
        # shape: [batch_size, num_missing, 2]
        probs = lls[:, :, :]

        # Fill expected counts for masked features only
        for feature_pos in mask_indices:
            original = data[:, feature_pos]
            prob_0 = probs[:, feature_pos, 0]
            prob_1 = probs[:, feature_pos, 1]
            expected_counts = prob_0 * 0 + prob_1 * 1

            all_expected_counts[batch_start:batch_end, feature_pos] = expected_counts.detach().cpu()
            all_originals[batch_start:batch_end, feature_pos] = original.detach().cpu()

        batch_start = batch_end

    # Compute R2 only on masked features
    r2s = compute_r_squared_per_feature(all_originals[:, mask_indices], 
                                        all_expected_counts[:, mask_indices])
    return mask_indices, r2s

# === Load and compile PC model ===
pc_path = '/scratch2/prateek/genetic_pc_github/results/b38/noneur/hclt/pc_14670_noneur_3202-128_5000epochs_ps0.005_2.jpc'
pc_model = juice.compile(juice.load(pc_path))
pc_model.to(device)

# === Load mask indices once ===
mask_file = "/scratch2/prateek/genetic_pc_github/results/b38/missing_indices_hum5.txt"
with open(mask_file, "r") as f:
    mask_indices = [int(line.strip()) for line in f if line.strip()]

# === Load MAF file (2-column: SNP <tab> MAF) into lists ===
maf_df = pd.read_csv(
    "/scratch2/prateek/genetic_pc_github/aux/b38_legend.maf.txt",
    sep=" ", header=None, names=["SNP", "MAF"]
)
snp_list = maf_df["SNP"].tolist()
maf_list = maf_df["MAF"].tolist()

# === Step 1: Compute base R2 ===
print("Computing base R2...")
base_data_path = '/scratch2/prateek/genetic_pc_github/results/b38/noneur/data/noneur_test.txt'
valid_data = np.loadtxt(base_data_path, dtype=np.int8, delimiter=' ')
valid_data_tensor = torch.tensor(valid_data, dtype=torch.long)

_, r2_base = run_r2_on_data_with_mask(valid_data_tensor, pc_model, mask_file)

# === Step 2: Compute bootstrap R2s ===
bootstrap_dir = '/scratch2/prateek/genetic_pc_github/results/b38/noneur/data/test_bootstraps'
r2_boots = []

for boot_id in range(1, 11):
    print(f"Processing bootstrap_{boot_id}.vcf")
    file_path = f"{bootstrap_dir}/bootstrap_{boot_id}.vcf"
    valid_data = vcf_to_haplotype_array(file_path)
    valid_data_tensor = torch.tensor(valid_data, dtype=torch.long)
    _, r2_boot = run_r2_on_data_with_mask(valid_data_tensor, pc_model, mask_file)
    r2_boots.append(r2_boot)

r2_boot_array = np.stack(r2_boots, axis=-1)  # shape (num_masked_features, 10)

# === Step 3: Build final DataFrame in target format using mask indices ===
rows = []
for idx, mask_idx in enumerate(mask_indices):
    snp_id = snp_list[mask_idx]
    snp_maf = maf_list[mask_idx]

    row = {
        "SNP Set": snp_id,
        "R2": r2_base[idx],
        "MAF": snp_maf
    }
    for boot_id in range(1, 11):
        row[f"R2_boot_{boot_id}"] = r2_boot_array[idx, boot_id - 1]
    rows.append(row)

output_df = pd.DataFrame(rows, columns=[
    "SNP Set", "R2",
    "R2_boot_1","R2_boot_2","R2_boot_3","R2_boot_4","R2_boot_5",
    "R2_boot_6","R2_boot_7","R2_boot_8","R2_boot_9","R2_boot_10",
    "MAF"
])

output_csv = "/scratch2/prateek/genetic_pc_github/plots/impute/results/multi/noneur_multi_pc_b38_chr15_results_hum5.csv"
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
output_df.to_csv(output_csv, index=False)
print(f"\nSaved per-SNP results to {output_csv}")

# === Step 4: Print averages ===
mean_base = np.nanmean(r2_base)
boot_means = [np.nanmean(r2_boot_array[:, j]) for j in range(r2_boot_array.shape[1])]
mean_boot = np.mean(boot_means)
sem_boot = np.std(boot_means, ddof=1)
ci_boot = 1.96 * sem_boot

print("\n=== Results ===")
print(f"Base mean R2 = {mean_base:.4f}")
print(f"Bootstrap mean R2 = {mean_boot:.4f} ± {ci_boot:.4f} (95% CI)")