# %%
import pyjuice as juice
import torch
import time
from torch.utils.data import TensorDataset, DataLoader
import pyjuice.nodes.distributions as dists
import numpy as np
import pandas as pd
import numpy as np
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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

snps = "14670"
amt = 1056
data = "b38"
split = "afr"

latents = 128
ps = 0.005
num_epochs = 5000
batch_size = 256


print("Number of CUDA devices:", torch.cuda.device_count())
print(torch.version.cuda)

device = torch.device("cuda:1")
np.random.seed(1)

print(device)

train_data = np.loadtxt(f"../results/{data}/{split}/data/{split}_train.txt", dtype=np.int8, delimiter=' ')
valid_data = np.loadtxt(f"../results/{data}/{split}/data/{split}_test.txt", dtype=np.int8, delimiter=' ')

train_data = torch.tensor(train_data, dtype=torch.long)
valid_data = torch.tensor(valid_data, dtype=torch.long)

print(train_data.shape)
print(valid_data.shape)
# %%

train_loader = DataLoader(
    dataset = TensorDataset(train_data),
    batch_size = batch_size,
    shuffle = True,
    drop_last = True
)
valid_loader = DataLoader(
    dataset = TensorDataset(valid_data),
    batch_size = batch_size,
    shuffle = False,
    drop_last = True
)

ns = juice.structures.HCLT(
    train_data[:amt].float().to(device),
    num_latents = latents,
    input_dist=dists.Categorical(num_cats=2),
)

pc = juice.compile(ns)
pc.to(device)

with torch.cuda.device(pc.device):
    for batch in train_loader:
        x = batch[0].to(device)
        lls = pc(x, record_cudagraph = True)
        lls.mean().backward()

    log_filename = f"../results/{data}/{split}/hclt/{snps}_{split}_{amt}_{latents}_{num_epochs}epochs_ps{ps}.log"
    with open(log_filename, "w") as log_file:
        for epoch in range(1, num_epochs+1):
            t0 = time.time()

            # Manually zeroing out the flows
            pc.init_param_flows(flows_memory = 0.0)

            train_ll = 0.0
            for batch in train_loader:
                x = batch[0].to(device)

                # We only run the forward and the backward pass, and accumulate the flows throughout the epoch
                lls = pc(x)
                lls.mean().backward()

                train_ll += lls.mean().detach().cpu().numpy().item()

            # Set step size to 1.0 for full-batch EM
            pc.mini_batch_em(step_size = 1.0, pseudocount = ps)

            train_ll /= len(train_loader)

            t1 = time.time()
            test_ll = 0.0
            for batch in valid_loader:
                x = batch[0].to(pc.device)
                lls = pc(x)
                test_ll += lls.mean().detach().cpu().numpy().item()

            test_ll /= len(valid_loader)
            t2 = time.time()

            log_line = f"[Epoch {epoch}/{num_epochs}][train LL: {train_ll:.2f}; val LL: {test_ll:.2f}].....[train forward+backward+step {t1-t0:.2f}; val forward {t2-t1:.2f}]"
            print(log_line)  # Print to console
            log_file.write(log_line + "\n")  # Save to log file
            log_file.flush()  # Ensure logs are written in real-time

            if epoch % 5000 == 0:
                juice.save(f'../results/{data}/{split}/hclt/pc_{snps}_{split}_{amt}-{latents}_{epoch}epochs_ps{ps}.jpc', pc)
