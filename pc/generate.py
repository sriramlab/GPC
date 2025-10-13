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
from tqdm import tqdm

data = "b38"
split = "8020"
snps = f"14670_{split}_4006"
latents = 128
ps = 0.005
num_epochs = 5000

print("Number of CUDA devices:", torch.cuda.device_count())
print(torch.version.cuda)

device = torch.device("cuda")
np.random.seed(1)

print(device)
print(os.getenv("TRITON_CACHE_DIR"))

ns = juice.load(f'/scratch2/prateek/genetic_pc_github/results/{data}/{split}/hclt/pc_{snps}-{latents}_{num_epochs}epochs_ps{ps}_2.jpc')
pc = juice.compile(ns)
pc.to(device)

print(f"Num. params: {ns.num_parameters()}")

samples = []

# First 50 iterations of 100 samples each
for i in tqdm(range(50), desc="Sampling 100s"):
    s = juice.queries.sample(pc, num_samples=100)
    for x in s.cpu():
        samples.append(x)

# Final 8 samples
for x in tqdm(juice.queries.sample(pc, num_samples=8).cpu(), desc="Sampling final 8"):
    samples.append(x)

# Convert to numpy
np_arrays = [tensor.numpy() for tensor in samples]
d = np.vstack(np_arrays)

# Save
np.savetxt(f'/scratch2/prateek/genetic_pc_github/results/{data}/{split}/hclt/{data}_hclt_{split}_samples_2.txt', d, fmt='%d')