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

snps = "805"
amt = 4006
data = "1KG"
split = "8020"

latents = 16
ps = 0.005
num_epochs = 100
batch_size = 256


print("Number of CUDA devices:", torch.cuda.device_count())
print(torch.version.cuda)

device = torch.device("cuda")
np.random.seed(1)

print(device)

train_data = np.loadtxt(f"data/805_train.txt", dtype=np.int8, delimiter=' ')
valid_data = np.loadtxt(f"data/805_test.txt", dtype=np.int8, delimiter=' ')

train_data = torch.tensor(train_data, dtype=torch.long)
valid_data = torch.tensor(valid_data, dtype=torch.long)

print(train_data.shape)
print(valid_data.shape)
# %%

train_loader = DataLoader(
    dataset = TensorDataset(train_data),
    batch_size = batch_size,
    shuffle = True
)
valid_loader = DataLoader(
    dataset = TensorDataset(valid_data),
    batch_size = batch_size,
    shuffle = False
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

    log_filename = f"{snps}_{split}_{amt}_{latents}_{num_epochs}epochs_ps{ps}.log"
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
            print(log_line)
            log_file.write(log_line + "\n")
            log_file.flush()

            if epoch % 100 == 0:
                juice.save(f'pc_{snps}_{split}_{amt}-{latents}_{epoch}epochs_ps{ps}.jpc', pc)