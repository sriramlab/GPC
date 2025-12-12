# %%
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import pyjuice as juice
import pyjuice.nodes.distributions as dists

# ============================================================
#                   CONFIGURATION
# ============================================================
snps = 14670
amt = 3202
latents = 128
ps = 0.005
num_epochs = 5000
batch_size = 256
valid_frac = 0.1
patience = 100

resume_retrain = False
resume_retrain_epochs = 315

# train_file = "/scratch2/prateek/genetic_pc_github/aux/b38_real_eur_and_noneur_train.txt"
train_file = "/scratch2/prateek/genetic_pc_github/results/b38/noneur/data/noneur_train.txt"

save_dir = "/scratch2/prateek/genetic_pc_github/results/b38/noneur/hclt/"
os.makedirs(save_dir, exist_ok=True)

log_filename = os.path.join(
    save_dir,
    f"hclt_{snps}_noneur{amt}_L{latents}_E{num_epochs}_ps{ps}_shuf.log"
)
model_checkpoint = os.path.join(
    save_dir,
    f"pc_{snps}_noneur{amt}_L{latents}_E{num_epochs}_ps{ps}_shuf.jpc"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
np.random.seed(1)

print("Using device:", device)
print("CUDA devices:", torch.cuda.device_count())
print("CUDA version:", torch.version.cuda)


# ============================================================
#                          DATA
# ============================================================
print("Loading training data...")
train_arr = np.loadtxt(train_file, dtype=np.int8, delimiter=" ")

print("Shuffling dataset before split...")
perm = np.random.permutation(len(train_arr))
train_arr = train_arr[perm]

train_tensor = torch.tensor(train_arr, dtype=torch.long)
num_valid = max(1, int(len(train_tensor) * valid_frac))
valid_tensor = train_tensor[-num_valid:]
train_tensor = train_tensor[:-num_valid]

print(f"Train samples: {len(train_tensor)},  Valid samples: {len(valid_tensor)}")

train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(TensorDataset(valid_tensor), batch_size=batch_size, shuffle=False)


# ============================================================
#                STAGE 1: TRAIN WITH EARLY STOPPING
# ============================================================
if not resume_retrain:

    print("Building initial HCLT structure...")
    ns = juice.structures.HCLT(
        train_tensor[:amt].float().to(device),
        num_latents=latents,
        input_dist=dists.Categorical(num_cats=2),
    )
    pc = juice.compile(ns).to(device)

    best_val_ll = -np.inf
    best_epoch = 0
    epochs_no_improve = 0

    ### NOTE: open log in append mode
    with open(log_filename, "a") as log_file:

        log_file.write("\n===== NEW RUN: STAGE 1 TRAINING =====\n")

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()

            pc.init_param_flows(flows_memory=0.0)

            train_ll_accum = 0.0
            for (batch,) in train_loader:
                x = batch.to(device)
                lls = pc(x)
                lls.mean().backward()
                train_ll_accum += lls.mean().item()

            pc.mini_batch_em(step_size=1.0, pseudocount=ps)
            train_ll = train_ll_accum / len(train_loader)
            t1 = time.time()

            val_ll_accum = 0.0
            for (batch,) in valid_loader:
                x = batch.to(device)
                lls = pc(x)
                val_ll_accum += lls.mean().item()

            val_ll = val_ll_accum / len(valid_loader)
            t2 = time.time()

            log_line = (
                f"[Epoch {epoch}/{num_epochs}] "
                f"[train LL: {train_ll:.2f}; val LL: {val_ll:.2f}] "
                f"[train time {t1-t0:.2f}s; val time {t2-t1:.2f}s]"
            )

            print(log_line)
            log_file.write(log_line + "\n")
            log_file.flush()

            if val_ll > best_val_ll:
                best_val_ll = val_ll
                best_epoch = epoch
                epochs_no_improve = 0
                print(f"   --> Validation improved. (best val LL = {best_val_ll:.2f})")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch}. No improvement for {patience} epochs.")
                break

    print(f"\n==== Best epoch: {best_epoch} ====\n")

else:
    best_epoch = resume_retrain_epochs
    print(f"\nSkipping Stage 1. Will run retraining for {best_epoch} epochs.\n")


# ============================================================
#        STAGE 2: RETRAIN FROM SCRATCH ON FULL DATASET
# ============================================================
print("Retraining on full dataset for", best_epoch, "epochs...")

full_tensor = torch.cat([train_tensor, valid_tensor], dim=0)
full_loader = DataLoader(TensorDataset(full_tensor), batch_size=batch_size, shuffle=True)

ns_full = juice.structures.HCLT(
    full_tensor[:amt].float().to(device),
    num_latents=latents,
    input_dist=dists.Categorical(num_cats=2),
)

pc_full = juice.compile(ns_full).to(device)

### Append to the log again
with open(log_filename, "a") as log_file:
    log_file.write("\n===== STAGE 2: RETRAINING START =====\n")

    for ep in range(1, best_epoch + 1):
        pc_full.init_param_flows(flows_memory=0.0)

        for (batch,) in full_loader:
            x = batch.to(device)
            lls = pc_full(x)
            lls.mean().backward()

        pc_full.mini_batch_em(step_size=1.0, pseudocount=ps)

        if ep % 50 == 0 or ep == best_epoch:
            msg = f"Retrain Epoch {ep}/{best_epoch}"
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

final_checkpoint = model_checkpoint.replace(".jpc", "_FULL_RETRAIN_SHUF.jpc")
juice.save(final_checkpoint, pc_full)

print(f"\nFinal full-data model saved to:\n  {final_checkpoint}")
