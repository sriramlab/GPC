"""Trains the chain-structured HMM baseline: an ablation of GPC with the
latent tree replaced by a chain."""

import os
import math
import re
import sys
import time
import argparse
import numpy as np
import torch
import pyjuice as juice

from hmm_config import CONFIGS, ROOT, resolve


# --- CONFIGURATION ---
class CFG:
    # Model Hyperparameters
    latents = 128
    ps = 0.005          # reference pseudocount, defined at REF_BATCH
    REF_BATCH = 256     # the batch size the published runs used
    batch_size = 256

    # Checkpoints are written every CKPT_EVERY epochs; a resume rewinds to the
    # last multiple, so this also bounds what a restart can lose.
    CKPT_EVERY = 500

    # Execution mode: the number of chunks comes from the dataset entry in
    # hmm_config (4 for 1KG/UKBB, 6 for b38, whose 14,670 SNPs are not divisible
    # by 4). The full region does not compile -- the layer-by-layer graph build
    # blows the C stack -- which is why the HMM is always trained in chunks.

    # System
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 1


sys.setrecursionlimit(100000)
torch.manual_seed(CFG.seed)
np.random.seed(CFG.seed)

_EPOCH_RE = re.compile(r"^\[Epoch (\d+)/")


# Parsed-data cache. Kept outside the repo: it is a pure derived artifact (a
# .npy of the same haplotypes already on disk as text), it is large -- the UKBB
# matrices are ~170 MB each -- and it would otherwise litter results/ with files
# that are not results.
CACHE_DIR = "/scratch2/prateek/tmp/hmm_data_cache"


def load_data(cfg):
    """Load the train/valid haplotype matrices, caching the parsed array.

    np.loadtxt on the UKBB matrices takes minutes and every chunk is its own
    process, so without the cache that cost is paid once per chunk.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    out = []
    for path in (cfg["train_path"], cfg["valid_path"]):
        cache = os.path.join(
            CACHE_DIR, os.path.relpath(path, ROOT).replace(os.sep, "_") + ".npy")
        if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
            arr = np.load(cache)
        else:
            print(f"Loading data from {path}...", flush=True)
            arr = np.loadtxt(path, dtype=np.int8, delimiter=' ')
            # np.save appends ".npy" unless handed a file object, so write
            # through one and rename atomically.
            tmp = f"{cache}.tmp{os.getpid()}"
            with open(tmp, "wb") as fh:
                np.save(fh, arr)
            os.replace(tmp, cache)
        out.append(torch.tensor(arr, dtype=torch.uint8))
    print(f"Train shape: {tuple(out[0].shape)} | Valid shape: {tuple(out[1].shape)}")
    return out


def _resume_point(log_path, ckpt_path, num_epochs):
    """How many epochs of `log_path` are actually backed by the checkpoint."""
    if not (os.path.exists(log_path) and os.path.exists(ckpt_path)):
        return 0
    with open(log_path) as fh:
        logged = [l for l in fh if l.startswith("[Epoch")]
    if not logged:
        return 0
    last = _EPOCH_RE.match(logged[-1])
    n = int(last.group(1)) if last else len(logged)
    if n >= num_epochs:
        return num_epochs
    return (n // CFG.CKPT_EVERY) * CFG.CKPT_EVERY


def _truncate_log(log_path, keep):
    """Drop log lines past the resume point so the log matches the weights."""
    with open(log_path) as fh:
        lines = [l for l in fh if l.startswith("[Epoch")]
    with open(log_path, "w") as fh:
        fh.writelines(lines[:keep])


def train_model(cfg, train_subset, valid_subset, chunk_id, num_epochs, outdir,
                batch_size, ps_mode, val_every=1):
    seq_length = train_subset.shape[1]
    n_train = cfg["n_train"]
    device = CFG.device

    # Matches the naming already on disk, e.g.
    #   pc_10K_0_8020_hmm_4006-128_5000epochs_ps0.005.jpc
    #   10K_0_8020_hmm_4006_128_5000epochs_ps0.005.log
    stem = f"{cfg['tag']}_{chunk_id}_{cfg['split']}_hmm_{n_train}"
    log_path = os.path.join(
        outdir, f"{stem}_{CFG.latents}_{num_epochs}epochs_ps{CFG.ps}.log")
    ckpt_path = os.path.join(
        outdir, f"pc_{stem}-{CFG.latents}_{num_epochs}epochs_ps{CFG.ps}.jpc")

    done = _resume_point(log_path, ckpt_path, num_epochs)
    if done >= num_epochs:
        print(f"chunk {chunk_id}: already at {num_epochs} epochs, skipping")
        return

    # Batches are split into `nb` near-equal groups rather than fixed-size strides
    # with a small remainder. `lls.mean()` weights every batch equally regardless
    # of its size, so a stride split lets a short trailing batch dominate the EM
    # update -- at bs=1024 the 1KG:afr remainder (32 of 1056 haplotypes) would
    # carry half of it. Near-equal groups remove that distortion entirely and
    # make the update essentially independent of the batch size.
    n_all = train_subset.shape[0]
    nb = max(1, math.ceil(n_all / batch_size))
    eff_batch = n_all / nb

    # The pseudocount is defined at REF_BATCH; rescale it so the EM update stays
    # equivalent when a different batch size is used (see the module docstring).
    pseudocount = CFG.ps * (CFG.REF_BATCH / eff_batch) if ps_mode == "scaled" else CFG.ps

    if done:
        print(f"chunk {chunk_id}: resuming from epoch {done} ({ckpt_path})")
        _truncate_log(log_path, done)
        pc = juice.compile(juice.load(ckpt_path)).to(device)
    else:
        # Starting over: drop any earlier log. The file is opened in append mode
        # below so that a resume keeps its history, but with done == 0 there is
        # nothing to keep -- leaving it would splice a stale partial run in front
        # of this one, and anything that measures progress by counting [Epoch
        # lines would then over-report and could call the chunk finished early.
        if os.path.exists(log_path):
            _truncate_log(log_path, 0)
        ns = juice.structures.HMM(seq_length=seq_length, num_latents=CFG.latents,
                                  homogeneous=False, num_emits=2)
        pc = juice.compile(ns).to(device)

    # Both matrices stay on the GPU for the whole run: uint8 keeps them small
    # (a UKBB chunk is ~53 MB) and shuffling is a device-side gather, so there
    # is no host-side collation and no per-batch host->device copy.
    tr = train_subset.to(device=device, dtype=torch.uint8)
    va = valid_subset.to(device=device, dtype=torch.uint8)
    n, nv = tr.shape[0], va.shape[0]
    bounds = [(i * n) // nb for i in range(nb + 1)]

    # Validation always uses the reference stride so that the logged val LL stays
    # directly comparable to the runs already on disk, whatever --batch-size is.
    nvb = (nv + CFG.REF_BATCH - 1) // CFG.REF_BATCH

    print(f"\nStarting training for chunk {chunk_id} ({seq_length} features)...")
    print(f"  batch {batch_size} -> {nb} batches/epoch of ~{eff_batch:.0f}, "
          f"pseudocount {pseudocount:g}")
    print(f"  log  -> {log_path}")
    print(f"  ckpt -> {ckpt_path}")

    with open(log_path, "a") as log_file:
        for epoch in range(done + 1, num_epochs + 1):
            t0 = time.time()
            pc.init_param_flows(flows_memory=0.0)

            # Training loop. The LL is accumulated on-device and read back once
            # per epoch; a per-batch .item() would sync the GPU on every batch.
            perm = torch.randperm(n, device=device)
            acc = torch.zeros((), device=device)
            for i in range(nb):
                m = pc(tr[perm[bounds[i]:bounds[i + 1]]].long()).mean()
                m.backward()
                acc += m.detach()

            pc.mini_batch_em(step_size=1.0, pseudocount=pseudocount)
            train_ll = (acc / nb).item()      # single sync; also drains the queue
            t1 = time.time()

            # Validation loop. The held-out LL is only ever logged, never used
            # to alter training, so evaluating it less often changes nothing
            # about the fitted model. It is worth skipping when the held-out
            # split is large: on the AATS split it costs as much as an epoch of
            # training. One line is still written per epoch, since the resume
            # logic counts logged epochs.
            if epoch % val_every == 0 or epoch == num_epochs or epoch == done + 1:
                with torch.no_grad():
                    vacc = torch.zeros((), device=device)
                    for s in range(0, nv, CFG.REF_BATCH):
                        vacc += pc(va[s:s + CFG.REF_BATCH].long()).mean()
                    valid_ll = (vacc / nvb).item()
            t2 = time.time()

            stats = (f"[Epoch {epoch}/{num_epochs}][train LL: {train_ll:.2f}; "
                     f"val LL: {valid_ll:.2f}].....[train forward+backward+step "
                     f"{t1-t0:.2f}; val forward {t2-t1:.2f}]")
            print(stats)
            log_file.write(stats + "\n")
            log_file.flush()

            if epoch % CFG.CKPT_EVERY == 0 or epoch == num_epochs:
                juice.save(ckpt_path, pc)

    # Cleanup to prevent memory fragmentation between chunks
    del pc, tr, va
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Train a (chunked) HMM inside the GPC/pyjuice framework.")
    parser.add_argument("config", choices=sorted(CONFIGS.keys()),
                        help="dataset/split key, e.g. 1KG:afr or UKBB:eur_and_noneur_train")
    parser.add_argument("--chunks", type=int, nargs="+", default=None,
                        help="only train these chunk ids (0-indexed), e.g. --chunks 0 2")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override the per-dataset default epoch count")
    parser.add_argument("--batch-size", type=int, default=CFG.batch_size,
                        help="haplotypes per circuit traversal (default 256). Epoch cost is "
                             "~proportional to the batch count, so larger is faster.")
    parser.add_argument("--val-every", type=int, default=1,
                        help="epochs between held-out evaluations (default 1). The value is "
                             "only logged, so raising this leaves the fitted model unchanged.")
    parser.add_argument("--pseudocount-mode", choices=("scaled", "fixed"), default="scaled",
                        help="'scaled' (default) keeps the EM update equivalent to the bs=256 "
                             "runs already on disk; 'fixed' applies 0.005 as-is.")
    args, _ = parser.parse_known_args()

    cfg = resolve(args.config)
    num_epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    outdir = cfg["model_dir"]
    os.makedirs(outdir, exist_ok=True)

    train_data, valid_data = load_data(cfg)
    assert train_data.shape[0] == cfg["n_train"], \
        f"expected {cfg['n_train']} training haplotypes, got {train_data.shape[0]}"

    num_chunks = cfg["num_chunks"]
    if num_chunks > 1:
        assert train_data.shape[1] % num_chunks == 0, \
            f"{train_data.shape[1]} features not divisible by {num_chunks} chunks"
        train_chunks = torch.chunk(train_data, num_chunks, dim=1)
        valid_chunks = torch.chunk(valid_data, num_chunks, dim=1)

        chunk_ids = args.chunks if args.chunks else list(range(num_chunks))
        for i in chunk_ids:
            train_model(cfg, train_chunks[i], valid_chunks[i], i, num_epochs, outdir,
                        args.batch_size, args.pseudocount_mode, args.val_every)
    else:
        # Train on the whole dataset as one block
        train_model(cfg, train_data, valid_data, 0, num_epochs, outdir,
                    args.batch_size, args.pseudocount_mode, args.val_every)


if __name__ == "__main__":
    main()
