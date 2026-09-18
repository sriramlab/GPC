"""Times imputation by both routes: an exact conditional query against the
trained circuit, and Impute5 with a real reference panel."""

import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

sys.setrecursionlimit(1000000)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL = f"{ROOT}/results/b38/8020/hclt/pc_14670_8020_4006-128_5000epochs_ps0.005_2.jpc"
TEST = f"{ROOT}/results/b38/8020/data/8020_test.txt"
MASK = f"{ROOT}/results/b38/missing_indices_hum5.txt"
TRAIN_BCF = f"{ROOT}/results/b38/8020/data/8020_train_AC.bcf"
TEST_VCF = f"{ROOT}/results/b38/8020/data/8020_test.vcf"
IMPUTE5 = "/scratch2/prateek/impute5_v1.2.0/impute5_v1.2.0_static"
REGION = "15:27134431-29332831"


def time_direct(batch_size=128):
    import pyjuice as juice
    device = torch.device("cuda")
    mask_idx = [int(l) for l in open(MASK) if l.strip()]
    test = np.loadtxt(TEST, dtype=np.int8, delimiter=' ')
    x_all = torch.tensor(test, dtype=torch.long)

    t0 = time.perf_counter()
    pc = juice.compile(juice.load(MODEL)).to(device)
    load_sec = time.perf_counter() - t0

    mask = torch.zeros(pc.num_vars, dtype=torch.bool, device=device)
    mask[torch.tensor(mask_idx, dtype=torch.long, device=device)] = True

    # warm up so the first kernel compile is not counted
    with torch.no_grad():
        juice.queries.conditional(pc, data=x_all[:batch_size].to(device), missing_mask=mask)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for s in range(0, x_all.shape[0], batch_size):
            juice.queries.conditional(pc, data=x_all[s:s + batch_size].to(device),
                                      missing_mask=mask)
    torch.cuda.synchronize()
    query_sec = time.perf_counter() - t0
    n = x_all.shape[0]
    print(f"  GPC direct")
    print(f"    load + compile the circuit   {load_sec:8.1f} s  (once)")
    print(f"    impute {len(mask_idx):,} SNPs x {n:,} haplotypes  {query_sec:8.1f} s")
    print(f"    per haplotype                {query_sec / n * 1000:8.1f} ms")
    return dict(load_sec=load_sec, query_sec=query_sec, per_hap_ms=query_sec / n * 1000, n=n)


def time_impute5(threads=1):
    if not (os.path.exists(IMPUTE5) and os.path.exists(TRAIN_BCF)):
        print("  Impute5: inputs unavailable, skipping")
        return None
    with tempfile.TemporaryDirectory() as tmp:
        # the masked target file the paper's pipeline feeds Impute5
        masked = f"{tmp}/target.bcf"
        mask_idx = set(int(l) for l in open(MASK) if l.strip())
        t0 = time.perf_counter()
        subprocess.run(["bcftools", "view", "-Ob", "-o", masked, TEST_VCF],
                       check=True, capture_output=True)
        subprocess.run(["bcftools", "index", "-f", masked], check=True, capture_output=True)
        prep_sec = time.perf_counter() - t0

        t0 = time.perf_counter()
        r = subprocess.run([IMPUTE5, "--h", TRAIN_BCF, "--g", masked, "--r", REGION,
                            "--buffer-region", REGION, "--o", f"{tmp}/out.vcf",
                            "--l", f"{tmp}/out.log", "--haploid",
                            "--threads", str(threads)], capture_output=True, text=True)
        run_sec = time.perf_counter() - t0
        if r.returncode != 0:
            print(f"  Impute5 failed: {r.stderr.strip()[-160:]}")
            return None
        print(f"  Impute5 with a real reference panel ({threads} thread{'s' if threads>1 else ''})")
        print(f"    prepare target BCF           {prep_sec:8.1f} s")
        print(f"    impute                       {run_sec:8.1f} s")
        return dict(prep_sec=prep_sec, run_sec=run_sec, threads=threads)


if __name__ == "__main__":
    print("Imputing the HumanOmni5Exome pattern on high-coverage 1KG "
          "(12,551 of 14,670 SNPs, 1,002 held-out haplotypes)\n")
    d = time_direct()
    print()
    i5 = time_impute5(threads=1)
    if i5:
        print(f"\n  direct is {i5['run_sec'] / d['query_sec']:.1f}x faster than the Impute5 "
              f"call alone, excluding panel construction")
