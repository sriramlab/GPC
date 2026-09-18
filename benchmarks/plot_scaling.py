"""Draws the scaling figure: training time, per-epoch cost and peak memory
against haplotype and SNP counts."""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "results", "final", "scaling_results.csv")
OUT = os.path.join(HERE, "results", "plots")

# Same identity as the imputation figures. RBM's orange sits light against a
# white surface, so every series also carries a distinct marker and a direct
# label; identity is never colour alone.
STYLE = {
    "gpc":  dict(label="GPC",  color="dodgerblue",   marker="s"),
    "rbm":  dict(label="RBM",  color="orange",       marker="^"),
    "wgan": dict(label="WGAN", color="g",            marker="D"),
    "hmm":  dict(label="HMM",  color="mediumorchid", marker="v"),
}
# The HMM is trained in fixed 2,445-SNP chunks, so its per-epoch cost is a
# per-chunk figure that does not scale with M the way the other rows do. Plotting
# it alongside them invites a false comparison, so it is left out of this figure.
ORDER = ["gpc", "rbm", "wgan"]

N_FIXED = 2000      # for the M sweeps: every method is in its normal regime here
M_FIXED = 8191      # for the N sweeps: the largest M all four methods support


def load():
    rows = [r for r in csv.DictReader(open(CSV)) if r["status"].startswith("ok")]
    out = {}
    for r in rows:
        out[(r["method"], int(r["N"]), int(r["M"]))] = dict(
            sec=float(r["sec_per_epoch"]),
            gib=float(r["peak_gpu_gib"]),
            hours=float(r["projected_total_hours"]) if r["projected_total_hours"] else None,
        )
    return out


def series(d, method, field, fix, sweep):
    """Points for one method along one axis, sorted by the swept variable."""
    pts = []
    for (m, n, M), v in d.items():
        if m != method or v[field] is None:
            continue
        if sweep == "N" and M == fix:
            pts.append((n, v[field]))
        elif sweep == "M" and n == fix:
            pts.append((M, v[field]))
    return sorted(pts)


def panel(ax, d, field, sweep, fix, ylabel, title, logy=True, logx=True):
    ends = []
    for method in ORDER:
        pts = series(d, method, field, fix, sweep)
        if not pts:
            continue
        xs, ys = zip(*pts)
        st = STYLE[method]
        ax.plot(xs, ys, marker=st["marker"], color=st["color"], linewidth=2,
                markersize=7, label=st["label"], zorder=3)
        ends.append([xs[-1], ys[-1], st])

    # Direct labels at the right-hand end, so identity survives in greyscale.
    # Nudge them apart when two series finish at nearly the same value, which
    # happens where the curves cross.
    if ends:
        span = max(e[1] for e in ends) - min(e[1] for e in ends) or 1.0
        ends.sort(key=lambda e: e[1])
        for i in range(1, len(ends)):
            gap = ends[i][1] - ends[i - 1][1]
            if gap < 0.06 * span:
                ends[i][1] = ends[i - 1][1] + 0.06 * span
        for x, y, st in ends:
            ax.annotate(st["label"], (x, y), textcoords="offset points",
                        xytext=(7, 0), va="center", fontsize=9, color=st["color"],
                        fontweight="bold")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Number of haplotypes" if sweep == "N" else "Number of SNPs", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)
    ax.tick_params(labelsize=10)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scale", choices=("log", "linear"), default="linear",
                    help="axis scaling for the runtime panels and both x axes")
    ap.add_argument("--per-epoch", action="store_true",
                    help="secondary figure: per-epoch cost alone, independent of any "
                         "epoch budget")
    args = ap.parse_args()
    lg = args.scale == "log"
    suffix = "" if lg else "_linear"

    d = load()
    os.makedirs(OUT, exist_ok=True)

    # One combined figure: total training time, per-epoch cost, and peak memory,
    # each against N and against M. Linear axes throughout.
    fig, axes = plt.subplots(3, 2, figsize=(11, 12))
    rows = [("hours", "Total training time (h)", "Training time"),
            ("sec",   "Seconds per epoch",       "Per-epoch cost"),
            ("gib",   "Peak GPU memory (GiB)",   "Memory")]
    for r, (field, ylab, name) in enumerate(rows):
        panel(axes[r][0], d, field, "N", M_FIXED, ylab,
              f"{name} vs haplotypes ($M={M_FIXED:,}$)", logy=lg, logx=lg)
        panel(axes[r][1], d, field, "M", N_FIXED, ylab,
              f"{name} vs SNPs ($N={N_FIXED:,}$)", logy=lg, logx=lg)

    for ax in axes.flat:
        ax.set_xlim(right=ax.get_xlim()[1] * (1.45 if lg else 1.30))

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"scaling{suffix}.{ext}"), dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}/scaling{suffix}.pdf / .png")

    if True:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Total training time, not per-epoch: the three methods need very different
    # epoch budgets (GPC 5,000, RBM 100,000, WGAN 2,001), so per-epoch cost alone
    # compares different amounts of work. Per-epoch values are given in the caption.
    panel(axes[0][0], d, "hours", "N", M_FIXED, "Total training time (h)",
          f"Training time vs haplotypes ($M={M_FIXED:,}$)", logy=lg, logx=lg)
    panel(axes[0][1], d, "hours", "M", N_FIXED, "Total training time (h)",
          f"Training time vs SNPs ($N={N_FIXED:,}$)", logy=lg, logx=lg)
    panel(axes[1][0], d, "gib", "N", M_FIXED, "Peak GPU memory (GiB)",
          f"Memory vs haplotypes ($M={M_FIXED:,}$)", logy=False, logx=lg)
    panel(axes[1][1], d, "gib", "M", N_FIXED, "Peak GPU memory (GiB)",
          f"Memory vs SNPs ($N={N_FIXED:,}$)", logy=False, logx=lg)

    for ax in axes.flat:
        ax.set_xlim(right=ax.get_xlim()[1] * 1.45)   # room for the direct labels

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"scaling{suffix}.{ext}"), dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}/scaling{suffix}.pdf / .png")

    if not lg:
        return
    # A plain-text view of the same numbers, so the figure is checkable.
    print(f"\n{'':6} {'N':>6} {'M':>7} {'s/epoch':>10} {'peak GiB':>9} {'total h':>9}")
    for method in ORDER:
        for (m, n, M), v in sorted(d.items()):
            if m == method:
                h = f"{v['hours']:9.1f}" if v["hours"] else " " * 9
                print(f"{m:6} {n:6} {M:7} {v['sec']:10.3f} {v['gib']:9.2f} {h}")


if __name__ == "__main__":
    main()
