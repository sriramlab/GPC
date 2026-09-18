"""Figure 1: from data, to the HMM chain, to the learned Chow-Liu tree, to the HCLT.

Reviewer 1 asked for an explicit HMM chain in Figure 1 so the advantage of the
tree topology is visible rather than implied. Styling follows the ICLR 2026
workshop slides (gpc_iclr_2026.pdf): X2 and X6 are the example correlated pair,
carried in red through all four panels.

    python plots/make_fig1_hclt.py
    -> overleaf/figs/fig_hclt_chain.pdf (+ .png for quick inspection)
"""

import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

# Palette sampled from the slides.
DARK = "#333333"      # observed-node fill
RED = "#C23F38"       # the example correlated pair
NAVY = "#1A1A2E"      # panel titles
GRAY = "#555555"      # secondary labels
TEAL = "#289D8F"

R = 0.40              # node radius


def node(ax, x, y, label, filled=False, red=False, fs=11):
    edge = RED if red else DARK
    face = (RED if red else DARK) if filled else "white"
    txt = "white" if filled else (RED if red else DARK)
    ax.add_patch(Circle((x, y), R, facecolor=face, edgecolor=edge,
                        linewidth=1.6, zorder=3))
    ax.text(x, y, label, ha="center", va="center", color=txt,
            fontsize=fs, fontweight="bold", zorder=4)


def edge(ax, p, q, style="solid", color=None, shrink=R * 72 / 2.2, lw=1.6):
    """Arrow between two node centres, trimmed to the node boundary."""
    color = color or DARK
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=13, linewidth=lw,
        color=color, linestyle=style, shrinkA=shrink, shrinkB=shrink, zorder=2))


def arc(ax, p, q, height, color, style=(0, (4, 3)), lw=1.5, arrow=True, shrink=1.0):
    """Dashed correlation arc drawn above/below a pair of nodes.

    Endpoints are already on the node boundary, so shrink stays small -- the
    default trim used for straight edges would leave the arc hanging in space.
    """
    patch = FancyArrowPatch(
        p, q, connectionstyle=f"arc3,rad={-height}",
        arrowstyle="-|>" if arrow else "-", mutation_scale=11,
        linewidth=lw, color=color, linestyle=style,
        shrinkA=shrink, shrinkB=shrink, zorder=2)
    ax.add_patch(patch)
    return patch


def arc_midpoint(p, q, height):
    """Half-way point of the arc drawn by arc(), in data coordinates.

    arc3 traces a quadratic Bezier whose control point sits at the chord
    midpoint displaced by rad * rot90(q - p); the curve at t = 0.5 is therefore
    the chord midpoint displaced by half of that.
    """
    (px, py), (qx, qy) = p, q
    mx, my = (px + qx) / 2, (py + qy) / 2
    dx, dy = qx - px, qy - py
    return mx - 0.5 * height * dy, my + 0.5 * height * dx


def cross(ax, x, y, color, size=0.26, lw=2.2):
    """A small x drawn over an arc to mark a dependency the model cannot express."""
    ax.plot([x - size, x + size], [y - size, y + size],
            color=color, lw=lw, solid_capstyle="round", zorder=5)
    ax.plot([x - size, x + size], [y + size, y - size],
            color=color, lw=lw, solid_capstyle="round", zorder=5)


def panel_title(ax, x, y, letter, text):
    ax.text(x, y, f"({letter})", ha="left", va="center", color=NAVY,
            fontsize=13, fontweight="bold")
    ax.text(x + 1.05, y, text, ha="left", va="center", color=NAVY,
            fontsize=12.5, fontweight="bold")


def main():
    fig, ax = plt.subplots(figsize=(13.5, 8.6))
    ax.set_xlim(0, 20)
    ax.axis("off")
    ax.set_aspect("equal")

    genotypes = ["0", "1", "1", "0", "1", "0"]
    red_idx = {1, 5}  # X2 and X6

    # ---------------------------------------------------------------- (a) data
    panel_title(ax, 0.2, 12.5, "a", "Genotype data")
    xs_a = [1.3 + 1.25 * i for i in range(6)]
    y_a = 9.7
    for i, (x, g) in enumerate(zip(xs_a, genotypes)):
        node(ax, x, y_a, g, filled=True, red=(i in red_idx))
        ax.text(x, y_a - 0.9, f"$X_{i+1}$", ha="center", va="center",
                fontsize=12, fontweight="bold", color=RED if i in red_idx else DARK)
    # adjacent (weak, local) correlations
    for i in range(5):
        arc(ax, (xs_a[i], y_a + R), (xs_a[i + 1], y_a + R), 0.55, DARK, lw=1.2,
            arrow=False)
    # the long-range pair the whole figure is about
    arc(ax, (xs_a[1], y_a + R), (xs_a[5], y_a + R), 0.42, RED, lw=1.8, arrow=False)
    ax.text((xs_a[1] + xs_a[5]) / 2, y_a + 2.05, "strong long-range correlation",
            ha="center", va="center", fontsize=10.5, style="italic", color=RED)

    # ------------------------------------------------------- (b) the HMM chain
    panel_title(ax, 10.4, 12.5, "b", "HMM: chain over latent variables")
    xs_b = [11.3 + 1.35 * i for i in range(6)]
    yz_b, yx_b = 11.0, 9.0
    for i, x in enumerate(xs_b):
        node(ax, x, yz_b, f"$Z_{i+1}$", red=(i in red_idx))
        node(ax, x, yx_b, f"$X_{i+1}$", filled=True, red=(i in red_idx))
        edge(ax, (x, yz_b), (x, yx_b), style=(0, (3, 2.5)),
             color=RED if i in red_idx else DARK, lw=1.3)
    for i in range(5):
        edge(ax, (xs_b[i], yz_b), (xs_b[i + 1], yz_b))
    # The same pair as in (a), but the chain has no edge for it: drawn faint and
    # struck through so the arc reads as the dependency, not as a modelled edge.
    missing = arc(ax, (xs_b[5], yx_b - R), (xs_b[1], yx_b - R), 0.30, RED,
                  style=(0, (1.5, 2.8)), lw=1.4, arrow=False)
    missing.set_alpha(0.45)
    del missing
    cross(ax, *arc_midpoint((xs_b[5], yx_b - R), (xs_b[1], yx_b - R), 0.30), color=RED)
    ax.text((xs_b[1] + xs_b[5]) / 2, yx_b - 2.15,
            "no direct edge: routed through $Z_3,\\ Z_4,\\ Z_5$",
            ha="center", va="center", fontsize=10.5, style="italic", color=RED)

    # --------------------------------------------------- (c) the Chow-Liu tree
    panel_title(ax, 0.2, 6.1, "c", "Learned Chow-Liu tree")
    clt = {1: (3.75, 4.7), 3: (2.3, 3.4), 4: (2.3, 2.1), 5: (2.3, 0.8),
           2: (5.2, 3.4), 6: (5.2, 2.1)}
    for a, b in [(1, 3), (3, 4), (4, 5), (1, 2), (2, 6)]:
        edge(ax, clt[a], clt[b])
    for i, p in clt.items():
        node(ax, *p, f"$Z_{i}$", red=(i - 1 in red_idx))
    ax.text(6.1, 2.75, "correlated pair\nnow adjacent", ha="left", va="center",
            fontsize=10.5, style="italic", color=RED)

    # ------------------------------------------------------------- (d) the HCLT
    panel_title(ax, 10.4, 6.1, "d", "HCLT: tree over latent variables")
    hclt = {1: (14.5, 4.7), 3: (12.8, 3.4), 4: (12.8, 2.1), 5: (12.8, 0.8),
            2: (16.2, 3.4), 6: (16.2, 2.1)}
    emit = {1: (13.0, 4.7), 3: (11.3, 3.4), 4: (11.3, 2.1), 5: (11.3, 0.8),
            2: (17.7, 3.4), 6: (17.7, 2.1)}
    for a, b in [(1, 3), (3, 4), (4, 5), (1, 2), (2, 6)]:
        edge(ax, hclt[a], hclt[b])
    for i in hclt:
        edge(ax, hclt[i], emit[i], color=RED if i - 1 in red_idx else DARK)
    for i, p in hclt.items():
        node(ax, *p, f"$Z_{i}$", red=(i - 1 in red_idx))
    for i, p in emit.items():
        node(ax, *p, f"$X_{i}$", filled=True, red=(i - 1 in red_idx))

    ax.set_ylim(0.1, 13.1)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "overleaf", "figs", "fig_hclt_chain")
    fig.savefig(f"{out}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(f"{out}.png", bbox_inches="tight", pad_inches=0.05, dpi=300)
    print(f"wrote {os.path.normpath(out)}.pdf / .png")


if __name__ == "__main__":
    main()
