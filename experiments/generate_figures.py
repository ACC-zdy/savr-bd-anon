"""
Generate publication-grade vector PDF figures for both the revised SaVR-BD journal manuscript
(paper/figures/) and the DASFAA 2027 LNCS manuscript (paper_dasfaa/figures/) from results/*.json.

Visual design principles:
1. External Top Legends: All legends sit strictly outside the plotting axes in dedicated header bars.
2. No Dashed-Line Spaghetti or Cluttered T-Caps: Multi-line trajectories use smooth solid lines
   with visual hierarchy (bold primary curves, muted thinner baseline curves), translucent +/- 1 SD
   shaded confidence ribbons (fill_between), and small white-rimmed circular markers.
3. Chart Type Diversity & De-bunching:
   - Fig 1 (Mechanism): Smooth analytical curves + shaded sensitivity zone + area-shaded bias curves.
   - Fig 2 (UCI a9a & Cauchy): Shaded ribbon curves + embedded zoom callout inset at sigma=0.10 on a9a
     to separate the VR-Barker cluster (0.082 vs 0.076 vs 0.069).
   - Fig 3 (Stiffness mismatch): Panel (a) log-log ribbon trajectories with zoomed linear inset at
     lambda in {0.03, 0.01}; Panel (b) Grouped Multi-Regime Bar Chart across lambda in {1.0, 0.1, 0.01}
     for Joint MMD^2 to HMC.
   - Fig 4 (UCI Wine Quality BNN): Two-tier broken Y-axis (split upper Langevin instability tier with
     DIV badges vs. lower 72%-height zoomed Barker basin tier) for both RMSE and NLL.
   - Fig 5 (Ablations): Grouped regime bar chart with 10x drop callout + smooth shaded confidence
     ribbon for rho* sensitivity plateau.
   - Fig 6 (Diagnostics): Ribbon convergence curves without T-caps + marker-free area-shaded ACF decay
     curves + symlog stiff-coordinate trajectory.
   - Fig 7 / fig_scalability (Table-cardinality scaling): Ribbon scaling curves + Horizontal Paired
     Latency Bar Chart comparing N=2,500 vs N=100,000 rows (40x table growth).
"""

from __future__ import annotations
import json
import shutil
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 8.8,
        "axes.labelsize": 9.1,
        "axes.titlesize": 9.2,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.4,
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.6,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
DASFAA_FIG_DIR = ROOT / "figures"

# Cohesive publication palette (Okabe-Ito / Nature-inspired high-contrast hierarchy)
COLORS = {
    "SaVR-BD": "#0055A4",          # Vivid deep royal blue (hero)
    "SaVR-BD (SVRG)": "#0055A4",   # Vivid deep royal blue
    "SaVR-BD (SAGA)": "#008080",   # Rich deep teal (hero single-scan)
    "UniVR-SGBD": "#1B9E77",       # Emerald green
    "NaiveVR-SGBD": "#8E44AD",     # Muted plum purple
    "HardGate-SGBD": "#A63603",    # Rust brown
    "c-SGBD": "#E69F00",           # Golden amber
    "sc-SGBD": "#D9B310",          # Ochre
    "v-SGBD": "#66A61E",           # Olive green
    "SVRG-SGLD": "#D95F02",        # Vermilion orange
    "SGLD-CV": "#C51B7D",          # Magenta rose
    "SGLD": "#64748B",             # Slate grey
}


def clean_axes(ax: plt.Axes, log_y: bool = False) -> None:
    ax.grid(True, which="major", color="#CBD5E1", alpha=0.50, linewidth=0.48)
    if log_y:
        ax.grid(True, which="minor", color="#E2E8F0", alpha=0.38, linewidth=0.32)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_color("#334155")


def plot_ribbon_curve(
    ax: plt.Axes,
    xs: np.ndarray | list[float],
    means: np.ndarray | list[float],
    stds: np.ndarray | list[float] | None,
    label: str,
    color: str,
    is_hero: bool = False,
    linestyle: str = "-",
    alpha_band: float = 0.13,
    show_dots: bool = True,
    zorder: int = 3,
    clip_low: float = 1e-6,
) -> None:
    x_arr = np.asarray(xs, dtype=float)
    m_arr = np.asarray(means, dtype=float)
    lw = 2.15 if is_hero else 1.35
    al = 1.0 if is_hero else 0.86
    ms = 4.2 if is_hero else 3.2

    ax.plot(
        x_arr,
        m_arr,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=lw,
        alpha=al,
        marker="o" if show_dots else None,
        markersize=ms,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.65 if is_hero else 0.45,
        zorder=zorder + (2 if is_hero else 0),
    )
    if stds is not None and alpha_band > 0:
        s_arr = np.asarray(stds, dtype=float)
        lo = np.maximum(m_arr - s_arr, clip_low)
        hi = m_arr + s_arr
        ax.fill_between(
            x_arr,
            lo,
            hi,
            color=color,
            alpha=alpha_band if is_hero else alpha_band * 0.65,
            linewidth=0.0,
            zorder=zorder - 1,
        )


def make_fig1_mechanism() -> None:
    d6 = json.loads((RES_DIR / "exp6_ablations.json").read_text(encoding="utf-8"))
    zs_dense = np.linspace(-5.5, 5.5, 400)
    sig = 1.0 / (1.0 + np.exp(-zs_dense))
    sens = 4.0 * sig * (1.0 - sig)
    gate = np.where(np.abs(zs_dense) > 1e-6, np.tanh(zs_dense / 2.0) / (zs_dense / 2.0), 1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.80))

    ax1.axvspan(-2.0, 2.0, color="#EFF6FF", alpha=0.85, label="Active calibration zone ($|z|<2$)")
    ax1.fill_between(zs_dense, 0, sens, color="#FDBA74", alpha=0.22)
    ax1.plot(zs_dense, sig, color="#1E293B", linestyle="-", linewidth=1.45, label=r"Flip prob. $\sigma(z)$")
    ax1.plot(zs_dense, sens, color="#D95F02", linestyle="--", linewidth=1.65, label=r"Sensitivity $4\sigma'(z)$")
    ax1.plot(zs_dense, gate, color="#0055A4", linestyle="-", linewidth=2.15, label=r"Smooth gate $g_2(z)$")
    ax1.set_ylim(-0.03, 1.06)
    ax1.set_xlabel("Exact logistic argument $z$")
    ax1.set_ylabel("Value / normalized weight")
    ax1.set_title("(a) Sigmoid sensitivity and smooth gate", pad=4.0)
    ax1.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        fontsize=6.9,
        columnspacing=0.8,
    )
    clean_axes(ax1)

    bdata = d6["single_step_bias"]
    zs = np.asarray(bdata["zs"], dtype=float)
    bias_specs = [
        ("NaiveVR-SGBD", COLORS["NaiveVR-SGBD"], "-", 1.35, 0.08, False),
        ("HardGate-SGBD", COLORS["HardGate-SGBD"], "--", 1.30, 0.06, False),
        ("UniVR-SGBD", COLORS["UniVR-SGBD"], "-", 1.45, 0.10, False),
        ("SaVR-BD", COLORS["SaVR-BD"], "-", 2.25, 0.22, True),
    ]
    for m, col, ls, lw, fill_a, is_hero in bias_specs:
        y_curve = np.asarray(bdata["curves"][m]) * 1e3
        ax2.fill_between(zs, 0, y_curve, color=col, alpha=fill_a)
        ax2.plot(
            zs,
            y_curve,
            label=m,
            color=col,
            linestyle=ls,
            linewidth=lw,
            marker="o" if is_hero else None,
            markevery=6,
            markersize=3.6,
            markerfacecolor=col,
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=5 if is_hero else 3,
        )
    ax2.annotate(
        r"$11.5\times$ bias cut",
        xy=(1.32, 1.05),
        xytext=(2.35, 9.4),
        fontsize=7.2,
        fontweight="bold",
        color="#0055A4",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#EFF6FF", edgecolor="#93C5FD", lw=0.6),
        arrowprops=dict(arrowstyle="->", color="#0055A4", lw=1.0, connectionstyle="arc3,rad=-0.15"),
    )
    ax2.set_ylim(-0.25, 12.0)
    ax2.set_xlabel("Exact logistic argument $z$")
    ax2.set_ylabel(r"Bias $|\mathbb{E}[\tilde{p}] - \sigma(z)| \times 10^3$")
    ax2.set_title(r"(b) Local flip-probability bias ($\tilde{\tau}=0.5$)", pad=4.0)
    ax2.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        fontsize=6.9,
        columnspacing=0.8,
    )
    clean_axes(ax2)

    fig.tight_layout(pad=0.6)
    fig.savefig(FIG_DIR / "fig1_mechanism.pdf")
    plt.close(fig)


def make_fig2_a9a_cauchy() -> None:
    d1 = json.loads((RES_DIR / "exp1_a9a.json").read_text(encoding="utf-8"))
    d2 = json.loads((RES_DIR / "exp2_cauchy.json").read_text(encoding="utf-8"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.95))

    sig1 = np.asarray(d1["sigmas"], dtype=float)
    keys1 = [f"{s:.0e}" for s in sig1]
    methods_1 = [
        ("SGLD", "--", False, 0.0),
        ("SVRG-SGLD", "--", False, 0.08),
        ("v-SGBD", ":", False, 0.0),
        ("c-SGBD", "-.", False, 0.08),
        ("NaiveVR-SGBD", "-", False, 0.08),
        ("UniVR-SGBD", "-", False, 0.10),
        ("SaVR-BD", "-", True, 0.16),
    ]
    a9a_last_means = {}
    a9a_last_stds = {}
    for m, ls, is_hero, ab in methods_1:
        means = [float(np.nanmean([r["mmd2"] for r in d1["results"][m][k]])) for k in keys1]
        stds = [float(np.nanstd([r["mmd2"] for r in d1["results"][m][k]])) for k in keys1]
        a9a_last_means[m] = means[-1]
        a9a_last_stds[m] = stds[-1]
        plot_ribbon_curve(
            ax1, sig1, means, stds, label=m, color=COLORS[m],
            is_hero=is_hero, linestyle=ls, alpha_band=ab, zorder=5 if is_hero else 3
        )
    ax1.axhline(d1["ref_mmd2_floor"], color="#111111", linestyle=":", linewidth=1.0, label="HMC floor")
    ax1.set_xscale("log")
    ax1.set_ylim(-0.02, 0.73)
    ax1.set_xlabel(r"Proposal scale $\sigma$ ($h=\sigma^2$ for Langevin)")
    ax1.set_ylabel(r"Gaussian $\mathrm{MMD}^2$ to HMC")
    ax1.set_title("(a) UCI a9a posterior fit ($d=124$)")
    clean_axes(ax1)

    # Embedded vertical bar zoom inset in the upper-left open canvas of Panel (a)
    # showing the VR-Barker separation at sigma = 10^-1 (0.082 vs 0.076 vs 0.069) without left y-tick protrusion
    ax_in = ax1.inset_axes([0.08, 0.50, 0.44, 0.42])
    ax_in.set_facecolor("#F8FAFC")
    vr_order = ["NaiveVR-SGBD", "UniVR-SGBD", "SaVR-BD"]
    vr_short = ["VR-naive", "UniVR", "SaVR-BD"]
    x_pos = np.arange(len(vr_order))
    for idx, m in enumerate(vr_order):
        mu = a9a_last_means[m]
        sd = a9a_last_stds[m]
        col = COLORS[m]
        ax_in.bar(idx, mu, width=0.55, color=col, alpha=0.90 if m == "SaVR-BD" else 0.74)
        ax_in.errorbar(idx, mu, yerr=sd, fmt="none", ecolor="#1E293B", elinewidth=0.85, capsize=1.8)
        ax_in.text(
            idx, mu + sd + 0.003, f"{mu:.3f}",
            ha="center", va="bottom", fontsize=5.9,
            fontweight="bold" if m == "SaVR-BD" else "normal", color="#0F172A"
        )
    ax_in.set_xticks(x_pos, vr_short, fontsize=5.9)
    ax_in.set_ylim(0.045, 0.108)
    ax_in.set_yticks([0.05, 0.08])
    ax_in.tick_params(axis="y", labelsize=5.8, pad=1.0)
    ax_in.tick_params(axis="x", pad=1.0)
    ax_in.set_title(r"Zoom at $\sigma=10^{-1}$", fontsize=6.3, fontweight="bold", pad=2.0, color="#1E293B")
    for sp in ("top", "right"):
        ax_in.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax_in.spines[sp].set_linewidth(0.6)
        ax_in.spines[sp].set_color("#64748B")

    sig2 = np.asarray(d2["sigmas"], dtype=float)
    keys2 = [f"{s:.0e}" for s in sig2]
    methods_2 = [
        ("SVRG-SGLD", "--", False, 0.08),
        ("v-SGBD", ":", False, 0.0),
        ("c-SGBD", "-.", False, 0.08),
        ("NaiveVR-SGBD", "-", False, 0.09),
        ("UniVR-SGBD", "-", False, 0.10),
        ("SaVR-BD", "-", True, 0.16),
    ]
    for m, ls, is_hero, ab in methods_2:
        means = [float(np.nanmean([r["mmd2"] for r in d2["results"][m][k]])) for k in keys2]
        stds = [float(np.nanstd([r["mmd2"] for r in d2["results"][m][k]])) for k in keys2]
        plot_ribbon_curve(
            ax2, sig2, means, stds, label=m, color=COLORS[m],
            is_hero=is_hero, linestyle=ls, alpha_band=ab, zorder=5 if is_hero else 3, clip_low=1e-4
        )
    ax2.axhline(d2["ref_mmd2_floor"], color="#111111", linestyle=":", linewidth=1.0, label="HMC floor")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_ylim(4.8e-3, 0.45)
    ax2.set_xlabel(r"Proposal scale $\sigma$")
    ax2.set_ylabel(r"Gaussian $\mathrm{MMD}^2$ to HMC")
    ax2.set_title("(b) Heavy-tailed Cauchy fit ($d=20$)")
    clean_axes(ax2, log_y=True)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=True,
        framealpha=0.95,
        fontsize=7.2,
        columnspacing=1.0,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.84], pad=0.6)
    fig.savefig(FIG_DIR / "fig2_a9a_cauchy.pdf")
    plt.close(fig)


def make_fig3_mismatch() -> None:
    d4 = json.loads((RES_DIR / "exp4_mismatch.json").read_text(encoding="utf-8"))
    lambdas = np.asarray(d4["lambdas"], dtype=float)
    keys = [f"lam={l:.2f}" for l in lambdas]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.90))

    methods = [
        ("SGLD-CV", ":", False, 0.0),
        ("SVRG-SGLD", "--", False, 0.10),
        ("c-SGBD", "-.", False, 0.08),
        ("NaiveVR-SGBD", "-", False, 0.08),
        ("UniVR-SGBD", "-", False, 0.10),
        ("SaVR-BD", "-", True, 0.16),
    ]

    bias_table = {}
    mmd_table = {}
    for m, ls, is_hero, ab in methods:
        means, stds = [], []
        mmd_means, mmd_stds = [], []
        for k in keys:
            vals = np.asarray([r["bias_coord0"] for r in d4["results"][m][k]], dtype=float)
            means.append(float(np.nanmean(vals)))
            stds.append(float(np.nanstd(vals)) / np.sqrt(len(vals)))
            mv = np.asarray([r["mmd2"] for r in d4["results"][m][k]], dtype=float)
            mmd_means.append(float(np.nanmean(mv)))
            mmd_stds.append(float(np.nanstd(mv)) / np.sqrt(len(mv)))
        bias_table[m] = (means, stds)
        mmd_table[m] = (mmd_means, mmd_stds)

        plot_ribbon_curve(
            ax1, lambdas, means, stds, label=m, color=COLORS[m],
            is_hero=is_hero, linestyle=ls, alpha_band=ab, zorder=5 if is_hero else 3, clip_low=1e-4
        )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.invert_xaxis()
    ax1.set_ylim(2.8e-4, 24.0)
    ax1.set_xlabel(r"Stiffness scale parameter $\lambda$")
    ax1.set_ylabel(r"$|\bar{\theta}_1 - \mathbb{E}_{\pi}[\theta_1]|$")
    ax1.set_title("(a) Stiff-coordinate posterior mean error")
    clean_axes(ax1, log_y=True)

    # Embedded compact vertical bar zoom inset in Panel (a) upper-left showing lambda=0.01 (in 10^-3 units)
    ax_in = ax1.inset_axes([0.08, 0.52, 0.38, 0.40])
    ax_in.set_facecolor("#F8FAFC")
    vr_order = ["NaiveVR-SGBD", "UniVR-SGBD", "SaVR-BD"]
    vr_short = ["VR-naive", "UniVR", "SaVR-BD"]
    for idx, m in enumerate(vr_order):
        mu_01 = bias_table[m][0][-1] * 1e3
        se_01 = bias_table[m][1][-1] * 1e3
        ax_in.bar(idx, mu_01, width=0.54, color=COLORS[m], alpha=0.90 if m == "SaVR-BD" else 0.74)
        ax_in.errorbar(idx, mu_01, yerr=se_01, fmt="none", ecolor="#1E293B", elinewidth=0.85, capsize=1.6)
        ax_in.text(
            idx, mu_01 + se_01 + 0.06, f"{mu_01:.2f}",
            ha="center", va="bottom", fontsize=5.8,
            fontweight="bold" if m == "SaVR-BD" else "normal"
        )
    ax_in.set_xticks(np.arange(3), vr_short, fontsize=5.7)
    ax_in.set_ylim(2.2, 3.85)
    ax_in.set_yticks([2.5, 3.2])
    ax_in.tick_params(axis="y", labelsize=5.7, pad=1.0)
    ax_in.tick_params(axis="x", pad=1.0)
    ax_in.set_title(r"$\lambda=0.01$ ($\times 10^{-3}$)", fontsize=6.0, fontweight="bold", pad=2.0, color="#1E293B")
    for sp in ("top", "right"):
        ax_in.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax_in.spines[sp].set_linewidth(0.6)
        ax_in.spines[sp].set_color("#64748B")

    # Panel (b): Grouped Multi-Regime Bar Chart of Joint MMD^2 across lambda in {1.0, 0.1, 0.01}
    # Eliminates overlapping 6-line log-log spaghetti!
    reg_indices = [0, 2, 4]  # lambda = 1.0, 0.1, 0.01
    reg_labels = [r"$\lambda=1.0$" + "\n(Well-cond.)", r"$\lambda=0.1$" + "\n(Stiff)", r"$\lambda=0.01$" + "\n(Extreme)"]
    bar_methods = ["SGLD-CV", "SVRG-SGLD", "c-SGBD", "NaiveVR-SGBD", "UniVR-SGBD", "SaVR-BD"]
    n_m = len(bar_methods)
    x_base = np.arange(len(reg_indices))
    total_w = 0.78
    bw = total_w / n_m

    for i_m, m in enumerate(bar_methods):
        offsets = x_base - total_w / 2.0 + (i_m + 0.5) * bw
        vals = [mmd_table[m][0][idx] for idx in reg_indices]
        ses = [mmd_table[m][1][idx] for idx in reg_indices]
        ax2.bar(
            offsets,
            vals,
            width=bw * 0.90,
            yerr=ses,
            color=COLORS[m],
            alpha=0.95 if m == "SaVR-BD" else 0.78,
            edgecolor="white",
            linewidth=0.4,
            error_kw=dict(elinewidth=0.7, ecolor="#1E293B", capsize=1.4),
        )

    ax2.set_yscale("log")
    ax2.set_ylim(1.2e-3, 0.65)
    ax2.set_xticks(x_base, reg_labels, fontsize=7.6)
    ax2.set_ylabel(r"Full-dimensional $\mathrm{MMD}^2$ to HMC")
    ax2.set_title("(b) Joint posterior discrepancy across regimes")
    clean_axes(ax2, log_y=True)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=6,
        frameon=True,
        framealpha=0.95,
        fontsize=7.0,
        columnspacing=0.8,
        handlelength=1.6,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.87], pad=0.6)
    fig.savefig(FIG_DIR / "fig3_mismatch.pdf")
    plt.close(fig)


def make_fig4_bnn() -> None:
    """
    Two-tier Broken Y-Axis (Split Upper Langevin Instability Tier vs. Lower Zoomed Barker Basin Tier)
    so that the 5 Barker curves in [0.812, 0.842] (RMSE) and [1.211, 1.247] (NLL) get 72% of the vertical
    height and separate cleanly instead of being compressed into a single braided rope at the bottom.
    """
    d5 = json.loads((RES_DIR / "exp5_bnn_uci.json").read_text(encoding="utf-8"))
    sigmas = np.asarray(d5["sigmas"], dtype=float)
    keys = [f"{s:.0e}" for s in sigmas]

    fig = plt.figure(figsize=(7.1, 3.05))
    outer = GridSpec(1, 2, figure=fig, wspace=0.25, left=0.08, right=0.98, bottom=0.15, top=0.78)

    gs_left = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], height_ratios=[1.0, 2.35], hspace=0.10)
    gs_right = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], height_ratios=[1.0, 2.35], hspace=0.10)

    ax1_top = fig.add_subplot(gs_left[0])
    ax1_bot = fig.add_subplot(gs_left[1], sharex=ax1_top)
    ax2_top = fig.add_subplot(gs_right[0])
    ax2_bot = fig.add_subplot(gs_right[1], sharex=ax2_top)

    method_specs = [
        ("SGLD", "--", False, 0.08),
        ("SVRG-SGLD", "-.", False, 0.08),
        ("v-SGBD", ":", False, 0.0),
        ("c-SGBD", "--", False, 0.06),
        ("NaiveVR-SGBD", "-", False, 0.08),
        ("UniVR-SGBD", "-", False, 0.08),
        ("SaVR-BD", "-", True, 0.16),
    ]

    for m, ls, is_hero, ab in method_specs:
        rmses, r_stds, nlls, n_stds, valid_s = [], [], [], [], []
        for s, k in zip(sigmas, keys):
            runs = [r for r in d5["results"][m][k] if not r["diverged"]]
            if len(runs) >= 3:
                valid_s.append(float(s))
                rmses.append(float(np.mean([r["rmse"] for r in runs])))
                r_stds.append(float(np.std([r["rmse"] for r in runs])))
                nlls.append(float(np.mean([r["nll"] for r in runs])))
                n_stds.append(float(np.std([r["nll"] for r in runs])))
        if not valid_s:
            continue
        is_langevin = m in ("SGLD", "SVRG-SGLD")
        target_ax1 = ax1_top if is_langevin else ax1_bot
        target_ax2 = ax2_top if is_langevin else ax2_bot

        # Also plot dummy on ax1_top for unified legend collection
        if not is_langevin:
            ax1_top.plot([], [], label=m, color=COLORS[m], linestyle=ls, linewidth=2.1 if is_hero else 1.35, marker="o", markersize=3.8 if is_hero else 3.0, markeredgecolor="white", markeredgewidth=0.5)

        plot_ribbon_curve(
            target_ax1, valid_s, rmses, r_stds, label=m if is_langevin else "_nolegend_",
            color=COLORS[m], is_hero=is_hero, linestyle=ls, alpha_band=ab, zorder=5 if is_hero else 3
        )
        plot_ribbon_curve(
            target_ax2, valid_s, nlls, n_stds, label="_nolegend_",
            color=COLORS[m], is_hero=is_hero, linestyle=ls, alpha_band=ab, zorder=5 if is_hero else 3
        )

    # Warm-start reference and Divergence zone badges
    ax1_bot.axhline(0.8255, color="#475569", linestyle=":", linewidth=0.95)
    ax1_bot.text(1.02e-3, 0.8272, "L-BFGS warm start (0.8255)", fontsize=6.2, color="#334155")

    for ax_t in (ax1_top, ax2_top):
        ax_t.axvspan(5.2e-3, 1.08e-2, color="#FEE2E2", alpha=0.65)
        ax_t.text(
            7.4e-3,
            0.898 if ax_t is ax1_top else 1.318,
            "Langevin\nDIV (5/5)",
            ha="center",
            va="center",
            fontsize=6.2,
            fontweight="bold",
            color="#B91C1C",
        )

    ax1_top.set_ylim(0.870, 0.924)
    ax1_bot.set_ylim(0.809, 0.844)
    ax2_top.set_ylim(1.280, 1.352)
    ax2_bot.set_ylim(1.208, 1.250)

    for ax_t, ax_b in ((ax1_top, ax1_bot), (ax2_top, ax2_bot)):
        ax_t.set_xscale("log")
        ax_b.set_xscale("log")
        clean_axes(ax_t)
        clean_axes(ax_b)
        ax_t.spines["bottom"].set_visible(False)
        ax_t.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_b.set_xlabel(r"Proposal scale $\sigma$")

        # Draw diagonal axis-break slashes on the left spine
        d = 0.015
        kwargs = dict(transform=ax_t.transAxes, color="#334155", clip_on=False, linewidth=0.8)
        ax_t.plot((-d, +d), (-d, +d), **kwargs)
        kwargs.update(transform=ax_b.transAxes)
        ax_b.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    ax1_top.set_title("(a) UCI Wine Quality BNN test RMSE", pad=5.0)
    ax2_top.set_title("(b) UCI Wine Quality BNN test NLL", pad=5.0)
    ax1_bot.set_ylabel("Held-out test RMSE", y=0.68)
    ax2_bot.set_ylabel("Held-out test NLL", y=0.68)

    handles, labels = ax1_top.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 1.01),
        ncol=4,
        frameon=True,
        framealpha=0.95,
        fontsize=7.2,
        columnspacing=1.0,
    )
    fig.savefig(FIG_DIR / "fig4_bnn_uci.pdf")
    plt.close(fig)


def make_fig5_ablations() -> None:
    d6 = json.loads((RES_DIR / "exp6_ablations.json").read_text(encoding="utf-8"))
    comp = d6["component_ablation"]
    rho = d6["rho_star_ablation"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.75))

    order = ["v-SGBD", "c-SGBD", "sc-SGBD", "NaiveVR-SGBD", "HardGate-SGBD", "UniVR-SGBD", "SaVR-BD"]
    short_labels = ["v-\nSGBD", "c-\nSGBD", "sc-\nSGBD", "VR\nnaive", "VR\nhard", "UniVR\nSGBD", "SaVR\nBD"]
    means = [comp[m]["mmd2_mean"] for m in order]
    stds = [comp[m]["mmd2_std"] for m in order]
    cols = [COLORS[m] for m in order]
    xpos = np.arange(len(order))

    ax1.axvspan(-0.5, 2.5, color="#FEF3C7", alpha=0.35)
    ax1.axvspan(2.5, 6.5, color="#EFF6FF", alpha=0.45)
    bars = ax1.bar(
        xpos,
        means,
        yerr=stds,
        color=cols,
        width=0.56,
        alpha=0.92,
        edgecolor="white",
        linewidth=0.5,
        error_kw=dict(elinewidth=0.85, ecolor="#1E293B", capsize=2.2),
    )
    ax1.set_yscale("log")
    ax1.set_ylim(7.5e-3, 0.56)
    ax1.set_xticks(xpos, short_labels, fontsize=6.9)
    ax1.set_ylabel(r"Gaussian $\mathrm{MMD}^2$ to HMC")
    ax1.set_title(r"(a) Component ablation ($\sigma=0.02$)", pad=4.0)
    for bar, val, st in zip(bars, means, stds):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            (val + st) * 1.12,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.0,
            fontweight="bold" if val < 0.015 else "normal",
            color="#0F172A",
        )
    clean_axes(ax1, log_y=True)

    r_keys = sorted(rho.keys(), key=float)
    r_vals = np.asarray([float(k) for k in r_keys], dtype=float)
    r_means = np.asarray([rho[k]["mmd2_mean"] for k in r_keys], dtype=float)
    r_stds = np.asarray([rho[k]["mmd2_std"] for k in r_keys], dtype=float)

    ax2.axvspan(1.85, 5.15, color="#ECFDF5", alpha=0.70, label=r"Robust plateau ($\rho^*\in[2,5]$)")
    plot_ribbon_curve(
        ax2, r_vals, r_means, r_stds, label=r"SaVR-BD ($\pm 1$ SD)",
        color=COLORS["SaVR-BD"], is_hero=True, alpha_band=0.18
    )
    ax2.axvline(2.0, color="#D95F02", linestyle="--", linewidth=1.2, label=r"Theory default ($\rho^*=2$)")
    ax2.set_ylim(0.0112, 0.0222)
    ax2.set_xlim(0.35, 5.2)
    ax2.set_xlabel(r"Saturation scale $\rho^*$")
    ax2.set_ylabel(r"Gaussian $\mathrm{MMD}^2$ to HMC")
    ax2.set_title(r"(b) Sensitivity to saturation scale $\rho^*$", pad=4.0)
    ax2.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=2,
        frameon=True,
        framealpha=0.95,
        fontsize=6.6,
        columnspacing=0.6,
    )
    clean_axes(ax2)

    fig.tight_layout(pad=0.6)
    fig.savefig(FIG_DIR / "fig5_ablations.pdf")
    plt.close(fig)


def make_fig6_diagnostics() -> None:
    d7 = json.loads((RES_DIR / "exp7_diagnostics.json").read_text(encoding="utf-8"))
    bcurves = d7["budget_curves"]
    acfs = d7["acf_curves"]
    lags = np.asarray(d7["acf_lags"], dtype=float)
    st = d7["stiff_trace"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.2, 2.85))

    spec_style = {
        "c-SGBD": ("c-SGBD", COLORS["c-SGBD"], "--", False, 0.08),
        "SGLD (best stable)": ("SGLD (tuned)", COLORS["SGLD"], ":", False, 0.06),
        "SVRG-SGLD": ("SVRG-SGLD", COLORS["SVRG-SGLD"], "--", False, 0.08),
        "NaiveVR-SGBD": ("NaiveVR-SGBD", COLORS["NaiveVR-SGBD"], "-", False, 0.08),
        "SaVR-BD (SVRG)": ("SaVR-BD (SVRG)", COLORS["SaVR-BD (SVRG)"], "-", True, 0.15),
        "SaVR-BD (SAGA)": ("SaVR-BD (SAGA)", COLORS["SaVR-BD (SAGA)"], "-", True, 0.15),
    }

    for key, (lbl, col, ls, is_hero, ab) in spec_style.items():
        cur = bcurves[key]
        plot_ribbon_curve(
            ax1,
            cur["passes"],
            cur["mmd2_mean"],
            cur["mmd2_std"],
            label=lbl,
            color=col,
            is_hero=is_hero,
            linestyle=ls,
            alpha_band=ab,
            zorder=5 if is_hero else 3,
            clip_low=1e-4,
        )
    ax1.axhline(d7["cauchy_ref_floor"], color="#111111", linestyle=":", linewidth=1.0, label="HMC floor")
    ax1.set_yscale("log")
    ax1.set_ylim(4.8e-3, 0.38)
    ax1.set_xlabel("Data passes (grad evals / $N$)")
    ax1.set_ylabel(r"Gaussian $\mathrm{MMD}^2$ to HMC")
    ax1.set_title("(a) Equal-budget accuracy")
    clean_axes(ax1, log_y=True)

    max_show = 40
    x_lags = lags[: max_show + 1]
    for key in ["SGLD (best stable)", "c-SGBD", "NaiveVR-SGBD", "SaVR-BD (SVRG)", "SaVR-BD (SAGA)"]:
        lbl, col, ls, is_hero, _ = spec_style[key]
        y_acf = np.asarray(acfs[key][: max_show + 1], dtype=float)
        if is_hero:
            ax2.fill_between(x_lags, 0, np.maximum(y_acf, 0), color=col, alpha=0.12)
        ax2.plot(
            x_lags,
            y_acf,
            label=lbl,
            color=col,
            linestyle=ls,
            linewidth=2.0 if is_hero else 1.3,
            alpha=1.0 if is_hero else 0.85,
            zorder=5 if is_hero else 3,
        )
    ax2.axhline(0.0, color="#475569", linestyle=":", linewidth=0.8)
    ax2.set_ylim(-0.06, 1.05)
    ax2.set_xlabel("Lag (iterations)")
    ax2.set_ylabel("Mean coordinate ACF")
    ax2.set_title("(b) Mixing autocorrelation")
    clean_axes(ax2)

    steps = np.arange(1, 181)
    tr_svrg = np.asarray(st["traces"]["SVRG-SGLD"][:180], dtype=float)
    tr_savr = np.asarray(st["traces"]["SaVR-BD"][:180], dtype=float)
    ax3.plot(steps, tr_svrg, color=COLORS["SVRG-SGLD"], linewidth=0.90, alpha=0.82, label="SVRG-SGLD")
    ax3.plot(steps, tr_savr, color=COLORS["SaVR-BD"], linewidth=1.45, label="SaVR-BD (SVRG)")
    ax3.axhspan(st["hmc_q025"], st["hmc_q975"], color="#1E293B", alpha=0.22, label="HMC 95% CI")
    ax3.axhline(st["hmc_mean"], color="#0F172A", linestyle=":", linewidth=0.9)
    ax3.set_yscale("symlog", linthresh=0.1)
    ax3.set_ylim(-25.0, 25.0)
    ax3.set_xlabel("Post-burn-in iteration")
    ax3.set_ylabel(r"Stiff coordinate $\theta_{t,1}$")
    ax3.set_title(r"(c) Stiff trace ($\lambda=0.03$)")
    clean_axes(ax3)

    h1, l1 = ax1.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    for hh, ll in zip(h3, l3):
        if ll == "HMC 95% CI":
            h1.append(hh)
            l1.append(ll)
    fig.legend(
        h1,
        l1,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=True,
        framealpha=0.95,
        fontsize=6.9,
        columnspacing=0.9,
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.84], pad=0.55)
    fig.savefig(FIG_DIR / "fig6_diagnostics.pdf")
    plt.close(fig)


def make_fig7_scalability_dasfaa() -> None:
    """
    Panel (a): Shaded ribbon scaling curves across N in [2.5k, 100k].
    Panel (b): Horizontal Paired Bar Chart comparing execution latency per 100 steps at
               N = 2,500 rows vs. N = 100,000 rows (40x table growth), visually demonstrating
               O(1) constant per-step query latency without another flat dashed line chart.
    """
    DASFAA_FIG_DIR.mkdir(parents=True, exist_ok=True)
    d8 = json.loads((RES_DIR / "exp8_scalability.json").read_text(encoding="utf-8"))
    res = d8["results"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.88))
    spec_style = {
        "SGLD": ("SGLD", COLORS["SGLD"], ":", False, 0.0),
        "SVRG-SGLD": ("SVRG-SGLD", COLORS["SVRG-SGLD"], "--", False, 0.08),
        "v-SGBD": ("v-SGBD", COLORS["v-SGBD"], ":", False, 0.06),
        "c-SGBD": ("c-SGBD", COLORS["c-SGBD"], "--", False, 0.08),
        "NaiveVR-SGBD": ("NaiveVR-SGBD", COLORS["NaiveVR-SGBD"], "-", False, 0.09),
        "SaVR-BD (SVRG)": ("SaVR-BD (SVRG)", COLORS["SaVR-BD (SVRG)"], "-", True, 0.15),
        "SaVR-BD (SAGA)": ("SaVR-BD (SAGA)", COLORS["SaVR-BD (SAGA)"], "-", True, 0.16),
    }

    for key, (lbl, col, ls, is_hero, ab) in spec_style.items():
        xs_valid, err_m, err_s = [], [], []
        for n_val in d8["N_grid"]:
            rec = res[key][str(n_val)]
            if rec["diverged_count"] < rec["n_seeds"] and np.isfinite(rec["param_err_mean"]):
                xs_valid.append(float(n_val))
                err_m.append(rec["param_err_mean"])
                err_s.append(rec["param_err_std"])
        if xs_valid:
            plot_ribbon_curve(
                ax1, xs_valid, err_m, err_s, label=lbl, color=col,
                is_hero=is_hero, linestyle=ls, alpha_band=ab, zorder=5 if is_hero else 3
            )

    ax1.annotate(
        "SGLD diverges\nat $N=10^5$",
        xy=(30000, 1.63),
        xytext=(5500, 5.2),
        fontsize=6.6,
        fontweight="bold",
        color="#64748B",
        arrowprops=dict(arrowstyle="->", color="#64748B", lw=0.85),
    )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylim(0.11, 15.0)
    ax1.set_xlabel("Table cardinality $N$ (rows)")
    ax1.set_ylabel(r"Posterior mean RMSE $\|\bar{\theta}-\theta^{\mathrm{MAP}}\|_2$")
    ax1.set_title("(a) Accuracy across table cardinality $N$")
    clean_axes(ax1, log_y=True)

    # Panel (b): Horizontal Paired Latency Bar Chart comparing N=2,500 vs N=100,000 (40x table growth)
    lat_order = ["v-SGBD", "SVRG-SGLD", "c-SGBD", "NaiveVR-SGBD", "SaVR-BD (SVRG)", "SaVR-BD (SAGA)"]
    lat_short = ["v-SGBD", "SVRG-SGLD", "c-SGBD", "VR-naive", "SaVR (SVRG)", "SaVR (SAGA)"]
    y_pos = np.arange(len(lat_order))
    lat_2k = [res[m]["2500"]["ms_per_100_steps"] for m in lat_order]
    lat_100k = [res[m]["100000"]["ms_per_100_steps"] for m in lat_order]

    bh = 0.36
    ax2.barh(y_pos - bh / 2, lat_2k, height=bh * 0.92, color="#94A3B8", alpha=0.85, label="$N=2{,}500$ rows")
    bars_100k = ax2.barh(
        y_pos + bh / 2,
        lat_100k,
        height=bh * 0.92,
        color=[COLORS[m] for m in lat_order],
        alpha=0.92,
        label="$N=100{,}000$ rows ($40\\times$)",
    )
    for bar, v100 in zip(bars_100k, lat_100k):
        ax2.text(
            v100 + 0.8,
            bar.get_y() + bar.get_height() / 2.0,
            f"{v100:.1f} ms",
            va="center",
            ha="left",
            fontsize=6.3,
            fontweight="bold",
            color="#0F172A",
        )

    ax2.set_yticks(y_pos, lat_short, fontsize=7.2)
    ax2.invert_yaxis()
    ax2.set_xlim(0.0, 52.5)
    ax2.set_xlabel("Latency / 100 steps (ms; grey: 2.5k, color: 100k)")
    ax2.set_title("(b) Constant $O(1)$ per-step latency ($40\\times$ rows)")
    clean_axes(ax2)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=True,
        framealpha=0.95,
        fontsize=7.1,
        columnspacing=0.9,
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.84], pad=0.6)
    fig.savefig(DASFAA_FIG_DIR / "fig_scalability.pdf")
    fig.savefig(FIG_DIR / "fig7_scalability.pdf")
    plt.close(fig)

    for fn in [
        "fig0_overview.pdf",
        "fig1_mechanism.pdf",
        "fig2_a9a_cauchy.pdf",
        "fig3_mismatch.pdf",
        "fig4_bnn_uci.pdf",
        "fig5_ablations.pdf",
        "fig6_diagnostics.pdf",
    ]:
        src = FIG_DIR / fn
        if src.exists() and src.parent != DASFAA_FIG_DIR:
            shutil.copy2(src, DASFAA_FIG_DIR / fn)


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    make_fig1_mechanism()
    make_fig2_a9a_cauchy()
    make_fig3_mismatch()
    make_fig4_bnn()
    make_fig5_ablations()
    make_fig6_diagnostics()
    make_fig7_scalability_dasfaa()
    print(f"Generated all redesigned vector figures in {FIG_DIR} and {DASFAA_FIG_DIR}")
