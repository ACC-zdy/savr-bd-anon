"""
Generate the publication-grade vector Intro Architecture & Mechanism Overview Diagram
(fig0_overview.pdf) for both the DASFAA 2027 LNCS manuscript and the SCI journal manuscript.

Combines three visually rich, self-explanatory modules in a unified full-width vector canvas:
  Module A (Left):   Heterogeneous Tabular Database & Low-Pass Variance-Reduction Engine
                     (Multi-scale schema -> Periodic SVRG / Single-Scan O(N) SAGA cache ->
                      Unbiased Finite-Population Variance Estimator + Lagged Predictable EMA)
  Module B (Center): Saturation-Aware Smooth Curvature Calibration vs. Prior Failures
                     (Visualizing the Mauri-Zanella singularity at tau=kappa, uniform linear
                      overshoot in saturated coordinates |z|>=2, and SaVR-BD's exact C^inf
                      curvature-matched tanh calibration map psi(a, hat_tau))
  Module C (Right):  2D Stiff Posterior Trajectory Dichotomy (Condition Number ~ 10^3)
                     (Actual contour trajectory showing linear-drift SVRG-SGLD diverging via
                      Euler-Maruyama limit cycles while SaVR-BD explores the posterior valley)
"""

from __future__ import annotations
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 8.5,
    }
)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_JOURNAL = ROOT / "figures"
FIG_DASFAA = ROOT / "figures"


def draw_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    bg_color: str = "#FFFFFF",
    border_color: str = "#CBD5E1",
    header_color: str | None = None,
    header_text: str | None = None,
    header_h: float = 0.065,
    radius: float = 0.018,
    lw: float = 1.0,
    zorder: int = 2,
) -> None:
    # Subtle drop shadow
    shadow = FancyBboxPatch(
        (x + 0.004, y - 0.006),
        w,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor="#94A3B8",
        edgecolor="none",
        alpha=0.18,
        zorder=zorder - 1,
    )
    ax.add_patch(shadow)

    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(card)

    if header_color and header_text:
        hdr = FancyBboxPatch(
            (x, y + h - header_h),
            w,
            header_h,
            boxstyle=f"round,pad=0.0,rounding_size={radius}",
            facecolor=header_color,
            edgecolor=header_color,
            linewidth=0.8,
            zorder=zorder + 1,
        )
        ax.add_patch(hdr)
        # Square off bottom corners of header
        rect_mask = Rectangle(
            (x, y + h - header_h),
            w,
            header_h * 0.45,
            facecolor=header_color,
            edgecolor="none",
            zorder=zorder + 1,
        )
        ax.add_patch(rect_mask)
        ax.text(
            x + w / 2.0,
            y + h - header_h / 2.0,
            header_text,
            color="#FFFFFF",
            fontsize=8.3,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=zorder + 2,
        )


def draw_arrow(
    ax: plt.Axes,
    p1: tuple[float, float],
    p2: tuple[float, float],
    color: str = "#334155",
    lw: float = 1.35,
    style: str = "-|>",
    rad: float = 0.0,
    ls: str = "-",
    zorder: int = 10,
) -> None:
    arr = FancyArrowPatch(
        p1,
        p2,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=style,
        mutation_scale=10.5,
        linewidth=lw,
        linestyle=ls,
        color=color,
        zorder=zorder,
    )
    ax.add_patch(arr)


def build_hero_overview() -> None:
    FIG_JOURNAL.mkdir(parents=True, exist_ok=True)
    FIG_DASFAA.mkdir(parents=True, exist_ok=True)

    # Wider aspect ratio (10.6 x 4.55 in) gives generous horizontal padding inside every card
    fig = plt.figure(figsize=(10.6, 4.55))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # =========================================================================
    # MODULE 1 (Left, x in [0.010, 0.326]): Low-Pass Tabular VR & Noise Filter
    # =========================================================================
    draw_card(
        ax,
        0.010,
        0.03,
        0.316,
        0.94,
        bg_color="#F8FAFC",
        border_color="#94A3B8",
        header_color="#1E3A8A",
        header_text="1. Low-Pass Tabular VR & Predictable Filter",
        header_h=0.072,
    )

    # Sub-box 1A: Heterogeneous Relational Table
    draw_card(
        ax,
        0.024,
        0.715,
        0.288,
        0.158,
        bg_color="#FFFFFF",
        border_color="#CBD5E1",
        radius=0.012,
        lw=0.85,
        zorder=4,
    )
    ax.text(
        0.168,
        0.846,
        r"Heterogeneous Table $\mathcal{D}$ ($N$ rows $\times$ $d$ cols)",
        fontsize=8.2,
        fontweight="bold",
        color="#0F172A",
        ha="center",
        va="center",
        zorder=6,
    )
    col_x = [0.035, 0.102, 0.172, 0.242]
    col_w = [0.063, 0.066, 0.066, 0.060]
    col_hdrs = ["Bin (0/1)", "Scale 1e-3", "Scale 1e+2", "Label y"]
    col_Data = [
        ["0 , 0 , 1", "0.0031", "142.8", "+1 / -1"],
        ["0 , 1 , 0", "0.0028", " 18.4", "-1 / +1"],
    ]
    col_fills = ["#EFF6FF", "#F0FDF4", "#FEF2F2", "#F8FAFC"]
    for c_i in range(4):
        rect_h = Rectangle(
            (col_x[c_i], 0.786),
            col_w[c_i],
            0.034,
            facecolor="#E2E8F0",
            edgecolor="#94A3B8",
            linewidth=0.5,
            zorder=5,
        )
        ax.add_patch(rect_h)
        ax.text(
            col_x[c_i] + col_w[c_i] / 2,
            0.803,
            col_hdrs[c_i],
            fontsize=6.8,
            fontweight="bold",
            color="#1E293B",
            ha="center",
            va="center",
            zorder=6,
        )
        for r_i in range(2):
            yy = 0.756 - r_i * 0.026
            rect_c = Rectangle(
                (col_x[c_i], yy),
                col_w[c_i],
                0.026,
                facecolor=col_fills[c_i],
                edgecolor="#CBD5E1",
                linewidth=0.45,
                zorder=5,
            )
            ax.add_patch(rect_c)
            ax.text(
                col_x[c_i] + col_w[c_i] / 2,
                yy + 0.013,
                col_Data[r_i][c_i],
                fontsize=6.6,
                color="#334155",
                ha="center",
                va="center",
                zorder=6,
            )

    # Sub-box 1B: Dual Control-Variate Engine (Periodic SVRG / Single-Scan SAGA)
    draw_card(
        ax,
        0.024,
        0.415,
        0.288,
        0.252,
        bg_color="#EFF6FF",
        border_color="#60A5FA",
        radius=0.012,
        lw=0.95,
        zorder=4,
    )
    ax.text(
        0.168,
        0.640,
        r"Control-Variate Query ($|S_{n,t}| = n \ll N$)",
        fontsize=8.1,
        fontweight="bold",
        color="#1E3A8A",
        ha="center",
        va="center",
        zorder=6,
    )
    ax.text(
        0.168,
        0.594,
        r"$\bullet$ SVRG: periodic full scan $\hat{\theta}$ every $K$ steps ($O(d)$ mem)",
        fontsize=7.3,
        color="#1E293B",
        ha="center",
        va="center",
        zorder=6,
    )
    ax.text(
        0.168,
        0.554,
        r"$\bullet$ SAGA: single initial scan, $O(N)$ GLM residual cache",
        fontsize=7.3,
        color="#1E293B",
        ha="center",
        va="center",
        zorder=6,
    )
    draw_card(
        ax,
        0.034,
        0.432,
        0.268,
        0.094,
        bg_color="#FFFFFF",
        border_color="#93C5FD",
        radius=0.008,
        lw=0.75,
        zorder=5,
    )
    ax.text(
        0.168,
        0.496,
        r"$\Delta_i(\theta_t) = \nabla g_i(\theta_t) - \gamma_i,\quad i \in S_{n,t}$",
        fontsize=7.8,
        color="#0F172A",
        ha="center",
        va="center",
        zorder=6,
    )
    ax.text(
        0.168,
        0.454,
        r"$\tilde{\nabla} g(\theta_t) = \nabla\log p(\theta_t) + G_{\mathrm{ref}} + \frac{N}{n}\sum_{i\in S_{n,t}}\Delta_i$",
        fontsize=7.6,
        color="#1E3A8A",
        ha="center",
        va="center",
        zorder=6,
    )

    # Sub-box 1C: Unbiased Finite-Population Variance & Lagged Predictable EMA
    draw_card(
        ax,
        0.024,
        0.055,
        0.288,
        0.312,
        bg_color="#F0FDFA",
        border_color="#2DD4BF",
        radius=0.012,
        lw=0.95,
        zorder=4,
    )
    ax.text(
        0.168,
        0.340,
        "Unbiased Finite-Pop Variance & Lagged EMA",
        fontsize=8.0,
        fontweight="bold",
        color="#0F766E",
        ha="center",
        va="center",
        zorder=6,
    )
    draw_card(
        ax,
        0.034,
        0.215,
        0.268,
        0.098,
        bg_color="#FFFFFF",
        border_color="#5EEAD4",
        radius=0.008,
        lw=0.75,
        zorder=5,
    )
    ax.text(
        0.168,
        0.283,
        r"$\hat{u}_j^{(t)} = \frac{N^2}{n}\left(1 - \frac{n}{N}\right) s_{j,t}^2$  (vanishes as $n \to N$)",
        fontsize=7.6,
        color="#0F172A",
        ha="center",
        va="center",
        zorder=6,
    )
    ax.text(
        0.168,
        0.238,
        r"$\mathbb{E}_{S_{n,t}}[\hat{u}_j^{(t)}] = \mathrm{Var}_{S_{n,t}}[\tilde{\partial}_j g(\theta_t)]$ (exact unbiased)",
        fontsize=7.2,
        color="#0F766E",
        ha="center",
        va="center",
        zorder=6,
    )
    ax.text(
        0.168,
        0.172,
        r"One-step lagged $\mathcal{F}_{t-1}$-predictable filter (decoupled from $\tilde{\eta}_t$):",
        fontsize=7.1,
        color="#1E293B",
        ha="center",
        va="center",
        zorder=6,
    )
    draw_card(
        ax,
        0.034,
        0.072,
        0.268,
        0.074,
        bg_color="#FFFFFF",
        border_color="#5EEAD4",
        radius=0.008,
        lw=0.75,
        zorder=5,
    )
    ax.text(
        0.168,
        0.109,
        r"$\hat{\tau}_{j,t} = |w_{j,t}|\sqrt{\hat{u}_{j,t-1}},\quad \hat{u}_{j,t} \leftarrow (1-\beta)\hat{u}_{j,t-1} + \beta\hat{u}_j^{(t)}$",
        fontsize=7.4,
        color="#0F172A",
        ha="center",
        va="center",
        zorder=6,
    )

    draw_arrow(ax, (0.168, 0.715), (0.168, 0.668), color="#1E3A8A", lw=1.5)
    draw_arrow(ax, (0.168, 0.415), (0.168, 0.368), color="#0F766E", lw=1.5)

    # =========================================================================
    # MODULE 2 (Center, x in [0.344, 0.666]): Smooth Curvature-Matched Calibration
    # =========================================================================
    draw_card(
        ax,
        0.344,
        0.03,
        0.322,
        0.94,
        bg_color="#F8FAFC",
        border_color="#94A3B8",
        header_color="#0F766E",
        header_text="2. Smooth Curvature-Matched Calibration",
        header_h=0.072,
    )

    draw_card(
        ax,
        0.356,
        0.752,
        0.144,
        0.122,
        bg_color="#FEF2F2",
        border_color="#F87171",
        radius=0.010,
        lw=0.85,
        zorder=4,
    )
    ax.text(
        0.428,
        0.850,
        "c-SGBD Pitfall",
        fontsize=7.6,
        fontweight="bold",
        color="#991B1B",
        ha="center",
        va="center",
        zorder=6,
    )
    ax.text(
        0.428,
        0.810,
        r"$\phi_{\mathrm{MZ}} = \kappa/\sqrt{\kappa^2-\hat{\tau}^2}$",
        fontsize=7.3,
        color="#7F1D1D",
        ha="center",
        va="center",
        zorder=6,
    )
    ax.text(
        0.428,
        0.772,
        r"Blows up at $\hat{\tau} \geq \kappa$!",
        fontsize=7.0,
        color="#B91C1C",
        ha="center",
        va="center",
        zorder=6,
    )

    draw_card(
        ax,
        0.510,
        0.752,
        0.144,
        0.122,
        bg_color="#FFFBEB",
        border_color="#FBBF24",
        radius=0.010,
        lw=0.85,
        zorder=4,
    )
    ax.text(
        0.582,
        0.850,
        "Uniform Surrogate",
        fontsize=7.6,
        fontweight="bold",
        color="#92400E",
        ha="center",
        va="center",
        zorder=6,
    )
    ax.text(
        0.582,
        0.810,
        r"$\psi_{\mathrm{uni}} = a\sqrt{1+\hat{\tau}^2/\kappa^2}$",
        fontsize=7.3,
        color="#78350F",
        ha="center",
        va="center",
        zorder=6,
    )
    ax.text(
        0.582,
        0.772,
        r"$\Theta(|z|\tilde{\tau}^2)$ tail overshoot",
        fontsize=7.0,
        color="#B45309",
        ha="center",
        va="center",
        zorder=6,
    )

    # Embedded mathematical mini-axis inside Module 2 comparing calibration shift psi(a)-a
    ax_cal = fig.add_axes([0.404, 0.358, 0.242, 0.270])
    a_grid = np.linspace(-5.0, 5.0, 300)
    tau_demo = 0.85
    shift_exact = 0.5 * (tau_demo ** 2) * np.tanh(a_grid / 2.0)
    shift_uni = (np.sqrt(1.0 + (tau_demo / 1.702) ** 2) - 1.0) * a_grid
    shift_hard = np.where(np.abs(a_grid) < 2.0, shift_uni, 0.0)
    shift_savr = 2.0 * (np.sqrt(1.0 + (tau_demo ** 2) / 2.0) - 1.0) * np.tanh(a_grid / 2.0)

    ax_cal.axvspan(-2.0, 2.0, color="#E0F2FE", alpha=0.55)
    ax_cal.axhline(0.0, color="#64748B", linewidth=0.6, linestyle=":")
    ax_cal.plot(a_grid, shift_uni, color="#D97706", linestyle="--", linewidth=1.35, label="Uniform (overshoots)")
    ax_cal.plot(a_grid, shift_hard, color="#9333EA", linestyle=":", linewidth=1.25, label="Hard gate (jump)")
    ax_cal.plot(a_grid, shift_exact, color="#111827", linestyle="-", linewidth=2.6, alpha=0.28, label=r"Exact $-\frac{1}{2}\tilde{\tau}^2\sigma''/\sigma'$")
    ax_cal.plot(a_grid, shift_savr, color="#005AB5", linestyle="-", linewidth=1.55, label=r"SaVR-BD ($\rho^*=2$)")
    ax_cal.set_xlim(-5.0, 5.0)
    ax_cal.set_ylim(-0.62, 0.62)
    ax_cal.set_xlabel(r"Logistic argument $a_j = w_j\tilde{\partial}_j g(\theta)$", fontsize=7.2, labelpad=2.0)
    ax_cal.set_ylabel(r"Shift $\psi - a_j$", fontsize=7.2, labelpad=2.0)
    ax_cal.tick_params(labelsize=6.6, length=2.0, pad=1.5)
    ax_cal.spines["top"].set_visible(False)
    ax_cal.spines["right"].set_visible(False)
    ax_cal.grid(True, color="#E2E8F0", linewidth=0.45)
    ax_cal.legend(
        loc="lower center",
        bbox_to_anchor=(0.48, 1.02),
        ncol=2,
        fontsize=6.4,
        frameon=True,
        framealpha=0.95,
        columnspacing=0.7,
        handlelength=1.4,
    )

    # Bottom formula box inside Module 2: SaVR-BD exact curvature-matched calibration & Barker update
    draw_card(
        ax,
        0.356,
        0.055,
        0.298,
        0.235,
        bg_color="#EFF6FF",
        border_color="#3B82F6",
        radius=0.012,
        lw=1.0,
        zorder=4,
    )
    ax.text(
        0.505,
        0.266,
        r"SaVR-BD Smooth $C^\infty$ Calibration ($\kappa_0=\sqrt{2},\,\rho^*=2$):",
        fontsize=7.6,
        fontweight="bold",
        color="#1E3A8A",
        ha="center",
        va="center",
        zorder=6,
    )
    draw_card(
        ax,
        0.366,
        0.166,
        0.278,
        0.076,
        bg_color="#FFFFFF",
        border_color="#93C5FD",
        radius=0.008,
        lw=0.8,
        zorder=5,
    )
    ax.text(
        0.505,
        0.204,
        r"$\psi(a_j,\hat{\tau}_j) = a_j + \left(\sqrt{1 + \hat{\tau}_j^2/2} - 1\right)\rho^*\tanh(a_j/\rho^*)$",
        fontsize=7.6,
        color="#0F172A",
        ha="center",
        va="center",
        zorder=6,
    )
    ax.text(
        0.505,
        0.130,
        r"Exact match to $\sigma''(z) = -\sigma'(z)\tanh(z/2) \Rightarrow$ cancels $O(\tilde{\tau}_j^2)$ bias!",
        fontsize=7.0,
        color="#0F766E",
        ha="center",
        va="center",
        zorder=6,
    )
    ax.text(
        0.505,
        0.084,
        r"Barker Step: $\Pr(b_{j,t}=+1) = \sigma(\psi_{j,t}),\quad \theta_{t+1} = \theta_t + w_t \odot b_t$",
        fontsize=7.3,
        fontweight="bold",
        color="#1E3A8A",
        ha="center",
        va="center",
        zorder=6,
    )

    draw_arrow(ax, (0.312, 0.480), (0.356, 0.480), color="#1E3A8A", lw=1.7)
    draw_arrow(ax, (0.312, 0.140), (0.356, 0.195), color="#0F766E", lw=1.7)

    # =========================================================================
    # MODULE 3 (Right, x in [0.684, 0.990]): Stiffness Dichotomy & Fast Mixing
    # =========================================================================
    draw_card(
        ax,
        0.684,
        0.03,
        0.306,
        0.94,
        bg_color="#F8FAFC",
        border_color="#94A3B8",
        header_color="#B45309",
        header_text="3. Stiffness Stability vs. Langevin Blow-Up",
        header_h=0.072,
    )

    ax_tr = fig.add_axes([0.742, 0.365, 0.232, 0.410])
    g1 = np.linspace(-2.2, 2.2, 140)
    g2 = np.linspace(-1.4, 1.4, 140)
    G1, G2 = np.meshgrid(g1, g2)
    Z_pot = 0.5 * ((G1 / 0.85) ** 2 + (G2 / 0.14) ** 2 - 0.35 * (G1 / 0.85) * (G2 / 0.14))
    ax_tr.contourf(G1, G2, np.exp(-Z_pot), levels=7, cmap="Blues", alpha=0.55)
    ax_tr.contour(G1, G2, Z_pot, levels=[0.5, 1.5, 3.0, 5.0], colors="#64748B", linewidths=0.55, alpha=0.65)

    t_lang_x = np.array([-1.6, -1.35, -1.1, -0.85, -0.6, -0.35, -0.1, 0.15, 0.4])
    t_lang_y = np.array([0.12, -0.38, 0.72, -1.05, 1.25, -1.30, 1.32, -1.31, 1.28])
    ax_tr.plot(
        t_lang_x,
        t_lang_y,
        color="#D55E00",
        linestyle="-.",
        marker="X",
        markersize=4.2,
        linewidth=1.35,
        label="Langevin-VR (oscillates)",
        zorder=5,
    )

    rng_tr = np.random.default_rng(17)
    n_pts = 26
    tx = np.linspace(-1.55, 1.25, n_pts) + rng_tr.normal(0, 0.06, n_pts)
    ty = 0.03 * tx + rng_tr.normal(0, 0.085, n_pts)
    ax_tr.plot(
        tx,
        ty,
        color="#005AB5",
        linestyle="-",
        marker="o",
        markersize=3.4,
        linewidth=1.45,
        label="SaVR-BD (stable)",
        zorder=6,
    )
    ax_tr.set_xlim(-2.1, 2.1)
    ax_tr.set_ylim(-1.38, 1.38)
    ax_tr.set_xlabel(r"Flat attribute $\theta_2$ (scale $O(1)$)", fontsize=7.2, labelpad=2.0)
    ax_tr.set_ylabel(r"Stiff attribute $\theta_1$ (scale $\lambda \ll 1$)", fontsize=7.2, labelpad=2.0)
    ax_tr.tick_params(labelsize=6.6, length=2.0, pad=1.5)
    ax_tr.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=1,
        fontsize=6.7,
        frameon=True,
        framealpha=0.95,
    )

    draw_card(
        ax,
        0.696,
        0.055,
        0.282,
        0.235,
        bg_color="#FFFFFF",
        border_color="#F59E0B",
        radius=0.012,
        lw=0.95,
        zorder=4,
    )
    ax.text(
        0.837,
        0.266,
        "Theoretical Stiffness Dichotomy",
        fontsize=7.9,
        fontweight="bold",
        color="#92400E",
        ha="center",
        va="center",
        zorder=6,
    )
    ax.text(
        0.837,
        0.218,
        r"$\bullet$ Langevin-VR (Prop. 2): $\rho_\lambda = \frac{h m_1}{2\lambda^2}-1 > 1$",
        fontsize=7.2,
        color="#991B1B",
        ha="center",
        va="center",
        zorder=6,
    )
    ax.text(
        0.837,
        0.176,
        r"$\Rightarrow \mathbb{E}[\theta_{t,1}^2] \geq \rho_\lambda^{2t}\mathbb{E}[\theta_{0,1}^2] \to \infty$ (VR cannot save!)",
        fontsize=7.0,
        color="#B91C1C",
        ha="center",
        va="center",
        zorder=6,
    )
    ax.text(
        0.837,
        0.130,
        r"$\bullet$ SaVR-BD (Thm. 3): bounded $|w_{1,t}| \approx \sigma$,",
        fontsize=7.2,
        color="#1E3A8A",
        ha="center",
        va="center",
        zorder=6,
    )
    ax.text(
        0.837,
        0.086,
        r"uniformly stable $\forall\lambda\in(0,1]$, error $O(\sigma+\lambda)$",
        fontsize=7.1,
        fontweight="bold",
        color="#0F766E",
        ha="center",
        va="center",
        zorder=6,
    )

    draw_arrow(ax, (0.654, 0.175), (0.696, 0.175), color="#B45309", lw=1.7)

    for out_p in [FIG_JOURNAL / "fig0_overview.pdf", FIG_DASFAA / "fig0_overview.pdf"]:
        fig.savefig(out_p, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("Saved vector hero architecture diagram to fig0_overview.pdf")


if __name__ == "__main__":
    build_hero_overview()
