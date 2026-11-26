"""
Complete reproducible benchmark suite for the revised SaVR-BD journal manuscript.

Runs six self-contained experiments:
  1. exp1_a9a:           Real UCI Adult/a9a logistic regression (N=8000, d=124) against HMC gold reference
  2. exp2_cauchy:        Heavy-tailed Cauchy-prior linear regression (N=2000, d=20) against HMC reference
  3. exp3_heterogeneous: Ill-conditioned logistic regression (N=4000, d=30, cond ~ 10^6) against HMC reference
  4. exp4_mismatch:      20-seed coordinate-stiffness mismatch sweep (lambda in {1, 0.3, 0.1, 0.03, 0.01})
  5. exp5_bnn_uci:       Real UCI Wine Quality Bayesian neural network regression (N=5000, d=417)
  6. exp6_ablations:     Single-step Theorem 1 bias verification, component decomposition, rho* sensitivity, SVRG vs SAGA

All outputs are saved with seed-level provenance to results/*.json.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import wilcoxon
from sklearn.datasets import fetch_openml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.savr_bd import (
    KAPPA_CURVATURE,
    KAPPA_PROBIT,
    CorrectedSGBD,
    HardGateVRSGBD,
    NaiveVRSGBD,
    SaVRBD,
    SaVRBDConfig,
    StableCorrectedSGBD,
    UniformVRSGBD,
    VanillaSGBD,
    calibrated_argument,
    sigmoid,
)
from src.sgld_baselines import SGLD, SGLDCV, SGLDConfig, SVRGSGLD


# ============================================================================
# Shared Metrics & Reference HMC
# ============================================================================
def mmd_rbf_unbiased(
    X: np.ndarray,
    Y: np.ndarray,
    bandwidth_sq: float | None = None,
    max_pts: int = 800,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Unbiased U-statistic Gaussian RBF MMD^2 (Gretton et al., 2012) with fixed
    or median-heuristic squared bandwidth. Returns (mmd2, bandwidth_sq).
    """
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(Y))):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = min(max_pts, X.shape[0], Y.shape[0])
    Xs = X[rng.choice(X.shape[0], m, replace=False)]
    Ys = Y[rng.choice(Y.shape[0], m, replace=False)]

    if bandwidth_sq is None:
        sq_ref = ((Ys[:, None, :] - Ys[None, :, :]) ** 2).sum(-1)
        iu = np.triu_indices(m, k=1)
        bandwidth_sq = float(np.median(sq_ref[iu]) + 1e-8)

    Kxx = np.exp(-((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(-1) / bandwidth_sq)
    Kyy = np.exp(-((Ys[:, None, :] - Ys[None, :, :]) ** 2).sum(-1) / bandwidth_sq)
    Kxy = np.exp(-((Xs[:, None, :] - Ys[None, :, :]) ** 2).sum(-1) / bandwidth_sq)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)
    mmd2 = float(Kxx.sum() / (m * (m - 1)) + Kyy.sum() / (m * (m - 1)) - 2.0 * Kxy.mean())
    return max(mmd2, 0.0), bandwidth_sq


def min_ess_per_1k_grads(samples: np.ndarray, grad_evals: int, max_dims: int = 64) -> float:
    """Minimum coordinatewise ESS via FFT autocorrelation, normalized per 1,000 gradient evaluations."""
    if not np.all(np.isfinite(samples)) or grad_evals <= 0:
        return 0.0
    n, d = samples.shape
    dims = np.linspace(0, d - 1, min(d, max_dims), dtype=int)
    ess_list = []
    for j in dims:
        x = samples[:, j] - samples[:, j].mean()
        v = float(np.var(x))
        if v < 1e-14 or not np.isfinite(v):
            ess_list.append(1.0)
            continue
        f = np.fft.rfft(x, n=2 * n)
        acf = np.fft.irfft(f * np.conj(f))[:n].real
        acf /= acf[0]
        neg = np.where(acf < 0.0)[0]
        cut = int(neg[0]) if len(neg) > 0 else n
        tau_int = 1.0 + 2.0 * float(acf[1:cut].sum())
        ess_list.append(n / max(tau_int, 1.0))
    return float(np.min(ess_list) / grad_evals * 1000.0)


def hmc_reference(
    grad_full: Callable[[np.ndarray], np.ndarray],
    neg_log_post: Callable[[np.ndarray], float],
    d: int,
    n_samples: int = 3000,
    n_burnin: int = 800,
    step_size: float = 0.012,
    n_leapfrog: int = 20,
    mass_diag: np.ndarray | None = None,
    theta0: np.ndarray | None = None,
    seed: int = 0,
) -> Tuple[np.ndarray, float]:
    """Preconditioned Hamiltonian Monte Carlo reference sampler."""
    rng = np.random.default_rng(seed)
    th = theta0.copy() if theta0 is not None else np.zeros(d, dtype=float)
    inv_m = 1.0 / mass_diag if mass_diag is not None else np.ones(d, dtype=float)
    sqrt_m = np.sqrt(mass_diag) if mass_diag is not None else np.ones(d, dtype=float)

    samples = np.zeros((n_samples, d), dtype=float)
    accepted = 0
    total = n_burnin + n_samples
    for t in range(total):
        p = rng.standard_normal(d) * sqrt_m
        th_p, p_p = th.copy(), p.copy()
        g = grad_full(th_p)
        p_p = p_p + 0.5 * step_size * g
        for _ in range(n_leapfrog - 1):
            th_p = th_p + step_size * (inv_m * p_p)
            p_p = p_p + step_size * grad_full(th_p)
        th_p = th_p + step_size * (inv_m * p_p)
        p_p = p_p + 0.5 * step_size * grad_full(th_p)

        h_old = neg_log_post(th) + 0.5 * float(np.sum((p ** 2) * inv_m))
        h_new = neg_log_post(th_p) + 0.5 * float(np.sum((p_p ** 2) * inv_m))
        if np.isfinite(h_new) and np.log(rng.random()) < (h_old - h_new):
            th = th_p
            accepted += 1
        if t >= n_burnin:
            samples[t - n_burnin] = th
    acc_rate = accepted / total
    return samples, acc_rate


def paired_bootstrap_ci(a: List[float], b: List[float], n_boot: int = 4000, seed: int = 0) -> Tuple[float, float]:
    arr = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    m = len(arr)
    boots = np.array([arr[rng.integers(0, m, m)].mean() for _ in range(n_boot)])
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


# ============================================================================
# Target Models
# ============================================================================
def make_logistic_model(X: np.ndarray, y: np.ndarray, alpha_prior: float = 1.0):
    def grad_lik(theta: np.ndarray, indices: np.ndarray | None = None, per_point: bool = False) -> np.ndarray:
        if indices is None:
            margin = np.clip(y * (X @ theta), -500.0, 500.0)
            weights = y / (1.0 + np.exp(margin))
            return (X * weights[:, None]) if per_point else (X.T @ weights)
        Xi = X[indices]
        yi = y[indices]
        margin = np.clip(yi * (Xi @ theta), -500.0, 500.0)
        weights = yi / (1.0 + np.exp(margin))
        return (Xi * weights[:, None]) if per_point else (Xi.T @ weights)

    def grad_prior(theta: np.ndarray) -> np.ndarray:
        return -alpha_prior * theta

    def neg_log_post(theta: np.ndarray) -> float:
        margin = np.clip(-y * (X @ theta), -500.0, 500.0)
        return float(0.5 * alpha_prior * (theta @ theta) + np.logaddexp(0.0, margin).sum())

    return grad_lik, grad_prior, neg_log_post


def heldout_logistic_pll(samples: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> float:
    if not np.all(np.isfinite(samples)):
        return float("nan")
    sub = samples[::5]
    logits = np.clip((sub @ X_test.T) * y_test[None, :], -500.0, 500.0)
    log_p = -np.logaddexp(0.0, -logits)  # (S, N_test)
    max_lp = np.max(log_p, axis=0)
    lme = max_lp + np.log(np.mean(np.exp(log_p - max_lp[None, :]), axis=0) + 1e-30)
    return float(np.mean(lme))


def build_sampler(
    name: str,
    grad_lik: Callable,
    grad_prior: Callable,
    N: int,
    d: int,
    sigma: float,
    mb: int,
    K: int,
    seed: int,
    ref_theta: np.ndarray | None = None,
    preconditioner: str = "identity",
):
    rng = np.random.default_rng(seed)
    if name in ("SGLD", "SGLD-CV", "SVRG-SGLD"):
        cfg = SGLDConfig(
            step_size=sigma ** 2,
            minibatch_size=mb,
            epoch_length=K,
            preconditioner=preconditioner,
        )
        cls_map = {"SGLD": SGLD, "SGLD-CV": SGLDCV, "SVRG-SGLD": SVRGSGLD}
        s = cls_map[name](grad_lik, grad_prior, N, d, cfg, rng=rng)
        return s

    b_cfg = SaVRBDConfig(
        step_scale=sigma,
        minibatch_size=mb,
        epoch_length=K,
        ema_beta=0.2,
        rho_star=2.0,
        kappa=KAPPA_CURVATURE,
        preconditioner=preconditioner,
    )
    b_map = {
        "v-SGBD": VanillaSGBD,
        "c-SGBD": CorrectedSGBD,
        "sc-SGBD": StableCorrectedSGBD,
        "NaiveVR-SGBD": NaiveVRSGBD,
        "HardGate-SGBD": HardGateVRSGBD,
        "UniVR-SGBD": UniformVRSGBD,
        "SaVR-BD": SaVRBD,
    }
    return b_map[name](grad_lik, grad_prior, N, d, b_cfg, rng=rng)


# ============================================================================
# Experiment 1: Real UCI a9a Logistic Regression (HMC Gold Reference)
# ============================================================================
def run_exp1_a9a(out_dir: Path) -> None:
    print("\n" + "=" * 72, flush=True)
    print("EXP 1: Real UCI a9a Bayesian Logistic Regression (HMC Reference)", flush=True)
    print("=" * 72, flush=True)

    X_raw, y_raw = fetch_openml("a9a", version=1, return_X_y=True, as_frame=False)
    if hasattr(X_raw, "toarray"):
        X_raw = X_raw.toarray()
    X_raw = np.asarray(X_raw, dtype=float)
    y_all = np.where(np.asarray(y_raw, dtype=float) > 0, 1.0, -1.0)

    rng_split = np.random.default_rng(42)
    perm = rng_split.permutation(X_raw.shape[0])
    train_idx, test_idx = perm[:8000], perm[8000:10000]
    X_tr_raw, y_tr = X_raw[train_idx], y_all[train_idx]
    X_te_raw, y_te = X_raw[test_idx], y_all[test_idx]

    mean = X_tr_raw.mean(axis=0)
    # a9a consists of binary {0,1} indicators; centering preserves bounded per-row norm
    # without inflating ultra-rare dummy columns by 70x.
    X_tr = np.hstack([X_tr_raw - mean, np.ones((len(train_idx), 1))])
    X_te = np.hstack([X_te_raw - mean, np.ones((len(test_idx), 1))])
    N, d = X_tr.shape
    print(f"Prepared UCI a9a: N_train={N}, N_test={len(test_idx)}, d={d}", flush=True)

    grad_lik, grad_prior, nlp = make_logistic_model(X_tr, y_tr, alpha_prior=1.0)
    grad_full = lambda th: grad_prior(th) + grad_lik(th, None, False)

    map_th = minimize(nlp, np.zeros(d), jac=lambda th: -grad_full(th), method="L-BFGS-B").x
    # Diagonal Hessian mass matrix for rapid HMC mixing on d=124
    s_map = 1.0 / (1.0 + np.exp(-np.clip(X_tr @ map_th, -500, 500)))
    w_map = s_map * (1.0 - s_map)
    hess_diag = 1.0 + np.sum((X_tr ** 2) * w_map[:, None], axis=0)

    t0 = time.time()
    hmc_1, acc_1 = hmc_reference(
        grad_full, nlp, d, n_samples=3500, n_burnin=800,
        step_size=0.14, n_leapfrog=20, mass_diag=hess_diag, theta0=map_th, seed=101
    )
    hmc_2, acc_2 = hmc_reference(
        grad_full, nlp, d, n_samples=3500, n_burnin=800,
        step_size=0.14, n_leapfrog=20, mass_diag=hess_diag, theta0=map_th, seed=202
    )
    ref_floor, bw_sq = mmd_rbf_unbiased(hmc_1, hmc_2, bandwidth_sq=None, seed=0)
    ref_pll = heldout_logistic_pll(hmc_1, X_te, y_te)
    print(
        f"HMC gold ref ready in {time.time()-t0:.1f}s | acc=({acc_1:.3f}, {acc_2:.3f}) | "
        f"Ref-vs-Ref MMD^2={ref_floor:.5f} | Ref PLL={ref_pll:.5f}",
        flush=True,
    )

    methods = [
        "SGLD", "SGLD-CV", "SVRG-SGLD",
        "v-SGBD", "c-SGBD", "NaiveVR-SGBD", "UniVR-SGBD", "SaVR-BD",
    ]
    sigmas = [3e-3, 1e-2, 3e-2, 1e-1]
    n_seeds = 7
    mb, K = 128, 62
    n_samples, n_burnin = 2000, 500

    results: Dict[str, Dict[str, List[Dict]]] = {m: {} for m in methods}
    for sig in sigmas:
        key = f"{sig:.0e}"
        print(f"-- a9a sigma={key} --", flush=True)
        for m in methods:
            runs = []
            for seed in range(n_seeds):
                init_rng = np.random.default_rng(1000 + seed)
                th0 = map_th + 0.05 * init_rng.standard_normal(d)
                sampler = build_sampler(m, grad_lik, grad_prior, N, d, sig, mb, K, seed=5000 + seed)
                if isinstance(sampler, SGLDCV):
                    sampler.initialize(th0, ref_theta=map_th)
                else:
                    sampler.initialize(th0)
                ts = time.time()
                samp = sampler.sample(n_samples, n_burnin)
                wall = time.time() - ts
                mmd2, _ = mmd_rbf_unbiased(samp, hmc_1, bandwidth_sq=bw_sq, seed=seed)
                pll = heldout_logistic_pll(samp, X_te, y_te)
                ess = min_ess_per_1k_grads(samp, sampler.grad_evals)
                fb = (
                    float(np.mean(sampler.diagnostics["fallback_fraction"]))
                    if hasattr(sampler, "diagnostics") and sampler.diagnostics["fallback_fraction"]
                    else 0.0
                )
                sat = (
                    float(np.mean(sampler.diagnostics["saturated_fraction"]))
                    if hasattr(sampler, "diagnostics") and sampler.diagnostics["saturated_fraction"]
                    else 0.0
                )
                runs.append({
                    "seed": seed,
                    "mmd2": mmd2,
                    "pll": pll,
                    "ess_1k": ess,
                    "fallback_frac": fb,
                    "sat_frac": sat,
                    "grad_evals": int(sampler.grad_evals),
                    "wall_s": wall,
                })
            results[m][key] = runs
            m_vals = [r["mmd2"] for r in runs]
            p_vals = [r["pll"] for r in runs]
            e_vals = [r["ess_1k"] for r in runs]
            print(
                f"  {m:13s} | MMD^2={np.nanmean(m_vals):.5f}+/-{np.nanstd(m_vals):.5f} | "
                f"PLL={np.nanmean(p_vals):.5f} | ESS/1k={np.mean(e_vals):.4f}",
                flush=True,
            )

    payload = {
        "experiment": "uci_a9a_hmc",
        "N_train": N,
        "N_test": int(len(test_idx)),
        "d": d,
        "minibatch_size": mb,
        "epoch_length": K,
        "n_seeds": n_seeds,
        "n_samples": n_samples,
        "n_burnin": n_burnin,
        "sigmas": sigmas,
        "hmc_acc": [acc_1, acc_2],
        "ref_mmd2_floor": ref_floor,
        "ref_pll": ref_pll,
        "bandwidth_sq": bw_sq,
        "methods": methods,
        "results": results,
    }
    (out_dir / "exp1_a9a.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ============================================================================
# Experiment 2: Heavy-Tailed Cauchy-Prior Linear Regression
# ============================================================================
def run_exp2_cauchy(out_dir: Path) -> None:
    print("\n" + "=" * 72, flush=True)
    print("EXP 2: Heavy-Tailed Cauchy-Prior Linear Regression (HMC Reference)", flush=True)
    print("=" * 72, flush=True)

    rng0 = np.random.default_rng(0)
    N, d, gamma = 2000, 20, 0.5
    X = rng0.standard_normal((N, d))
    true_th = rng0.standard_cauchy(d) * gamma
    y = X @ true_th + rng0.standard_normal(N)

    def grad_lik(th, idx=None, per_point=False):
        if idx is None:
            res = y - X @ th
            return (res[:, None] * X) if per_point else (X.T @ res)
        Xi, yi = X[idx], y[idx]
        res = yi - Xi @ th
        return (res[:, None] * Xi) if per_point else (Xi.T @ res)

    def grad_prior(th):
        return -2.0 * th / (gamma ** 2 + th ** 2)

    def nlp(th):
        return float(np.log1p((th / gamma) ** 2).sum() + 0.5 * np.sum((y - X @ th) ** 2))

    grad_full = lambda th: grad_prior(th) + grad_lik(th, None, False)
    map_th = minimize(nlp, np.zeros(d), jac=lambda th: -grad_full(th), method="L-BFGS-B").x

    hmc_1, acc_1 = hmc_reference(
        grad_full, nlp, d, n_samples=3500, n_burnin=800,
        step_size=0.012, n_leapfrog=25, theta0=map_th, seed=11
    )
    hmc_2, acc_2 = hmc_reference(
        grad_full, nlp, d, n_samples=3500, n_burnin=800,
        step_size=0.012, n_leapfrog=25, theta0=map_th, seed=22
    )
    ref_floor, bw_sq = mmd_rbf_unbiased(hmc_1, hmc_2, bandwidth_sq=None, seed=0)
    print(f"Cauchy HMC acc=({acc_1:.3f}, {acc_2:.3f}) | Ref floor MMD^2={ref_floor:.5f}", flush=True)

    methods = [
        "SGLD", "SGLD-CV", "SVRG-SGLD",
        "v-SGBD", "c-SGBD", "NaiveVR-SGBD", "UniVR-SGBD", "SaVR-BD",
    ]
    sigmas = [5e-3, 1e-2, 2e-2, 3e-2]
    n_seeds = 10
    mb, K = 32, 62
    n_samples, n_burnin = 1500, 500

    results = {m: {} for m in methods}
    for sig in sigmas:
        key = f"{sig:.0e}"
        print(f"-- Cauchy sigma={key} --", flush=True)
        for m in methods:
            runs = []
            for seed in range(n_seeds):
                init_rng = np.random.default_rng(2000 + seed)
                th0 = map_th + 0.02 * init_rng.standard_normal(d)
                sampler = build_sampler(m, grad_lik, grad_prior, N, d, sig, mb, K, seed=6000 + seed)
                if isinstance(sampler, SGLDCV):
                    sampler.initialize(th0, ref_theta=map_th)
                else:
                    sampler.initialize(th0)
                ts = time.time()
                samp = sampler.sample(n_samples, n_burnin)
                wall = time.time() - ts
                mmd2, _ = mmd_rbf_unbiased(samp, hmc_1, bandwidth_sq=bw_sq, seed=seed)
                ess = min_ess_per_1k_grads(samp, sampler.grad_evals)
                fb = (
                    float(np.mean(sampler.diagnostics["fallback_fraction"]))
                    if hasattr(sampler, "diagnostics") and sampler.diagnostics["fallback_fraction"]
                    else 0.0
                )
                sat = (
                    float(np.mean(sampler.diagnostics["saturated_fraction"]))
                    if hasattr(sampler, "diagnostics") and sampler.diagnostics["saturated_fraction"]
                    else 0.0
                )
                runs.append({
                    "seed": seed,
                    "mmd2": mmd2,
                    "ess_1k": ess,
                    "fallback_frac": fb,
                    "sat_frac": sat,
                    "wall_s": wall,
                })
            results[m][key] = runs
            m_vals = [r["mmd2"] for r in runs]
            print(f"  {m:13s} | MMD^2={np.nanmean(m_vals):.5f}+/-{np.nanstd(m_vals):.5f}", flush=True)

    payload = {
        "experiment": "cauchy_heavy_tail",
        "N": N,
        "d": d,
        "gamma": gamma,
        "minibatch_size": mb,
        "epoch_length": K,
        "n_seeds": n_seeds,
        "sigmas": sigmas,
        "ref_mmd2_floor": ref_floor,
        "methods": methods,
        "results": results,
    }
    (out_dir / "exp2_cauchy.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ============================================================================
# Experiment 3: Heterogeneous Hessian Logistic Regression
# ============================================================================
def run_exp3_heterogeneous(out_dir: Path) -> None:
    print("\n" + "=" * 72, flush=True)
    print("EXP 3: Heterogeneous-Scale Logistic Regression (HMC Reference)", flush=True)
    print("=" * 72, flush=True)

    rng0 = np.random.default_rng(7)
    N, d = 4000, 30
    feat_scale = 10.0 ** np.linspace(-1.5, 1.5, d)  # feature std spans 3 orders -> Hessian cond ~ 10^6
    X = rng0.standard_normal((N, d)) * feat_scale[None, :]
    true_th = rng0.standard_normal(d) / feat_scale
    probs = sigmoid(X @ true_th)
    y = np.where(rng0.random(N) < probs, 1.0, -1.0)

    grad_lik, grad_prior, nlp = make_logistic_model(X, y, alpha_prior=1.0)
    grad_full = lambda th: grad_prior(th) + grad_lik(th, None, False)
    map_th = minimize(nlp, np.zeros(d), jac=lambda th: -grad_full(th), method="L-BFGS-B").x

    s_map = sigmoid(X @ map_th)
    hess_diag = 1.0 + np.sum((X ** 2) * (s_map * (1.0 - s_map))[:, None], axis=0)
    hmc_1, acc_1 = hmc_reference(
        grad_full, nlp, d, n_samples=3000, n_burnin=800,
        step_size=0.10, n_leapfrog=20, mass_diag=hess_diag, theta0=map_th, seed=31
    )
    hmc_2, acc_2 = hmc_reference(
        grad_full, nlp, d, n_samples=3000, n_burnin=800,
        step_size=0.10, n_leapfrog=20, mass_diag=hess_diag, theta0=map_th, seed=32
    )
    ref_floor, bw_sq = mmd_rbf_unbiased(hmc_1, hmc_2, bandwidth_sq=None, seed=0)
    print(f"Heterogeneous HMC acc=({acc_1:.3f}, {acc_2:.3f}) | Ref floor MMD^2={ref_floor:.5f}", flush=True)

    methods = ["SGLD", "SVRG-SGLD", "v-SGBD", "c-SGBD", "NaiveVR-SGBD", "UniVR-SGBD", "SaVR-BD"]
    sigmas = [3e-3, 1e-2, 3e-2, 1e-1]
    n_seeds = 7
    mb, K = 64, 62
    results = {m: {} for m in methods}

    for sig in sigmas:
        key = f"{sig:.0e}"
        print(f"-- Heterogeneous sigma={key} --", flush=True)
        for m in methods:
            runs = []
            for seed in range(n_seeds):
                init_rng = np.random.default_rng(3000 + seed)
                th0 = map_th + 0.01 * init_rng.standard_normal(d)
                sampler = build_sampler(m, grad_lik, grad_prior, N, d, sig, mb, K, seed=7000 + seed)
                sampler.initialize(th0)
                samp = sampler.sample(2000, 500)
                mmd2, _ = mmd_rbf_unbiased(samp, hmc_1, bandwidth_sq=bw_sq, seed=seed)
                runs.append({"seed": seed, "mmd2": mmd2})
            results[m][key] = runs
            m_vals = [r["mmd2"] for r in runs]
            print(f"  {m:13s} | MMD^2={np.nanmean(m_vals):.5f}+/-{np.nanstd(m_vals):.5f}", flush=True)

    payload = {
        "experiment": "heterogeneous_logistic",
        "N": N,
        "d": d,
        "sigmas": sigmas,
        "ref_mmd2_floor": ref_floor,
        "methods": methods,
        "results": results,
    }
    (out_dir / "exp3_heterogeneous.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ============================================================================
# Experiment 4: 20-Seed Coordinate-Stiffness Mismatch Sweep
# ============================================================================
def run_exp4_mismatch(out_dir: Path) -> None:
    print("\n" + "=" * 72, flush=True)
    print("EXP 4: 20-Seed Coordinate-Stiffness Mismatch Sweep", flush=True)
    print("=" * 72, flush=True)

    rng0 = np.random.default_rng(0)
    N, d = 3000, 10
    X_base = rng0.standard_normal((N, d))
    true_base = rng0.standard_normal(d)
    p_base = sigmoid(X_base @ true_base)
    y = np.where(rng0.random(N) < p_base, 1.0, -1.0)

    lambdas = [1.0, 0.3, 0.1, 0.03, 0.01]
    sigma_fixed = 0.03  # fixed proposal scale -> h = sigma^2 = 9e-4 for Langevin
    mb, K = 64, 50
    n_seeds = 20
    methods = ["SGLD-CV", "SVRG-SGLD", "c-SGBD", "NaiveVR-SGBD", "UniVR-SGBD", "SaVR-BD"]
    results = {m: {} for m in methods}

    for lam in lambdas:
        key = f"lam={lam:.2f}"
        X_lam = X_base.copy()
        X_lam[:, 0] /= lam  # coordinate 1 curvature scales as lam^{-2}
        grad_lik, grad_prior, nlp = make_logistic_model(X_lam, y, alpha_prior=1.0)
        grad_full = lambda th: grad_prior(th) + grad_lik(th, None, False)
        map_lam = minimize(nlp, np.zeros(d), jac=lambda th: -grad_full(th), method="L-BFGS-B").x

        s_map = sigmoid(X_lam @ map_lam)
        hess_diag = 1.0 + np.sum((X_lam ** 2) * (s_map * (1.0 - s_map))[:, None], axis=0)
        hmc_lam, acc_lam = hmc_reference(
            grad_full, nlp, d, n_samples=2500, n_burnin=600,
            step_size=0.12, n_leapfrog=18, mass_diag=hess_diag, theta0=map_lam, seed=int(1000 * lam) + 5
        )
        _, bw_sq = mmd_rbf_unbiased(hmc_lam[:1200], hmc_lam[1200:2400], bandwidth_sq=None, seed=0)
        hmc_mean0 = float(hmc_lam[:, 0].mean())
        print(f"-- Mismatch {key} (HMC acc={acc_lam:.3f}, E_pi[theta_1]={hmc_mean0:.4f}) --", flush=True)

        for m in methods:
            runs = []
            for seed in range(n_seeds):
                init_rng = np.random.default_rng(4000 + seed)
                th0 = map_lam + 0.002 * init_rng.standard_normal(d)
                sampler = build_sampler(m, grad_lik, grad_prior, N, d, sigma_fixed, mb, K, seed=8000 + seed)
                if isinstance(sampler, SGLDCV):
                    sampler.initialize(th0, ref_theta=map_lam)
                else:
                    sampler.initialize(th0)
                samp = sampler.sample(2000, 500)
                err_mean0 = float(abs(np.mean(samp[:, 0]) - hmc_mean0))
                mmd2, _ = mmd_rbf_unbiased(samp, hmc_lam, bandwidth_sq=bw_sq, seed=seed)
                runs.append({
                    "seed": seed,
                    "bias_coord0": err_mean0,
                    "mmd2": mmd2,
                    "hmc_mean0": hmc_mean0,
                })
            results[m][key] = runs
            b_vals = [r["bias_coord0"] for r in runs]
            m_vals = [r["mmd2"] for r in runs]
            print(
                f"  {m:13s} | |E[th_1]-pi_1|={np.nanmean(b_vals):.5f} | MMD^2={np.nanmean(m_vals):.5f}",
                flush=True,
            )

    # Statistical tests at lam = 0.01 and 0.10
    stats_summary = {}
    for lam in [0.1, 0.03, 0.01]:
        k = f"lam={lam:.2f}"
        savr_b = [r["bias_coord0"] for r in results["SaVR-BD"][k]]
        svrg_b = [r["bias_coord0"] for r in results["SVRG-SGLD"][k]]
        uni_b = [r["bias_coord0"] for r in results["UniVR-SGBD"][k]]
        stats_summary[k] = {
            "savr_vs_svrg_wilcoxon_p": float(wilcoxon(savr_b, svrg_b).pvalue),
            "savr_vs_svrg_ci95": list(paired_bootstrap_ci(savr_b, svrg_b)),
            "savr_vs_uni_wilcoxon_p": float(wilcoxon(savr_b, uni_b).pvalue),
            "savr_vs_uni_ci95": list(paired_bootstrap_ci(savr_b, uni_b)),
        }

    payload = {
        "experiment": "stiffness_mismatch_20seed",
        "N": N,
        "d": d,
        "sigma_fixed": sigma_fixed,
        "lambdas": lambdas,
        "n_seeds": n_seeds,
        "methods": methods,
        "stats_summary": stats_summary,
        "results": results,
    }
    (out_dir / "exp4_mismatch.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ============================================================================
# Experiment 5: Real UCI Wine Quality Bayesian Neural Network Regression
# ============================================================================
class VectorizedBNN:
    """
    One-hidden-layer tanh Bayesian Neural Network with Student-t heavy-tailed weight prior:
      input(d_in) -> Linear(H) -> tanh -> Linear(1)
    Vectorized per-datapoint backpropagation with zero Python loops over batch items.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        hidden: int = 32,
        prior_df: float = 3.0,
        prior_scale: float = 1.0,
        obs_var: float = 0.5,
    ):
        self.X = X
        self.y = y.ravel()
        self.N, self.d_in = X.shape
        self.H = hidden
        self.nu = prior_df
        self.s2 = prior_scale ** 2
        self.sigma_y2 = obs_var

        self.n_w1 = self.d_in * self.H
        self.n_b1 = self.H
        self.n_w2 = self.H
        self.dim = self.n_w1 + self.n_b1 + self.n_w2 + 1

    def unpack(self, theta: np.ndarray):
        i1 = self.n_w1
        i2 = i1 + self.n_b1
        i3 = i2 + self.n_w2
        W1 = theta[:i1].reshape(self.d_in, self.H)
        b1 = theta[i1:i2]
        w2 = theta[i2:i3]
        b2 = theta[i3]
        return W1, b1, w2, b2

    def forward(self, X_in: np.ndarray, theta: np.ndarray) -> np.ndarray:
        W1, b1, w2, b2 = self.unpack(theta)
        h = np.tanh(X_in @ W1 + b1)
        return h @ w2 + b2

    def grad_prior(self, theta: np.ndarray) -> np.ndarray:
        # Student-t(nu, scale) prior log-density: -((nu+1)/2) sum log(1 + theta^2 / (nu * s2))
        return -(self.nu + 1.0) * theta / (self.nu * self.s2 + theta ** 2)

    def grad_lik(self, theta: np.ndarray, indices: np.ndarray | None = None, per_point: bool = False) -> np.ndarray:
        Xs = self.X if indices is None else self.X[indices]  # (n, d_in)
        ys = self.y if indices is None else self.y[indices]  # (n,)
        W1, b1, w2, b2 = self.unpack(theta)

        pre1 = np.clip(Xs @ W1 + b1, -20.0, 20.0)  # (n, H)
        h = np.tanh(pre1)                          # (n, H)
        pred = h @ w2 + b2                         # (n,)
        d_out = (ys - pred) / self.sigma_y2        # (n,) d log p(y_i | th) / d pred_i

        d_pre1 = (d_out[:, None] * w2[None, :]) * (1.0 - h ** 2)  # (n, H)

        if per_point:
            n = Xs.shape[0]
            out = np.empty((n, self.dim), dtype=float)
            i1 = self.n_w1
            i2 = i1 + self.n_b1
            i3 = i2 + self.n_w2
            # Batched outer product: (n, d_in, 1) * (n, 1, H) -> (n, d_in * H)
            out[:, :i1] = (Xs[:, :, None] * d_pre1[:, None, :]).reshape(n, i1)
            out[:, i1:i2] = d_pre1
            out[:, i2:i3] = d_out[:, None] * h
            out[:, i3] = d_out
            return out

        g = np.empty(self.dim, dtype=float)
        i1 = self.n_w1
        i2 = i1 + self.n_b1
        i3 = i2 + self.n_w2
        g[:i1] = (Xs.T @ d_pre1).ravel()
        g[i1:i2] = d_pre1.sum(axis=0)
        g[i2:i3] = h.T @ d_out
        g[i3] = float(d_out.sum())
        return g

    def neg_log_post(self, theta: np.ndarray) -> float:
        pred = self.forward(self.X, theta)
        nll = 0.5 * float(np.sum((self.y - pred) ** 2)) / self.sigma_y2
        nprior = 0.5 * (self.nu + 1.0) * float(np.sum(np.log1p((theta ** 2) / (self.nu * self.s2))))
        return nll + nprior

    def evaluate_test(self, samples: np.ndarray, X_te: np.ndarray, y_te: np.ndarray) -> Tuple[float, float]:
        if not np.all(np.isfinite(samples)):
            return float("nan"), float("nan")
        sub = samples[::4]
        preds = np.array([self.forward(X_te, th) for th in sub])  # (S, N_te)
        mean_pred = preds.mean(axis=0)
        rmse = float(np.sqrt(np.mean((mean_pred - y_te) ** 2)))

        log_p = -0.5 * ((preds - y_te[None, :]) ** 2) / self.sigma_y2 - 0.5 * np.log(
            2.0 * np.pi * self.sigma_y2
        )
        max_lp = np.max(log_p, axis=0)
        lme = max_lp + np.log(np.mean(np.exp(log_p - max_lp[None, :]), axis=0) + 1e-30)
        test_nll = float(-np.mean(lme))
        return rmse, test_nll


def run_exp5_bnn_uci(out_dir: Path) -> None:
    print("\n" + "=" * 72, flush=True)
    print("EXP 5: Real UCI Wine Quality BNN Regression (Natural Multi-Scale Physicochemical Features)", flush=True)
    print("=" * 72, flush=True)

    X_raw, y_raw = fetch_openml(data_id=287, return_X_y=True, as_frame=False)
    X_raw = np.asarray(X_raw, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)

    rng_sp = np.random.default_rng(42)
    perm = rng_sp.permutation(X_raw.shape[0])
    tr_idx, te_idx = perm[:5000], perm[5000:]
    # Center features at training mean while retaining real physicochemical scale heterogeneity
    # (natural stds span 0.003 for density up to 56.5 for total sulfur dioxide, a 19,000x ratio;
    # we divide globally by median feature std so typical coordinate is O(1) while preserving natural condition ratio!)
    X_mean = X_raw[tr_idx].mean(axis=0)
    med_std = float(np.median(X_raw[tr_idx].std(axis=0)))
    X_tr = (X_raw[tr_idx] - X_mean) / med_std
    X_te = (X_raw[te_idx] - X_mean) / med_std

    y_mean, y_std = float(y_raw[tr_idx].mean()), float(y_raw[tr_idx].std())
    y_tr = (y_raw[tr_idx] - y_mean) / y_std
    y_te = (y_raw[te_idx] - y_mean) / y_std

    feat_stds = X_tr.std(axis=0)
    print(
        f"UCI Wine Quality: N_tr={len(tr_idx)}, N_te={len(te_idx)}, d_in={X_tr.shape[1]}, "
        f"natural feature std ratio max/min = {feat_stds.max()/feat_stds.min():.1f}x",
        flush=True,
    )

    model = VectorizedBNN(X_tr, y_tr, hidden=32, prior_df=3.0, prior_scale=1.0, obs_var=0.65)
    print(f"BNN total parameters d={model.dim}", flush=True)

    # Warm-start MAP finder (capped at 120 L-BFGS iterations to locate posterior basin)
    rng_w = np.random.default_rng(0)
    th_init = rng_w.standard_normal(model.dim) * 0.05
    grad_full = lambda th: model.grad_prior(th) + model.grad_lik(th, None, False)
    warm_th = minimize(
        model.neg_log_post, th_init, jac=lambda th: -grad_full(th),
        method="L-BFGS-B", options={"maxiter": 120}
    ).x
    map_rmse, _ = model.evaluate_test(warm_th[None, :], X_te, y_te)
    print(f"Warm-start point Test RMSE={map_rmse:.4f}", flush=True)

    methods = ["SGLD", "SGLD-CV", "SVRG-SGLD", "v-SGBD", "c-SGBD", "NaiveVR-SGBD", "UniVR-SGBD", "SaVR-BD"]
    sigmas = [1e-3, 3e-3, 6e-3, 1e-2]
    n_seeds = 5
    mb, K = 64, 78
    n_samples, n_burnin = 1200, 300

    results = {m: {} for m in methods}
    for sig in sigmas:
        key = f"{sig:.0e}"
        print(f"-- BNN sigma={key} --", flush=True)
        for m in methods:
            runs = []
            for seed in range(n_seeds):
                init_rng = np.random.default_rng(9000 + seed)
                th0 = warm_th + 0.005 * init_rng.standard_normal(model.dim)
                sampler = build_sampler(
                    m, model.grad_lik, model.grad_prior, model.N, model.dim,
                    sig, mb, K, seed=9500 + seed
                )
                if isinstance(sampler, SGLDCV):
                    sampler.initialize(th0, ref_theta=warm_th)
                else:
                    sampler.initialize(th0)
                ts = time.time()
                samp = sampler.sample(n_samples, n_burnin)
                wall = time.time() - ts
                rmse, nll = model.evaluate_test(samp, X_te, y_te)
                ess = min_ess_per_1k_grads(samp, sampler.grad_evals)
                diverged = bool(not np.isfinite(rmse) or rmse > 5.0)
                runs.append({
                    "seed": seed,
                    "rmse": rmse if not diverged else float("nan"),
                    "nll": nll if not diverged else float("nan"),
                    "diverged": diverged,
                    "ess_1k": ess,
                    "wall_s": wall,
                })
            results[m][key] = runs
            r_vals = [r["rmse"] for r in runs if not r["diverged"]]
            n_vals = [r["nll"] for r in runs if not r["diverged"]]
            div_cnt = sum(1 for r in runs if r["diverged"])
            r_str = f"{np.mean(r_vals):.4f}+/-{np.std(r_vals):.4f}" if r_vals else "DIVERGED"
            n_str = f"{np.mean(n_vals):.4f}" if n_vals else "DIVERGED"
            print(f"  {m:13s} | RMSE={r_str} | NLL={n_str} | div={div_cnt}/{n_seeds}", flush=True)

    payload = {
        "experiment": "uci_wine_quality_bnn",
        "N_train": int(len(tr_idx)),
        "N_test": int(len(te_idx)),
        "d_in": int(X_tr.shape[1]),
        "dim": model.dim,
        "feature_std_ratio": float(feat_stds.max() / feat_stds.min()),
        "sigmas": sigmas,
        "n_seeds": n_seeds,
        "methods": methods,
        "results": results,
    }
    (out_dir / "exp5_bnn_uci.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ============================================================================
# Experiment 6: Theorem 1 Bias Curve + Component, rho*, and SAGA Ablations
# ============================================================================
def run_exp6_ablations(out_dir: Path) -> None:
    print("\n" + "=" * 72, flush=True)
    print("EXP 6: Theorem 1 Local Bias Verification & Component / rho* / SAGA Ablations", flush=True)
    print("=" * 72, flush=True)

    # Part A: Single-step bias curve over z in [-5, 5] at tau = 0.5
    rng_b = np.random.default_rng(123)
    zs = np.linspace(-5.0, 5.0, 41)
    tau = 0.5
    eps = rng_b.normal(0.0, tau, size=(250000, 1))
    a_mat = zs[None, :] + eps
    p_exact = sigmoid(zs)

    bias_curves = {}
    for label, mode, kap in [
        ("NaiveVR-SGBD", "none", KAPPA_CURVATURE),
        ("HardGate-SGBD", "hard", KAPPA_PROBIT),
        ("UniVR-SGBD", "uniform", KAPPA_PROBIT),
        ("SaVR-BD", "smooth", KAPPA_CURVATURE),
    ]:
        psi = calibrated_argument(a_mat, np.full_like(a_mat, tau), rho_star=2.0, kappa=kap, gate_mode=mode)
        err = np.abs(sigmoid(psi).mean(axis=0) - p_exact)
        bias_curves[label] = [float(v) for v in err]
        print(f"  Single-step bias ({label:14s}): mean={err.mean():.6f}, max={err.max():.6f}", flush=True)

    # Part B: Component decomposition + rho* sweep + SVRG vs SAGA on Cauchy heavy-tail (sigma = 0.02)
    rng0 = np.random.default_rng(0)
    N, d, gamma = 2000, 20, 0.5
    X = rng0.standard_normal((N, d))
    true_th = rng0.standard_cauchy(d) * gamma
    y = X @ true_th + rng0.standard_normal(N)

    def grad_lik(th, idx=None, per_point=False):
        if idx is None:
            res = y - X @ th
            return (res[:, None] * X) if per_point else (X.T @ res)
        Xi, yi = X[idx], y[idx]
        res = yi - Xi @ th
        return (res[:, None] * Xi) if per_point else (Xi.T @ res)

    def grad_prior(th):
        return -2.0 * th / (gamma ** 2 + th ** 2)

    def nlp(th):
        return float(np.log1p((th / gamma) ** 2).sum() + 0.5 * np.sum((y - X @ th) ** 2))

    grad_full = lambda th: grad_prior(th) + grad_lik(th, None, False)
    map_th = minimize(nlp, np.zeros(d), jac=lambda th: -grad_full(th), method="L-BFGS-B").x
    hmc_ref_s, _ = hmc_reference(grad_full, nlp, d, n_samples=3500, n_burnin=800, step_size=0.012, n_leapfrog=25, theta0=map_th, seed=11)
    _, bw_sq = mmd_rbf_unbiased(hmc_ref_s[:1500], hmc_ref_s[1500:3000], bandwidth_sq=None, seed=0)

    sig = 0.02
    mb, K = 32, 62
    n_seeds = 8

    comp_methods = [
        "v-SGBD", "c-SGBD", "sc-SGBD",
        "NaiveVR-SGBD", "HardGate-SGBD", "UniVR-SGBD", "SaVR-BD",
    ]
    comp_res = {}
    for m in comp_methods:
        mmds, esss = [], []
        for s in range(n_seeds):
            th0 = map_th + 0.02 * np.random.default_rng(2000 + s).standard_normal(d)
            sampler = build_sampler(m, grad_lik, grad_prior, N, d, sig, mb, K, seed=6000 + s)
            sampler.initialize(th0)
            samp = sampler.sample(1500, 500)
            mv, _ = mmd_rbf_unbiased(samp, hmc_ref_s, bandwidth_sq=bw_sq, seed=s)
            mmds.append(mv)
            esss.append(min_ess_per_1k_grads(samp, sampler.grad_evals))
        comp_res[m] = {
            "mmd2_mean": float(np.mean(mmds)),
            "mmd2_std": float(np.std(mmds)),
            "ess_mean": float(np.mean(esss)),
            "ess_std": float(np.std(esss)),
        }
        print(f"  Component {m:14s}: MMD^2={np.mean(mmds):.5f}+/-{np.std(mmds):.5f}", flush=True)

    # rho* sensitivity sweep (actually passing rho_star properly!)
    rho_grid = [0.5, 1.0, 2.0, 3.0, 5.0]
    rho_res = {}
    for rho in rho_grid:
        mmds, esss = [], []
        for s in range(n_seeds):
            th0 = map_th + 0.02 * np.random.default_rng(2000 + s).standard_normal(d)
            cfg = SaVRBDConfig(step_scale=sig, minibatch_size=mb, epoch_length=K, rho_star=rho, kappa=KAPPA_CURVATURE, gate_mode="smooth")
            sampler = SaVRBD(grad_lik, grad_prior, N, d, cfg, rng=np.random.default_rng(6000 + s))
            sampler.initialize(th0)
            samp = sampler.sample(1500, 500)
            mv, _ = mmd_rbf_unbiased(samp, hmc_ref_s, bandwidth_sq=bw_sq, seed=s)
            mmds.append(mv)
            esss.append(min_ess_per_1k_grads(samp, sampler.grad_evals))
        rho_res[str(rho)] = {
            "mmd2_mean": float(np.mean(mmds)),
            "mmd2_std": float(np.std(mmds)),
            "ess_mean": float(np.mean(esss)),
            "ess_std": float(np.std(esss)),
        }
        print(f"  rho*={rho:3.1f}: MMD^2={np.mean(mmds):.5f}+/-{np.std(mmds):.5f}", flush=True)

    # SVRG vs SAGA variant comparison
    vr_variants = {}
    for var_name in ["svrg", "saga"]:
        mmds, esss = [], []
        for s in range(n_seeds):
            th0 = map_th + 0.02 * np.random.default_rng(2000 + s).standard_normal(d)
            cfg = SaVRBDConfig(step_scale=sig, minibatch_size=mb, epoch_length=K, rho_star=2.0, variant=var_name)
            sampler = SaVRBD(grad_lik, grad_prior, N, d, cfg, rng=np.random.default_rng(6000 + s))
            sampler.initialize(th0)
            samp = sampler.sample(1500, 500)
            mv, _ = mmd_rbf_unbiased(samp, hmc_ref_s, bandwidth_sq=bw_sq, seed=s)
            mmds.append(mv)
            esss.append(min_ess_per_1k_grads(samp, sampler.grad_evals))
        vr_variants[var_name] = {
            "mmd2_mean": float(np.mean(mmds)),
            "mmd2_std": float(np.std(mmds)),
            "ess_mean": float(np.mean(esss)),
            "ess_std": float(np.std(esss)),
        }
        print(f"  Variant {var_name:6s}: MMD^2={np.mean(mmds):.5f}, ESS/1k={np.mean(esss):.4f}", flush=True)

    payload = {
        "experiment": "ablations",
        "single_step_bias": {"zs": [float(z) for z in zs], "tau": tau, "curves": bias_curves},
        "component_ablation": comp_res,
        "rho_star_ablation": rho_res,
        "variant_ablation": vr_variants,
    }
    (out_dir / "exp6_ablations.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ============================================================================
# Experiment 7: Equal Compute-Budget Pareto, Trajectory Trace, and ACF Diagnostics
# ============================================================================
def compute_acf_1d(x: np.ndarray, max_lag: int = 60) -> List[float]:
    x_c = np.asarray(x, dtype=float) - float(np.mean(x))
    v = float(np.var(x_c))
    if v < 1e-14 or not np.isfinite(v):
        return [1.0] + [0.0] * max_lag
    n = len(x_c)
    f = np.fft.rfft(x_c, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[: max_lag + 1].real / (v * n)
    return [float(val) for val in acf]


def run_exp7_diagnostics(out_dir: Path) -> None:
    print("\n" + "=" * 72, flush=True)
    print("EXP 7: Compute-Budget Convergence, Stiffness Trace, and ACF Diagnostics", flush=True)
    print("=" * 72, flush=True)

    # Part A: Equal data-pass & wall-clock convergence on heavy-tailed Cauchy (N=2000, d=20)
    rng0 = np.random.default_rng(0)
    N, d, gamma = 2000, 20, 0.5
    X = rng0.standard_normal((N, d))
    true_th = rng0.standard_cauchy(d) * gamma
    y = X @ true_th + rng0.standard_normal(N)

    def grad_lik(th, idx=None, per_point=False):
        if idx is None:
            res = y - X @ th
            return (res[:, None] * X) if per_point else (X.T @ res)
        Xi, yi = X[idx], y[idx]
        res = yi - Xi @ th
        return (res[:, None] * Xi) if per_point else (Xi.T @ res)

    def grad_prior(th):
        return -2.0 * th / (gamma ** 2 + th ** 2)

    def nlp(th):
        return float(np.log1p((th / gamma) ** 2).sum() + 0.5 * np.sum((y - X @ th) ** 2))

    grad_full = lambda th: grad_prior(th) + grad_lik(th, None, False)
    map_th = minimize(nlp, np.zeros(d), jac=lambda th: -grad_full(th), method="L-BFGS-B").x
    hmc_1, _ = hmc_reference(
        grad_full, nlp, d, n_samples=3500, n_burnin=800,
        step_size=0.012, n_leapfrog=25, theta0=map_th, seed=11
    )
    hmc_2, _ = hmc_reference(
        grad_full, nlp, d, n_samples=3500, n_burnin=800,
        step_size=0.012, n_leapfrog=25, theta0=map_th, seed=22
    )
    ref_floor, bw_sq = mmd_rbf_unbiased(hmc_1, hmc_2, bandwidth_sq=None, seed=0)

    pass_grid = [8, 16, 30, 50, 75, 100, 130]  # cumulative likelihood passes (grad_evals / N)
    max_evals = max(pass_grid) * N
    mb, K = 32, 62
    n_seeds = 8

    # Note: SGLD at sigma=0.02 oscillates outside support, so we also grant SGLD its best stable scale sigma=0.005
    budget_specs = [
        ("SGLD (best stable)", "SGLD", 0.005, "svrg"),
        ("SVRG-SGLD", "SVRG-SGLD", 0.02, "svrg"),
        ("c-SGBD", "c-SGBD", 0.02, "svrg"),
        ("NaiveVR-SGBD", "NaiveVR-SGBD", 0.02, "svrg"),
        ("SaVR-BD (SVRG)", "SaVR-BD", 0.02, "svrg"),
        ("SaVR-BD (SAGA)", "SaVR-BD", 0.02, "saga"),
    ]

    budget_curves = {}
    acf_curves = {}
    for label, base_name, sig_val, var_mode in budget_specs:
        seed_mmds = {p: [] for p in pass_grid}
        seed_walls = {p: [] for p in pass_grid}
        seed_acfs = []
        for s in range(n_seeds):
            th0 = map_th + 0.02 * np.random.default_rng(2000 + s).standard_normal(d)
            if label == "SaVR-BD (SAGA)":
                cfg = SaVRBDConfig(
                    step_scale=sig_val, minibatch_size=mb, epoch_length=K,
                    ema_beta=0.2, rho_star=2.0, kappa=KAPPA_CURVATURE, variant="saga"
                )
                sampler = SaVRBD(grad_lik, grad_prior, N, d, cfg, rng=np.random.default_rng(6000 + s))
            else:
                sampler = build_sampler(base_name, grad_lik, grad_prior, N, d, sig_val, mb, K, seed=6000 + s)

            t0 = time.perf_counter()
            sampler.initialize(th0)
            traj = []
            ptr = 0
            while sampler.grad_evals < max_evals:
                th_cur = sampler.step()
                traj.append(th_cur)
                while ptr < len(pass_grid) and sampler.grad_evals >= pass_grid[ptr] * N:
                    elapsed = time.perf_counter() - t0
                    arr = np.asarray(traj[max(1, len(traj) // 4):], dtype=float)
                    if len(arr) >= 40:
                        mv, _ = mmd_rbf_unbiased(arr, hmc_1, bandwidth_sq=bw_sq, max_pts=600, seed=s)
                    else:
                        mv = float("nan")
                    seed_mmds[pass_grid[ptr]].append(mv)
                    seed_walls[pass_grid[ptr]].append(elapsed)
                    ptr += 1
            full_post = np.asarray(traj[len(traj) // 4:], dtype=float)
            # Average ACF across first 5 coordinates
            acf_5 = np.mean([compute_acf_1d(full_post[:, j], max_lag=50) for j in range(5)], axis=0)
            seed_acfs.append(acf_5)

        budget_curves[label] = {
            "passes": pass_grid,
            "mmd2_mean": [float(np.nanmean(seed_mmds[p])) for p in pass_grid],
            "mmd2_std": [float(np.nanstd(seed_mmds[p])) for p in pass_grid],
            "wall_ms_mean": [float(np.mean(seed_walls[p]) * 1000.0) for p in pass_grid],
        }
        acf_curves[label] = [float(v) for v in np.mean(seed_acfs, axis=0)]
        print(
            f"  Budget {label:18s} | @130 passes MMD^2={budget_curves[label]['mmd2_mean'][-1]:.5f} "
            f"| wall={budget_curves[label]['wall_ms_mean'][-1]:.1f} ms",
            flush=True,
        )

    # Part B: Stiff-coordinate trace at lambda = 0.03 (N=3000, d=10, sigma=0.03)
    rng_m = np.random.default_rng(0)
    Nm, dm = 3000, 10
    X_base = rng_m.standard_normal((Nm, dm))
    true_base = rng_m.standard_normal(dm)
    y_m = np.where(rng_m.random(Nm) < sigmoid(X_base @ true_base), 1.0, -1.0)

    lam_demo = 0.03
    X_lam = X_base.copy()
    X_lam[:, 0] /= lam_demo
    gl_m, gp_m, nlp_m = make_logistic_model(X_lam, y_m, alpha_prior=1.0)
    gf_m = lambda th: gp_m(th) + gl_m(th, None, False)
    map_m = minimize(nlp_m, np.zeros(dm), jac=lambda th: -gf_m(th), method="L-BFGS-B").x
    s_map = sigmoid(X_lam @ map_m)
    hess_diag = 1.0 + np.sum((X_lam ** 2) * (s_map * (1.0 - s_map))[:, None], axis=0)
    hmc_m, _ = hmc_reference(
        gf_m, nlp_m, dm, n_samples=2500, n_burnin=600,
        step_size=0.12, n_leapfrog=18, mass_diag=hess_diag, theta0=map_m, seed=35
    )
    hmc_q025, hmc_mean, hmc_q975 = [float(v) for v in np.quantile(hmc_m[:, 0], [0.025, 0.5, 0.975])]
    hmc_mean = float(np.mean(hmc_m[:, 0]))

    traces = {}
    for m_name in ["SVRG-SGLD", "SaVR-BD"]:
        th0 = map_m + 0.002 * np.random.default_rng(4000).standard_normal(dm)
        smp = build_sampler(m_name, gl_m, gp_m, Nm, dm, 0.03, 64, 50, seed=8000)
        smp.initialize(th0)
        tr = smp.sample(300, 200)
        traces[m_name] = [float(v) for v in tr[:, 0]]

    payload = {
        "experiment": "compute_budget_and_acf_diagnostics",
        "cauchy_ref_floor": ref_floor,
        "budget_curves": budget_curves,
        "acf_lags": list(range(51)),
        "acf_curves": acf_curves,
        "stiff_trace": {
            "lambda": lam_demo,
            "sigma": 0.03,
            "hmc_mean": hmc_mean,
            "hmc_q025": hmc_q025,
            "hmc_q975": hmc_q975,
            "traces": traces,
        },
    }
    (out_dir / "exp7_diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ============================================================================
# Experiment 8: Large-N Tabular Cardinality Scalability (N up to 100,000) & Full a9a
# ============================================================================
def binary_ece_and_brier(samples: np.ndarray, X_te: np.ndarray, y_te: np.ndarray, n_bins: int = 10) -> Tuple[float, float]:
    """Compute held-out Brier score and 10-bin Expected Calibration Error (ECE) for binary labels y in {-1, +1}."""
    if not np.all(np.isfinite(samples)):
        return float("nan"), float("nan")
    sub = samples[::4]
    logits = np.clip(sub @ X_te.T, -500.0, 500.0)
    probs = sigmoid(logits).mean(axis=0)  # posterior predictive P(y = +1 | x)
    y01 = (y_te > 0).astype(float)
    brier = float(np.mean((probs - y01) ** 2))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        mask = (probs >= bins[b]) & (probs < bins[b + 1] if b < n_bins - 1 else probs <= bins[b + 1])
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(probs[mask])) - float(np.mean(y01[mask])))
    return brier, float(ece)


def run_exp8_scalability(out_dir: Path) -> None:
    print("\n" + "=" * 72, flush=True)
    print("EXP 8: Large-N Tabular Scalability (N up to 100,000) & Full UCI Adult/a9a", flush=True)
    print("=" * 72, flush=True)

    # Part A: Cardinality scaling N in {2500, 10000, 30000, 100000} on heterogeneous multi-scale table (d=40)
    N_grid = [2500, 10000, 30000, 100000]
    d = 40
    mb = 128
    K_svrg = 200
    sigma_fixed = 0.008  # fixed proposal scale across table sizes (h = 6.4e-5: stable at N=2500, stiff for N>=10000)
    n_burnin, n_samples = 200, 600
    n_seeds = 5

    rng_fixed = np.random.default_rng(2027)
    feat_scale = 10.0 ** np.linspace(-0.8, 0.9, d)  # multi-unit column scales (50x scale range -> 2500x Hessian range)
    true_th = rng_fixed.standard_normal(d) * 0.5
    X_test = rng_fixed.standard_normal((3000, d)) * feat_scale[None, :]
    y_test = np.where(rng_fixed.random(3000) < sigmoid(X_test @ true_th), 1.0, -1.0)

    scale_methods = ["SGLD", "SVRG-SGLD", "v-SGBD", "c-SGBD", "NaiveVR-SGBD", "SaVR-BD (SVRG)", "SaVR-BD (SAGA)"]
    cardinality_res = {m: {} for m in scale_methods}

    for N in N_grid:
        rng_n = np.random.default_rng(100 + N)
        X_tr = rng_n.standard_normal((N, d)) * feat_scale[None, :]
        y_tr = np.where(rng_n.random(N) < sigmoid(X_tr @ true_th), 1.0, -1.0)
        grad_lik, grad_prior, nlp = make_logistic_model(X_tr, y_tr, alpha_prior=1.0)
        grad_full = lambda th: grad_prior(th) + grad_lik(th, None, False)
        map_th = minimize(nlp, np.zeros(d), jac=lambda th: -grad_full(th), method="L-BFGS-B", options={"maxiter": 400}).x

        print(f"-- Table Cardinality N={N} (d={d}, mb={mb}, sigma={sigma_fixed}) --", flush=True)
        for m in scale_methods:
            rmses, nlls, briers, eces, ms_per_100, fbs, divs = [], [], [], [], [], [], 0
            for s in range(n_seeds):
                th0 = map_th + 0.03 * np.random.default_rng(7000 + s).standard_normal(d)
                if m == "SaVR-BD (SAGA)":
                    cfg = SaVRBDConfig(
                        step_scale=sigma_fixed, minibatch_size=mb, epoch_length=K_svrg,
                        ema_beta=0.2, rho_star=2.0, kappa=KAPPA_CURVATURE, variant="saga"
                    )
                    smp = SaVRBD(grad_lik, grad_prior, N, d, cfg, rng=np.random.default_rng(8500 + s))
                elif m == "SaVR-BD (SVRG)":
                    smp = build_sampler("SaVR-BD", grad_lik, grad_prior, N, d, sigma_fixed, mb, K_svrg, seed=8500 + s)
                else:
                    smp = build_sampler(m, grad_lik, grad_prior, N, d, sigma_fixed, mb, K_svrg, seed=8500 + s)

                smp.initialize(th0)
                t0 = time.perf_counter()
                samp = smp.sample(n_samples, n_burnin)
                step_wall = time.perf_counter() - t0
                ms100 = (step_wall / (n_samples + n_burnin)) * 100.0 * 1000.0

                # Genuine divergence or macroscopic limit-cycle blow-up outside parameter basin (||theta - theta_MAP||_inf > 5.0)
                if not np.all(np.isfinite(samp)) or float(np.max(np.abs(samp - map_th[None, :]))) > 5.0:
                    divs += 1
                    continue

                param_err = float(np.linalg.norm(samp.mean(axis=0) - map_th))
                pll = heldout_logistic_pll(samp, X_test, y_test)
                brier, ece = binary_ece_and_brier(samp, X_test, y_test)
                fb = (
                    float(np.mean(smp.diagnostics["fallback_fraction"]))
                    if hasattr(smp, "diagnostics") and smp.diagnostics["fallback_fraction"]
                    else 0.0
                )
                rmses.append(param_err)
                nlls.append(-pll)
                briers.append(brier)
                eces.append(ece)
                ms_per_100.append(ms100)
                fbs.append(fb)

            cardinality_res[m][str(N)] = {
                "diverged_count": divs,
                "n_seeds": n_seeds,
                "param_err_mean": float(np.mean(rmses)) if rmses else float("nan"),
                "param_err_std": float(np.std(rmses)) if rmses else float("nan"),
                "test_nll_mean": float(np.mean(nlls)) if nlls else float("nan"),
                "test_nll_std": float(np.std(nlls)) if nlls else float("nan"),
                "brier_mean": float(np.mean(briers)) if briers else float("nan"),
                "ece_mean": float(np.mean(eces)) if eces else float("nan"),
                "ms_per_100_steps": float(np.mean(ms_per_100)) if ms_per_100 else float("nan"),
                "fallback_frac": float(np.mean(fbs)) if fbs else 0.0,
            }
            st_str = (
                f"err={np.mean(rmses):.4f} | NLL={np.mean(nlls):.4f} | ECE={np.mean(eces):.4f} | {np.mean(ms_per_100):.1f}ms/100st"
                if rmses else f"OSC/DIV ({divs}/{n_seeds})"
            )
            print(f"  {m:16s} | {st_str}", flush=True)

    payload = {
        "experiment": "large_N_tabular_scalability",
        "N_grid": N_grid,
        "d": d,
        "minibatch_size": mb,
        "epoch_length": K_svrg,
        "sigma_fixed": sigma_fixed,
        "methods": scale_methods,
        "results": cardinality_res,
    }
    (out_dir / "exp8_scalability.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["1", "2", "3", "4", "5", "6", "7", "8", "all"], default="all")
    parser.add_argument("--out", type=str, default=str(ROOT / "results"))
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_all = time.time()
    if args.exp in ("1", "all"):
        run_exp1_a9a(out_dir)
    if args.exp in ("2", "all"):
        run_exp2_cauchy(out_dir)
    if args.exp in ("3", "all"):
        run_exp3_heterogeneous(out_dir)
    if args.exp in ("4", "all"):
        run_exp4_mismatch(out_dir)
    if args.exp in ("5", "all"):
        run_exp5_bnn_uci(out_dir)
    if args.exp in ("6", "all"):
        run_exp6_ablations(out_dir)
    if args.exp in ("7", "all"):
        run_exp7_diagnostics(out_dir)
    if args.exp in ("8", "all"):
        run_exp8_scalability(out_dir)
    print(f"\nALL REQUESTED EXPERIMENTS FINISHED IN {time.time()-t_all:.1f}s", flush=True)
