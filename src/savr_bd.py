"""
Saturation-Aware Variance Reduction for Stochastic Gradient Barker Dynamics (SaVR-BD).

Implements:
- SaVRBD: SVRG/SAGA variance-reduced Barker dynamics with predictable
  finite-population noise estimation and smooth saturation-aware curvature
  calibration.
- VanillaSGBD (v-SGBD): Uncorrected stochastic-gradient Barker dynamics.
- CorrectedSGBD (c-SGBD): Faithful Mauri & Zanella (2024) bias-corrected SGBD
  with inverse shrinkage kappa / sqrt(kappa^2 - tau^2) and sign fallback when
  tau >= kappa.
- StableCorrectedSGBD (sc-SGBD): Non-VR SGBD with singularity-free surrogate
  inflation sqrt(kappa^2 + tau^2) / kappa.
- NaiveVRSGBD: SVRG + Barker dynamics without noise calibration (phi = 1).
- UniformVRSGBD (UniVR-SGBD): SVRG + Barker dynamics with uniform inflation.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import time
import numpy as np

# Probit-logistic global scale constant (Mauri & Zanella, 2024; Bowling et al., 2009)
KAPPA_PROBIT = 1.702
# Exact origin curvature-matching constant from sigma''(z) = -sigma'(z) tanh(z/2)
KAPPA_CURVATURE = np.sqrt(2.0)


@dataclass
class SaVRBDConfig:
    """Configuration for Barker-family stochastic-gradient samplers."""

    step_scale: float = 0.01       # sigma: bimodal reference step scale
    minibatch_size: int = 128      # n: minibatch size per step
    epoch_length: int = 100        # K: SVRG full-gradient refresh period
    ema_beta: float = 0.2          # beta: EMA smoothing rate on gradient-noise variance
    bimodal_spread: float = 0.1    # relative std of bimodal reference N(+-sigma, (0.1*sigma)^2)
    rho_star: float = 2.0          # natural logistic saturation scale (rho* = 2 matches tanh(a/2))
    kappa: float = KAPPA_CURVATURE # noise scale constant in phi_base(tau) = sqrt(1 + tau^2 / kappa^2)
    gate_mode: str = "smooth"      # "smooth" (tanh/x), "hard" (|a| < rho*), "uniform", or "none"
    variant: str = "svrg"          # "svrg" or "saga"
    preconditioner: str = "identity"  # "identity" or "rmsprop" (diagonal coordinate scaling)
    rmsprop_alpha: float = 0.99
    rmsprop_eps: float = 1e-5


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    nonneg = x >= 0.0
    out[nonneg] = 1.0 / (1.0 + np.exp(-x[nonneg]))
    ex = np.exp(x[~nonneg])
    out[~nonneg] = ex / (1.0 + ex)
    return out


def sigmoid_deriv(x: np.ndarray) -> np.ndarray:
    """First derivative sigma'(x) = sigma(x)(1 - sigma(x)) in (0, 1/4]."""
    s = sigmoid(x)
    return s * (1.0 - s)


def sample_reference(
    sigma: float,
    d: int,
    rng: np.random.Generator,
    spread: float = 0.1,
) -> np.ndarray:
    """
    Sample coordinate step magnitudes from the bimodal reference of Mauri & Zanella (2024):
        mu_sigma = 0.5 * N(-sigma, (spread * sigma)^2) + 0.5 * N(sigma, (spread * sigma)^2).
    """
    magnitudes = np.abs(rng.normal(sigma, spread * sigma, size=d))
    signs = 2 * rng.integers(0, 2, size=d) - 1
    return signs * magnitudes


def calibrated_argument(
    a: np.ndarray,
    tau_a: np.ndarray,
    rho_star: float = 2.0,
    kappa: float = KAPPA_CURVATURE,
    gate_mode: str = "smooth",
) -> np.ndarray:
    """
    Compute calibrated logistic argument psi(a, tau_a) for flipping probability sigma(psi).

    Modes:
    - "none":    psi = a
    - "uniform": psi = a * sqrt(1 + tau_a^2 / kappa^2)
    - "hard":    psi = a * sqrt(1 + tau_a^2 / kappa^2) if |a| < rho_star else a
    - "smooth":  psi = a + (sqrt(1 + tau_a^2 / kappa^2) - 1) * rho_star * tanh(a / rho_star)
                 Equivalently psi = a * phi_SaVR(a, tau_a) with C^infinity gate
                 g(a) = tanh(a / rho_star) / (a / rho_star) in (0, 1].
    """
    if gate_mode == "none":
        return a
    tau_pos = np.maximum(tau_a, 0.0)
    phi_base = np.sqrt(1.0 + (tau_pos / kappa) ** 2)
    if gate_mode == "uniform":
        return a * phi_base
    if gate_mode == "hard":
        return np.where(np.abs(a) < rho_star, a * phi_base, a)
    if gate_mode == "smooth":
        if rho_star <= 0.0:
            return a
        return a + (phi_base - 1.0) * rho_star * np.tanh(a / rho_star)
    raise ValueError(f"Unknown gate_mode: {gate_mode}")


def finite_pop_grad_variance(per_point_diff: np.ndarray, N: int) -> np.ndarray:
    """
    Unbiased finite-population variance estimator of the scaled minibatch sum
        (N / n) * sum_{i in S_n} diff_i
    under uniform sampling without replacement of size n from N items:
        hat_u_j = (N^2 / n) * (1 - n / N) * s_{j}^2,
    where s_j^2 is the Bessel-corrected sample variance (ddof=1) across S_n.
    """
    n = per_point_diff.shape[0]
    if n < 2 or n >= N:
        return np.zeros(per_point_diff.shape[1], dtype=float)
    fpc_scale = (float(N) ** 2 / float(n)) * (1.0 - float(n) / float(N))
    sample_var = per_point_diff.var(axis=0, ddof=1)
    return np.maximum(fpc_scale * sample_var, 0.0)


class SaVRBD:
    """
    Saturation-Aware Variance-Reduced Stochastic Gradient Barker Dynamics.
    """

    def __init__(
        self,
        log_likelihood_grad: Callable[..., np.ndarray],
        log_prior_grad: Callable[[np.ndarray], np.ndarray],
        data_size: int,
        dim: int,
        config: Optional[SaVRBDConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.grad_lik = log_likelihood_grad
        self.grad_prior = log_prior_grad
        self.N = int(data_size)
        self.d = int(dim)
        self.config = config or SaVRBDConfig()
        self.rng = rng if rng is not None else np.random.default_rng()

        self.theta: Optional[np.ndarray] = None
        self.checkpoint_theta: Optional[np.ndarray] = None
        self.checkpoint_lik_grad: Optional[np.ndarray] = None
        self.u_hat: np.ndarray = np.zeros(self.d, dtype=float)  # predictable gradient-noise variance EMA
        self._has_u_history: bool = False
        self.step_in_epoch: int = 0
        self.grad_evals: int = 0
        self._precond_v: Optional[np.ndarray] = None

        # SAGA gradient table
        self._saga_table: Optional[np.ndarray] = None
        self._saga_sum: Optional[np.ndarray] = None

        self.diagnostics: Dict[str, List[float]] = {
            "saturated_fraction": [],
            "mean_tau_a": [],
            "mean_effective_phi": [],
            "fallback_fraction": [],
        }

    def _refresh_svrg_checkpoint(self, theta: np.ndarray) -> None:
        self.checkpoint_theta = theta.copy()
        self.checkpoint_lik_grad = self.grad_lik(self.checkpoint_theta, None, per_point=False)
        self.grad_evals += self.N
        self.step_in_epoch = 0
        # At theta_t == checkpoint_theta, SVRG likelihood noise is identically zero
        self.u_hat = np.zeros(self.d, dtype=float)
        self._has_u_history = False

    def initialize(self, theta0: np.ndarray) -> None:
        self.theta = np.asarray(theta0, dtype=float).copy()
        self.grad_evals = 0
        self._precond_v = np.ones(self.d, dtype=float)
        for k in self.diagnostics:
            self.diagnostics[k] = []

        if self.config.variant == "svrg":
            self._refresh_svrg_checkpoint(self.theta)
        elif self.config.variant == "saga":
            self._saga_table = np.zeros((self.N, self.d), dtype=float)
            block = max(1, min(2048, self.N))
            for start in range(0, self.N, block):
                end = min(start + block, self.N)
                idx = np.arange(start, end)
                self._saga_table[idx] = self.grad_lik(self.theta, idx, per_point=True)
            self._saga_sum = self._saga_table.sum(axis=0)
            self.grad_evals += self.N
            self.u_hat = np.zeros(self.d, dtype=float)
            self._has_u_history = False
            self.step_in_epoch = 0
        else:
            raise ValueError(f"Unsupported VR variant: {self.config.variant}")

    def _coordinate_scales(self, grad_est: np.ndarray) -> np.ndarray:
        """Return per-coordinate reference scale multiplier (1 for isotropic, inv_rms for rmsprop)."""
        if self.config.preconditioner == "identity":
            return np.ones(self.d, dtype=float)
        if self.config.preconditioner == "rmsprop":
            alpha = self.config.rmsprop_alpha
            self._precond_v = alpha * self._precond_v + (1.0 - alpha) * (grad_est ** 2)
            raw = 1.0 / np.sqrt(self._precond_v + self.config.rmsprop_eps)
            # Normalize median scale to 1 so step_scale sigma retains consistent physical meaning
            return raw / (np.median(raw) + 1e-12)
        raise ValueError(f"Unknown preconditioner: {self.config.preconditioner}")

    def step(self) -> np.ndarray:
        cfg = self.config
        n = min(cfg.minibatch_size, self.N)

        if cfg.variant == "svrg":
            if self.step_in_epoch >= cfg.epoch_length:
                self._refresh_svrg_checkpoint(self.theta)

            if self.step_in_epoch == 0:
                grad_est = self.grad_prior(self.theta) + self.checkpoint_lik_grad
                u_cur = np.zeros(self.d, dtype=float)
                exact_step = True
            else:
                idx = self.rng.choice(self.N, size=n, replace=False)
                gp_cur = self.grad_lik(self.theta, idx, per_point=True)
                gp_ck = self.grad_lik(self.checkpoint_theta, idx, per_point=True)
                self.grad_evals += 2 * n
                diff = gp_cur - gp_ck
                grad_est = (
                    self.grad_prior(self.theta)
                    + self.checkpoint_lik_grad
                    + (self.N / n) * diff.sum(axis=0)
                )
                u_cur = finite_pop_grad_variance(diff, self.N)
                exact_step = False

        elif cfg.variant == "saga":
            idx = self.rng.choice(self.N, size=n, replace=False)
            gp_cur = self.grad_lik(self.theta, idx, per_point=True)
            self.grad_evals += n
            gp_old = self._saga_table[idx]
            diff = gp_cur - gp_old
            grad_est = (
                self.grad_prior(self.theta)
                + self._saga_sum
                + (self.N / n) * diff.sum(axis=0)
            )
            u_cur = finite_pop_grad_variance(diff, self.N)
            # Update SAGA table after forming unbiased estimator
            self._saga_table[idx] = gp_cur
            self._saga_sum = self._saga_sum + diff.sum(axis=0)
            exact_step = (self.step_in_epoch == 0)

        else:
            raise ValueError(f"Unknown variant: {cfg.variant}")

        coord_scale = self._coordinate_scales(grad_est)
        w = coord_scale * sample_reference(cfg.step_scale, self.d, self.rng, cfg.bimodal_spread)
        a = w * grad_est

        # Predictable noise std from F_{t-1}-measurable u_hat (zero at exact checkpoint step)
        tau_a = (
            np.zeros(self.d, dtype=float)
            if exact_step
            else np.abs(w) * np.sqrt(np.maximum(self.u_hat, 0.0))
        )

        psi = calibrated_argument(
            a,
            tau_a,
            rho_star=cfg.rho_star,
            kappa=cfg.kappa,
            gate_mode=cfg.gate_mode,
        )
        flip_probs = sigmoid(psi)
        b = 2.0 * (self.rng.random(self.d) < flip_probs).astype(float) - 1.0
        self.theta = self.theta + w * b
        self.step_in_epoch += 1

        # Update predictable variance state for subsequent steps
        if not exact_step:
            if not self._has_u_history:
                self.u_hat = u_cur
                self._has_u_history = True
            else:
                self.u_hat = (1.0 - cfg.ema_beta) * self.u_hat + cfg.ema_beta * u_cur

        # Diagnostics
        sat_frac = float(np.mean(np.abs(a) >= cfg.rho_star)) if cfg.rho_star > 0 else 0.0
        eff_phi = np.where(np.abs(a) > 1e-8, psi / a, np.sqrt(1.0 + (tau_a / cfg.kappa) ** 2))
        self.diagnostics["saturated_fraction"].append(sat_frac)
        self.diagnostics["mean_tau_a"].append(float(np.mean(tau_a)))
        self.diagnostics["mean_effective_phi"].append(float(np.mean(eff_phi)))
        self.diagnostics["fallback_fraction"].append(0.0)
        return self.theta.copy()

    def sample(
        self,
        n_samples: int,
        n_burnin: int = 0,
        thin: int = 1,
        verbose: bool = False,
    ) -> np.ndarray:
        total = n_burnin + n_samples * thin
        out = np.zeros((n_samples, self.d), dtype=float)
        idx = 0
        t0 = time.time()
        for t in range(total):
            self.step()
            if t >= n_burnin and (t - n_burnin) % thin == 0:
                out[idx] = self.theta
                idx += 1
            if verbose and (t + 1) % 1000 == 0:
                sat = (
                    float(np.mean(self.diagnostics["saturated_fraction"][-100:]))
                    if self.diagnostics["saturated_fraction"]
                    else 0.0
                )
                print(f"  step {t+1}/{total} ({time.time()-t0:.1f}s) | sat={sat:.1%}")
        return out


class VanillaSGBD(SaVRBD):
    """Uncorrected stochastic-gradient Barker dynamics (v-SGBD; Mauri & Zanella, 2024)."""

    def initialize(self, theta0: np.ndarray) -> None:
        self.theta = np.asarray(theta0, dtype=float).copy()
        self.grad_evals = 0
        self._precond_v = np.ones(self.d, dtype=float)
        self.u_hat = np.zeros(self.d, dtype=float)
        self._has_u_history = False
        self.step_in_epoch = 0
        for k in self.diagnostics:
            self.diagnostics[k] = []

    def step(self) -> np.ndarray:
        cfg = self.config
        n = min(cfg.minibatch_size, self.N)
        idx = self.rng.choice(self.N, size=n, replace=False)
        gp = self.grad_lik(self.theta, idx, per_point=True)
        self.grad_evals += n
        grad_est = self.grad_prior(self.theta) + (self.N / n) * gp.sum(axis=0)
        coord_scale = self._coordinate_scales(grad_est)
        w = coord_scale * sample_reference(cfg.step_scale, self.d, self.rng, cfg.bimodal_spread)
        a = w * grad_est
        flip_probs = sigmoid(a)
        b = 2.0 * (self.rng.random(self.d) < flip_probs).astype(float) - 1.0
        self.theta = self.theta + w * b
        self.step_in_epoch += 1
        self.diagnostics["fallback_fraction"].append(0.0)
        return self.theta.copy()


class CorrectedSGBD(VanillaSGBD):
    """
    Faithful Mauri & Zanella (2024) bias-corrected SGBD (c-SGBD):
    Uses inverse shrinkage phi_MZ(tau) = kappa / sqrt(kappa^2 - tau^2) when tau < kappa,
    and falls back to the extreme-noise sign estimator 1{a > 0} when tau >= kappa.
    """

    def step(self) -> np.ndarray:
        cfg = self.config
        n = min(cfg.minibatch_size, self.N)
        idx = self.rng.choice(self.N, size=n, replace=False)
        gp = self.grad_lik(self.theta, idx, per_point=True)
        self.grad_evals += n
        grad_est = self.grad_prior(self.theta) + (self.N / n) * gp.sum(axis=0)
        u_cur = finite_pop_grad_variance(gp, self.N)

        coord_scale = self._coordinate_scales(grad_est)
        w = coord_scale * sample_reference(cfg.step_scale, self.d, self.rng, cfg.bimodal_spread)
        a = w * grad_est

        # Use predictable EMA variance (initialized on first step)
        u_pred = u_cur if not self._has_u_history else self.u_hat
        tau_a = np.abs(w) * np.sqrt(np.maximum(u_pred, 0.0))

        below = tau_a < (KAPPA_PROBIT * 0.9999)
        fallback_frac = float(np.mean(~below))
        flip_probs = np.empty(self.d, dtype=float)
        flip_probs[below] = sigmoid(
            a[below] * KAPPA_PROBIT / np.sqrt(KAPPA_PROBIT ** 2 - tau_a[below] ** 2)
        )
        flip_probs[~below] = np.where(
            a[~below] > 0.0, 1.0, np.where(a[~below] < 0.0, 0.0, 0.5)
        )

        b = 2.0 * (self.rng.random(self.d) < flip_probs).astype(float) - 1.0
        self.theta = self.theta + w * b
        self.step_in_epoch += 1

        if not self._has_u_history:
            self.u_hat = u_cur
            self._has_u_history = True
        else:
            self.u_hat = (1.0 - cfg.ema_beta) * self.u_hat + cfg.ema_beta * u_cur

        self.diagnostics["fallback_fraction"].append(fallback_frac)
        self.diagnostics["mean_tau_a"].append(float(np.mean(tau_a)))
        return self.theta.copy()


class StableCorrectedSGBD(VanillaSGBD):
    """Non-VR SGBD with singularity-free inflation sqrt(1 + tau^2 / kappa^2)."""

    def step(self) -> np.ndarray:
        cfg = self.config
        n = min(cfg.minibatch_size, self.N)
        idx = self.rng.choice(self.N, size=n, replace=False)
        gp = self.grad_lik(self.theta, idx, per_point=True)
        self.grad_evals += n
        grad_est = self.grad_prior(self.theta) + (self.N / n) * gp.sum(axis=0)
        u_cur = finite_pop_grad_variance(gp, self.N)

        coord_scale = self._coordinate_scales(grad_est)
        w = coord_scale * sample_reference(cfg.step_scale, self.d, self.rng, cfg.bimodal_spread)
        a = w * grad_est

        u_pred = u_cur if not self._has_u_history else self.u_hat
        tau_a = np.abs(w) * np.sqrt(np.maximum(u_pred, 0.0))
        phi = np.sqrt(1.0 + (tau_a / KAPPA_PROBIT) ** 2)
        flip_probs = sigmoid(a * phi)

        b = 2.0 * (self.rng.random(self.d) < flip_probs).astype(float) - 1.0
        self.theta = self.theta + w * b
        self.step_in_epoch += 1

        if not self._has_u_history:
            self.u_hat = u_cur
            self._has_u_history = True
        else:
            self.u_hat = (1.0 - cfg.ema_beta) * self.u_hat + cfg.ema_beta * u_cur

        self.diagnostics["fallback_fraction"].append(0.0)
        self.diagnostics["mean_tau_a"].append(float(np.mean(tau_a)))
        return self.theta.copy()


class NaiveVRSGBD(SaVRBD):
    """SVRG/SAGA Barker dynamics with no noise calibration (gate_mode='none')."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = SaVRBDConfig(**{**self.config.__dict__, "gate_mode": "none"})


class UniformVRSGBD(SaVRBD):
    """SVRG/SAGA Barker dynamics with uniform Mauri-Zanella surrogate inflation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = SaVRBDConfig(
            **{**self.config.__dict__, "gate_mode": "uniform", "kappa": KAPPA_PROBIT}
        )


class HardGateVRSGBD(SaVRBD):
    """SVRG/SAGA Barker dynamics with hard indicator gate |a| < rho*."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = SaVRBDConfig(
            **{**self.config.__dict__, "gate_mode": "hard", "kappa": KAPPA_PROBIT}
        )
