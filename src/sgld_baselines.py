"""
Langevin SG-MCMC baselines for comparison with Barker-family samplers:
- SGLD (Welling & Teh, 2011)
- SGLD-CV (Baker et al., 2019)
- SVRG-SGLD (Dubey et al., 2016)

All methods evaluate exact O(d) prior gradients at the current state theta_t
and apply stochastic / control-variate estimation strictly to the data
log-likelihood sum, matching the SaVR-BD accounting convention.
"""

from dataclasses import dataclass
from typing import Callable, Optional
import time
import numpy as np


@dataclass
class SGLDConfig:
    """Configuration for Langevin-family SG-MCMC samplers."""

    step_size: float = 1e-4
    minibatch_size: int = 128
    epoch_length: int = 100
    preconditioner: str = "identity"  # "identity" or "rmsprop"
    rmsprop_alpha: float = 0.99
    rmsprop_eps: float = 1e-5


class SGLD:
    """Unadjusted Stochastic Gradient Langevin Dynamics."""

    def __init__(
        self,
        log_likelihood_grad: Callable[..., np.ndarray],
        log_prior_grad: Callable[[np.ndarray], np.ndarray],
        data_size: int,
        dim: int,
        config: Optional[SGLDConfig] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.grad_lik = log_likelihood_grad
        self.grad_prior = log_prior_grad
        self.N = int(data_size)
        self.d = int(dim)
        self.config = config or SGLDConfig()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.theta: Optional[np.ndarray] = None
        self._v: Optional[np.ndarray] = None
        self.grad_evals: int = 0

    def initialize(self, theta0: np.ndarray) -> None:
        self.theta = np.asarray(theta0, dtype=float).copy()
        self._v = np.ones(self.d, dtype=float)
        self.grad_evals = 0

    def _estimate_gradient(self, theta: np.ndarray) -> np.ndarray:
        n = min(self.config.minibatch_size, self.N)
        idx = self.rng.choice(self.N, size=n, replace=False)
        gp = self.grad_lik(theta, idx, per_point=True)
        self.grad_evals += n
        return self.grad_prior(theta) + (self.N / n) * gp.sum(axis=0)

    def _step_with_grad(self, grad: np.ndarray) -> np.ndarray:
        h = self.config.step_size
        noise = self.rng.standard_normal(self.d)
        if self.config.preconditioner == "identity":
            if not np.all(np.isfinite(grad)) or np.max(np.abs(self.theta)) > 1e5:
                self.theta = np.full(self.d, np.nan)
                return self.theta.copy()
            drift = 0.5 * h * grad
            next_th = self.theta + drift + np.sqrt(h) * noise
            if not np.all(np.isfinite(next_th)) or np.max(np.abs(next_th)) > 1e5:
                self.theta = np.full(self.d, np.nan)
            else:
                self.theta = next_th
        elif self.config.preconditioner == "rmsprop":
            alpha = self.config.rmsprop_alpha
            self._v = alpha * self._v + (1.0 - alpha) * (grad ** 2)
            inv_std = 1.0 / (np.sqrt(self._v) + self.config.rmsprop_eps)
            self.theta = self.theta + 0.5 * h * (inv_std * grad) + np.sqrt(h * inv_std) * noise
        else:
            raise ValueError(f"Unknown preconditioner: {self.config.preconditioner}")
        return self.theta.copy()

    def step(self) -> np.ndarray:
        grad = self._estimate_gradient(self.theta)
        return self._step_with_grad(grad)

    def sample(
        self,
        n_samples: int,
        n_burnin: int = 0,
        thin: int = 1,
        verbose: bool = False,
    ) -> np.ndarray:
        total_steps = n_burnin + n_samples * thin
        out = np.zeros((n_samples, self.d), dtype=float)
        idx = 0
        t0 = time.time()
        for t in range(total_steps):
            self.step()
            if t >= n_burnin and (t - n_burnin) % thin == 0:
                out[idx] = self.theta
                idx += 1
            if verbose and (t + 1) % 1000 == 0:
                print(f"  SGLD step {t+1}/{total_steps} ({time.time()-t0:.1f}s)")
        return out


class SGLDCV(SGLD):
    """SGLD with fixed control variates centered at a reference point (Baker et al., 2019)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_theta: Optional[np.ndarray] = None
        self.ref_lik_grad: Optional[np.ndarray] = None

    def initialize(self, theta0: np.ndarray, ref_theta: Optional[np.ndarray] = None) -> None:
        super().initialize(theta0)
        self.ref_theta = (
            np.asarray(ref_theta, dtype=float).copy()
            if ref_theta is not None
            else self.theta.copy()
        )
        self.ref_lik_grad = self.grad_lik(self.ref_theta, None, per_point=False)
        self.grad_evals += self.N

    def _estimate_gradient(self, theta: np.ndarray) -> np.ndarray:
        n = min(self.config.minibatch_size, self.N)
        idx = self.rng.choice(self.N, size=n, replace=False)
        gp_cur = self.grad_lik(theta, idx, per_point=True)
        gp_ref = self.grad_lik(self.ref_theta, idx, per_point=True)
        self.grad_evals += 2 * n
        lik_est = self.ref_lik_grad + (self.N / n) * (gp_cur - gp_ref).sum(axis=0)
        return self.grad_prior(theta) + lik_est


class SVRGSGLD(SGLD):
    """SGLD with periodic SVRG checkpoint refreshes (Dubey et al., 2016)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_theta: Optional[np.ndarray] = None
        self.checkpoint_lik_grad: Optional[np.ndarray] = None
        self.step_in_epoch: int = 0

    def _refresh_checkpoint(self, theta: np.ndarray) -> None:
        self.checkpoint_theta = theta.copy()
        self.checkpoint_lik_grad = self.grad_lik(self.checkpoint_theta, None, per_point=False)
        self.grad_evals += self.N
        self.step_in_epoch = 0

    def initialize(self, theta0: np.ndarray) -> None:
        super().initialize(theta0)
        self._refresh_checkpoint(self.theta)

    def step(self) -> np.ndarray:
        if self.step_in_epoch >= self.config.epoch_length:
            self._refresh_checkpoint(self.theta)

        if self.step_in_epoch == 0:
            grad = self.grad_prior(self.theta) + self.checkpoint_lik_grad
        else:
            n = min(self.config.minibatch_size, self.N)
            idx = self.rng.choice(self.N, size=n, replace=False)
            gp_cur = self.grad_lik(self.theta, idx, per_point=True)
            gp_ck = self.grad_lik(self.checkpoint_theta, idx, per_point=True)
            self.grad_evals += 2 * n
            lik_est = self.checkpoint_lik_grad + (self.N / n) * (gp_cur - gp_ck).sum(axis=0)
            grad = self.grad_prior(self.theta) + lik_est

        out = self._step_with_grad(grad)
        self.step_in_epoch += 1
        return out
