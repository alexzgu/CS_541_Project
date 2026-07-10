"""Sticky HDP-HMM (weak-limit) with blocked-Gibbs inference — spec §7.

Boundaries are the derived events z_t != z_{t-1} of a latent state path; no
break variable exists, hence no class imbalance. Run strictly UNSUPERVISED:
token labels/sequences never touch the state path (no forced alignment).

Sweep (spec §7.2): FFBS path draw -> per-dim Normal-Inverse-Gamma emission
updates -> Dirichlet transitions with sticky bias -> CRT table counts with
Fox et al.'s override correction -> global weights beta.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class PCA:
    def __init__(self, dim: int):
        self.dim = dim
        self.mean: np.ndarray | None = None
        self.components: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCA":
        self.mean = X.mean(axis=0)
        Xc = X - self.mean
        # SVD on a subsample for speed; components are what matter
        if len(Xc) > 20000:
            idx = np.random.default_rng(0).choice(len(Xc), 20000, replace=False)
            Xs = Xc[idx]
        else:
            Xs = Xc
        _, s, Vt = np.linalg.svd(Xs, full_matrices=False)
        self.components = Vt[: self.dim]
        proj = Xs @ self.components.T
        self.scale = proj.std(axis=0) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) @ self.components.T) / self.scale


def _kmeans_init(X: np.ndarray, k: int, iters: int = 8, seed: int = 0) -> np.ndarray:
    """Tiny k-means for a sensible initial state path."""
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), size=min(k, len(X)), replace=False)]
    z = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1) if len(X) * len(C) < 2e7 else None
        if d is None:  # chunked distance for long songs
            z = np.concatenate([
                ((X[s : s + 4096, None, :] - C[None]) ** 2).sum(-1).argmin(1)
                for s in range(0, len(X), 4096)
            ])
        else:
            z = d.argmin(1)
        for j in range(len(C)):
            pts = X[z == j]
            if len(pts):
                C[j] = pts.mean(0)
    return z


@dataclass
class StickyHDPHMMResult:
    boundary_prob: np.ndarray            # [T] P(z_t != z_{t-1})
    boundaries: list[dict] = field(default_factory=list)  # {frame, prob, std_frames}
    last_path: np.ndarray | None = None  # [T] final sample (unit discovery)
    n_active_states: int = 0


class StickyHDPHMM:
    def __init__(self, L: int = 120, alpha: float = 4.0, gamma: float = 4.0,
                 rho: float = 0.95, sweeps: int = 30, burnin: int = 10,
                 seed: int = 541):
        self.L, self.alpha, self.gamma = L, alpha, gamma
        self.kappa = rho / (1.0 - rho) * alpha
        self.sweeps, self.burnin = sweeps, burnin
        self.rng = np.random.default_rng(seed)
        # NIG prior per dimension (data is whitened): mu0, lam0, a0, b0
        self.mu0, self.lam0, self.a0, self.b0 = 0.0, 1.0, 2.0, 0.5

    # -- emission model ------------------------------------------------------
    def _sample_emissions(self, X: np.ndarray, z: np.ndarray):
        D = X.shape[1]
        mu = np.empty((self.L, D))
        var = np.empty((self.L, D))
        for j in range(self.L):
            pts = X[z == j]
            n = len(pts)
            if n:
                xbar = pts.mean(0)
                S = ((pts - xbar) ** 2).sum(0)
            else:
                xbar = np.zeros(D)
                S = np.zeros(D)
            lam_n = self.lam0 + n
            mu_n = (self.lam0 * self.mu0 + n * xbar) / lam_n
            a_n = self.a0 + n / 2
            b_n = self.b0 + 0.5 * S + (self.lam0 * n / (2 * lam_n)) * (xbar - self.mu0) ** 2
            var[j] = b_n / self.rng.gamma(a_n, 1.0, size=D)      # InvGamma draw
            mu[j] = self.rng.normal(mu_n, np.sqrt(var[j] / lam_n))
        return mu, var

    def _log_lik(self, X: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
        """[T, L] diagonal-Gaussian log-likelihoods."""
        # -(x-mu)^2/(2 var) - 0.5 log(2 pi var), summed over dims
        const = -0.5 * np.log(2 * np.pi * var).sum(1)             # [L]
        ll = np.empty((len(X), self.L))
        for s in range(0, len(X), 2048):                          # memory-chunked
            Xc = X[s : s + 2048]
            diff = Xc[:, None, :] - mu[None]                      # [t, L, D]
            ll[s : s + len(Xc)] = const - 0.5 * (diff ** 2 / var[None]).sum(-1)
        return ll

    # -- FFBS ----------------------------------------------------------------
    def _ffbs(self, ll: np.ndarray, pi: np.ndarray, beta: np.ndarray) -> np.ndarray:
        T = len(ll)
        alpha = np.empty((T, self.L))
        w = np.exp(ll[0] - ll[0].max()) * beta
        alpha[0] = w / w.sum()
        for t in range(1, T):
            pred = alpha[t - 1] @ pi
            w = np.exp(ll[t] - ll[t].max()) * pred
            s = w.sum()
            alpha[t] = w / s if s > 0 else np.full(self.L, 1.0 / self.L)
        z = np.empty(T, dtype=int)
        z[T - 1] = self.rng.choice(self.L, p=alpha[T - 1])
        for t in range(T - 2, -1, -1):
            w = alpha[t] * pi[:, z[t + 1]]
            s = w.sum()
            p = w / s if s > 0 else alpha[t]
            z[t] = self.rng.choice(self.L, p=p)
        return z

    # -- transitions / global weights ----------------------------------------
    def _sample_transitions(self, z: np.ndarray, beta: np.ndarray) -> np.ndarray:
        n = np.zeros((self.L, self.L))
        np.add.at(n, (z[:-1], z[1:]), 1.0)
        conc = self.alpha * beta[None, :] + self.kappa * np.eye(self.L) + n
        pi = self.rng.gamma(np.maximum(conc, 1e-6))
        pi /= pi.sum(1, keepdims=True)
        return pi, n

    def _sample_beta(self, n: np.ndarray, beta: np.ndarray) -> np.ndarray:
        # CRT table counts m_jk ~ sum Bern(w/(w+i-1)), w = alpha*beta_k (+kappa on diag)
        m = np.zeros((self.L, self.L))
        rho = self.kappa / (self.alpha + self.kappa)
        for j in range(self.L):
            for k in np.flatnonzero(n[j] > 0):
                w = self.alpha * beta[k] + (self.kappa if j == k else 0.0)
                i = np.arange(int(n[j, k]))
                m[j, k] = (self.rng.random(len(i)) < (w / (w + i))).sum()
        # sticky override correction (Fox et al. 2011)
        mbar = m.copy()
        for j in range(self.L):
            if m[j, j] > 0:
                p = rho / (rho + beta[j] * (1 - rho))
                mbar[j, j] = m[j, j] - self.rng.binomial(int(m[j, j]), p)
        conc = self.gamma / self.L + mbar.sum(0)
        beta_new = self.rng.gamma(np.maximum(conc, 1e-6))
        return beta_new / beta_new.sum()

    # -- driver ---------------------------------------------------------------
    def fit(self, X: np.ndarray, min_prob: float = 0.3) -> StickyHDPHMMResult:
        T = len(X)
        beta = np.full(self.L, 1.0 / self.L)
        z = _kmeans_init(X, self.L, seed=int(self.rng.integers(1 << 31)))
        boundary_counts = np.zeros(T)
        n_kept = 0
        pi = np.full((self.L, self.L), 1.0 / self.L)
        for sweep in range(self.sweeps + self.burnin):
            mu, var = self._sample_emissions(X, z)
            ll = self._log_lik(X, mu, var)
            z = self._ffbs(ll, pi, beta)
            pi, n = self._sample_transitions(z, beta)
            beta = self._sample_beta(n, beta)
            if sweep >= self.burnin:
                boundary_counts[1:] += (z[1:] != z[:-1])
                n_kept += 1
        prob = boundary_counts / max(1, n_kept)
        boundaries = self._extract_boundaries(prob, min_prob)
        return StickyHDPHMMResult(
            boundary_prob=prob,
            boundaries=boundaries,
            last_path=z,
            n_active_states=len(np.unique(z)),
        )

    @staticmethod
    def _extract_boundaries(prob: np.ndarray, min_prob: float, radius: int = 2) -> list[dict]:
        """Cluster posterior boundary mass into events: local windows around
        peaks; location = mass-weighted mean, dispersion = weighted std."""
        out = []
        p = prob.copy()
        while True:
            t = int(p.argmax())
            if p[t] < min_prob:
                break
            lo, hi = max(0, t - radius), min(len(p), t + radius + 1)
            w = p[lo:hi]
            idx = np.arange(lo, hi)
            mass = w.sum()
            mean = float((idx * w).sum() / mass)
            std = float(np.sqrt(((idx - mean) ** 2 * w).sum() / mass))
            out.append({"frame": mean, "prob": float(min(1.0, mass)), "std_frames": std})
            p[lo:hi] = 0.0
        out.sort(key=lambda b: b["frame"])
        return out
