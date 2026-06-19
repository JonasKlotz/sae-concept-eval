"""
NMFBaseline: wraps the existing NMF class with the SAE encode/decode interface
so it can be dropped into the evaluation pipeline as a classical baseline.

CLIP embeddings are L2-normalised and therefore bipolar (can be negative).
We apply ReLU before passing to NMF so that only positive embedding components
are factorised — this mirrors what TopK SAEs do implicitly through the ReLU
in their encoder, and is the standard treatment for NMF on bipolar data.

Interface contract (matching SAE base class):
    encode(x) -> (codes, codes)   -- two identical tensors (no pre-sparsification stage)
    decode(z)  -> x_hat
    W_enc      -- D.T, shape (input_dim, nb_concepts), so W_enc.norm(dim=0) == D.norm(dim=1)
    nb_concepts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nmf import NMF
from .utils import matrix_nnls


class NMFBaseline(nn.Module):
    """
    NMF dictionary learning wrapped with the SAE encode/decode interface.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input embeddings (768 for CLIP ViT-L-14).
    nb_concepts : int
        Number of NMF components (dictionary atoms).
    solver : str
        NMF solver passed to the underlying NMF class ('hals', 'mu', 'pgd', 'anls').
    nnls_max_iter : int
        Max projected-gradient iterations for inference-time NNLS encoding.
    """

    def __init__(self, input_dim: int, nb_concepts: int, solver: str = "hals", nnls_max_iter: int = 100):
        super().__init__()
        self.nb_concepts = nb_concepts
        self.input_dim = input_dim
        self.solver = solver
        self.nnls_max_iter = nnls_max_iter
        # D: (nb_concepts, input_dim) — stored as buffer so state_dict works
        self.register_buffer("D", torch.zeros(nb_concepts, input_dim))
        self._fitted = False

    # ------------------------------------------------------------------
    # SAE interface
    # ------------------------------------------------------------------

    @property
    def W_enc(self):
        """Shape (input_dim, nb_concepts), matching SAE W_enc convention."""
        return self.D.T

    def encode(self, x: torch.Tensor):
        """
        Project x onto the NMF dictionary via NNLS (zero-initialised projected
        gradient descent — deterministic given fixed D).

        Parameters
        ----------
        x : torch.Tensor, shape (batch, input_dim)
            Raw embeddings (may be negative; ReLU is applied internally).

        Returns
        -------
        (codes, codes) : tuple of torch.Tensor, shape (batch, nb_concepts)
            Both elements are identical (no separate pre-sparsification stage).
        """
        assert self._fitted, "NMFBaseline must be fitted before encoding."
        a = F.relu(x.to(self.D.device))  # (batch, input_dim)
        z = self._nnls(a)               # (batch, nb_concepts)
        return z, z

    def _nnls(self, a: torch.Tensor) -> torch.Tensor:
        """
        Solve min ||Z @ D - A||^2 s.t. Z >= 0 via projected gradient descent.
        Zero-initialised so results are deterministic given the same D.
        """
        D_T = self.D.T                          # (input_dim, nb_concepts)
        Q = D_T.T @ D_T                         # (nb_concepts, nb_concepts)
        P = a @ D_T                             # (batch, nb_concepts)
        lr = 1.0 / (Q.square().sum().sqrt() + 1e-8)

        Z = torch.zeros_like(P)                 # zero init → deterministic
        for _ in range(self.nnls_max_iter):
            grad = Z @ Q - P
            Z = torch.clamp(Z - lr * grad, min=0)
        return Z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct embeddings from NMF codes."""
        return z @ self.D  # (batch, input_dim)

    def forward(self, x: torch.Tensor):
        pre_codes, codes = self.encode(x)
        x_hat = self.decode(codes)
        return pre_codes, codes, x_hat

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, embeddings: torch.Tensor, max_iter: int = 500):
        """
        Fit NMF on the full embedding matrix.

        Parameters
        ----------
        embeddings : torch.Tensor, shape (N, input_dim)
            Training embeddings (can be negative; ReLU applied internally).
        max_iter : int
            NMF optimisation iterations.
        """
        a = F.relu(embeddings.to(self.D.device).float())

        nmf = NMF(
            nb_concepts=self.nb_concepts,
            device=str(self.D.device),
            solver=self.solver,
        )
        _, D = nmf.fit(a, max_iter=max_iter)
        self.D = D.detach()
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def state_dict(self, **kwargs):
        sd = super().state_dict(**kwargs)
        sd["_fitted"] = self._fitted
        return sd

    def load_state_dict(self, state_dict, strict=True):
        self._fitted = state_dict.pop("_fitted", True)
        return super().load_state_dict(state_dict, strict=strict)
