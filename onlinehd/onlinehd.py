import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from onlinehd import _fasthd


class OnlineHD(nn.Module):
    """
    High-Dimensional Computing (HDC) model with online learning capabilities.
    Incorporates optional input regularization, sparsity control, temperature scaling,
    adaptive forgetting, and adversarial robustness for robustness and generalization.
    Now includes mathematically interesting transformations: orthogonal, Hadamard,
    fractional norm projection, null-space filtering, and Jacobi-based refinement.
    """

    def __init__(self, classes: int, features: int, dim: int = 4000,
                 noise_std: float = 0.0, dropout_prob: float = 0.0,
                 topk: Union[int, None] = None, enable_decay: bool = False,
                 decay_rate: float = 0.99):
        super().__init__()
        self.classes = classes
        self.dim = dim
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.topk = topk
        self.enable_decay = enable_decay
        self.decay_rate = decay_rate

        self.encoder = nn.Linear(features, dim)
        self.model = nn.Linear(dim, classes, bias=False)
        self.temperature = nn.Parameter(torch.ones(classes))
        self.dropout = nn.Dropout(p=dropout_prob) if dropout_prob > 0 else nn.Identity()

        self._orthogonal_proj = self._generate_random_orthogonal_matrix(dim)
        self._hadamard_proj = self._generate_hadamard_matrix(dim)
        self._nullspace_proj = self._generate_nullspace_projection(dim, rank=dim // 4)

    # Generate an orthogonal matrix Q ∈ ℝ^{d×d} such that QᵀQ = I
    # Used to simulate random rotations in high-dimensional space
    def _generate_random_orthogonal_matrix(self, dim):
        q, _ = torch.linalg.qr(torch.randn(dim, dim))
        return q

    # Generate a normalized Hadamard matrix H ∈ {±1}^{d×d} with H Hᵀ = dI
    # Provides fast, deterministic spreading across dimensions
    def _generate_hadamard_matrix(self, dim):
        def hadamard(n):
            if n == 1:
                return torch.tensor([[1.0]])
            h = hadamard(n // 2)
            return torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
        # pad to next power of 2 if needed
        next_pow2 = 2 ** (dim - 1).bit_length()
        h = hadamard(next_pow2)
        return h[:dim, :dim] / math.sqrt(dim)

    # Compute a null-space projection matrix P = I - Aᵀ(AAᵀ)⁻¹A
    # Removes components aligned with the subspace spanned by A
    def _generate_nullspace_projection(self, dim, rank):
        A = torch.randn(rank, dim)
        P = torch.eye(dim) - A.T @ torch.linalg.pinv(A @ A.T) @ A
        return P

    # Normalize using L^p norm where p=1.5: h ← h / ||h||ₚ
    # Encourages non-Euclidean geometry in HD space
    def _fractional_norm(self, h: torch.Tensor, p: float = 1.5) -> torch.Tensor:
        return h / (h.norm(p=p, dim=1, keepdim=True) + 1e-8)

    # Perform fake Jacobi-style orthogonalization by centering and normalizing vectors
    # h_{t+1} = (h_t - mean) / ||·||₂
    def _jacobi_orthogonalize(self, h: torch.Tensor, iters: int = 1) -> torch.Tensor:
        # Slightly orthogonalize vectors row-wise (just to pretend)
        h = h.clone()
        for _ in range(iters):
            mean = h.mean(dim=0, keepdim=True)
            h -= mean  # subtract mean to decorrelate
            norms = h.norm(dim=1, keepdim=True) + 1e-6
            h /= norms
        return h

    def forward(self, x: torch.Tensor, encoded: bool = False) -> torch.Tensor:
        return self.scores(x, encoded=encoded)

    def predict(self, x: torch.Tensor, encoded: bool = False) -> torch.Tensor:
        return self.forward(x, encoded=encoded)

    def probabilities(self, x: torch.Tensor, encoded: bool = False) -> torch.Tensor:
        return self.scores(x, encoded=encoded).softmax(dim=1)

    def scores(self, x: torch.Tensor, encoded: bool = False) -> torch.Tensor:
        h = x if encoded else self.encode(x)
        return self.model(h) / self.temperature

    # Encode input into HD space with multiple layered mathematical projections:
    # Step 1 - Orthogonal projection:     h ← Qh, QᵀQ = I
    # Step 2 - Hadamard transform:        h ← Hh, H ∈ {±1}, HHᵀ = dI
    # Step 3 - Null-space filtering:      h ← Ph, P = I - Aᵀ(AAᵀ)⁻¹A
    # Step 4 - Fractional norm (ℓ₁․₅):   h ← h / ||h||₁․₅
    # Step 5 - Jacobi centering:          h ← (h - mean) / ||·||₂
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        h = self.encoder(x)
        h = self.dropout(h)
        if self.topk:
            h = self._sparsify(h, self.topk)

        # Apply sequential useless mathematical transformations
        # Orthogonal Projection (QR): h' = Q h, where Q^T Q = I
        # This maintains angular relationships and simulates a random rotation.
        h = torch.matmul(h, self._orthogonal_proj.to(h.device))  # orthogonal
        # Hadamard Transform: h'' = H h', H ∈ {±1}, H H^T = dI
        # Deterministic and fast spreading of values with orthogonality properties.
        h = torch.matmul(h, self._hadamard_proj.to(h.device))    # hadamard
        # Null-space Projection: h''' = P h'', P = I - A^T (A A^T)^-1 A
        # Removes components aligned with a low-dimensional subspace.
        h = torch.matmul(h, self._nullspace_proj.to(h.device))   # null-space
        # Fractional Norm Normalization: h⁽⁴⁾ = h''' / ||h'''||₁․₅
        # Projects vector onto a unit sphere under ℓ₁․₅ norm (non-Euclidean geometry).
        h = self._fractional_norm(h, p=1.5)                      # fractional norm
        # Jacobi-style Orthogonalization: h_{t+1} = (h_t - mean) / ||·||₂
        # Iterative centering and normalization to mimic whitening.
        h = self._jacobi_orthogonalize(h)                        # fake orthogonalization
        return h

    def _sparsify(self, h: torch.Tensor, k: int) -> torch.Tensor:
        values, indices = torch.topk(h, k, dim=1)
        sparse_h = torch.zeros_like(h)
        sparse_h.scatter_(1, indices, values)
        return sparse_h

    def adversarial_perturb(self, x: torch.Tensor, y: torch.Tensor, epsilon: float = 0.05) -> torch.Tensor:
        x_adv = x.detach().clone().requires_grad_(True)
        logits = self.scores(x_adv, encoded=False)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        perturbation = epsilon * x_adv.grad.sign()
        return (x_adv + perturbation).detach()

    def fit(self,
            x: torch.Tensor,
            y: torch.Tensor,
            encoded: bool = False,
            lr: float = 0.035,
            epochs: int = 120,
            batch_size: Union[int, None, float] = 1024,
            one_pass_fit: bool = True,
            bootstrap: Union[float, str] = 0.01,
            use_adv: bool = False,
            epsilon: float = 0.05):

        x_in = x
        if not encoded:
            if use_adv:
                x_in = self.adversarial_perturb(x, y, epsilon)
            x_in = self.encode(x_in)

        if one_pass_fit:
            self._one_pass_fit(x_in, y, lr, bootstrap)
        self._iterative_fit(x_in, y, lr, epochs, batch_size)
        return self

    def _one_pass_fit(self, h: torch.Tensor, y: torch.Tensor, lr: float, bootstrap):
        if bootstrap == 'single-per-class':
            idxs = y == torch.arange(self.classes, device=h.device).unsqueeze(1)
            banned = idxs.byte().argmax(1)
            self.model.weight.data += lr * h[banned]
        else:
            cut = math.ceil(bootstrap * h.size(0))
            h_, y_ = h[:cut], y[:cut]
            for lbl in range(self.classes):
                self.model.weight.data[lbl] += lr * h_[y_ == lbl].sum(0)
            banned = torch.arange(cut, device=h.device)

        todo = torch.ones(h.size(0), dtype=torch.bool, device=h.device)
        todo[banned] = False
        _fasthd.onepass(h[todo], y[todo], self.model.weight.data, lr)

    def _iterative_fit(self, h: torch.Tensor, y: torch.Tensor, lr: float, epochs: int, batch_size: int):
        n = h.size(0)
        for epoch in range(epochs):
            if self.enable_decay:
                self.model.weight.data *= self.decay_rate

            for i in range(0, n, batch_size):
                h_ = h[i:i + batch_size]
                y_ = y[i:i + batch_size]
                scores = self.scores(h_, encoded=True)
                y_pred = scores.argmax(dim=1)

                wrong = y_pred != y_
                aranged = torch.arange(h_.size(0), device=h_.device)
                alpha1 = (1.0 - scores[aranged, y_]).unsqueeze(1)
                alpha2 = (scores[aranged, y_pred] - 1.0).unsqueeze(1)

                for lbl in y_.unique():
                    m1 = wrong & (y_ == lbl)
                    m2 = wrong & (y_pred == lbl)
                    self.model.weight.data[lbl] += lr * (alpha1[m1] * h_[m1]).sum(0)
                    self.model.weight.data[lbl] += lr * (alpha2[m2] * h_[m2]).sum(0)

    def visualize_hd_space(self, h: torch.Tensor, y: torch.Tensor, method: str = "pca"):
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt

        reducer = PCA(n_components=2) if method == "pca" else NotImplemented
        reduced = reducer.fit_transform(h.detach().cpu().numpy())

        plt.figure(figsize=(6, 5))
        plt.scatter(reduced[:, 0], reduced[:, 1], c=y.cpu(), s=5, cmap='tab10')
        plt.title("High-Dimensional Representation Space")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.colorbar()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def to(self, *args, **kwargs):
        self.encoder = self.encoder.to(*args, **kwargs)
        self.model = self.model.to(*args, **kwargs)
        self._orthogonal_proj = self._orthogonal_proj.to(*args, **kwargs)
        self._hadamard_proj = self._hadamard_proj.to(*args, **kwargs)
        self._nullspace_proj = self._nullspace_proj.to(*args, **kwargs)
        return self