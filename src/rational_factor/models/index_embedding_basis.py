import torch
from .basis_functions import Basis, NonnegativeBasis
from .density_model import ConditionalDensityModel
from .parameters import Parameters


class NormalziedIndexEmbeddingBasis(torch.nn.Module, Basis, NonnegativeBasis):
    """Basis of conditional densities indexed by learned embeddings.

    Each basis function is ``φ_i(y) = p(y | e_i)``, where ``p`` is a shared
    :class:`~rational_factor.models.density_model.ConditionalDensityModel` and
    ``e_i`` is a learned index embedding. The conditional model (flow, spline
    density, etc.) is supplied by the caller; this class only owns the
    embedding table and the index-broadcast evaluation.
    """

    def __init__(
        self,
        model: ConditionalDensityModel,
        n_basis: int,
        embedding: torch.nn.Embedding | None = None,
        coeffs: Parameters = None,
        *,
        embedding_init_std: float = 0.05,
    ):
        if n_basis < 1:
            raise ValueError("n_basis must be at least 1")
        if embedding is None:
            embedding = torch.nn.Embedding(n_basis, model.conditioner_dim)
            torch.nn.init.normal_(embedding.weight, mean=0.0, std=embedding_init_std)
        if embedding.num_embeddings != n_basis:
            raise ValueError(
                f"embedding num_embeddings {embedding.num_embeddings} "
                f"must match n_basis {n_basis}"
            )
        if embedding.embedding_dim != model.conditioner_dim:
            raise ValueError(
                f"embedding dim {embedding.embedding_dim} must match "
                f"model conditioner_dim {model.conditioner_dim}"
            )

        torch.nn.Module.__init__(self)
        self.model = model
        self.index_embedding = embedding
        self.owner = self
        Basis.__init__(
            self,
            dim=model.dim,
            batch_size=1,
            n_basis=n_basis,
            params=(),
            coeffs=coeffs,
        )

    def dtype_device(self):
        weight = self.index_embedding.weight
        return weight.dtype, weight.device

    def _index_conditioners(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        idx = torch.arange(self._n_basis, device=device)
        return self.index_embedding(idx).to(dtype=dtype)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Evaluate ``φ_i(y) = p(y | e_i)``, shape ``(n_data, n_basis)``."""
        dtype, device = self.dtype_device()
        y = torch.as_tensor(y, dtype=dtype, device=device)
        if y.ndim == 1:
            y = y.unsqueeze(-1) if self._dim == 1 else y.unsqueeze(0)
        if y.ndim != 2 or y.shape[1] != self._dim:
            raise ValueError(
                f"y must have shape (n_data, {self._dim}), got {tuple(y.shape)}"
            )

        n_data, m = y.shape[0], self._n_basis
        cond = self._index_conditioners(y.dtype, y.device)
        y_rep = y.unsqueeze(1).expand(-1, m, -1).reshape(n_data * m, self._dim)
        c_rep = cond.unsqueeze(0).expand(n_data, -1, -1).reshape(n_data * m, -1)
        dens = self.model(y_rep, conditioner=c_rep).reshape(n_data, m)
        return dens * self.coeffs().to(dtype=dens.dtype, device=dens.device)

    def normalized(self):
        return True

    def Omega1(self, lows: torch.Tensor = None, highs: torch.Tensor = None):
        if lows is not None or highs is not None:
            raise NotImplementedError("Restricted-domain moments are not implemented")
        dtype, device = self.dtype_device()
        return torch.ones(self._batch_size, self._n_basis, dtype=dtype, device=device)