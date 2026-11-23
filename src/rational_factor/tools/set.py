import torch


class Hyperrectangle:
    """Axis-aligned box in ``R^dim``, optionally storing a batch of boxes.

    ``low`` / ``high`` have shape ``(dim,)`` for a single box, or
    ``(batch, dim)`` for many boxes sharing the same ambient dimension.
    """

    def __init__(self, low: torch.Tensor, high: torch.Tensor):
        low = torch.as_tensor(low)
        high = torch.as_tensor(high)
        if low.shape != high.shape:
            raise ValueError(
                f"low and high must have the same shape, got "
                f"{tuple(low.shape)} vs {tuple(high.shape)}"
            )
        if low.ndim not in (1, 2):
            raise ValueError(
                f"low/high must have shape (dim,) or (batch, dim), got {tuple(low.shape)}"
            )
        if low.shape[-1] < 1:
            raise ValueError("hyperrectangle dimension must be positive")
        if bool((low > high).any()):
            raise ValueError("low must not exceed high in any coordinate")
        self.low = low
        self.high = high

    @property
    def dim(self) -> int:
        return int(self.low.shape[-1])

    @property
    def batch_shape(self) -> torch.Size:
        return self.low.shape[:-1]

    @property
    def is_batched(self) -> bool:
        return self.low.ndim == 2

    def __contains__(self, x: torch.Tensor):
        x = torch.as_tensor(x, dtype=self.low.dtype, device=self.low.device)
        if x.shape[-1] != self.dim:
            return False if not self.is_batched else self.low.new_zeros(self.batch_shape, dtype=torch.bool)
        inside = (x >= self.low) & (x <= self.high)
        out = torch.all(inside, dim=-1)
        return bool(out) if out.ndim == 0 else out

    def contains_any(self, x: torch.Tensor) -> torch.Tensor:
        """Boolean mask over leading batch dims: True iff ``x`` is in any box.

        ``x`` has shape ``(..., dim)``. For a single box the result matches
        coordinate-wise containment; for a batched hyperrectangle the box
        axis is reduced with a logical OR.
        """
        x = torch.as_tensor(x, dtype=self.low.dtype, device=self.low.device)
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"x last dim must be {self.dim}, got {tuple(x.shape)}"
            )
        if self.is_batched:
            # (..., 1, dim) vs (n_boxes, dim) -> (..., n_boxes, dim)
            inside = (x[..., None, :] >= self.low) & (x[..., None, :] <= self.high)
            return inside.all(dim=-1).any(dim=-1)
        inside = (x >= self.low) & (x <= self.high)
        return inside.all(dim=-1)
    
    def union(self, other: "Hyperrectangle") -> "Hyperrectangle":
        """Stack boxes along the batch axis into a single batched hyperrectangle."""
        if self.dim != other.dim:
            raise ValueError(
                f"union requires matching dim, got {self.dim} and {other.dim}"
            )
        low_a = self.low if self.is_batched else self.low.unsqueeze(0)
        high_a = self.high if self.is_batched else self.high.unsqueeze(0)
        low_b = other.low if other.is_batched else other.low.unsqueeze(0)
        high_b = other.high if other.is_batched else other.high.unsqueeze(0)
        return Hyperrectangle(
            torch.cat([low_a, low_b], dim=0),
            torch.cat([high_a, high_b], dim=0),
        )

    def __str__(self):
        return f"Hyperrectangle(low={self.low}, high={self.high})"

    def __repr__(self):
        return str(self)
