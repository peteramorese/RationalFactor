import torch

from rational_factor.models.index_embedding_model import IndexEmbeddingTransform


class LatentReflectionSpaceSplitter(torch.nn.Module):
    """Split the domain by reflecting through a latent hyperplane.

    For each index embedding of ``tf``, maps ``y`` to ``z = T(y)``, reflects
    across ``z[reflection_axis] = 0``, and inverts to the partner
    ``y_partner = T^{-1}(z')``. Points with ``z[axis] >= 0`` belong to set 0;
    points with ``z[axis] < 0`` belong to set 1. ``ladj`` is the log absolute
    Jacobian of the partner map ``y → y_partner`` on both sides.
    """

    def __init__(self, tf: IndexEmbeddingTransform, reflection_axis: int = 0):
        super().__init__()
        features = tf.features
        if features is not None and not (0 <= reflection_axis < features):
            raise ValueError(
                f"reflection_axis {reflection_axis} out of range for "
                f"features={features}"
            )
        self.tf = tf
        self.reflection_axis = reflection_axis

    def partner(self, y: torch.Tensor):
        """Return partners and split metadata for a batch of points.

        Parameters
        ----------
        y :
            Input points of shape ``(batch, features)``.

        Returns
        -------
        y_partner : Tensor, shape ``(batch, n_mappings, features)``
        set_index : LongTensor, shape ``(batch, n_mappings)``
            ``0`` if ``z[axis] >= 0``, else ``1``.
        ladj : Tensor, shape ``(batch, n_mappings)``
            ``log |det Dτ|(y)`` for the partner map ``τ: y → y_partner``.
        d_to_boundary : Tensor, shape ``(batch, n_mappings)``
            Signed latent coordinate ``z[reflection_axis]``.
        """
        if y.ndim != 2:
            raise ValueError(
                f"y must have shape (batch, features), got {tuple(y.shape)}"
            )

        z, ladj_y = self.tf.forward(y)
        d_to_boundary = z[..., self.reflection_axis]
        set_index = (d_to_boundary < 0).to(dtype=torch.long)

        z_reflected = z.clone()
        z_reflected[..., self.reflection_axis] = -z_reflected[..., self.reflection_axis]
        y_partner, ladj_inv = self.tf.inverse(z_reflected)

        # Volume change along y --T→ z --reflect→ z' --T^{-1}→ y_partner.
        ladj = ladj_y + ladj_inv
        return y_partner, set_index, ladj, d_to_boundary
