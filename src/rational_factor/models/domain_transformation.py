import copy

import torch
from nflows.transforms.base import Transform, CompositeTransform
from nflows.transforms.permutations import RandomPermutation
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform, MaskedPiecewiseRationalQuadraticAutoregressiveTransform


class DomainTF(Transform):
    """Domain transform with nflows ``Transform`` API (``context`` optional).

    Unconditional maps (e.g. ``IdentityTF``, ``ErfSeparableTF``) ignore ``context``.
    Flow maps may set ``context_features`` and condition on ``context``.
    """

    def __init__(self, dim: int, context_features: int | None = None):
        super().__init__()
        self.dim = dim
        self.context_features = context_features

    def marginal(self, marginal_dims: tuple[int, ...]):
        raise NotImplementedError("Marginal is not implemented")


class IdentityTF(DomainTF):
    def __init__(self, dim: int):
        super().__init__(dim)

    def forward(self, inputs: torch.Tensor, context: torch.Tensor | None = None):
        return inputs, inputs.new_zeros(inputs.shape[0])

    def inverse(self, inputs: torch.Tensor, context: torch.Tensor | None = None):
        return inputs, inputs.new_zeros(inputs.shape[0])

    def marginal(self, marginal_dims: tuple[int, ...]):
        marginal_dims = tuple(marginal_dims)
        assert all(0 <= i < self.dim for i in marginal_dims), "marginal_dims must be in [0, dim)"
        return IdentityTF(len(marginal_dims))


class StackedTF(DomainTF):
    def __init__(self, tfs: list[DomainTF]):
        stacked_dim = sum(tf.dim for tf in tfs)
        ctxs = {
            tf.context_features
            for tf in tfs
            if tf.context_features is not None
        }
        if len(ctxs) > 1:
            raise ValueError(
                "StackedTF children must share the same context_features "
                f"(got {sorted(ctxs)})"
            )
        context_features = next(iter(ctxs)) if ctxs else None
        super().__init__(stacked_dim, context_features=context_features)
        self.tfs = torch.nn.ModuleList(tfs)

    def forward(self, inputs: torch.Tensor, context: torch.Tensor | None = None):
        assert inputs.shape[1] == self.dim, "inputs must have shape (n_data, dim)"

        z_parts = []
        ladj = inputs.new_zeros(inputs.shape[0])
        cursor = 0

        for tf in self.tfs:
            x_part = inputs[:, cursor : cursor + tf.dim]
            z_part, ladj_part = tf.forward(x_part, context=context)
            z_parts.append(z_part)
            ladj = ladj + ladj_part
            cursor += tf.dim

        z = torch.cat(z_parts, dim=1)
        return z, ladj

    def inverse(self, inputs: torch.Tensor, context: torch.Tensor | None = None):
        assert inputs.shape[1] == self.dim, "inputs must have shape (n_data, dim)"

        x_parts = []
        ladj = inputs.new_zeros(inputs.shape[0])
        cursor = 0

        for tf in self.tfs:
            z_part = inputs[:, cursor : cursor + tf.dim]
            x_part, ladj_part = tf.inverse(z_part, context=context)
            x_parts.append(x_part)
            ladj = ladj + ladj_part
            cursor += tf.dim

        x = torch.cat(x_parts, dim=1)
        return x, ladj

    def marginal(self, marginal_dims: tuple[int, ...]):
        marginal_dims = tuple(marginal_dims)
        assert len(marginal_dims) == len(set(marginal_dims)), "marginal_dims must be unique"
        assert all(0 <= i < self.dim for i in marginal_dims), "marginal_dims must be in [0, dim)"

        owners: list[tuple[int, int]] = []
        for tf_idx, tf in enumerate(self.tfs):
            for local_i in range(tf.dim):
                owners.append((tf_idx, local_i))

        locals_per_tf: dict[int, list[int]] = {}
        last_tf = -1
        for g in marginal_dims:
            tf_idx, local_i = owners[g]
            if tf_idx < last_tf:
                raise ValueError(
                    "marginal_dims must not interleave stacked transforms; "
                    "kept dims must appear in block order"
                )
            last_tf = tf_idx
            locals_per_tf.setdefault(tf_idx, []).append(local_i)

        marg_tfs = [
            self.tfs[tf_idx].marginal(tuple(local_dims))
            for tf_idx, local_dims in locals_per_tf.items()
        ]
        if len(marg_tfs) == 1:
            return marg_tfs[0]
        return StackedTF(marg_tfs)


class ErfSeparableTF(DomainTF):
    """Maps x to z via a parameterized Gaussian CDF per dimension: z_d = Phi((x_d - loc_d) / scale_d)."""

    def __init__(self, dim: int, loc: torch.Tensor, scale: torch.Tensor, trainable: bool = True, min_scale: float = 1e-3, numerical_tolerance: float = 1e-20):
        super().__init__(dim)
        # (dim, 2): column 0 = location, column 1 = raw scale (softplus applied in forward)
        self.trainable = trainable
        self.min_scale = min_scale
        scale = torch.as_tensor(scale, dtype=loc.dtype, device=loc.device).clamp(min=min_scale)
        if trainable:
            scale_params = torch.sqrt(scale)
            self.params = torch.nn.Parameter(torch.hstack([loc.unsqueeze(1), scale_params.unsqueeze(1)]))
        else:
            self.register_buffer("params", torch.hstack([loc.unsqueeze(1), scale.unsqueeze(1)]))

        self.numerical_tolerance = numerical_tolerance

    @classmethod
    def copy_from_trainable(cls, other: "ErfSeparableTF"):
        return cls(
            other.dim,
            other.params[:, 0].detach().clone(),
            torch.square(other.params[:, 1]).detach().clone(),
            trainable=False,
            min_scale=getattr(other, "min_scale", 1e-3),
            numerical_tolerance=getattr(other, "numerical_tolerance", 1e-20),
        )

    @classmethod
    def from_data(cls, x_data: torch.Tensor, trainable: bool = True, min_scale: float = 1e-3):
        dim = x_data.shape[1]
        mean = x_data.mean(dim=0)
        std = x_data.std(dim=0).clamp_min(min_scale)
        return cls(dim, mean, std, trainable=trainable, min_scale=min_scale)

    def loc_scale(self):
        if self.trainable:
            loc = self.params[:, 0]   # (dim,)
            scale = torch.square(self.params[:, 1]) # (dim,)
        else:
            loc = self.params[:, 0]   # (dim,)
            scale = self.params[:, 1]  # (dim,)
        return loc, torch.clamp(scale, min=self.min_scale)

    def forward(self, inputs: torch.Tensor, context: torch.Tensor | None = None):
        loc, scale = self.loc_scale()
        sqrt_2 = torch.sqrt(inputs.new_tensor(2.0))
        u = (inputs - loc) / (scale * sqrt_2)
        z = 0.5 * (1.0 + torch.special.erf(u))
        # Keep outputs strictly inside (0, 1) for downstream unit-box maps (e.g. bounded RQS).
        z = z.clamp(self.numerical_tolerance, 1.0 - self.numerical_tolerance)
        ladj = (-torch.log(scale) - 0.5 * torch.log(inputs.new_tensor(2.0 * torch.pi)) - u ** 2).sum(dim=-1)
        return z, ladj

    def inverse(self, inputs: torch.Tensor, context: torch.Tensor | None = None):
        loc, scale = self.loc_scale()
        sqrt_2 = torch.sqrt(inputs.new_tensor(2.0))
        u = torch.special.erfinv(2.0 * inputs.clamp(self.numerical_tolerance, 1.0 - self.numerical_tolerance) - 1.0)
        x = loc + scale * sqrt_2 * u
        ladj = (torch.log(scale) + 0.5 * (torch.log(inputs.new_tensor(2.0 * torch.pi)) + u ** 2)).sum(dim=-1)
        return x, ladj

    def marginal(self, marginal_dims: tuple[int, ...]):
        marginal_dims = tuple(marginal_dims)
        assert all(0 <= i < self.dim for i in marginal_dims), "marginal_dims must be in [0, dim)"

        loc, scale = self.loc_scale()
        return ErfSeparableTF(
            dim=len(marginal_dims),
            loc=loc[list(marginal_dims)].detach().clone(),
            scale=scale[list(marginal_dims)].detach().clone(),
            trainable=False,
            min_scale=self.min_scale,
            numerical_tolerance=self.numerical_tolerance,
        )


class MaskedAffineNFTF(DomainTF):
    def __init__(
        self,
        dim: int,
        n_layers: int = 5,
        hidden_features: int = 128,
        trainable: bool = True,
        init_wo_warping: bool = False,
        context_features: int | None = None,
    ):
        super().__init__(dim, context_features=context_features)

        transforms = []
        for _ in range(n_layers):
            transforms.append(RandomPermutation(features=dim))
            maf = MaskedAffineAutoregressiveTransform(
                features=dim,
                hidden_features=hidden_features,
                context_features=context_features,
                num_blocks=2,
                use_residual_blocks=True,
                random_mask=False,
                activation=torch.tanh,
                dropout_probability=0.0,
                use_batch_norm=False,
            )
            if init_wo_warping:
                self._init_maf_wo_warping(maf)
            transforms.append(maf)

        self.T = CompositeTransform(transforms)

        if not trainable:
            raise NotImplementedError("Initializing as non trainable is not implemented")

    @staticmethod
    def _init_maf_wo_warping(maf):
        last_linear = None
        for m in maf.modules():
            if isinstance(m, torch.nn.Linear):
                last_linear = m

        if last_linear is None:
            raise RuntimeError("Could not find final Linear layer in MAF.")

        torch.nn.init.zeros_(last_linear.weight)
        torch.nn.init.zeros_(last_linear.bias)

    @classmethod
    def copy_from_trainable(cls, other: "MaskedAffineNFTF"):
        new_module = copy.deepcopy(other)
        for p in new_module.parameters():
            p.requires_grad_(False)
        return new_module

    def forward(self, inputs: torch.Tensor, context: torch.Tensor | None = None):
        assert inputs.shape[1] == self.dim, "inputs must have shape (n_data, dim)"
        return self.T(inputs, context=context)

    def inverse(self, inputs: torch.Tensor, context: torch.Tensor | None = None):
        assert inputs.shape[1] == self.dim, "inputs must have shape (n_data, dim)"
        return self.T.inverse(inputs, context=context)


class _ClampInputsTransform(Transform):
    """Clamp inputs into a closed interval; log-det contribution is zero.

    Absorbs floating-point drift before bounded spline layers that raise
    ``InputOutsideDomain`` when inputs leave ``[left, right]``.
    """

    def __init__(self, left: float = 0.0, right: float = 1.0, eps: float = 0.0):
        super().__init__()
        if not left < right:
            raise ValueError(f"Need left < right, got left={left}, right={right}")
        if eps < 0:
            raise ValueError(f"eps must be non-negative, got {eps}")
        if 2.0 * eps >= right - left:
            raise ValueError(f"eps={eps} too large for interval [{left}, {right}]")
        self.left = float(left + eps)
        self.right = float(right - eps)

    def forward(self, inputs, context=None):
        outputs = inputs.clamp(self.left, self.right)
        ladj = inputs.new_zeros(inputs.shape[0])
        return outputs, ladj

    def inverse(self, inputs, context=None):
        outputs = inputs.clamp(self.left, self.right)
        ladj = inputs.new_zeros(inputs.shape[0])
        return outputs, ladj


class MaskedRQSNFTF(DomainTF):
    def __init__(
        self,
        dim: int,
        n_layers: int = 5,
        hidden_features: int = 128,
        trainable: bool = True,
        num_bins: int = 8,
        tails: str | None = "linear",
        tail_bound: float = 3.0,
        context_features: int | None = None,
        domain_eps: float = 0.0,
    ):
        super().__init__(dim, context_features=context_features)
        self.tails = tails
        self.tail_bound = float(tail_bound)
        self.domain_eps = float(domain_eps)
        # Bounded RQS (tails=None) is only defined on the unit interval.
        self._bounded_domain = tails is None

        transforms = []
        for _ in range(n_layers):
            transforms.append(RandomPermutation(features=dim))
            if self._bounded_domain:
                # Clamp before each bounded RQS so cascade drift cannot raise
                # nflows.InputOutsideDomain.
                transforms.append(
                    _ClampInputsTransform(0.0, 1.0, eps=domain_eps)
                )
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=dim,
                    hidden_features=hidden_features,
                    context_features=context_features,
                    num_bins=num_bins,
                    tails=tails,
                    tail_bound=tail_bound,
                    num_blocks=2,
                    use_residual_blocks=True,
                    random_mask=False,
                    activation=torch.tanh,
                    dropout_probability=0.0,
                    use_batch_norm=False,
                )
            )

        self.T = CompositeTransform(transforms)

        if not trainable:
            for p in self.parameters():
                p.requires_grad_(False)

    def _clamp_unit_box(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self._bounded_domain:
            return inputs
        lo = self.domain_eps
        hi = 1.0 - self.domain_eps
        return inputs.clamp(lo, hi)

    @classmethod
    def copy_from_trainable(cls, other: "MaskedRQSNFTF"):
        new_module = copy.deepcopy(other)
        for p in new_module.parameters():
            p.requires_grad_(False)
        return new_module

    def forward(self, inputs: torch.Tensor, context: torch.Tensor | None = None):
        assert inputs.shape[1] == self.dim, "inputs must have shape (n_data, dim)"
        inputs = self._clamp_unit_box(inputs)
        outputs, ladj = self.T(inputs, context=context)
        return self._clamp_unit_box(outputs), ladj

    def inverse(self, inputs: torch.Tensor, context: torch.Tensor | None = None):
        assert inputs.shape[1] == self.dim, "inputs must have shape (n_data, dim)"
        inputs = self._clamp_unit_box(inputs)
        outputs, ladj = self.T.inverse(inputs, context=context)
        return self._clamp_unit_box(outputs), ladj


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int = 128,
        num_hidden_layers: int = 2,
        activation=torch.nn.Tanh,
        zero_init_last: bool = True,
    ):
        super().__init__()

        layers = []
        last = in_features
        for _ in range(num_hidden_layers):
            layers.append(torch.nn.Linear(last, hidden_features))
            layers.append(activation())
            last = hidden_features

        layers.append(torch.nn.Linear(last, out_features))
        self.net = torch.nn.Sequential(*layers)

        if zero_init_last:
            final = self.net[-1]
            torch.nn.init.zeros_(final.weight)
            torch.nn.init.zeros_(final.bias)

    def forward(self, x):
        return self.net(x)


class AdditiveCouplingTransform(Transform):
    """
    NICE-style additive coupling layer.

    y_masked = x_masked
    y_free   = x_free + t_theta(x_masked)

    Exact inverse:
    x_free = y_free - t_theta(y_masked)

    log |det J| = 0 exactly.
    """

    def __init__(
        self,
        features: int,
        mask: torch.Tensor,
        hidden_features: int = 128,
        num_hidden_layers: int = 2,
        activation=torch.nn.Tanh,
        zero_init: bool = True,
    ):
        super().__init__()
        assert mask.shape == (features,)
        assert mask.dtype in (torch.float32, torch.float64, torch.bool)

        self.features = features
        self.register_buffer("mask", mask.float())
        self.register_buffer("inv_mask", 1.0 - mask.float())

        self.shift_net = MLP(
            in_features=features,
            out_features=features,
            hidden_features=hidden_features,
            num_hidden_layers=num_hidden_layers,
            activation=activation,
            zero_init_last=zero_init,
        )

    def forward(self, inputs, context=None):
        assert inputs.shape[-1] == self.features

        x_id = inputs * self.mask
        shift = self.shift_net(x_id) * self.inv_mask

        outputs = inputs + shift
        ladj = inputs.new_zeros(inputs.shape[0])
        return outputs, ladj

    def inverse(self, inputs, context=None):
        assert inputs.shape[-1] == self.features

        y_id = inputs * self.mask
        shift = self.shift_net(y_id) * self.inv_mask

        outputs = inputs - shift
        ladj = inputs.new_zeros(inputs.shape[0])
        return outputs, ladj


class HouseholderTransform(Transform):
    """
    Orthogonal mixing using a product of Householder reflections.

    H(v) = I - 2 vv^T / (v^T v)

    Each reflection is orthogonal, so |det H| = 1.
    Therefore log |det J| = 0 exactly.

    A product of K reflections gives a learned orthogonal matrix.
    """

    def __init__(
        self,
        features: int,
        num_reflections: int = 4,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.features = features
        self.num_reflections = num_reflections
        self.eps = eps

        self.vectors = torch.nn.Parameter(
            torch.randn(num_reflections, features) / features**0.5
        )

    def _apply_reflection(self, x, v):
        # x: (batch, features)
        # v: (features,)
        denom = torch.sum(v * v).clamp_min(self.eps)
        projection = (x @ v)[:, None] * v[None, :] / denom
        return x - 2.0 * projection

    def forward(self, inputs, context=None):
        assert inputs.shape[-1] == self.features

        outputs = inputs
        for k in range(self.num_reflections):
            outputs = self._apply_reflection(outputs, self.vectors[k])

        ladj = inputs.new_zeros(inputs.shape[0])
        return outputs, ladj

    def inverse(self, inputs, context=None):
        assert inputs.shape[-1] == self.features

        # Each Householder reflection is self-inverse.
        # The inverse of the product applies them in reverse order.
        outputs = inputs
        for k in reversed(range(self.num_reflections)):
            outputs = self._apply_reflection(outputs, self.vectors[k])

        ladj = inputs.new_zeros(inputs.shape[0])
        return outputs, ladj


class VolumePreservingNFTF(DomainTF):
    """
    Expressive exact volume-preserving normalizing-flow-style domain transform.

    Architecture:
        additive coupling
        Householder orthogonal mixing
        additive coupling
        Householder orthogonal mixing
        ...

    Every layer has log |det J| = 0 exactly.
    """

    def __init__(
        self,
        dim: int,
        n_layers: int = 6,
        hidden_features: int = 128,
        num_hidden_layers: int = 2,
        num_householder_reflections: int = 4,
        trainable: bool = True,
        use_random_permutation: bool = False,
        zero_init: bool = True,
    ):
        super().__init__(dim)

        transforms = []

        base_mask = torch.arange(dim) % 2
        base_mask = base_mask.float()

        for layer_idx in range(n_layers):
            # Alternate masks so both halves get updated.
            if layer_idx % 2 == 0:
                mask = base_mask
            else:
                mask = 1.0 - base_mask

            transforms.append(
                AdditiveCouplingTransform(
                    features=dim,
                    mask=mask,
                    hidden_features=hidden_features,
                    num_hidden_layers=num_hidden_layers,
                    activation=torch.nn.Tanh,
                    zero_init=zero_init,
                )
            )

            # Orthogonal learned mixing.
            transforms.append(
                HouseholderTransform(
                    features=dim,
                    num_reflections=num_householder_reflections,
                )
            )

            # Optional fixed permutation. Also exact volume-preserving.
            if use_random_permutation:
                transforms.append(RandomPermutation(features=dim))

        self.T = CompositeTransform(transforms)

        if not trainable:
            for p in self.parameters():
                p.requires_grad_(False)

    @classmethod
    def copy_from_trainable(cls, other: "VolumePreservingNFTF"):
        new_module = copy.deepcopy(other)
        for p in new_module.parameters():
            p.requires_grad_(False)
        return new_module

    def forward(self, inputs: torch.Tensor, context: torch.Tensor | None = None):
        assert inputs.shape[1] == self.dim, "inputs must have shape (n_data, dim)"
        z, ladj = self.T(inputs, context=context)

        # Should be exactly zero except for dtype/device shape.
        return z, ladj

    def inverse(self, inputs: torch.Tensor, context: torch.Tensor | None = None):
        assert inputs.shape[1] == self.dim, "inputs must have shape (n_data, dim)"
        x, ladj = self.T.inverse(inputs, context=context)

        # Should be exactly zero except for dtype/device shape.
        return x, ladj
