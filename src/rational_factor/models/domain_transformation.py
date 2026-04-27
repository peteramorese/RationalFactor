import copy

import torch
from nflows.transforms.base import Transform, CompositeTransform
from nflows.transforms.permutations import RandomPermutation
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform, MaskedPiecewiseRationalQuadraticAutoregressiveTransform


class DomainTF(torch.nn.Module):
    def __init__(self, dim : int):
        super().__init__()

        self.dim = dim
    
    def forward(self, x : torch.Tensor):
        """
        Forward transformation of the domain (from the physical space to the latent space)

        Returns:
            z : torch.Tensor corresponding latent state
            ladj : torch.Tensor log absolute determinant of the Jacobian of the transformation
        """
        raise NotImplementedError("Forward TF is not implemented")

    def inverse(self, z : torch.Tensor):
        '''
        Inverse transformation of the domain (from the latent space to the physical space)

        Returns:
            x : torch.Tensor corresponding physical state
            ladj : torch.Tensor log absolute determinant of the Jacobian of the transformation
        '''
        raise NotImplementedError("Inverse TF is not implemented")
    
    def marginal(self, marginal_dims : tuple[int, ...]):
        raise NotImplementedError("Marginal is not implemented")


class IdentityTF(DomainTF):
    def __init__(self, dim : int):
        super().__init__(dim)

    def forward(self, x : torch.Tensor):
        return x, x.new_zeros(x.shape[0])

    def inverse(self, z : torch.Tensor):
        return z, z.new_zeros(z.shape[0])


class ErfSeparableTF(DomainTF):
    """Maps x to z via a parameterized Gaussian CDF per dimension: z_d = Phi((x_d - loc_d) / scale_d)."""

    def __init__(self, dim : int, loc : torch.Tensor, scale : torch.Tensor, trainable : bool = True, min_scale : float = 1e-3, numerical_tolerance : float = 1e-20):
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
    def copy_from_trainable(cls, other : 'ErfSeparableTF'):
        return cls(
            other.dim,
            other.params[:, 0].detach().clone(),
            torch.square(other.params[:, 1]).detach().clone(),
            trainable=False,
            min_scale=getattr(other, "min_scale", 1e-3),
            numerical_tolerance=getattr(other, "numerical_tolerance", 1e-20),
        )

    @classmethod
    def from_data(cls, x_data : torch.Tensor, trainable : bool = True, min_scale : float = 1e-3):
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

    def forward(self, x : torch.Tensor):
        loc, scale = self.loc_scale()
        sqrt_2 = torch.sqrt(x.new_tensor(2.0))
        u = (x - loc) / (scale * sqrt_2)
        z = 0.5 * (1.0 + torch.special.erf(u))
        ladj = (-torch.log(scale) - 0.5 * torch.log(x.new_tensor(2.0 * torch.pi)) - u ** 2).sum(dim=-1)
        return z, ladj

    def inverse(self, z : torch.Tensor):
        loc, scale = self.loc_scale()
        sqrt_2 = torch.sqrt(z.new_tensor(2.0))
        u = torch.special.erfinv(2.0 * z.clamp(self.numerical_tolerance, 1.0 - self.numerical_tolerance) - 1.0)
        x = loc + scale * sqrt_2 * u
        ladj = (torch.log(scale) + 0.5 * (torch.log(z.new_tensor(2.0 * torch.pi)) + u ** 2)).sum(dim=-1) 
        return x, ladj
    
    def marginal(self, marginal_dims : tuple[int, ...]):
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
    def __init__(self, dim : int, n_layers : int = 5, hidden_features : int = 128, trainable : bool = True, init_wo_warping : bool = False):
        super().__init__(dim)

        transforms = []
        for _ in range(n_layers):
            transforms.append(RandomPermutation(features=dim))
            maf = MaskedAffineAutoregressiveTransform(
                features=dim,
                hidden_features=hidden_features,
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
    def copy_from_trainable(cls, other : 'MaskedAffineNFTF'):
        new_module = copy.deepcopy(other)
        for p in new_module.parameters():
            p.requires_grad_(False)
        return new_module
    
    def forward(self, x : torch.Tensor):
        assert x.shape[1] == self.dim, "x must have shape (n_data, dim)"
        return self.T(x)

    def inverse(self, z : torch.Tensor):
        assert z.shape[1] == self.dim, "z must have shape (n_data, dim)"
        return self.T.inverse(z)


class MaskedRQSNFTF(DomainTF):
    def __init__(self, dim: int, n_layers: int = 5, hidden_features: int = 128, trainable: bool = True, num_bins: int = 8, tails: str = "linear", tail_bound: float = 3.0):
        super().__init__(dim)

        transforms = []
        for _ in range(n_layers):
            transforms.append(RandomPermutation(features=dim))
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=dim,
                    hidden_features=hidden_features,
                    context_features=None,
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

    @classmethod
    def copy_from_trainable(cls, other: "MaskedRQSNFTF"):
        new_module = copy.deepcopy(other)
        for p in new_module.parameters():
            p.requires_grad_(False)
        return new_module

    def forward(self, x: torch.Tensor):
        assert x.shape[1] == self.dim, "x must have shape (n_data, dim)"
        return self.T(x)

    def inverse(self, z: torch.Tensor):
        assert z.shape[1] == self.dim, "z must have shape (n_data, dim)"
        return self.T.inverse(z)


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

    def forward(self, x: torch.Tensor):
        assert x.shape[1] == self.dim, "x must have shape (n_data, dim)"
        z, ladj = self.T(x)

        # Should be exactly zero except for dtype/device shape.
        return z, ladj

    def inverse(self, z: torch.Tensor):
        assert z.shape[1] == self.dim, "z must have shape (n_data, dim)"
        x, ladj = self.T.inverse(z)

        # Should be exactly zero except for dtype/device shape.
        return x, ladj