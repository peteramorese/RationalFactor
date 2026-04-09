import copy

import torch
from nflows.transforms.base import CompositeTransform
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


class ErfSeparableTF(DomainTF):
    """Maps x to z via a parameterized Gaussian CDF per dimension: z_d = Phi((x_d - loc_d) / scale_d)."""

    def __init__(self, dim : int, loc : torch.Tensor, scale : torch.Tensor, trainable : bool = True, min_scale : float = 1e-3):
        super().__init__(dim)
        # (dim, 2): column 0 = location, column 1 = raw scale (softplus applied in forward)
        self.trainable = trainable
        if trainable:
            scale_params = torch.sqrt(scale)
            self.params = torch.nn.Parameter(torch.hstack([loc.unsqueeze(1), scale_params.unsqueeze(1)]))
            #self.params = torch.nn.Parameter(torch.hstack([loc.unsqueeze(1), scale.unsqueeze(1)]))
            #self.min_scale = min_scale
        else:
            self.register_buffer("params", torch.hstack([loc.unsqueeze(1), scale.unsqueeze(1)]))

    @classmethod
    def copy_from_trainable(cls, other : 'ErfSeparableTF'):
        return cls(other.dim, other.params[:, 0].detach().clone(), torch.square(other.params[:, 1]).detach().clone(), trainable=False)
        #return cls(other.dim, other.params[:, 0].detach().clone(), other.params[:, 1].detach().clone(), trainable=False)

    @classmethod
    def from_data(cls, x_data : torch.Tensor, trainable : bool = True):
        dim = x_data.shape[1]
        mean = x_data.mean(dim=0)
        std = x_data.std(dim=0)
        return cls(dim, mean, std, trainable)

    def _loc_scale(self):
        if self.trainable:
            loc = self.params[:, 0]   # (dim,)
            #scale = torch.nn.functional.softplus(self.params[:, 1]) + self.min_scale # (dim,)
            scale = torch.square(self.params[:, 1]) # (dim,)
            return loc, scale
        else:
            loc = self.params[:, 0]   # (dim,)
            scale = self.params[:, 1]  # (dim,)
            return loc, scale

    def forward(self, x : torch.Tensor):
        loc, scale = self._loc_scale()
        sqrt_2 = torch.sqrt(x.new_tensor(2.0))
        u = (x - loc) / (scale * sqrt_2)
        z = 0.5 * (1.0 + torch.special.erf(u))
        ladj = (-torch.log(scale) - 0.5 * torch.log(x.new_tensor(2.0 * torch.pi)) - u ** 2).sum(dim=-1)
        if torch.isnan(z).any():
            print("z is nan")
            print("x: ", x)
            print("loc: ", loc)
            print("scale: ", scale)
            print("u: ", u)
            raise ValueError("z is nan")
        return z, ladj

    def inverse(self, z : torch.Tensor):
        loc, scale = self._loc_scale()
        sqrt_2 = torch.sqrt(z.new_tensor(2.0))
        u = torch.special.erfinv(2.0 * z.clamp(1e-6, 1.0 - 1e-6) - 1.0)
        x = loc + scale * sqrt_2 * u
        ladj = (torch.log(scale) + 0.5 * (torch.log(z.new_tensor(2.0 * torch.pi)) + u ** 2)).sum(dim=-1) 
        return x, ladj


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