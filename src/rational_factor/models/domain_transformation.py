import torch
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import RandomPermutation
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform


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

    def __init__(self, dim : int, trainable : bool = True):
        super().__init__(dim)
        # (dim, 2): column 0 = location, column 1 = raw scale (softplus applied in forward)
        if trainable:
            self.params = torch.nn.Parameter(torch.randn(dim, 2))
        else:
            self.register_buffer("params", torch.randn(dim, 2))

    def forward(self, x : torch.Tensor):
        loc = self.params[:, 0]   # (dim,)
        scale = torch.softplus(self.params[:, 1])  # (dim,)
        sqrt_2 = torch.sqrt(x.new_tensor(2.0))
        u = (x - loc) / (scale * sqrt_2)
        z = 0.5 * (1.0 + torch.special.erf(u))
        ladj = (-torch.log(scale) - 0.5 * torch.log(2 * torch.pi) - u ** 2).sum(dim=-1)
        return z, ladj

    def inverse(self, z : torch.Tensor):
        loc = self.params[:, 0]
        scale = torch.softplus(self.params[:, 1])
        sqrt_2 = torch.sqrt(z.new_tensor(2.0))
        u = torch.special.erfinv(2.0 * z.clamp(1e-6, 1.0 - 1e-6) - 1.0)
        x = loc + scale * sqrt_2 * u
        ladj = (torch.log(scale) + 0.5 * (torch.log(2 * torch.pi) + u ** 2)).sum(dim=-1) 
        return x, ladj


class MaskedAffineNFTF(DomainTF):
    def __init__(self, dim : int, n_layers : int = 5, hidden_features : int = 128, trainable : bool = True):
        super().__init__(dim)

        if trainable:
            self.params = torch.nn.Parameter(torch.randn(dim, 2))

            transforms = []
            for _ in range(n_layers):
                transforms.append(RandomPermutation(features=dim))
                transforms.append(
                    MaskedAffineAutoregressiveTransform(
                        features=dim,
                        hidden_features=hidden_features,
                        num_blocks=2,
                        use_residual_blocks=True,
                        random_mask=False,
                        activation=torch.relu,
                        dropout_probability=0.0,
                        use_batch_norm=False,
                    )
                )

            self.T = CompositeTransform(transforms)

        else:
            raise NotImplementedError("Non-trainable MaskedAffineNFTF is not implemented")
    
    def forward(self, x : torch.Tensor):
        assert x.shape[1] == self.dim, "x must have shape (n_data, dim)"
        return self.T(x)

    def inverse(self, z : torch.Tensor):
        assert z.shape[1] == self.dim, "z must have shape (n_data, dim)"
        return self.T.inverse(z)
