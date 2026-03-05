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
            x : torch.Tensor corresponding latent state
            ladj : torch.Tensor log absolute determinant of the Jacobian of the transformation
        """
        raise NotImplementedError("Forward TF is not implemented")

    def inverse(self, x : torch.Tensor):
        '''
        Inverse transformation of the domain (from the latent space to the physical space)

        Returns:
            z : torch.Tensor corresponding physical state
            ladj : torch.Tensor log absolute determinant of the Jacobian of the transformation
        '''
        raise NotImplementedError("Inverse TF is not implemented")


class MaskedAffineNFTF(DomainTF):
    def __init__(self, dim : int, n_layers : int = 5, hidden_features : int = 128):
        super().__init__(dim)

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
    
    def forward(self, x : torch.Tensor):
        assert x.shape[1] == self.dim, "x must have shape (n_data, dim)"
        return self.T(x)

    def inverse(self, z : torch.Tensor):
        assert z.shape[1] == self.dim, "z must have shape (n_data, dim)"
        return self.T.inverse(z)
