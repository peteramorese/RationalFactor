import torch
from rational_factor.models.domain_transformation import DomainTF
from nflows.transforms import CompositeTransform, RandomPermutation, MaskedPiecewiseRationalQuadraticAutoregressiveTransform

class ConditionalDomainTF(torch.nn.Module):
    def __init__(self, dim: int, conditioner_dim: int):
        super().__init__()
        self.dim = dim
        self.conditioner_dim = conditioner_dim

    def forward(self, x : torch.Tensor, conditioner : torch.Tensor):
        raise NotImplementedError("Forward TF is not implemented")

    def inverse(self, z : torch.Tensor, conditioner : torch.Tensor):
        raise NotImplementedError("Inverse TF is not implemented")

class ConditionalMaskedRQSNFTF(ConditionalDomainTF):
    def __init__(self, dim: int, conditioner_dim: int, n_layers: int = 5, hidden_features: int = 128, trainable: bool = True, num_bins: int = 8, tails: str = "linear", tail_bound: float = 3.0):
        super().__init__(dim, conditioner_dim)

        transforms = []
        for _ in range(n_layers):
            transforms.append(RandomPermutation(features=dim))
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=dim,
                    hidden_features=hidden_features,
                    context_features=conditioner_dim,
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

    def forward(self, x: torch.Tensor, conditioner : torch.Tensor):
        assert x.shape[1] == self.dim, "x must have shape (n_data, dim)"
        assert conditioner.shape[1] == self.conditioner_dim, "conditioner must have shape (n_data, conditioner_dim)"
        return self.T(x, context=conditioner)

    def inverse(self, z: torch.Tensor, conditioner : torch.Tensor):
        assert z.shape[1] == self.dim, "z must have shape (n_data, dim)"
        assert conditioner.shape[1] == self.conditioner_dim, "conditioner must have shape (n_data, conditioner_dim)"
        return self.T.inverse(z, context=conditioner)

