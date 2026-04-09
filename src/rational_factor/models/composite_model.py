import torch
from .domain_transformation import DomainTF
from .density_model import DensityModel, ConditionalDensityModel
from collections.abc import Sequence


class CompositeDensityModel(DensityModel):
    def __init__(self, domain_tfs: DomainTF | Sequence[DomainTF], density_model: DensityModel):
        domain_tfs = _as_transform_sequence(domain_tfs)
        _validate_transform_sequence(domain_tfs, density_model.dim)
        super().__init__(density_model.dim)
        self.domain_tfs = torch.nn.ModuleList(domain_tfs)
        self.density_model = density_model
    
    def log_density(self, x: torch.Tensor):
        z_i = x
        total_ladj = x.new_zeros(x.shape[0])
        for tf in self.domain_tfs:
            z_i, ladj = tf(z_i)
            total_ladj = total_ladj + ladj
        return self._clip_log_density(self.density_model.log_density(z_i) + total_ladj)
    
    def valid(self):
        return self.density_model.valid()
    
    def sample(self, n_samples: int):
        z_samples = self.density_model.sample(n_samples)
        x = z_samples
        for tf in reversed(self.domain_tfs):
            x, _ = tf.inverse(x)
        return x

class CompositeConditionalModel(ConditionalDensityModel):
    def __init__(self, domain_tfs: DomainTF | Sequence[DomainTF], conditional_density_model: ConditionalDensityModel):
        domain_tfs = _as_transform_sequence(domain_tfs)
        _validate_transform_sequence(domain_tfs, conditional_density_model.dim)
        super().__init__(conditional_density_model.dim, conditional_density_model.conditioner_dim)
        self.domain_tfs = torch.nn.ModuleList(domain_tfs)
        self.conditional_density_model = conditional_density_model

    def log_density(self, xp : torch.Tensor, *, conditioner: torch.Tensor):
        z = conditioner
        zp = xp
        total_ladj = xp.new_zeros(xp.shape[0])
        for tf in self.domain_tfs:
            z, _ = tf(z)
            zp, ladj = tf(zp)
            total_ladj = total_ladj + ladj
        return self._clip_log_density(self.conditional_density_model.log_density(zp, conditioner=z) + total_ladj)
    
    def valid(self):
        return self.conditional_density_model.valid()
    
    def sample(self, conditioner: torch.Tensor):
        z = conditioner
        for tf in self.domain_tfs:
            z, _ = tf(z)
        zp = self.conditional_density_model.sample(z)
        xpi = zp
        for tf in reversed(self.domain_tfs):
            xpi, _ = tf.inverse(xpi)
        return xpi

def _as_transform_sequence(domain_tfs: DomainTF | Sequence[DomainTF]) -> list[DomainTF]:
    if isinstance(domain_tfs, DomainTF):
        return [domain_tfs]
    return list(domain_tfs)


def _validate_transform_sequence(domain_tfs: list[DomainTF], dim: int):
    assert len(domain_tfs) > 0, "At least one domain transformation must be provided"
    for tf in domain_tfs:
        assert tf.dim == dim, "Domain TF and density model must have the same dimension"
