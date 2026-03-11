import torch
from .domain_transformation import DomainTF
from .density_model import DensityModel, ConditionalDensityModel

class CompositeDensityModel(DensityModel):
    def __init__(self, domain_tf : DomainTF, density_model : DensityModel):
        assert domain_tf.dim == density_model.dim, "Domain TF and density model must have the same dimension"
        super().__init__(density_model.dim)
        self.domain_tf = domain_tf
        self.density_model = density_model
    
    def log_density(self, x : torch.Tensor):
        z, ladj = self.domain_tf(x)
        return self.density_model.log_density(z) + ladj
    
    def valid(self):
        return self.density_model.valid()
    
    def sample(self, n_samples : int):
        z_samples = self.density_model.sample(n_samples)
        x, _ = self.domain_tf.inverse(z_samples)
        return x

class CompositeConditionalModel(ConditionalDensityModel):
    def __init__(self, domain_tf : DomainTF, conditional_density_model : ConditionalDensityModel):
        assert domain_tf.dim == conditional_density_model.dim, "Domain TF and conditional density model must have the same dimension"
        super().__init__(conditional_density_model.dim)
        self.domain_tf = domain_tf
        self.conditional_density_model = conditional_density_model

    def log_density(self, x : torch.Tensor, xp : torch.Tensor):
        z, _ = self.domain_tf(x)
        #print("z: ", z)
        zp, ladj = self.domain_tf(xp)
        #print("zp: ", zp)
        return self.conditional_density_model.log_density(z, zp) + ladj
    
    def valid(self):
        return self.conditional_density_model.valid()
    
    def sample(self, x : torch.Tensor, n_samples : int):
        z, _ = self.domain_tf(x)
        zp = self.conditional_density_model.sample(z, n_samples)
        xp, _ = self.domain_tf.inverse(zp)
        return xp