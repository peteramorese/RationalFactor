import torch

class LinearRFF(torch.nn.Module):
    def __init__(self, d : int, n_phi : int, n_psi : int):
        super().__init__()

        self.d = d
        self.n_phi = n_phi
        self.n_psi = n_psi

        self.phi_params = None
        self.psi_params = None

        self.F = torch.nn.Parameter(torch.randn(n_phi)) # f
        self.G = torch.nn.Parameter(torch.randn(n_phi)) # g
        self.H0 = torch.nn.Parameter(torch.randn(n_phi)) # h0

        self.separable_params = None
    
    def _register_params_separable(self, n_params_per_basis : int):
        if self.separable_params is not None:
            raise ValueError("Params already registered")
        self.phi_params = torch.nn.Parameter(torch.randn(self.d, self.n_phi, n_params_per_basis))
        self.psi_params = torch.nn.Parameter(torch.randn(self.d, self.n_psi, n_params_per_basis))
        self.separable_params = True

    def _register_params_similar(self, n_params_per_basis : int):
        if self.separable_params is not None:
            raise ValueError("Params already registered")
        self.phi_params = torch.nn.Parameter(torch.randn(self.n_phi, n_params_per_basis))
        self.psi_params = torch.nn.Parameter(torch.randn(self.n_psi, n_params_per_basis))
        self.separable_params = False