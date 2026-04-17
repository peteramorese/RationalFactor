
class GaussianBasis(SeparableBasis, NonnegativeBasis):
    def __init__(self, params_init : torch.Tensor, trainable : bool = True, min_std : float = 1e-5, block_size : int = None):
        assert params_init.shape[2] == 2, "params_init must have shape (d, n_basis, 2)"
        super().__init__(params_init, trainable)
        self.min_std = min_std
        self.block_size = block_size

    @classmethod
    def random_init(cls, d : int, n_basis : int, offsets : torch.Tensor = torch.zeros(2), min_std : float = 1e-5, variance: float = 1.0, device = None, block_size : int = None):
        if device is None:
            device = offsets.device
        else:
            offsets = offsets.to(device)
        offsets = offsets.repeat(d, n_basis, 1)
        return cls(torch.randn(d, n_basis, 2, device=device) * torch.sqrt(torch.tensor(variance, device=device)) + offsets, min_std=min_std, block_size=block_size)

    @classmethod
    def set_init(cls, d : int, n_basis : int, offsets : torch.Tensor = torch.zeros(2), min_std : float = 1e-5, block_size : int = None):
        offsets = offsets.repeat(d, n_basis, 1)
        return cls(offsets, min_std=min_std, block_size=block_size)

    def freeze_params(self):
        return GaussianBasis(self.params.detach().clone(), trainable=False, min_std=self.min_std, block_size=self.block_size)

    def means_stds(self):
        return self.params[..., 0], torch.nn.functional.softplus(self.params[..., 1] - 1.0) + self.min_std
    
    def forward(self, y: torch.Tensor):
        assert y.shape[1] == self.dim(), "y must have shape (n_data, d)"
        y = y[:, :, None]  # (n_data, d, n_basis)
        mu, std = self.means_stds()

        log_dim_factors = (
            -0.5 * torch.log(y.new_tensor(2.0 * torch.pi))
            - torch.log(std)
            - (y - mu) ** 2 / (2 * std ** 2)
        )
        return torch.exp(log_dim_factors.sum(dim=1))  # (n_data, n_basis)

    def normalized(self):
        return True

    def Omega1(self):
        return torch.ones(
            self.n_basis_functions(),
            dtype=self.params.dtype,
            device=self.params.device,
        )

    def Omega2(self, other: 'GaussianBasis'):
        assert isinstance(other, GaussianBasis), "other must be GaussianBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"

        # (d, n1), (d, n2)
        mu1, std1 = self.means_stds()
        mu2, std2 = other.means_stds()

        # Broadcast to (d, n1, n2)
        diff = mu1[:, :, None] - mu2[:, None, :]
        var_sum = (std1[:, :, None] ** 2) + (std2[:, None, :] ** 2)

        # log of 1D Gaussian pdf evaluated at diff with variance var_sum
        log_dim_ip = -0.5 * (torch.log(2 * torch.pi * var_sum) + (diff * diff) / var_sum)

        log_Omega = log_dim_ip.sum(dim=0)
        return torch.exp(log_Omega)

    def Omega3_contract(
        self,
        other1: "GaussianBasis",
        other2: "GaussianBasis",
        left_i: torch.Tensor,
        left_j: torch.Tensor,
        block_size: int | None = None,
    ):
        """
        Computes v[k] = sum_{i,j} left_i[i] * left_j[j] * Omega3[i,j,k]
        without materializing Omega3.
        """
        assert isinstance(other1, GaussianBasis), "other1 must be GaussianBasis"
        assert isinstance(other2, GaussianBasis), "other2 must be GaussianBasis"
        assert self.dim() == other1.dim() == other2.dim(), "Basis functions must have the same dimension"
        assert left_i.dim() == 1 and left_i.shape[0] == self.n_basis_functions(), "left_i has wrong shape"
        assert left_j.dim() == 1 and left_j.shape[0] == other1.n_basis_functions(), "left_j has wrong shape"

        mu0, std0 = self.means_stds()
        mu1, std1 = other1.means_stds()
        mu2, std2 = other2.means_stds()

        var0 = std0.square()
        var1 = std1.square()
        var2 = std2.square()

        inv0 = var0.reciprocal()
        inv1 = var1.reciprocal()
        inv2 = var2.reciprocal()

        n0 = self.n_basis_functions()
        n1 = other1.n_basis_functions()
        n2 = other2.n_basis_functions()

        log2pi = torch.log(mu0.new_tensor(2.0 * torch.pi))

        if block_size is None:
            block_size = self.block_size
        if block_size is None:
            # Full vectorized path (equivalent to building Omega3 then contracting).
            mu_i = mu0[:, :, None, None]
            mu_j = mu1[:, None, :, None]
            mu_k = mu2[:, None, None, :]

            inv_i = inv0[:, :, None, None]
            inv_j = inv1[:, None, :, None]
            inv_k = inv2[:, None, None, :]

            var_i = var0[:, :, None, None]
            var_j = var1[:, None, :, None]
            var_k = var2[:, None, None, :]

            S = inv_i + inv_j + inv_k
            T = mu_i * inv_i + mu_j * inv_j + mu_k * inv_k
            U = mu_i.square() * inv_i + mu_j.square() * inv_j + mu_k.square() * inv_k

            log_dim = (
                -0.5 * (3.0 * log2pi + torch.log(var_i) + torch.log(var_j) + torch.log(var_k))
                + 0.5 * (log2pi - torch.log(S))
                - 0.5 * (U - T.square() / S)
            )
            omega_full = torch.exp(log_dim.sum(dim=0))
            return torch.einsum("i,j,ijk->k", left_i, left_j, omega_full)

        assert block_size > 0, "block_size must be positive"
        denom = torch.zeros(n2, dtype=mu0.dtype, device=mu0.device)

        for j_start in range(0, n1, block_size):
            j_end = min(j_start + block_size, n1)
            left_j_blk = left_j[j_start:j_end]
            for k_start in range(0, n2, block_size):
                k_end = min(k_start + block_size, n2)
                log_chunk = torch.zeros((n0, j_end - j_start, k_end - k_start), dtype=mu0.dtype, device=mu0.device)
                for r in range(self.dim()):
                    mu_i = mu0[r, :, None, None]
                    mu_j = mu1[r, None, j_start:j_end, None]
                    mu_k = mu2[r, None, None, k_start:k_end]

                    inv_i = inv0[r, :, None, None]
                    inv_j = inv1[r, None, j_start:j_end, None]
                    inv_k = inv2[r, None, None, k_start:k_end]

                    var_i = var0[r, :, None, None]
                    var_j = var1[r, None, j_start:j_end, None]
                    var_k = var2[r, None, None, k_start:k_end]

                    S = inv_i + inv_j + inv_k
                    T = mu_i * inv_i + mu_j * inv_j + mu_k * inv_k
                    U = mu_i.square() * inv_i + mu_j.square() * inv_j + mu_k.square() * inv_k

                    log_chunk += (
                        -0.5 * (3.0 * log2pi + torch.log(var_i) + torch.log(var_j) + torch.log(var_k))
                        + 0.5 * (log2pi - torch.log(S))
                        - 0.5 * (U - T.square() / S)
                    )

                omega_chunk = torch.exp(log_chunk)
                denom[k_start:k_end] += torch.einsum("i,j,ijk->k", left_i, left_j_blk, omega_chunk)

        return denom

    def Omega22(self, other: "GaussianBasis"):
        assert isinstance(other, GaussianBasis), "other must be GaussianBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"

        # (d, n_phi), (d, n_psi)
        mu1, std1 = self.means_stds()
        mu2, std2 = other.means_stds()

        var1 = std1 * std1
        var2 = std2 * std2
        inv_var1 = 1.0 / var1
        inv_var2 = 1.0 / var2

        # Broadcast everything to (d, n_phi, n_phi, n_psi, n_psi)
        mu_i = mu1[:, :, None, None, None]
        mu_j = mu1[:, None, :, None, None]
        mu_k = mu2[:, None, None, :, None]
        mu_l = mu2[:, None, None, None, :]

        inv_i = inv_var1[:, :, None, None, None]
        inv_j = inv_var1[:, None, :, None, None]
        inv_k = inv_var2[:, None, None, :, None]
        inv_l = inv_var2[:, None, None, None, :]

        var_i = var1[:, :, None, None, None]
        var_j = var1[:, None, :, None, None]
        var_k = var2[:, None, None, :, None]
        var_l = var2[:, None, None, None, :]

        S = inv_i + inv_j + inv_k + inv_l
        T = mu_i * inv_i + mu_j * inv_j + mu_k * inv_k + mu_l * inv_l
        U = mu_i.square() * inv_i + mu_j.square() * inv_j + mu_k.square() * inv_k + mu_l.square() * inv_l

        two_pi = mu1.new_tensor(2.0 * torch.pi)
        log2pi = torch.log(two_pi)

        log_pref = -0.5 * (4.0 * log2pi + torch.log(var_i) + torch.log(var_j) + torch.log(var_k) + torch.log(var_l))
        log_gauss_int = 0.5 * (log2pi - torch.log(S))

        quad = -0.5 * (U - (T * T) / S)

        log_dim = log_pref + log_gauss_int + quad              # (d, nf, nf, ng, ng)
        log_Omega = log_dim.sum(dim=0)                         # (nf, nf, ng, ng)

        return torch.exp(log_Omega)


    def marginal(self, marginal_dims: tuple[int, ...]) -> 'GaussianBasis':
        marginal_dims = tuple(marginal_dims)
        assert all(0 <= i < self.dim() for i in marginal_dims), "marginal_dims must be in [0, d)"
        return GaussianBasis(
            self.params[marginal_dims, :, :].detach().clone(),
            trainable=self.params.requires_grad,
            min_std=self.min_std,
            block_size=self.block_size,
        )

class BetaBasis(SeparableBasis, NonnegativeBasis):
    def __init__(self, params_init: torch.Tensor, trainable: bool = True, min_concentration: float = 1.0, eps: float = 1e-6):
        assert min_concentration > 0.0, "min_concentration must be positive"
        assert params_init.shape[2] == 2, "params_init must have shape (d, n_basis, 2)"
        super().__init__(params_init, trainable)
        self.min_concentration = min_concentration
        self.eps = eps

    @classmethod
    def random_init(cls, d: int, n_basis: int, offsets: torch.Tensor = torch.zeros(2), variance: float = 1.0, min_concentration: float = 1.0, eps: float = 1e-6, device = None):
        if device is None:
            device = offsets.device
        else:
            offsets = offsets.to(device)
        offsets = offsets.repeat(d, n_basis, 1)
        return cls(
            torch.randn(d, n_basis, 2, device=device) * torch.sqrt(torch.tensor(variance, device=device)) + offsets,
            min_concentration=min_concentration,
            eps=eps,
        )

    @classmethod
    def set_init(cls, d: int, n_basis: int, offsets: torch.Tensor = torch.zeros(2), min_concentration: float = 1.0, eps: float = 1e-6, device = None):
        if device is None:
            device = offsets.device
        else:
            offsets = offsets.to(device)
        offsets = offsets.repeat(d, n_basis, 1)
        return cls(
            offsets,
            min_concentration=min_concentration,
            eps=eps,
        )

    def freeze_params(self):
        return BetaBasis(
            self.params.detach().clone(),
            trainable=False,
            min_concentration=self.min_concentration,
            eps=self.eps,
        )

    def alphas_betas(self):
        alpha = torch.nn.functional.softplus(self.params[..., 0] - 1.0) + self.min_concentration
        beta = torch.nn.functional.softplus(self.params[..., 1] - 1.0) + self.min_concentration
        return alpha, beta

    @staticmethod
    def _log_beta_fn(a: torch.Tensor, b: torch.Tensor):
        return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)

    def forward(self, y: torch.Tensor):
        assert y.shape[1] == self.dim(), "y must have shape (n_data, d)"

        #y_input = y.clone()

        y = y.clamp(self.eps, 1.0 - self.eps)
        y = y[:, :, None]  # (n_data, d, n_basis)

        alpha, beta = self.alphas_betas()  # (d, n_basis), (d, n_basis)

        log_dim_factors = (
            (alpha - 1.0) * torch.log(y)
            + (beta - 1.0) * torch.log1p(-y)
            - self._log_beta_fn(alpha, beta)
        )

        return torch.exp(log_dim_factors.sum(dim=1))  # (n_data, n_basis)

    def normalized(self):
        return True

    def Omega1(self):
        return torch.ones(
            self.n_basis_functions(),
            dtype=self.params.dtype,
            device=self.params.device,
        )

    def Omega2(self, other: "BetaBasis"):
        assert isinstance(other, BetaBasis), "other must be BetaBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"

        a1, b1 = self.alphas_betas()   # (d, n1)
        a2, b2 = other.alphas_betas()  # (d, n2)

        # Broadcast to (d, n1, n2)
        a_sum = a1[:, :, None] + a2[:, None, :] - 1.0
        b_sum = b1[:, :, None] + b2[:, None, :] - 1.0

        log_dim_ip = (
            self._log_beta_fn(a_sum, b_sum)
            - self._log_beta_fn(a1[:, :, None], b1[:, :, None])
            - self._log_beta_fn(a2[:, None, :], b2[:, None, :])
        )

        log_Omega = log_dim_ip.sum(dim=0)  # (n1, n2)
        return torch.exp(log_Omega)

    def Omega3_contract(
        self,
        other1: "BetaBasis",
        other2: "BetaBasis",
        left_i: torch.Tensor,
        left_j: torch.Tensor,
        block_size: int | None = None,
    ):
        """
        Computes v[k] = sum_{i,j} left_i[i] * left_j[j] * Omega3[i,j,k]
        without materializing Omega3.
        """
        assert isinstance(other1, BetaBasis), "other1 must be BetaBasis"
        assert isinstance(other2, BetaBasis), "other2 must be BetaBasis"
        assert self.dim() == other1.dim() == other2.dim(), "Basis functions must have the same dimension"
        assert left_i.dim() == 1 and left_i.shape[0] == self.n_basis_functions(), "left_i has wrong shape"
        assert left_j.dim() == 1 and left_j.shape[0] == other1.n_basis_functions(), "left_j has wrong shape"

        a0, b0 = self.alphas_betas()
        a1, b1 = other1.alphas_betas()
        a2, b2 = other2.alphas_betas()

        n0 = self.n_basis_functions()
        n1 = other1.n_basis_functions()
        n2 = other2.n_basis_functions()

        if block_size is None:
            a_i = a0[:, :, None, None]
            a_j = a1[:, None, :, None]
            a_k = a2[:, None, None, :]
            b_i = b0[:, :, None, None]
            b_j = b1[:, None, :, None]
            b_k = b2[:, None, None, :]
            a_sum = a_i + a_j + a_k - 2.0
            b_sum = b_i + b_j + b_k - 2.0
            log_dim = (
                self._log_beta_fn(a_sum, b_sum)
                - self._log_beta_fn(a_i, b_i)
                - self._log_beta_fn(a_j, b_j)
                - self._log_beta_fn(a_k, b_k)
            )
            omega_full = torch.exp(log_dim.sum(dim=0))
            return torch.einsum("i,j,ijk->k", left_i, left_j, omega_full)

        assert block_size > 0, "block_size must be positive"
        denom = torch.zeros(n2, dtype=a0.dtype, device=a0.device)

        for j_start in range(0, n1, block_size):
            j_end = min(j_start + block_size, n1)
            left_j_blk = left_j[j_start:j_end]
            for k_start in range(0, n2, block_size):
                k_end = min(k_start + block_size, n2)
                log_chunk = torch.zeros((n0, j_end - j_start, k_end - k_start), dtype=a0.dtype, device=a0.device)
                for r in range(self.dim()):
                    a_i = a0[r, :, None, None]
                    a_j = a1[r, None, j_start:j_end, None]
                    a_k = a2[r, None, None, k_start:k_end]

                    b_i = b0[r, :, None, None]
                    b_j = b1[r, None, j_start:j_end, None]
                    b_k = b2[r, None, None, k_start:k_end]

                    a_sum = a_i + a_j + a_k - 2.0
                    b_sum = b_i + b_j + b_k - 2.0

                    log_chunk += (
                        self._log_beta_fn(a_sum, b_sum)
                        - self._log_beta_fn(a_i, b_i)
                        - self._log_beta_fn(a_j, b_j)
                        - self._log_beta_fn(a_k, b_k)
                    )

                omega_chunk = torch.exp(log_chunk)
                denom[k_start:k_end] += torch.einsum("i,j,ijk->k", left_i, left_j_blk, omega_chunk)

        return denom

    def Omega22(self, other: "BetaBasis"):
        assert isinstance(other, BetaBasis), "other must be BetaBasis"
        assert self.dim() == other.dim(), "Basis functions must have the same dimension"

        a1, b1 = self.alphas_betas()   # (d, n_phi)
        a2, b2 = other.alphas_betas()  # (d, n_psi)

        # Broadcast to (d, n_phi, n_phi, n_psi, n_psi)
        a_i = a1[:, :, None, None, None]
        a_j = a1[:, None, :, None, None]
        a_k = a2[:, None, None, :, None]
        a_l = a2[:, None, None, None, :]

        b_i = b1[:, :, None, None, None]
        b_j = b1[:, None, :, None, None]
        b_k = b2[:, None, None, :, None]
        b_l = b2[:, None, None, None, :]

        a_sum = a_i + a_j + a_k + a_l - 3.0
        b_sum = b_i + b_j + b_k + b_l - 3.0

        log_dim = (
            self._log_beta_fn(a_sum, b_sum)
            - self._log_beta_fn(a_i, b_i)
            - self._log_beta_fn(a_j, b_j)
            - self._log_beta_fn(a_k, b_k)
            - self._log_beta_fn(a_l, b_l)
        )

        log_Omega = log_dim.sum(dim=0)  # (n_phi, n_phi, n_psi, n_psi)
        return torch.exp(log_Omega)

    def marginal(self, marginal_dims: tuple[int, ...]) -> "BetaBasis":
        marginal_dims = tuple(marginal_dims)
        assert all(0 <= i < self.dim() for i in marginal_dims), "marginal_dims must be in [0, d)"

        return BetaBasis(
            self.params[marginal_dims, :, :].detach().clone(),
            trainable=self.params.requires_grad,
            min_concentration=self.min_concentration,
            eps=self.eps,
        )