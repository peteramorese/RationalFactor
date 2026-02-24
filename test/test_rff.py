import torch
import rational_factor.systems.truth_models as truth_models
from rational_factor.systems.base import sample_trajectories, create_transition_data_matrix
from torch.utils.data import DataLoader, TensorDataset
from rational_factor.models.basis_functions import GaussianBasis
from rational_factor.models.rational_factor import LinearRFF, LinearFF
import rational_factor.models.train as train
import rational_factor.models.loss as loss

def make_mvnormal_init_sampler(mean: torch.Tensor, covariance: torch.Tensor):
    """
    Create an initial state sampler that draws n_samples i.i.d. from a multivariate normal.

    Args:
        mean: (d,) mean vector
        covariance: (d, d) covariance matrix

    Returns:
        callable(n_samples: int) -> torch.Tensor of shape (n_samples, d)
    """
    mean = torch.as_tensor(mean, dtype=torch.float32)
    covariance = torch.as_tensor(covariance, dtype=torch.float32)
    dist = torch.distributions.MultivariateNormal(mean, covariance)

    def sampler(n_samples: int) -> torch.Tensor:
        return dist.sample((n_samples,))

    return sampler


if __name__ == "__main__":
    
    ###
    use_gpu = torch.cuda.is_available()
    n_basis = 100
    n_epochs = 1000
    batch_size = 256
    learning_rate = 1e-2
    n_timesteps_train = 10
    n_trajectories_train = 1000
    ###

    # Create system
    system = truth_models.VanDerPol(dt=0.3, mu=0.9, covariance=0.1*torch.eye(2))

    # Generate data set from trajectories
    mean = torch.tensor([0.2, 0.1])
    cov = torch.diag(torch.tensor([0.2, 0.2]))
    dist = torch.distributions.MultivariateNormal(mean, cov)
    def init_state_sampler(n_samples : int):
        return dist.sample((n_samples,))

    traj_data = sample_trajectories(system, init_state_sampler, n_timesteps=n_timesteps_train, n_trajectories=n_trajectories_train)
    x0_data = TensorDataset(traj_data[0])
    x_k, x_kp1 = create_transition_data_matrix(traj_data, separate=True)
    xp_data = TensorDataset(x_k, x_kp1)

    x0_dataloader = DataLoader(x0_data, batch_size=256, shuffle=True, pin_memory=use_gpu)
    xp_dataloader = DataLoader(xp_data, batch_size=512, shuffle=True, pin_memory=use_gpu)

    # Create basis functions
    phi_basis = GaussianBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 2.0]))
    psi_basis = GaussianBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 2.0]))
    psi0_basis = GaussianBasis.random_init(system.dim(), n_basis=n_basis, offsets=torch.tensor([0.0, 2.0]))

    # Create and train the transition model
    tran_model = LinearRFF(phi_basis, psi_basis)
    print("Training transition model")
    tran_model = train.train(tran_model, 
        xp_dataloader, 
        loss.rff_mle_loss, 
        torch.optim.Adam(tran_model.parameters(), lr=learning_rate), epochs=n_epochs)
    print("Done! \n")


    init_model = LinearFF(tran_model, psi0_basis)
    print("Training initial model")
    init_model = train.train(init_model, 
        x0_dataloader, 
        loss.ff_mle_loss, 
        torch.optim.Adam(init_model.parameters(), lr=learning_rate), epochs=n_epochs)
    print("Done! \n")