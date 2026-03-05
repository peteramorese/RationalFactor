import torch
from torch.utils.data import DataLoader
from .density_model import DensityModel, ConditionalDensityModel
import time
from copy import deepcopy

def train(model : DensityModel | ConditionalDensityModel, data_loader : DataLoader, labeled_loss_fns : dict[str, callable], optimizer, epochs=100, verbose=True, use_best=True):
    
    torch.autograd.set_detect_anomaly(True)

    model.train()

    loss_labels = list(labeled_loss_fns.keys())
    loss_fns = list(labeled_loss_fns.values())

    def train_step(*args):
        optimizer.zero_grad()
        losses = [loss_fn(model, *args) for loss_fn in loss_fns]
        total_loss = sum(losses)
        total_loss.backward()
        optimizer.step()
        return total_loss.item(), losses

    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        start_time = time.time()
        total_sum_loss = 0.0
        sum_losses = [0.0 for _ in loss_fns]
        for batch in data_loader:
            total_loss, losses = train_step(*batch)
            total_sum_loss += total_loss
            for i, loss in enumerate(losses):
                sum_losses[i] += loss.item()

        end_time = time.time()
        avg_total_loss = total_sum_loss / len(data_loader)
        epoch_time = end_time - start_time

        if use_best and model.valid() and avg_total_loss < best_loss:
            best_loss = avg_total_loss
            best_state = deepcopy(model.state_dict())

        if verbose:
            avg_losses = [loss_sum / len(data_loader) for loss_sum in sum_losses]
            loss_details = ", ".join(
                f"{label}:{value:.4f}"
                for label, value in zip(loss_labels, avg_losses)
            )
            print(
                f"Epoch {epoch+1}, Time: {epoch_time:.2f}s, Loss: tot:{avg_total_loss:.4f}, "
                f"{loss_details}"
            )
        else:
            print(f"Epoch {epoch+1}, Loss: {avg_total_loss:.4f}, Time: {epoch_time:.2f}s")
        
    if use_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n Restored best model (total loss={best_loss:.4f})")

    return model