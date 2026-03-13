import torch
from torch.utils.data import DataLoader
from .density_model import DensityModel, ConditionalDensityModel
import time
from copy import deepcopy

def train(model : DensityModel | ConditionalDensityModel, data_loader : DataLoader, labeled_loss_fns : dict[str, callable], optimizer, epochs=100, verbose=True, use_best : str = "total"):
    
    torch.autograd.set_detect_anomaly(True)

    model.train()

    loss_labels = list(labeled_loss_fns.keys())
    loss_fns = list(labeled_loss_fns.values())

    assert use_best in loss_labels, f"use_best must be one of {loss_labels}"

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
        avg_losses = [loss_sum / len(data_loader) for loss_sum in sum_losses]
        loss_dict = {label: value for label, value in zip(loss_labels, avg_losses)}
        loss_dict["total"] = avg_total_loss

        epoch_time = end_time - start_time

        valid = model.valid()
        if use_best and valid and loss_dict[use_best] < best_loss:
            best_loss = loss_dict[use_best]
            best_state = deepcopy(model.state_dict())

        if verbose:
            loss_details = ", ".join(
                f"{label}:{value:.4f}"
                for label, value in zip(loss_labels, avg_losses)
            )
            if not valid:
                loss_details += " <invld>"
            print(
                f"Epoch {epoch+1}, Time: {epoch_time:.2f}s, Loss: tot:{avg_total_loss:.4f}, "
                f"{loss_details}"
            )
        else:
            print(f"Epoch {epoch+1}, Loss: {avg_total_loss:.4f}, Time: {epoch_time:.2f}s")
        
    if use_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n Restored best model ({use_best} loss={best_loss:.4f})")

    return model

def train_to_valid(model : DensityModel | ConditionalDensityModel, labeled_loss_fns : dict[str, callable], optimizer, epochs=100, verbose=True, use_best : str = "total"):
    
    torch.autograd.set_detect_anomaly(True)

    model.train()

    loss_labels = list(labeled_loss_fns.keys())
    loss_fns = list(labeled_loss_fns.values())

    assert use_best in loss_labels, f"use_best must be one of {loss_labels}"

    def train_step():
        optimizer.zero_grad()
        losses = [loss_fn(model) for loss_fn in loss_fns]
        total_loss = sum(losses)
        total_loss.backward()
        optimizer.step()
        return total_loss.item(), losses

    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        start_time = time.time()
        loss = 0.0
        losses = [0.0 for _ in loss_fns]
        loss, losses = train_step()

        end_time = time.time()
        loss_dict = {label: value for label, value in zip(loss_labels, losses)}
        loss_dict["total"] = loss

        epoch_time = end_time - start_time

        valid = model.valid()
        if valid:
            print("Found valid model!")
            return model

        if use_best and loss_dict[use_best] < best_loss:
            best_loss = loss_dict[use_best]
            best_state = deepcopy(model.state_dict())

        if verbose:
            loss_details = ", ".join(
                f"{label}:{value:.4f}"
                for label, value in zip(loss_labels, losses)
            )
            if not valid:
                loss_details += " <invld>"
            print(
                f"Epoch {epoch+1}, Time: {epoch_time:.2f}s, Loss: tot:{loss:.4f}, "
                f"{loss_details}"
            )
        else:
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Time: {epoch_time:.2f}s")
        
    if use_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n Restored best model ({use_best} loss={best_loss:.4f})")

    return model