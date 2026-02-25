import torch
from torch.utils.data import DataLoader
from .rational_factor import LinearRFF
import time

def train(model, data_loader : DataLoader, labeled_loss_fns : dict[str, callable], optimizer, epochs=100, verbose=True):
    
    model.train()

    loss_labels = list(labeled_loss_fns.keys())
    loss_fns = list(labeled_loss_fns.values())

    def train_step(*args):
        optimizer.zero_grad()
        #loss = loss_fn(model, *args)
        losses = [loss_fn(model, *args) for loss_fn in loss_fns]
        total_loss = sum(losses)
        total_loss.backward()
        optimizer.step()
        return total_loss.item(), losses

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
    return model