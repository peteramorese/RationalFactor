import torch
from torch.utils.data import DataLoader
from .density_model import DensityModel, ConditionalDensityModel
import time
from copy import deepcopy
from datetime import datetime, timedelta

class TrainingTimer:
    def __init__(self, n_groups : int, iterations : int, epochs_per_group : int):
        self.n_groups = n_groups
        self.iterations = iterations
        if isinstance(epochs_per_group, list):
            assert len(epochs_per_group) == n_groups, "epochs_per_group must be a list of the same length as n_groups"
            self.epochs_per_group = epochs_per_group
        else:
            self.epochs_per_group = [epochs_per_group for _ in range(n_groups)]

        self.total_epochs = iterations * sum(self.epochs_per_group)

    def initialize(self):
        self.times = [0.0 for _ in range(self.n_groups)]
        self.completed_epochs = [0 for _ in range(self.n_groups)]
        self.curr_date_time = time.time()
        self.curr_iter = 0
        self.curr_group = 0
        self.curr_epoch = 0
    
    def update_epoch(self, time_taken : float = None):
        self.curr_epoch += 1
        if self.curr_epoch > self.epochs_per_group[self.curr_group]:
            print("Exceeded number of epochs per group (predicted time will be wrong)")
        
        self.times[self.curr_group] += time_taken if time_taken is not None else time.time() - self.curr_date_time
        self.completed_epochs[self.curr_group] += 1
        self.curr_date_time = time.time()

    def update_group(self):
        assert self.curr_group < self.n_groups, "Exceeded number of groups"
        self.curr_group += 1
        self.curr_epoch = 0

    def update_iteration(self):
        self.curr_iter += 1
        if self.curr_iter > self.iterations:
            print("Exceeded number of iterations (predicted time will be wrong)")
        
        self.curr_epoch = 0
        self.curr_group = 0
    
    def get_predicted_time(self):
        avg_times = [
            elapsed / total_epochs if total_epochs > 0 else 0.0
            for elapsed, total_epochs in zip(self.times, self.completed_epochs)
        ]
        remaining_epochs = [self.iterations * epochs_per_group - total_epochs for total_epochs, epochs_per_group in zip(self.completed_epochs, self.epochs_per_group)]
        remaining_times = [avg_time * epochs for avg_time, epochs in zip(avg_times, remaining_epochs)]
        total_remaining_time = sum(remaining_times)
        etr = timedelta(seconds=total_remaining_time) 
        eta = datetime.now() + timedelta(seconds=total_remaining_time)
        return etr, eta 
    
    def get_predicted_etr_str(self):
        etr, _ = self.get_predicted_time()
        total = int(etr.total_seconds())
        h, r = divmod(total, 3600)
        m, s = divmod(r, 60)
        return f"{h}:{m:02d}:{s:02d}"
    
    def total_since_start(self):
        return sum(self.times)

def train(model : DensityModel | ConditionalDensityModel, data_loader : DataLoader, labeled_loss_fns : dict[str, callable], optimizer, epochs=100, verbose=True, use_best : str = "total"):
    
    torch.autograd.set_detect_anomaly(True)

    model.train()

    loss_labels = list(labeled_loss_fns.keys())
    loss_fns = list(labeled_loss_fns.values())

    assert use_best in loss_labels, f"use_best must be one of {loss_labels}"

    training_timer = TrainingTimer(n_groups=1, iterations=1, epochs_per_group=epochs)
    training_timer.initialize()

    def train_step(*args):
        optimizer.zero_grad()
        losses = [loss_fn(model, *args) for loss_fn in loss_fns]
        total_loss = sum(losses)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        return total_loss.item(), losses

    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        start_time = time.time()
        total_sum_loss = 0.0
        sum_losses = [0.0 for _ in loss_fns]
        for batch in data_loader:
            dev = next(model.parameters()).device
            if isinstance(batch, list):
                batch = [b.to(dev) for b in batch]
            elif isinstance(batch, tuple):
                batch = tuple(b.to(dev) for b in batch)
            else:
                batch = batch.to(dev)

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

        training_timer.update_epoch()

        if verbose:
            loss_details = ", ".join(
                f"{label}:{value:.4f}"
                for label, value in zip(loss_labels, avg_losses)
            )
            if not valid:
                loss_details += " <invld>"
            print(
                f"Epoch {epoch+1}, Time: {epoch_time:.2f}s, Loss: tot:{avg_total_loss:.4f}, "
                f"{loss_details}, etr: {training_timer.get_predicted_etr_str()}"
            )
        else:
            print(f"Epoch {epoch+1}, Loss: {avg_total_loss:.4f}, Time: {epoch_time:.2f}s")
        
    if verbose:
        print(f"Completed training in {training_timer.total_since_start():.2f} seconds")
    if use_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n Restored best model ({use_best} loss={best_loss:.4f})")

    return model, best_loss, training_timer.total_since_start()

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

def set_requires_grad(params : list[torch.nn.Parameter], requires_grad : bool):
    for param in params:
        param.requires_grad_(requires_grad)

def train_iterate(model : DensityModel | ConditionalDensityModel, data_loader : DataLoader, labeled_loss_fns : dict[str, callable], labeled_optimizers : dict[str, torch.optim.Optimizer], epochs_per_group=10, iterations=10, verbose=True, use_best : str = "total"):
    
    torch.autograd.set_detect_anomaly(False)

    model.train()

    loss_labels = list(labeled_loss_fns.keys())
    loss_fns = list(labeled_loss_fns.values())
    optimizer_labels = list(labeled_optimizers.keys())
    optimizers = list(labeled_optimizers.values())

    assert use_best in loss_labels, f"use_best must be one of {loss_labels}"

    def train_step(*args, optimizer : torch.optim.Optimizer, param_groups : list[list[torch.nn.Parameter]]): 
        optimizer.zero_grad()
        losses = [loss_fn(model, *args) for loss_fn in loss_fns]
        total_loss = sum(losses)
        total_loss.backward()
        for params in param_groups:
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        optimizer.step()
        return total_loss.item(), losses

    best_loss = float('inf')
    best_state = None

    if isinstance(epochs_per_group, list):
        assert len(epochs_per_group) == len(labeled_optimizers), "epochs_per_group must be a list of the same length as n_groups"
    else:
        epochs_per_group = [epochs_per_group for _ in range(len(labeled_optimizers))]

    training_timer = TrainingTimer(n_groups=len(optimizers), iterations=iterations, epochs_per_group=epochs_per_group)
    training_timer.initialize()

    for iteration in range(iterations):
        for optimizer, optimizer_label, epochs in zip(optimizers, optimizer_labels, epochs_per_group):

            # Set the gradient capabilities of current parameters
            set_requires_grad(model.parameters(), False)
            param_groups = [g['params'] for g in optimizer.param_groups]

            for params in param_groups:
                set_requires_grad(params, True)


            for epoch in range(epochs):
                start_time = time.time()
                total_sum_loss = 0.0
                sum_losses = [0.0 for _ in loss_fns]
                
                for batch in data_loader:
                    dev = next(model.parameters()).device
                    if isinstance(batch, list):
                        batch = [b.to(dev) for b in batch]
                    elif isinstance(batch, tuple):
                        batch = tuple(b.to(dev) for b in batch)
                    else:
                        batch = batch.to(dev)

                    total_loss, losses = train_step(*batch, optimizer=optimizer, param_groups=param_groups)
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

                training_timer.update_epoch()

                if verbose:
                    loss_details = ", ".join(
                        f"{label}:{value:.4f}"
                        for label, value in zip(loss_labels, avg_losses)
                    )
                    if not valid:
                        loss_details += " <invld>"
                    print(
                        f"Itr: {iteration+1}, Optimizing: {optimizer_label}, Epoch {epoch+1}, Time: {epoch_time:.2f}s, Loss: tot:{avg_total_loss:.4f}, "
                        f"{loss_details}, etr: {training_timer.get_predicted_etr_str()}"
                    )
                else:
                    print(f"Itr: {iteration+1}, Optimizing: {optimizer_label}, Epoch {epoch+1}, Loss: {avg_total_loss:.4f}, Time: {epoch_time:.2f}s")

            training_timer.update_group()
            
        training_timer.update_iteration()
        
    if verbose:
        print(f"Completed training in {training_timer.total_since_start():.2f} seconds")
    if use_best and best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n Restored best model ({use_best} loss={best_loss:.4f})")

    return model, best_loss, training_timer.total_since_start()
