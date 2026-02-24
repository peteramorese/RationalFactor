import torch
from torch.utils.data import DataLoader
from .rational_factor import LinearRFF
import time

def train(model, data_loader : DataLoader, loss_fn, optimizer, epochs=100):
    
    model.train()

    def train_step(*args):
        optimizer.zero_grad()
        loss = loss_fn(model, *args)
        loss.backward()
        optimizer.step()
        return loss.item()

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        for batch in data_loader:
            loss = train_step(*batch)
            total_loss += loss
        end_time = time.time()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(data_loader)}, Time: {end_time-start_time:.2f}s")
    return model