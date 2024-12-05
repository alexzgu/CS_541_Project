import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from .model import SkipConnectionNetwork
from .load import get_training_set
import torch


accuracy = Accuracy(task="multiclass", num_classes=105)

def train(epochs=1000, batch_size=128):
    dataset = get_training_set()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SkipConnectionNetwork()

    criterion = nn.CrossEntropyLoss()
    accuracy = Accuracy(task="multiclass", num_classes=105)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        for batch_X, batch_Y in dataloader:
            # Forward pass
            predictions = model(batch_X)
            loss = criterion(predictions, batch_Y.float())
            accuracy(torch.argmax(predictions, dim=1), torch.argmax(batch_Y, dim=1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy.compute():.2f}")







