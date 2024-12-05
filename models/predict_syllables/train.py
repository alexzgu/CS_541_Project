import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
from .model import SkipConnectionNetwork
from .load import get_training_set


accuracy = Accuracy(task="multiclass", num_classes=105)

def train(epochs=10):
    dataloader = get_training_set()
    model = SkipConnectionNetwork()

    criterion = nn.CrossEntropyLoss
    accuracy = Accuracy()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        for batch_X, batch_Y in dataloader:
            # Forward pass
            predictions = model(batch_X)
            loss = criterion(predictions, batch_Y)
            accuracy(predictions, batch_Y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy.compute():.2f}")







