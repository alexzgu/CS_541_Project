import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
from .model import LSTMClassifier
from .load import get_lstm_dataloader
import torch
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
model_directory = f'{current_directory}/pretrained'
accuracy = Accuracy(task="multiclass", num_classes=110)

def train(epochs=10, batch_size=32):
    dataloader = get_lstm_dataloader(batch_size)
    model = LSTMClassifier()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    model.train()

    for epoch in range(epochs):
        for padded_sequences, labels, lengths in dataloader:
            # Forward pass
            outputs = model(padded_sequences, lengths)
            loss = criterion(outputs, labels)
            accuracy(torch.argmax(outputs, dim=1), torch.argmax(labels, dim=1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy.compute():.2f}")

        torch.save(model.state_dict(), f'model_directory/model_{epoch}_{accuracy.compute():.2f}')

