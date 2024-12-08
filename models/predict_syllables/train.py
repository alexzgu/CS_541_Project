import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
from .model import LSTMClassifier
from .load import get_lstm_dataloader
import numpy as np
import torch
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
model_directory = f'{current_directory}/pretrained'
accuracy = Accuracy(task="multiclass", num_classes=110)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accuracy = Accuracy(task="multiclass", num_classes=110).to(device)


# Send model to GPU

def train(
        model_name,
        segment_length_ms,
        epochs=100,
        batch_size=128,
        lr=0.001,
        weight_decay=0.001,
):
    print(device)
    dataloader = get_lstm_dataloader(batch_size, segment_length_ms)

    model = LSTMClassifier()
    if model_name is not None:
        model.load_state_dict(torch.load(f'{model_directory}/{model_name}'))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()

    for epoch in range(epochs):
        for padded_sequences, labels, lengths in dataloader:
            padded_sequences, labels = padded_sequences.to(device), labels.to(device)
            # Forward pass
            outputs = model(padded_sequences, lengths)
            loss = criterion(outputs, labels)
            accuracy(torch.argmax(outputs, dim=1), torch.argmax(labels, dim=1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy.compute():.2f}")

        model_path = f'{model_directory}/model_{segment_length_ms}ms_{epoch + 1}'
        torch.save(model.state_dict(), f'{model_path}_{accuracy.compute():.2f}')

        test_accuracy = test(model_path, segment_length_ms)
        torch.save(model.state_dict(), f'{model_path}_{test_accuracy:.2f}_test')


def test(model_name, segment_length_ms):
    print(device)
    batch_size = 128
    dataloader = get_lstm_dataloader(batch_size, segment_length_ms, 80, 93)

    model = LSTMClassifier()
    model.load_state_dict(torch.load(f'{model_directory}/{model_name}', map_location=torch.device(device)))
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    accuracies = []

    for padded_sequences, labels, lengths in dataloader:
        padded_sequences, labels = padded_sequences.to(device), labels.to(device)
        # Forward pass
        outputs = model(padded_sequences, lengths)
        loss = criterion(outputs, labels)
        accuracy(torch.argmax(outputs, dim=1), torch.argmax(labels, dim=1))

        print(f"Test Loss: {loss.item():.4f}, Test Accuracy: {accuracy.compute():.2f}")
        accuracies.append(accuracy.compute())

    return np.mean(accuracies)
