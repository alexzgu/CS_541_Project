import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import numpy as np
from torchvision import transforms


# =============================
# Define the TemporalCNN Model
# =============================

class TemporalCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=1):
        super(TemporalCNN, self).__init__()
        # Example architecture; replace with your actual model.
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(32 * 25, 128)  # Adjust based on input size
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Example forward pass; adjust according to your model.
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ==================================
# Define the Inference Functionality
# ==================================

def load_model(model_path, device):
    """
    Loads the model from the specified path.

    Args:
        model_path (str): Path to the saved model weights.
        device (torch.device): Device to load the model onto.

    Returns:
        model (nn.Module): Loaded model ready for inference.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' does not exist.")

    # Initialize the model architecture
    model = TemporalCNN()

    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model


def preprocess_input(input_data, transform=None):
    """
    Preprocesses the input data to match the model's expected input.

    Args:
        input_data (np.ndarray or list): Raw input data.
        transform (callable, optional): Optional transform to be applied on the input.

    Returns:
        torch.Tensor: Preprocessed input tensor.
    """
    if isinstance(input_data, list):
        input_data = np.array(input_data)

    if not isinstance(input_data, np.ndarray):
        raise TypeError("Input data should be a NumPy array or a list.")

    # Example: Normalize the input data
    input_tensor = torch.from_numpy(input_data).float()

    # Add channel dimension if necessary
    if input_tensor.dim() == 2:
        input_tensor = input_tensor.unsqueeze(0)  # Shape: (1, D, T)

    if transform:
        input_tensor = transform(input_tensor)

    return input_tensor


def predict(model, input_tensor, device):
    """
    Performs inference using the loaded model.

    Args:
        model (nn.Module): The trained model.
        input_tensor (torch.Tensor): Preprocessed input tensor.
        device (torch.device): Device to perform computation on.

    Returns:
        torch.Tensor: Raw model outputs.
    """
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        # If using a sigmoid activation for binary classification
        probabilities = torch.sigmoid(outputs)

    return outputs.cpu().numpy(), probabilities.cpu().numpy()


# ====================
# Command-Line Interface
# ====================

def parse_args():
    parser = argparse.ArgumentParser(description="Model Inference Script for TemporalCNN")
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                        help='Path to the saved model weights.')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input data file (e.g., .npy file).')
    parser.add_argument('--output_activation', action='store_true',
                        help='Apply sigmoid activation to model outputs.')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU for inference if available.')
    return parser.parse_args()


def main():
    args = parse_args()

    # Device configuration
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load the model
    model = load_model(args.model_path, device)
    print(f'Model loaded from {args.model_path}')

    # Load and preprocess the input data
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file '{args.input}' does not exist.")

    # Example: Load input data from a NumPy file
    input_data = np.load(args.input)

    # Preprocess the input data
    input_tensor = preprocess_input(input_data)

    # Perform prediction
    outputs, probabilities = predict(model, input_tensor, device)

    # Display the results
    print("Raw Model Outputs:")
    print(outputs)

    if args.output_activation:
        print("Probabilities (after sigmoid activation):")
        print(probabilities)

    # Optionally, save the outputs to a file
    # Example: Save probabilities to a NumPy file
    # np.save('model_predictions.npy', probabilities)


if __name__ == "__main__":
    main()

