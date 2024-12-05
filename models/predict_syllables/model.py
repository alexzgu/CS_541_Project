import torch.nn as nn

class SkipConnectionNetwork(nn.Module):
    def __init__(
        self,
        input_size=7680,
        hidden_size=512,
        num_classes=105,
        num_layers=10
    ):
        super(SkipConnectionNetwork, self).__init__()
        self.relu = nn.ReLU()

        self.input_layer = nn.Linear(input_size, hidden_size)

        # Define the 8 hidden layers with skip connections
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 2)
        ])

        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Input layer
        x = self.relu(self.input_layer(x))

        # Hidden layers with skip connections
        skip = x  # Save initial input for skip connection
        for layer in self.hidden_layers:
            x = self.relu(layer(x) + skip)
            skip = x  # Update skip connection

        # Output layer
        x = self.output_layer(x)
        return x


# Instantiate the model
model = SkipConnectionNetwork()

# Print model architecture
print(model)
