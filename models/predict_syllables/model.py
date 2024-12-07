from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size=768,
        hidden_size=252,
        num_layers=2,
        num_classes=110,
        dropout=0.33
    ):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        # Pack the sequence
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=True)

        # Pass through LSTM
        packed_out, (h_n, c_n) = self.lstm(packed_x)  # h_n is the last hidden state

        # Optionally unpack (if you need all time steps for something)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # Use the last hidden state for classification
        out = h_n[-1]  # (batch_size, hidden_size)
        out = self.fc(out)  # (batch_size, num_classes)
        return out
