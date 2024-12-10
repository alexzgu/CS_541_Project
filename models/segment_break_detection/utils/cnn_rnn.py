import torch
import torch.nn as nn
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from typing import List
import torch.nn.functional as F
import torch.optim as optim

class CNNRNN(nn.Module):
    def __init__(self, n_mels=32, hidden_size=128, num_layers=3):
        super(CNNRNN, self).__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv1d(n_mels, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv1d(64, 32, kernel_size=3, padding=1),
        #     nn.ReLU()
        # )
        # self.rnn = nn.LSTM(
        #     input_size=32,
        #     hidden_size=hidden_size,
        #     num_layers=num_layers,
        #     batch_first=True,
        #     bidirectional=True
        # )

        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.rnn = nn.LSTM(
            input_size=32,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # binary labels
        )

    def forward(self, x, mask):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        x = x * mask.unsqueeze(-1)
        return x