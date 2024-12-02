import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# load in data with dataloader
data_dir = '../../data'
syllable_directory = f'{data_dir}/clean/syllables'
audio_clip_directory = f'{syllable_directory}/clips'  # this contains indexed .mp3 files, each of which is a song
clip_index_file = f'{syllable_directory}/segment_index.csv'
# the clip index file contains the following columns:
# index: the index of the corresponding audio clip
# token: the token (i.e., label) corresponding to the clip (string)

# I want a custom dataset class such that each sample pairs the audio clip with its corresponding token
# I will then use a DataLoader to load in the data

class SyllableDataset(Dataset):
    def __init__(self, clip_index_file: str, audio_clip_directory: str):
        self.clip_index = pd.read_csv(clip_index_file)
        self.audio_clip_directory = audio_clip_directory

    def __len__(self):
        return len(self.clip_index)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        clip_name = f"{self.clip_index.iloc[idx, 1]}.mp3"
        clip_path = f"{self.audio_clip_directory}/{clip_name}"
        clip = torch.load(clip_path)
        token = self.clip_index.iloc[idx, 4]

        return clip, token

syllable_dataset = SyllableDataset(clip_index_file, audio_clip_directory)
syllable_dataloader = DataLoader(syllable_dataset, batch_size=4, shuffle=True)

# I will now define a simple model to classify the syllables
class SyllableClassifier(nn.Module):
    def __init__(self):
        super(SyllableClassifier, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

# I will now train the model
model = SyllableClassifier()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for i, data in enumerate(syllable_dataloader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")
print("Finished Training")

# I will now save the model
torch.save(model.state_dict(), "syllable_classifier.pth")