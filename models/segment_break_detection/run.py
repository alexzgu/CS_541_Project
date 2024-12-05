import pandas as pd
import torch
import os

from models.wave2vec2.audio import to_tensors

# Load the CSV files
data_dir = '../../data'
label_dir = f'{data_dir}/clean/segment_breaks'
input_dir = f'{data_dir}/clean/audio/vocals'  # vocal audio
# go through entire directory of label_dir, and get all file names
csv_files = os.listdir(label_dir)

num_training_examples = 2

data = []
for file in csv_files[:num_training_examples]:
    file = os.path.join(label_dir, file)
    file_name = os.path.basename(file).split(".")[0]
    df = pd.read_csv(file)
    data.append((df, file_name))

print("Loaded CSV files")

# Extract the audio data
audio_data = []
for df, song_id in data:
    start_times = df["start"]
    end_times = df["end"]
    audio_file = os.path.join(input_dir, f"{song_id}.mp3")

    print(f"Extracting audio data from {audio_file}")
    # extract waveform and sampling rate
    feature_vectors = to_tensors(audio_file, segment_length_ms=10)
    audio_data.append((feature_vectors, start_times, end_times))

print("Extracted audio data")
print(len(audio_data))
for i in range(2):
    print(audio_data[i][0].shape)
    print(data[i][0])
    print(data[i][1])
    print(audio_data[i][1])
    print(audio_data[i][2])
    print("==============================================")
raise Exception("Stop here")

# Create the RNN model
class RNNModel(torch.nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.rnn = torch.nn.LSTM(input_size=feature_vectors.shape[1], hidden_size=128, num_layers=1)
        self.fc = torch.nn.Linear(128, 1)

    def forward(self, x):
        h0 = torch.zeros(1, 128).to(x.device)
        c0 = torch.zeros(1, 128).to(x.device)
        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out[-1, :])
        return out

# Train the RNN model
print("initializing RNN model")
model = RNNModel()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training RNN model")
epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}==============================================")
    for i, (feature, target) in enumerate(zip(audio_data, data)):
        feature = torch.tensor(feature[0])
        target = torch.tensor(target[0]["break"])
        optimizer.zero_grad()
        output = model(feature)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
