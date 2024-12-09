import torch
from .model import LSTMClassifier
from .load import get_predicted_segments
from .syllables import syllables
import torch.nn as nn
import numpy as np
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
model_directory = f'{current_directory}/pretrained'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(model_name):
    model = LSTMClassifier(hidden_size=144, dropout=0.5)
    model.load_state_dict(torch.load(f'{model_directory}/{model_name}', map_location=torch.device(device)))

    index, tensors = get_predicted_segments()

    for num, row in index.iterrows():
        file_id = row['file']
        max_length = tensors[file_id].shape[0]
        start = min(int(row['start'] / 20), max_length)
        end = min(int(row['end'] / 20), max_length)

        if start==end:
            end += 1

        input_tensor = tensors[file_id][start:end]
        lengths = [end - start]

        prediction = model(input_tensor.unsqueeze(0), lengths).cpu().detach().numpy()

        print(syllables[np.argmax(prediction)])





