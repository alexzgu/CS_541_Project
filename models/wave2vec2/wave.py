from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch

def to_tensors(waveform):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

    # Convert to mono if stereo
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Preprocess audio for Wav2Vec2
    input_values = processor(waveform.squeeze(0), return_tensors="pt", sampling_rate=16000).input_values

    # Forward pass through Wav2Vec2 model to get hidden states
    with torch.no_grad():
        outputs = model(input_values)

    # Extract feature vectors
    hidden_states = outputs.last_hidden_state  # Shape: (batch_size, time_steps, feature_dim)

    # Convert hidden states to NumPy array
    feature_vectors = hidden_states.squeeze(0).cpu().numpy()  # Shape: (time_steps, feature_dim)

    return feature_vectors