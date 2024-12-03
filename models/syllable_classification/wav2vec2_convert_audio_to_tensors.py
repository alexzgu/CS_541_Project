from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio
import torch
import os

# load in data with dataloader
data_dir = '../../data'
syllable_dir = f'{data_dir}/clean/syllables'
audio_dir = f'{data_dir}/clean/audio/vocals'
clip_index_file = f'{syllable_dir}/segment_index.csv'

def encode_audio_to_wav2vec_vectors(audio_path, segment_length_ms):
    """
    Encodes an audio file into Wav2Vec2 feature vectors.

    Args:
        audio_path (str): Path to the audio file.
        segment_length_ms (int): Desired length of each encoding segment in milliseconds.

    Returns:
        numpy.ndarray: Array of Wav2Vec2 feature vectors.
    """

    # Load Wav2Vec2 processor and model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

    # Load audio file using torchaudio
    waveform, sampling_rate = torchaudio.load(audio_path)

    # Wav2Vec2 expects 16000 samples to correspond to 20 milliseconds segments of audio
    model_stride_ms = 20  # Wav2Vec2 model stride in milliseconds
    model_sampling_rate = 16000 # Wav2Vec2 expected sampling rate per 20ms/stride
    desired_sampling_rate = int(model_sampling_rate * (segment_length_ms / model_stride_ms))

    # Resample the audio to have our desired segmentation length
    if sampling_rate != desired_sampling_rate:
        resample_transform = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=desired_sampling_rate)
        waveform = resample_transform(waveform)

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


# Example usage
audio_path = "../../data/clean/audio/vocals/0.mp3"
feature_vectors = encode_audio_to_wav2vec_vectors(audio_path, 20)

print(f"Shape of Wav2Vec2 feature vectors: {feature_vectors.shape}")



# Process all files in directory
# tensors = {}
# for file in os.listdir(audio_dir):
#     if file.endswith('.mp3'):
#         file_path = os.path.join(audio_dir, file)
#         tensors[file] = encode_audio_to_wav2vec_vectors(file_path, 10)
#         # store the dictionary in the following path: f'{syllable_dir}/tensors.pt'
#         torch.save(tensors, f'{syllable_dir}/wav2vec2_embeddings.pt')
