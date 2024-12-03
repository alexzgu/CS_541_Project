from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio
import torch

def to_tensors(waveform, sampling_rate, segment_length_ms=None, num_vectors=None):
    """
        Encodes an audio file into Wav2Vec2 feature vectors.

        Args:
            waveform: the torchaudio waveform to encode
            sampling_rate (int): the sampling rate of the waveform
            segment_length_ms (int): Desired length of each encoding segment in milliseconds.
            num_vectors (int): Number of vectors to encode the audio file into.

        Returns:
            numpy.ndarray: Array of Wav2Vec2 feature vectors.
    """

    if segment_length_ms is None == num_vectors is None:
        raise Exception("Must define exactly one of segment_length_ms and num_vectors")

    # Wav2Vec2 expects a sampling rate of 16000 Hz and encodes 20ms segments of audio per vector
    model_stride_ms = 20  # Wav2Vec2 model stride in milliseconds
    model_sampling_rate = 16000  # Wav2Vec2 expected sampling rate

    desired_sampling_rate = None

    if segment_length_ms is not None:
        desired_sampling_rate = int(model_sampling_rate * (model_stride_ms / segment_length_ms))
    if num_vectors is not None:
        desired_samples = num_vectors * (model_sampling_rate * model_stride_ms * 0.001) + 320 # last 320 is a fudge factor, idk why it's needed
        desired_sampling_rate = int(sampling_rate * (desired_samples / waveform.size(1)))

    # Resample the audio to have our desired segmentation length
    if sampling_rate != desired_sampling_rate:
        resample_transform = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=desired_sampling_rate)
        waveform = resample_transform(waveform)

    tensors = _convert_to_wave2vec2(waveform)

    if num_vectors is not None and num_vectors != len(tensors):
        raise Exception("Internal error, output number of vectors is incorrect")

    return tensors


def _convert_to_wave2vec2(waveform):
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