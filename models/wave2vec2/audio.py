import torchaudio
import wave

def to_tensors(audio_path, segment_length_ms=None, num_vectors=None):
    """
    Encodes an audio file into Wav2Vec2 feature vectors.

    Args:
        audio_path (str): Path to the audio file.
        segment_length_ms (int): Desired length of each encoding segment in milliseconds.
        num_vectors (int): Number of vectors to encode the audio file into.

    Returns:
        numpy.ndarray: Array of Wav2Vec2 feature vectors.
    """

    if segment_length_ms is None == num_vectors is None:
        raise Exception("Must define exactly one of segment_length_ms and num_vectors")

    # Load audio file using torchaudio
    waveform, sampling_rate = torchaudio.load(audio_path)

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

    tensors = wave.to_tensors(waveform)

    if num_vectors is not None and num_vectors != len(tensors):
        raise Exception("Internal error, output number of vectors is incorrect")

    return tensors
