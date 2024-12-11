#!/usr/bin/env python3

"""
Script Name: preprocess.py

Description:
    This script processes an input audio file and its corresponding CSV label file
    to generate a preprocessed NumPy array (input.npy) suitable for model inference.

Usage:
    python preprocess.py --audio /path/to/audio.mp3 \
                         --csv /path/to/labels.csv \
                         --output /path/to/output/input.npy \
                         [--interval_width 20] \
                         [--n_mels 32] \
                         [--debug]

Arguments:
    --audio            Path to the input audio file (e.g., MP3, WAV).
    --csv              Path to the corresponding CSV label file.
    --output           Path to save the generated input.npy file.
    --interval_width   (Optional) Width of the fixed intervals in ms. Default is 20 ms.
    --n_mels           (Optional) Number of mel bands. Default is 32.
    --debug            (Optional) If set, enables debug mode with verbose output.
"""

import argparse
import os
import torch
import torchaudio
import pandas as pd
import numpy as np


def compute_spectrogram(waveform: torch.Tensor, sr: int, interval_width: int, n_mels: int) -> torch.Tensor:
    """
    Compute mel spectrogram from waveform.

    Args:
        waveform (torch.Tensor): The audio waveform tensor.
        sr (int): Sampling rate of the audio.
        interval_width (int): Width of each interval in milliseconds.
        n_mels (int): Number of mel bands.

    Returns:
        torch.Tensor: Normalized mel spectrogram with shape [n_frames, n_mels].
    """
    # Convert to mono by averaging channels if necessary
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Compute mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        hop_length=int(sr * interval_width / 1000),
        n_mels=n_mels
    )
    mel_spec = mel_transform(waveform)  # Shape: [1, n_mels, n_frames]

    # Convert to log scale
    mel_spec = torch.log(mel_spec + 1e-9)  # Prevent log(0)

    # Normalize
    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)

    # Transpose to shape [n_frames, n_mels]
    mel_spec = mel_spec.squeeze(0).transpose(0, 1)

    return mel_spec


def process_files(audio_path: str, csv_path: str, output_path: str, interval_width: int, n_mels: int, debug: bool):
    """
    Process the audio and CSV files to generate input.npy.

    Args:
        audio_path (str): Path to the audio file.
        csv_path (str): Path to the CSV label file.
        output_path (str): Path to save the generated input.npy.
        interval_width (int): Width of each interval in milliseconds.
        n_mels (int): Number of mel bands.
        debug (bool): If True, print debug statements.
    """
    if debug:
        print(f"Loading audio file from: {audio_path}")
    waveform, sr = torchaudio.load(audio_path)

    if debug:
        print(f"Audio waveform shape: {waveform.shape}")
        print(f"Sampling rate: {sr}")

    if debug:
        print("Computing mel spectrogram...")
    mel_spec = compute_spectrogram(waveform, sr, interval_width, n_mels)

    if debug:
        print(f"Mel spectrogram shape (n_frames, n_mels): {mel_spec.shape}")

    if debug:
        print(f"Loading CSV labels from: {csv_path}")
    df = pd.read_csv(csv_path)
    labels = torch.tensor(df['break'].values, dtype=torch.bool)

    if debug:
        print(f"Labels shape: {labels.shape}")
        print(f"Labels: {labels}")

    # Identify the last occurrence of a break
    if labels.any():
        last_break = labels.nonzero(as_tuple=True)[0][-1]
        if debug:
            print(f"Last break found at index: {last_break}")
    else:
        last_break = len(labels) - 1  # No breaks, consider entire range
        if debug:
            print("No breaks found in labels. Using entire range.")

    # Truncate mel_spec and labels to last_break + 2
    truncate_idx = last_break + 2
    mel_spec_truncated = mel_spec[:truncate_idx, :]
    labels_truncated = labels[:truncate_idx]

    if debug:
        print(f"Truncated mel spectrogram shape: {mel_spec_truncated.shape}")
        print(f"Truncated labels shape: {labels_truncated.shape}")

    # Prepare data for saving
    # Depending on model requirements, you might need to combine mel_spec and labels
    # For inference, labels are typically not needed. Here, we'll save only mel_spec.

    if debug:
        print(f"Saving preprocessed data to: {output_path}")

    # Convert to NumPy array
    mel_spec_np = mel_spec_truncated.numpy()

    # Save to input.npy
    np.save(output_path, mel_spec_np)

    if debug:
        print("Preprocessing completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio and CSV to generate input.npy for inference.")
    parser.add_argument('--audio', type=str, required=True, help='Path to the input audio file (e.g., MP3, WAV).')
    parser.add_argument('--csv', type=str, required=True, help='Path to the corresponding CSV label file.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the output input.npy file.')
    parser.add_argument('--interval_width', type=int, default=20,
                        help='Width of fixed intervals in ms. Default is 20 ms.')
    parser.add_argument('--n_mels', type=int, default=32, help='Number of mel bands. Default is 32.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with verbose output.')

    args = parser.parse_args()

    # Validate input paths
    if not os.path.isfile(args.audio):
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Process the files
    process_files(
        audio_path=args.audio,
        csv_path=args.csv,
        output_path=args.output,
        interval_width=args.interval_width,
        n_mels=args.n_mels,
        debug=args.debug
    )


if __name__ == "__main__":
    main()
