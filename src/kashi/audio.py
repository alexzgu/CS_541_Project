"""Audio I/O. Every input (mp4, mp3, webm, wav, ...) is routed through ffmpeg
into 16 kHz mono float32, so the rest of the pipeline only ever sees one format."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np


def ffmpeg_to_wav(input_path: str | Path, out_path: str | Path, sr: int = 16000) -> Path:
    """Extract/convert any media file (incl. mp4 video) to mono PCM wav."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(input_path),
        "-vn", "-ac", "1", "-ar", str(sr),
        "-f", "wav", str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def load_audio(path: str | Path, sr: int = 16000) -> np.ndarray:
    """Load any audio/video file as mono float32 at `sr` (via soundfile, ffmpeg fallback)."""
    import soundfile as sf

    path = Path(path)
    try:
        wave, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
        wave = wave.mean(axis=1)
        if file_sr != sr:
            raise ValueError("resample needed")
        return wave
    except Exception:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "audio.wav"
            ffmpeg_to_wav(path, tmp, sr=sr)
            wave, _ = sf.read(str(tmp), dtype="float32", always_2d=True)
            return wave.mean(axis=1)


def duration_s(path: str | Path) -> float:
    """Media duration in seconds via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return float(out)


def rms_per_frame(wave: np.ndarray, sr: int, frame_ms: int) -> np.ndarray:
    """Root-mean-square energy per analysis frame, shape [T]."""
    hop = int(sr * frame_ms / 1000)
    n = len(wave) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    frames = wave[: n * hop].reshape(n, hop)
    return np.sqrt((frames**2).mean(axis=1) + 1e-12).astype(np.float32)


def log_rms_db(wave: np.ndarray, sr: int, frame_ms: int) -> np.ndarray:
    """Per-frame energy in dB relative to full scale."""
    return (20.0 * np.log10(rms_per_frame(wave, sr, frame_ms) + 1e-12)).astype(np.float32)


def content_key(path: str | Path) -> str:
    """Stable cache key: dataset songs keep their numeric id, anything else
    hashes path+size+mtime."""
    path = Path(path)
    if path.stem.isdigit():
        return path.stem
    st = path.stat()
    payload = f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()
    return hashlib.sha1(payload).hexdigest()[:16]
