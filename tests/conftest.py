from pathlib import Path

import pytest

from kashi.config import Config

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def cfg(tmp_path):
    """Default config with data/artifacts/runs redirected into tmp."""
    return Config.load(overrides=[
        f"paths.data_dir={tmp_path / 'data'}",
        f"paths.artifacts_dir={tmp_path / 'artifacts'}",
        f"paths.runs_dir={tmp_path / 'runs'}",
    ])


@pytest.fixture()
def repo_cfg():
    """Config pointing at the real repo data (read-only use in tests)."""
    return Config.load()


@pytest.fixture()
def tone_wav(tmp_path):
    """10 s fixture: three voiced tone bursts separated by silence, 16 kHz mono."""
    import numpy as np
    import soundfile as sf

    sr = 16000
    t = np.arange(sr) / sr
    burst = (0.4 * np.sin(2 * np.pi * 220 * t) * np.hanning(sr)).astype(np.float32)
    silence = np.zeros(sr, dtype=np.float32)
    wave = np.concatenate([silence, burst, silence, burst, silence, burst,
                           silence, silence, burst, silence])
    path = tmp_path / "clip.wav"
    sf.write(path, wave, sr)
    return path
