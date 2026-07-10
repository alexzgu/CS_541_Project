"""CPU smoke test: 10 s synthetic clip through the full two_stage pipeline
(mel encoder, no separator, energy segmenter, untrained LSTM classifier) —
no network, no GPU, no checkpoints."""

from kashi.config import Config
from kashi.pipeline import transcribe


def test_transcribe_smoke(tmp_path, tone_wav):
    import torch

    torch.manual_seed(0)  # untrained classifier: deterministic init
    cfg = Config.load(overrides=[
        f"paths.data_dir={tmp_path / 'data'}",
        f"paths.artifacts_dir={tmp_path / 'artifacts'}",
        f"paths.runs_dir={tmp_path / 'runs'}",
        "pipeline.mode=two_stage",
        "pipeline.separator=none",
        "pipeline.encoder=mel",
        "pipeline.segmenter=energy",
        "pipeline.classifier=lstm",
        "classifier.lstm.checkpoint=",          # untrained weights: wiring only
        "classifier.lstm.input_size=64",        # mel dim
    ])
    stages = []
    result = transcribe(
        cfg, tone_wav, out_dir=tmp_path / "out",
        formats=["srt", "vtt", "ass", "csv"],
        progress=lambda stage, frac: stages.append(stage),
    )
    # all four outputs written and non-empty
    for fmt in ("srt", "vtt", "ass", "csv"):
        path = result.out_files[fmt]
        assert path.is_file() and path.stat().st_size > 0
    # the three tone bursts produce non-silence segments
    lyric = [s for s in result.segments if not s.is_silence]
    assert len(lyric) >= 3
    # contiguous coverage from 0 to ~10 s
    assert result.segments[0].start == 0.0
    assert abs(result.segments[-1].end - 10.0) < 0.1
    for a, b in zip(result.segments[:-1], result.segments[1:]):
        assert abs(a.end - b.start) < 1e-6
    assert "extract-audio" in result.timings and "encode" in result.timings
    assert stages[-1] == "done"


def test_transcribe_silence_only_classifier(tmp_path, tone_wav):
    cfg = Config.load(overrides=[
        f"paths.data_dir={tmp_path / 'data'}",
        f"paths.artifacts_dir={tmp_path / 'artifacts'}",
        f"paths.runs_dir={tmp_path / 'runs'}",
        "pipeline.mode=two_stage",
        "pipeline.separator=none",
        "pipeline.encoder=mel",
        "pipeline.segmenter=energy",
        "pipeline.classifier=silence_only",
    ])
    result = transcribe(cfg, tone_wav, out_dir=tmp_path / "out2", formats=["csv"])
    assert all(s.is_silence for s in result.segments)
