"""Web app smoke: upload the tone fixture, poll to completion, download files.
Runs the CPU-only two_stage config (mel encoder, silence_only classifier)."""

import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from kashi.config import Config  # noqa: E402
from kashi.web.app import create_app  # noqa: E402


@pytest.fixture()
def client(tmp_path):
    import torch

    torch.manual_seed(0)  # untrained LSTM: deterministic non-silence output
    cfg = Config.load(overrides=[
        f"paths.data_dir={tmp_path / 'data'}",
        f"paths.artifacts_dir={tmp_path / 'artifacts'}",
        f"paths.runs_dir={tmp_path / 'runs'}",
        "pipeline.mode=two_stage",
        "pipeline.separator=none",
        "pipeline.encoder=mel",
        "pipeline.segmenter=energy",
        "pipeline.classifier=lstm",
        "classifier.lstm.checkpoint=",
        "classifier.lstm.input_size=64",
    ])
    return TestClient(create_app(cfg))


def test_upload_poll_download(client, tone_wav):
    with open(tone_wav, "rb") as f:
        res = client.post("/jobs", files={"file": ("clip.wav", f, "audio/wav")})
    assert res.status_code == 200
    job_id = res.json()["job_id"]

    for _ in range(120):
        state = client.get(f"/jobs/{job_id}").json()
        if state["state"] in ("done", "error"):
            break
        time.sleep(0.25)
    assert state["state"] == "done", state.get("error")
    assert state["frac"] == 1.0
    assert "encode" in state["timings"]

    for fmt in ("srt", "vtt", "ass", "csv"):
        r = client.get(f"/jobs/{job_id}/files/{fmt}")
        assert r.status_code == 200 and len(r.content) > 0
    assert client.get(f"/jobs/{job_id}/media").status_code == 200
    assert client.get(f"/jobs/{job_id}/files/nope").status_code == 404
    assert client.get("/jobs/deadbeef").status_code == 404


def test_index_served(client):
    res = client.get("/")
    assert res.status_code == 200 and "kashi" in res.text
