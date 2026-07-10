import pytest

from kashi.config import Config
from kashi.registry import create, names, register


def test_layering_and_overrides(tmp_path):
    user = tmp_path / "user.toml"
    user.write_text("[data]\nframe_ms = 10\n")
    cfg = Config.load(user, overrides=["segmenter.energy.top_db=25.5", "pipeline.separator=uvr"])
    assert cfg.frame_ms == 10
    assert cfg["segmenter.energy.top_db"] == 25.5
    assert cfg["pipeline.separator"] == "uvr"
    # untouched defaults survive
    assert cfg["classifier.lstm.hidden_size"] == 144


def test_set_parsing():
    cfg = Config.load(overrides=[
        "qa.quarantine=false",
        "eval.tolerances_ms=10,30",
        "train.classifier.lr=0.01",
    ])
    assert cfg["qa.quarantine"] is False
    assert cfg["eval.tolerances_ms"] == [10, 30]
    assert cfg["train.classifier.lr"] == 0.01


def test_dump_roundtrip(tmp_path):
    import tomllib

    cfg = Config.load(overrides=["pipeline.encoder=mel"])
    parsed = tomllib.loads(cfg.dump_toml())
    assert parsed["pipeline"]["encoder"] == "mel"
    assert parsed["data"]["frame_ms"] == 20


def test_registry_selection(cfg):
    assert "energy" in names("segmenter")
    seg = create(cfg, "segmenter", "energy")
    assert seg.min_frames >= 1
    with pytest.raises(KeyError, match="No segmenter named"):
        create(cfg, "segmenter", "nope")


def test_registry_swap_via_config(cfg):
    from kashi.components.classifiers import SilenceOnlyClassifier

    cfg2 = Config.load(overrides=["pipeline.classifier=silence_only"])
    assert isinstance(create(cfg2, "classifier"), SilenceOnlyClassifier)


def test_custom_registration(cfg):
    @register("classifier", "_test_dummy")
    class Dummy:
        def __init__(self, cfg):
            pass

    assert "_test_dummy" in names("classifier")
    assert isinstance(create(cfg, "classifier", "_test_dummy"), Dummy)
