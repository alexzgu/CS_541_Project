from pathlib import Path

from kashi.subtitles import Segment, group_lines, read_csv, to_ass, to_srt, to_vtt, write_csv
from kashi.tokens import SILENCE

SEGS = [
    Segment(0.0, 1.0, SILENCE),
    Segment(1.0, 1.2, "か"),
    Segment(1.2, 1.45, "た"),
    Segment(1.45, 3.0, SILENCE),
    Segment(3.0, 3.3, "す", confidence=0.9),
]


def test_csv_roundtrip(tmp_path):
    path = write_csv(SEGS, tmp_path / "x.csv")
    back = read_csv(path)
    assert [s.token for s in back] == [s.token for s in SEGS]
    assert back[1].start == 1.0 and back[1].end == 1.2
    assert back[4].confidence == 0.9


def test_group_lines_breaks_on_silence():
    lines = group_lines(SEGS)
    assert [ln.text for ln in lines] == ["かた", "す"]
    assert lines[0].start == 1.0 and lines[0].end == 1.45


def test_writers():
    srt = to_srt(SEGS)
    assert "00:00:01,000 --> 00:00:01,450" in srt and "かた" in srt
    vtt = to_vtt(SEGS)
    assert vtt.startswith("WEBVTT") and "00:00:03.000" in vtt
    ass = to_ass(SEGS)
    assert "{\\k20}か" in ass and "{\\k25}た" in ass  # centisecond karaoke tags
    assert "Dialogue: 0,0:00:01.00,0:00:01.45,Karaoke" in ass


def test_display_lead_shifts_display_formats_only(tmp_path):
    from kashi.subtitles import read_csv, write_outputs

    out = write_outputs(SEGS, tmp_path, "x", ["csv", "srt"], display_lead_ms=100)
    srt = out["srt"].read_text()
    assert "00:00:00,900 --> 00:00:01,350" in srt  # 1.0-1.45 line leads by 100 ms
    back = read_csv(out["csv"])
    assert back[1].start == 1.0 and back[1].end == 1.2  # csv keeps true timings


def test_dag_skip_and_rerun(cfg, tmp_path):
    from kashi import dag

    src = tmp_path / "in.txt"
    out = tmp_path / "out.txt"
    src.write_text("v1")
    calls = []

    @dag.stage("_test", inputs=lambda c: [src], outputs=lambda c: [out],
               config_keys=["data.frame_ms"])
    def _test(c):
        calls.append(1)
        out.write_text(src.read_text())

    assert dag.run(cfg, "_test") == ["_test"]
    assert dag.run(cfg, "_test") == []          # fresh -> no-op
    src.write_text("v2-different")              # input changed -> stale
    assert dag.run(cfg, "_test") == ["_test"]
    assert len(calls) == 2
    del dag._STAGES["_test"]


def test_dag_training_barrier(cfg, tmp_path):
    from kashi import dag

    ran = []

    @dag.stage("_train", inputs=lambda c: [], outputs=lambda c: [tmp_path / "never"],
               is_training=True)
    def _train(c):
        ran.append(1)

    assert dag.run(cfg, "_train") == []
    assert dag.run(cfg, "_train", allow_train=True) == ["_train"]
    del dag._STAGES["_train"]
