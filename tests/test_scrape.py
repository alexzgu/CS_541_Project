"""VTT parser differential test: Python port must be byte-identical to the
Rust oracle's committed outputs (karaoke_subtitle_dataset/data/parsed)."""

from pathlib import Path

import pytest

from kashi.data.scrape import parse_vtt_to_csv

DS = Path("/home/eizigi/Documents/GitHub/karaoke_subtitle_dataset/data")

pytestmark = pytest.mark.skipif(
    not (DS / "indexed" / "vtts").is_dir(), reason="dataset repo not present"
)


@pytest.mark.parametrize("song_id", [0, 7, 45, 92])
def test_byte_identical_to_rust_oracle(tmp_path, song_id):
    vtt = DS / "indexed" / "vtts" / f"{song_id}.vtt"
    oracle = DS / "parsed" / f"{song_id}.csv"
    if not vtt.is_file() or not oracle.is_file():
        pytest.skip("song not present")
    mine = parse_vtt_to_csv(vtt, tmp_path / f"{song_id}.csv")
    assert mine.read_bytes() == oracle.read_bytes()
