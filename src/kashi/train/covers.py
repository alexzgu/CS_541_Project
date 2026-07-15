"""S15 covers: cross-cover agreement-filtered pseudo-transcripts.

Many singers cover the same song, so the LYRICS agree across a playlist even
though tempo, key, and intros/outros differ. Spike-decode every cover, then
keep only crops whose token n-grams are reproduced by other covers of the same
song: n-grams are time-free, so tempo differences don't matter, and per-cover
intros/outros/hallucinations have no cross-cover support and are dropped.

Support(crop) = fraction of its token 3-grams that appear in >= MIN_OTHERS
other covers of the group. Kept crops land in the same manifest format as
`kashi loop ctc-harvest`, so the training notebook ingests them unchanged
(PSEUDO_DIR flag).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import audio as audio_mod
from ..registry import create
from ..tokens import SILENCE_ID
from .pseudo import _test_ytids, spikes_to_crops

NGRAM = 3
MIN_OTHERS = 2       # a gram counts as supported if >= this many OTHER covers have it


def grams_of(tokens: list[int], n: int = NGRAM) -> set[tuple[int, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def support_index(sequences: dict[str, list[int]], n: int = NGRAM) -> dict[tuple, set[str]]:
    """gram -> set of cover ids containing it (full-song decoded sequences)."""
    idx: dict[tuple, set[str]] = defaultdict(set)
    for cid, seq in sequences.items():
        for g in grams_of(seq, n):
            idx[g].add(cid)
    return idx


def crop_support(tokens: list[int], cover_id: str, idx: dict[tuple, set[str]],
                 n: int = NGRAM, min_others: int = MIN_OTHERS) -> float:
    gs = grams_of(tokens, n)
    if not gs:
        return 0.0
    ok = sum(1 for g in gs if len(idx.get(g, set()) - {cover_id}) >= min_others)
    return ok / len(gs)


def _groups(covers_src: Path, sep_dir: Path) -> dict[str, list[tuple[str, Path]]]:
    """playlist -> [(stem, separated vocals path)] via the download layout."""
    out: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for f in sorted(sep_dir.glob("*/vocals.mp3")):
        stem = f.parent.name
        hits = list(covers_src.glob(f"*/{glob_escape(stem)}.*"))
        group = hits[0].parent.name if hits else "ungrouped"
        out[group].append((stem, f))
    return out


def glob_escape(s: str) -> str:
    return re.sub(r"([\[\]?*])", r"[\1]", s)


def harvest_covers(cfg, sep_dir: str | Path = "data/unlabeled/covers_sep/htdemucs",
                   covers_src: str | Path = "data/unlabeled/covers",
                   out_dir: str | Path | None = None,
                   min_support: float = 0.5) -> dict:
    """Decode every separated cover, build per-group support, write crops that
    pass the agreement filter. Same output format as harvest_ctc."""
    sep_dir, covers_src = Path(sep_dir), Path(covers_src)
    out_dir = Path(out_dir or cfg.artifacts_dir / "pseudo_covers")
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)

    leak = _test_ytids(cfg)
    decoder = create(cfg, "decoder")
    if getattr(decoder, "emissions", "") != "ctc":
        raise SystemExit("harvest_covers needs decoder.segmental.emissions = 'ctc'")
    frame_s = cfg.frame_ms / 1000.0
    sr = cfg.sample_rate

    groups = _groups(covers_src, sep_dir)
    manifest = out_dir / "manifest.jsonl"
    report: dict = {"groups": {}, "min_support": min_support}
    kept = tot = 0
    hours = 0.0
    for group, items in sorted(groups.items()):
        seqs: dict[str, list[int]] = {}
        crops_by_cover: dict[str, tuple[np.ndarray, list[dict]]] = {}
        for stem, f in items:
            m = re.search(r"\[([A-Za-z0-9_-]{11})\]", stem)
            if m and m.group(1) in leak:
                print(f"[covers] LEAK EXCLUDED: {stem}")
                continue
            try:
                wave = audio_mod.load_audio(f, sr=sr)
                T = max(1, int(len(wave) / sr / frame_s))
                logp = decoder._ctc_log_probs(wave, sr, T)
                path = logp.argmax(-1)
                probs = np.exp(logp[np.arange(len(path)), path])
            except Exception as e:  # noqa: BLE001
                print(f"[covers] {stem}: FAILED ({e})")
                continue
            spikes = [int(c) for c in path[np.insert(np.diff(path) != 0, 0, True)]
                      if c != SILENCE_ID]
            seqs[stem] = spikes
            crops_by_cover[stem] = (wave, spikes_to_crops(path, probs, frame_s,
                                                          blank_id=SILENCE_ID))
        idx = support_index(seqs)
        g_kept = g_tot = 0
        with open(manifest, "a") as mf:
            for stem, (wave, crops) in crops_by_cover.items():
                for k, c in enumerate(crops):
                    g_tot += 1
                    sup = crop_support(c["tokens"], stem, idx)
                    if sup < min_support:
                        continue
                    rel = f"crops/{group}_{stem}_{k}.npy"
                    np.save(out_dir / rel,
                            wave[int(c["t0"] * sr): int(c["t1"] * sr)].astype(np.float16))
                    mf.write(json.dumps({"file": rel, "song": f"{group}/{stem}",
                                         "support": round(sup, 3), **c}) + "\n")
                    g_kept += 1
                    hours += c["dur_s"] / 3600
        report["groups"][group] = {"covers": len(seqs), "crops": g_tot, "kept": g_kept}
        kept += g_kept
        tot += g_tot
        print(f"[covers] {group}: {len(seqs)} covers, {g_kept}/{g_tot} crops pass "
              f"support>={min_support}", flush=True)
    report.update({"kept": kept, "total": tot, "hours": round(hours, 2)})
    (out_dir / "harvest_covers.json").write_text(json.dumps(report, indent=1))
    print(f"[covers] done: {kept}/{tot} crops ({hours:.2f} h) -> {out_dir}")
    return report
