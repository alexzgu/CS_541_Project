"""Corpus expansion (P2b): stage labels from the companion dataset repo.

Sources (all label-only; audio download is a separate, user-gated step):
  t1        jp_t1_dataset.zip      59 songs, kanji/kana per-mora
  ro        roumanji_dataset.zip   151 songs, romaji per-mora (57 overlap t1)
  t2-extra  dataset repo data/     the 27 T2 songs never cleaned locally

Staged under data/imported/<set>/subtitle_files/<youtube_id>.csv in the raw
schema (start,end,line,unformatted,token) + data/imported/index.tsv.
Admission into the main manifest happens only once audio exists and realign
QA passes (ROADMAP P2b). The 57 t1∩ro songs feed the dual-track consistency
check (romaji→kana mapping must reproduce the kana track).
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import pandas as pd

from ..tokens import ROMAJI_TO_KANA, katakana_to_hiragana, keep_kana
from . import manifest

SETS = {
    "t1": ("jp_t1_dataset.zip", "jp_t1_data"),
    "ro": ("roumanji_dataset.zip", "roumanji_data"),
}

RAW_COLS = ["start", "end", "line", "unformatted", "token"]


def _zip_prefix(zf: zipfile.ZipFile, want: str) -> str:
    roots = {n.split("/")[0] for n in zf.namelist() if "/" in n}
    if want in roots:
        return want
    if len(roots) == 1:
        return next(iter(roots))
    raise FileNotFoundError(f"cannot find {want} in zip (roots: {roots})")


def _read_index(zf: zipfile.ZipFile, prefix: str) -> dict[int, dict]:
    for cand in (f"{prefix}/final_dataset/index.tsv", f"{prefix}/indexed/index.tsv"):
        try:
            raw = zf.read(cand).decode("utf-8")
        except KeyError:
            continue
        rows = list(csv.DictReader(io.StringIO(raw), delimiter="\t"))
        return {int(r["Index"]): r for r in rows}
    raise FileNotFoundError(f"no index.tsv under {prefix}")


def _stage4_frames(zf: zipfile.ZipFile, prefix: str):
    for name in sorted(zf.namelist()):
        if f"{prefix}/stage_4_processed/" in name and name.endswith(".csv"):
            idx = int(Path(name).stem)
            with zf.open(name) as f:
                yield idx, pd.read_csv(f)


def _to_raw(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "unformatted" not in out.columns:
        out["unformatted"] = out["token"]
    out = out[[c for c in RAW_COLS if c in out.columns]]
    for c in RAW_COLS:
        if c not in out.columns:
            out[c] = -1
    out = out[RAW_COLS].dropna(subset=["token"])
    return out.sort_values(["start", "end"]).reset_index(drop=True)


def import_sets(cfg, sets: list[str]) -> dict:
    repo = cfg.path("dataset_repo")
    out_root = cfg.data_dir / "imported"
    known_vids = {r["ID"] for r in manifest.read_index(cfg)}
    staged_vids: dict[str, str] = {}  # vid -> set that staged it
    index_rows: list[dict] = []
    summary: dict = {}

    for set_name in sets:
        if set_name == "t2-extra":
            n = _import_t2_extra(cfg, repo, out_root, known_vids, index_rows)
            summary[set_name] = n
            continue
        zip_name, prefix_want = SETS[set_name]
        zpath = repo / zip_name
        if not zpath.is_file():
            print(f"[import] missing {zpath}, skipped")
            continue
        n_new = n_dual = n_dupe = 0
        with zipfile.ZipFile(zpath) as zf:
            prefix = _zip_prefix(zf, prefix_want)
            index = _read_index(zf, prefix)
            for idx, df in _stage4_frames(zf, prefix):
                meta = index.get(idx)
                if meta is None:
                    continue
                vid = meta["ID"]
                raw = _to_raw(df)
                if set_name == "ro" and (vid in known_vids or staged_vids.get(vid) == "t1"):
                    # romaji twin of an existing kana song -> dual-track data
                    dest = out_root / "ro_dual" / f"{vid}.csv"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    raw.to_csv(dest, index=False)
                    n_dual += 1
                    continue
                if vid in known_vids or vid in staged_vids:
                    n_dupe += 1
                    continue
                dest = out_root / set_name / "subtitle_files" / f"{vid}.csv"
                dest.parent.mkdir(parents=True, exist_ok=True)
                raw.to_csv(dest, index=False)
                staged_vids[vid] = set_name
                index_rows.append({"Set": set_name, "Title": meta.get("Title", ""),
                                   "ID": vid, "Language": meta.get("Language", "")})
                n_new += 1
        summary[set_name] = {"staged": n_new, "dual_track": n_dual, "duplicates": n_dupe}
        print(f"[import] {set_name}: staged {n_new}, dual-track {n_dual}, duplicate {n_dupe}")

    if index_rows:
        idx_file = out_root / "index.tsv"
        exists = idx_file.is_file()
        with open(idx_file, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["Set", "Title", "ID", "Language"], delimiter="\t")
            if not exists:
                w.writeheader()
            w.writerows(index_rows)
    return summary


def _import_t2_extra(cfg, repo: Path, out_root: Path, known_vids: set[str],
                     index_rows: list[dict]) -> dict:
    """The 27 T2 songs whose raw labels exist in the dataset repo but were
    never staged into this project."""
    src_index = repo / "data" / "indexed" / "index.tsv"
    stage4 = repo / "data" / "stage_4_processed"
    if not src_index.is_file() or not stage4.is_dir():
        print("[import] dataset repo working data/ not found for t2-extra")
        return {}
    have_local = set(manifest.labeled_ids(cfg)) | {
        int(p.stem) for p in (cfg.data_dir / "raw" / "subtitles" / "subtitle_files").glob("*.csv")
    }
    n = 0
    with open(src_index, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            idx = int(r["Index"])
            if idx in have_local:
                continue
            src = stage4 / f"{idx}.csv"
            if not src.is_file():
                continue
            raw = _to_raw(pd.read_csv(src))
            dest = out_root / "t2-extra" / "subtitle_files" / f"{r['ID']}.csv"
            dest.parent.mkdir(parents=True, exist_ok=True)
            raw.to_csv(dest, index=False)
            index_rows.append({"Set": "t2-extra", "Title": r.get("Title", ""),
                               "ID": r["ID"], "Language": r.get("Language", "")})
            n += 1
    print(f"[import] t2-extra: staged {n}")
    return {"staged": n}


# ---------------------------------------------------------------------------
# Romaji -> kana and the dual-track consistency check
# ---------------------------------------------------------------------------

import re as _re

_RO_ALIASES = {
    "si": "し", "ti": "ち", "tu": "つ", "hu": "ふ", "zi": "じ",
    "sya": "しゃ", "syu": "しゅ", "syo": "しょ",
    "tya": "ちゃ", "tyu": "ちゅ", "tyo": "ちょ",
    "zya": "じゃ", "zyu": "じゅ", "zyo": "じょ",
    "jya": "じゃ", "jyu": "じゅ", "jyo": "じょ",
    "o": "お", "wo": "を",
}
# roumanji encodes the sokuon as a bare consonant before the doubled mora
_SOKUON = {"k", "s", "t", "p", "c", "d", "g", "b", "z", "j", "f", "h", "m", "r", "w", "y"}
_PUNCT = _re.compile(r"[??!!..,、。「」『』()()~〜…・\"'’”\s]+")
_FURIGANA = _re.compile(r"^(.+?)[((](.+?)[))]$")
_SMALL = set("ゃゅょぁぃぅぇぉ")


_MORA_TABLE: dict[str, str] = {}
_MORA_TABLE.update(_RO_ALIASES)
_MORA_TABLE.update(ROMAJI_TO_KANA)          # canonical Hepburn (first wins upstream)
_MORA_TABLE.update({"la": "ら", "li": "り", "lu": "る", "le": "れ", "lo": "ろ",
                    "lya": "りゃ", "lyu": "りゅ", "lyo": "りょ", "-": "ー"})
_CONSONANTS = set("kstpcdgbzjfhmrwyl")
_SENTINEL = _re.compile(r"^<.*>$")


def romaji_to_morae(token: str) -> list[str] | None:
    """Greedy romaji -> mora list. [] for punctuation/sentinel tokens,
    None for unparseable (English/out-of-scope) tokens."""
    raw = str(token).strip().lower()
    if _SENTINEL.match(raw):
        return []
    t = _PUNCT.sub("", raw)
    if not t:
        return []
    out: list[str] = []
    i = 0
    while i < len(t):
        c = t[i]
        # sokuon: doubled consonant (incl. 'tch' as in 'matcha')
        if c in _CONSONANTS and c != "n" and i + 1 < len(t) and (
            t[i + 1] == c or (c == "t" and t[i + 1 : i + 3].startswith("ch"))
        ):
            out.append("っ")
            i += 1
            continue
        # moraic nasal: n before non-vowel/non-y, at end, or doubled
        if c == "n" and (i + 1 == len(t) or t[i + 1] not in "aiueoy"):
            out.append("ん")
            i += 1 + (i + 1 < len(t) and t[i + 1] == "n" and
                      (i + 2 == len(t) or t[i + 2] not in "aiueoy"))
            continue
        for L in (3, 2, 1):
            piece = t[i : i + L]
            if piece in _MORA_TABLE:
                out.append(_MORA_TABLE[piece])
                i += L
                break
        else:
            return None
    return out


def romaji_to_kana(token: str) -> str | None:
    """Single-token convenience wrapper: joined morae or None."""
    m = romaji_to_morae(token)
    return "".join(m) if m is not None else None


_VOWEL_OF = {"あ": "あ", "い": "い", "う": "う", "え": "え", "お": "お"}


def _long_vowel_expand(prev: str) -> str:
    """ー -> the vowel it prolongs (phonetic normal form, matches romaji 'rii')."""
    from ..tokens import decompose

    try:
        _, _, v = decompose(prev)
        return {"a": "あ", "i": "い", "u": "う", "e": "え", "o": "お"}.get(v, "ー")
    except KeyError:
        return "ー"


def _split_morae(reading: str) -> list[str]:
    """Kana string -> morae (small ゃゅょ attach; ー becomes its vowel)."""
    out: list[str] = []
    for ch in reading:
        if ch in _SMALL and out:
            out[-1] += ch
        elif ch == "ー" and out:
            out.append(_long_vowel_expand(out[-1]))
        else:
            out.append(ch)
    return out


def _row_reading(token: str) -> str:
    """Raw row token -> hiragana reading. Handles the t1 style 漢字(かな)
    (inline furigana), katakana, and punctuation; '' if no reading."""
    t = str(token).strip()
    m = _FURIGANA.match(t)
    if m:
        t = m.group(2)
    t = _PUNCT.sub("", katakana_to_hiragana(t))
    t = keep_kana(t)
    return t


def _kana_sequence(df: pd.DataFrame) -> list[str]:
    """Native-track mora sequence: per-row readings (furigana-aware) expanded
    to morae, majority hemisphere when two are present."""
    df = df.copy()
    df["reading"] = df["token"].map(_row_reading)
    df = df[df["reading"].str.len() > 0]
    top = df[(df["line"] >= 0) & (df["line"] < 50)]
    bot = df[(df["line"] >= 50) | (df["line"] == -1)]
    pick = top if len(top) > len(bot) else bot
    toks: list[str] = []
    for reading in pick.sort_values(["start", "end"])["reading"]:
        toks.extend(_split_morae(reading))
    return toks


# Orthography-vs-phonetics: kana subs write particles は/へ/を; romaji subs
# write their sounds wa/e/o. Phonetically identical — zero-cost in the check.
_PHON_SAME = {frozenset(p) for p in
              [("は", "わ"), ("へ", "え"), ("を", "お"), ("ぢ", "じ"), ("づ", "ず")]}


def _phonetically_same(a: str, b: str) -> bool:
    return a == b or frozenset((a, b)) in _PHON_SAME


def dual_track_report(cfg) -> dict:
    """Compare romaji→kana against the native kana track on the t1∩ro songs."""
    from ..eval.metrics import levenshtein

    out_root = cfg.data_dir / "imported"
    dual_dir = out_root / "ro_dual"
    per_song = {}
    if not dual_dir.is_dir():
        return {"error": "no dual-track data; run `kashi dataset import --sets t1,ro` first"}
    for ro_file in sorted(dual_dir.glob("*.csv")):
        vid = ro_file.stem
        kana_file = out_root / "t1" / "subtitle_files" / f"{vid}.csv"
        if not kana_file.is_file():
            continue
        kana_df = pd.read_csv(kana_file)
        # time ranges of latin-script (English) rows: excluded from BOTH sides
        latin = kana_df[kana_df["token"].map(
            lambda t: _row_reading(t) == "" and any(c.isascii() and c.isalpha() for c in str(t))
        )]
        latin_ranges = list(zip(latin["start"], latin["end"]))

        def in_latin(t0: float, t1: float) -> bool:
            mid = (t0 + t1) / 2
            return any(s - 0.05 <= mid <= e + 0.05 for s, e in latin_ranges)

        ro = pd.read_csv(ro_file)
        mapped, unmappable = [], 0
        for _, r in ro.sort_values(["start", "end"]).iterrows():
            if in_latin(float(r["start"]), float(r["end"])):
                continue
            m = romaji_to_morae(r["token"])
            if m is None:
                unmappable += 1
            else:
                mapped.extend(m)
        kana_df = kana_df[~kana_df.apply(
            lambda r: in_latin(float(r["start"]), float(r["end"])), axis=1)]
        kana = _kana_sequence(kana_df)
        d = levenshtein(mapped, kana, same=_phonetically_same)
        rate = d / max(1, len(kana))
        per_song[vid] = {"mismatch_rate": rate, "unmappable": unmappable,
                         "ro_tokens": len(mapped), "kana_tokens": len(kana)}
    if not per_song:
        return {"error": "no overlapping songs found"}
    import numpy as np

    rates = [v["mismatch_rate"] for v in per_song.values()]
    report = {
        "songs": len(per_song),
        "median_mismatch_rate": float(np.median(rates)),
        "mean_mismatch_rate": float(np.mean(rates)),
        "total_unmappable": sum(v["unmappable"] for v in per_song.values()),
        "per_song": per_song,
    }
    print(f"[dual-track] {report['songs']} songs, median mismatch "
          f"{report['median_mismatch_rate']:.3f}, mean {report['mean_mismatch_rate']:.3f}, "
          f"unmappable romaji tokens {report['total_unmappable']}")
    return report
