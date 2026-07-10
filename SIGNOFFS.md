# Sign-offs needed

Decisions only you can make. Everything else proceeds without you. Reply per ID
(e.g. "S1 delete, S2 yes, S5 later"). Defaults happen only after your OK.

## S1 — Delete legacy code (unblocks: repo hygiene; ~100+ MB and most of the old confusion)

Everything below is superseded by the `kashi` package (its replacement noted). None of it is referenced by current code or tests.

| Item | Superseded by |
|---|---|
| `data_preparation/preparing_raw_data/` (vendored UVR5-GUI + demucs trees) | pip `audio-separator` / `demucs` via `kashi separate` |
| `models/paper_based_approach/` (never functional) | — |
| `models/segment_break_detection/` (notebooks, dead CLI, constant-predictor ckpts, 72 MB tensors tarball) | `kashi/nn/segmenter.py` + baselines (c)/(d) |
| `data_processing/` scripts + `models/predict_syllables/*.py`, `models/wave2vec2/` | `kashi/data/`, `kashi/nn/`, `kashi/train/` (ports verified: acc 0.5319 repro) — **keep `models/predict_syllables/pretrained/`** (checkpoints still used) |
| `isolate_vocals/` incl. vendored `audio_separator/` + its venvs | `kashi separate` |
| `data_processing/container_files/` (SLURM), `separate_scripts/` EDA notebooks | — |
| root `Pipfile`, `requirements.txt`, `out.csv`, `.venv-cpu/`, `tmp/` | `pyproject.toml` + `.venv` |
| `src/kashi/**/__pycache__` bytecode of the deleted 07-09 scaffold | rebuilt package (this was its only record; now superseded) |

**Options:** (a) delete now · (b) move under `legacy/` now, delete after P3 fully closes · (c) leave as-is.
**Default on OK:** (a), one commit, `models/tensors/songs_20ms/` (11 GB, the 10 ms cache) kept until you decide separately.

## S2 — Download audio for 70 staged songs (unblocks: corpus 66 → ~163 labeled songs)

97 songs' labels are staged under `data/imported/`. **27 (`t2-extra`) need NO download** — their audio + separated vocals are already local; I'll admit them without asking further. The other **70 (t1: 59, ro-unique: 11)** have labels but no audio anywhere on disk; getting it means `yt-dlp` from YouTube (~500 MB, one command, some videos may be delisted — those get dropped).

**Question:** OK to download these 70 via yt-dlp? (This is the YouTube-ToS comfort question — same method the original dataset used.)

## S3 — Scraping NEW data (unblocks: P5 pseudo-labeling pool; more labeled songs later)

Beyond S2: fetching content we have no labels for — (a) ~100+ unlabeled JP songs (audio only) for the self-training loop, (b) optionally new karaoke-subtitled playlists (labels + audio) via `kashi dataset scrape`.

**Question:** yes/no, and any volume cap or channel preference. Can wait until P5 starts.

## S4 — Romaji-labeled songs as first-class training data

The ro-set labels are phonetic (wa/e/o where kana writes は/へ/を). We transcribe *sound*, and the phonetic kernel already treats these pairs as near-identical (0.95), so mixing them in is consistent.
**Recommendation: yes.** Alternative: keep them a separate track used only for the dual-track consistency check.

## S5 — Your ears: gold windows on the test split (~2–4 h total, whenever)

Gold currently covers only train-side songs (0/6/16/19, from your old hand-corrected files) — final reporting needs verified *test-side* references. Per song (suggest 81, 85, 90, 92):

```
kashi gold export 81          # writes a prefilled Audacity label track + prints the exact import command
# open the song + label track in Audacity, fix boundaries/tokens/breaths by ear
kashi gold import 81 <file> --window-start <s> --window-end <e>
```

## S6 — First git commit

Everything (package, tests, configs, docs, CI, recovered `google drive/` notebooks) is uncommitted. Proposed: commit `src/ tests/ configs/ docs/ .github/ pyproject.toml README.md ROADMAP.md SIGNOFFS.md data/gold/ data/imported/ google drive/` and add `data/clean_v2/`, `data/staging/` to `.gitignore` (regenerable). Caches (`artifacts/`, `runs/`, venvs) are already ignored.
**Question:** OK to commit? Include `google drive/` and `data/imported/` (few MB) or ignore them?

## S7 — Cloud GPU budget (only gates P5.4)

Continued SSL pretraining on singing / larger encoders need ≥24 GB VRAM. Everything else runs on your 6 GB laptop.
**Default: skip — stay local.** Say the word only if you want to rent GPU time.
