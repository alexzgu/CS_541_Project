# S12 — How the label-timing offsets were measured

**Claim under review:** some songs' reference subtitles are shifted by a constant
number of milliseconds relative to the audio. **Proposed fix:** shift every row of
an affected song by one constant. This document gives the complete measurement,
every step, with the exact parameters used. No model is involved anywhere — the
measurement cannot be circular with the CTC transcriber we are evaluating.

Code: `src/kashi/data/offset_audit.py` (measurement) and `s12_review/make_review.py`
(this package). Reproduce with:

```bash
.venv/bin/python -m kashi.data.offset_audit          # all 93 songs -> runs/ref_offset_full.{txt,json}
.venv/bin/python s12_review/make_review.py           # this folder's figures/CSVs/data.js
```

## Inputs

- **Audio**: the separated vocal stem of each song, `data/clean/audio/vocals/<id>.mp3`,
  resampled to 16 kHz mono. Vocals (not the full mix) so that drums and bass cannot
  create onsets that have nothing to do with the lyrics.
- **Labels**: the reference subtitles `data/clean/subtitles/<id>.csv`
  (`start,end,token,...` rows). Only rows that the evaluation itself uses count:
  `<silence>` rows and `exclude=true` rows are ignored.
- **Calibration set**: the four hand-corrected gold songs {0, 6, 16, 19}
  (`data/gold/subtitles/`), whose timing we trust by construction.

## Step 1 — acoustic onset envelope (where does vocal energy *appear*?)

Not a mel spectrogram, but close in spirit — a **linear-frequency STFT** feeds a
standard *spectral flux* onset detector (mel-weighting would only change bin
weights; flux is summed over bins anyway):

1. STFT with a 512-sample Hann window (= 32 ms at 16 kHz) and a 160-sample hop
   (= 10 ms). 10 ms is the audit's time resolution.
2. Log-compress the magnitudes: `M = log1p(50 · |STFT|)`. The `log1p` compresses
   dynamic range so quiet syllables still register; the 50 sets the knee.
3. Per frequency bin, take the first difference over time and keep only increases:
   `F(f, t) = max(0, M(f, t) − M(f, t−1))`. Energy *appearing* is an onset cue;
   energy decaying is not.
4. Sum over frequency: `env(t) = Σ_f F(f, t)`, then standardize (zero mean, unit
   variance). `env` peaks when the singer starts a new sound.

## Step 2 — label onset train (where do the *labels* say sounds start?)

For every counted row, place a Gaussian bump (σ = 20 ms) at its `start` time on
the same 10 ms grid, sum, standardize. The bump width makes the correlation in
step 3 tolerant of ±tens-of-ms per-row jitter while still localizing the peak.

## Step 3 — cross-correlation (how far apart are the two?)

Slide the label train against the envelope over lags **−200 … +200 ms** and
compute the normalized dot product at each lag. If the labels are perfectly
placed, the correlation peaks at lag ≈ 0 (up to the instrument bias below). Sign
convention: **a peak at negative lag means the labels sit EARLIER than the
audio events they describe.**

The **peak lag** is the song's raw offset. The **sharpness**
`(corr[peak] − median(corr)) / std(corr)` says how unambiguous the peak is —
a clean song reads 2–3; a flat, unreliable curve reads < 1.

## Step 4 — instrument calibration on gold (what does "correct" read?)

Spectral flux cannot peak at the true onset: the 32 ms analysis window has to
fill with the new sound before the magnitude rises, so flux peaks 20–40 ms
*after* the true onset. Consequently even perfect labels read *negative*
(early) on this instrument. Measured on the four gold songs:
**−40, −40, −40, −30 ms → baseline −40 ms.** Every reading below has this
baseline subtracted; gold therefore reads ≈ 0 by construction, and what remains
for other songs is genuine label fault.

## Step 5 — decision rule

A song is corrected iff **|fault| ≥ 50 ms AND sharpness ≥ 1.5**. The correction
shifts every row (`start` and `end`) later by `−fault` (all measured faults are
"early", so shifts are positive). 50 ms is the evaluation's own timing tolerance;
the sharpness floor rejects unreliable measurements rather than applying them.

## Results (gold-calibrated fault, ms; negative = labels early)

Corpus (93 songs): median −30 ms, IQR [−40, −30] — v1 labels run slightly early
across the board (left alone; within the instrument's noise and the eval
tolerance). **17 songs cross the correction rule:**

| set | song: fault |
|---|---|
| test | 81: −50 · 89: −130 · 90: −90 · 91: −70 |
| test, unchanged | 83: −30 · 85: −30 · 92: 0 |
| train | 2: −50 · 12: −50 · 13: −90 · 14: −90 · 15: −50 · 39: −90 · 47: −60 · 57: −60 · 60: −50 · 70: −70 · 80: −50 · 82: −90 · 87: −70 |
| rejected measurement | 88: reads +170 but sharpness 0.6 (flat curve; the song has 2,356 label rows — ~4× normal density — so bumps blanket the timeline and no lag stands out). Left unchanged; flagged for manual inspection. |

Full table: `runs/ref_offset_full.txt` / `.json`.

## Independent corroboration (two other lines of evidence agree)

1. **Onset-shift sweep** (model-side, done first): advancing the CTC decoder's
   predicted onsets improves agreement with *test* references up to −4 frames
   (−80 ms) but with *gold* only to −1 frame (−20 ms). Model vs gold says the
   model is ~20 ms late; test needing 80 ms therefore says the test refs are
   ~60 ms early on average — matching this audit's test median (−50), and
   per-song: 92 (fault 0) preferred the gold-like shift in the sweep too, while
   89 (−130) was broken at every shift.
2. **Batch control**: original-batch train songs and t2-extra train songs show
   the same distribution — so this is per-video source-subtitle variance, not an
   artifact of the 2026-07-09 import (that hypothesis was tested and refuted).

## What this correction does NOT do

- It does not touch per-row jitter (rows individually off in both directions) —
  only the constant per-song component. Gold-gated snapping for jitter was
  tested earlier in the project and rejected; constant shifts are a different,
  structure-preserving transform.
- It does not change tokens, so SER and all transcript metrics are unaffected.
  Only timed-token F1 / boundary metrics re-baseline.
- Nothing in `data/clean` changes until S12-final approval. The corrected files
  staged here live in `s12_review/subtitles_corrected/` only.

## How to review (this folder)

Open `index.html` (any browser; if the video doesn't load from `file://`, run
`python -m http.server` in this folder and open `http://localhost:8000`). Pick a
song — the video plays with the **uncorrected** (blue) and **corrected** (aqua)
karaoke tracks side by side. Use the *vocals only* source and 0.5×/0.75× speed
to judge ±50–130 ms by ear; click any token to jump to it. Each song's page also
shows the measurement figure: the correlation curve with its peak, and an
8-second excerpt where you can see the blue label ticks sitting left of the gray
energy peaks and the aqua ticks sitting on them.
