# kashi — project summary (as of 2026-07-16)

Textless syllable-level transcription of Japanese songs into timed hiragana
karaoke subtitles. No transcript is consumed at inference; forced alignment is
permitted as a training bootstrap only (S9). Hardware: one RTX A3000 Laptop
(6 GB), 16 GB RAM; heavy jobs proved to fit locally.

## Headline

Frozen-test (7 songs, never tuned on): **SER 0.797 → 0.252, timed-token F1
0.106 → 0.725** since the revival began (2026-07-09). Roughly half the timing
gain came from fixing the labels, not the model — measured, not assumed.

## System

`src/kashi/` package: registry components (separator/encoder/segmenter/
classifier/decoder) selected by TOML config (`--set` overrides), DAG staging,
CLI (`kashi dataset|separate|encode|train|fit|eval|transcribe|realign|discover|
loop|colab|serve`), FastAPI web app (mp4 → SRT/VTT/ASS/CSV). 76 tests + CI.

Production pipeline: ffmpeg 16 kHz mono → (optional separator) → HuBERT-CTC
emissions → **spike decoding** (greedy peaky CTC; onset = spike − 1 frame;
extent capped at next onset / 0.6 s) → silence gap-fill → writers.
`subtitles.display_lead_ms = 100` shifts srt/vtt/ass earlier at write time
(viewers perceive leading subtitles as in-time); csv/eval keep true timings.

Token inventory: 110 (109 morae incl. youon + ふぁ/ふぃ/ふぇ/ふぉ/でぃ;
`<silence>` = CTC blank = index 109). No っ token — its time folds into the
host mora (v2 policy).

## Evaluation protocol

- Frozen test `PAPER_TEST_IDS = (81,83,85,89,90,91,92)`; train = other 86.
- Metrics: SER (Levenshtein), timed-token F1 (@50 ms), boundary F1@20/50 ms,
  phonetic partial credit. `runs/leaderboard.csv` is the append-only record.
- **Adopt-only-if-better** on SER AND timed-F1; every promotion shipped with a
  control (equal-compute labeled-only run, round-2 self-training, α=0 twin).
- Hyperparameters are tuned on gold + train samples, never on test (the one
  test-tuned knob we almost shipped — onset shift −4 — turned out to be
  fitting label noise; gold caught it).
- Gold subset (hand-corrected songs 0/6/16/19) arbitrates label-quality
  questions; it rejected boundary-snapping three times and settled S11/S12.

## Dataset (93 labeled songs; corpus was 66)

- **S12 timing audit**: model-free measurement (spectral-flux onset envelope ×
  Gaussian label-onset train, cross-correlated; gold calibrates the instrument
  at −40 ms) found per-song constant offsets 30–130 ms EARLY on 18 songs;
  corroborated independently by the decoder onset-shift sweep. Applied with
  backups; timed-F1 jumped 0.314 → 0.699 with SER unchanged — song 89 went
  0.017 → 0.906 (the model had been right; the labels were 130 ms off).
- **De-styling**: karaoke color-fade animation frames (uniform 67 ms rows,
  runs to 89) merged in 7 songs; song 88: 3,264 → 422 rows. Genuine doubles
  (いい) protected by a duration criterion, not run length.
- **clean_v2 normalization** (adopted): sokuon merge + decomposable chunk
  split (ない→な+い) recovered 4,290 of 6,131 excluded rows; champion SER on
  the fairer refs: 0.316 → 0.263 (it was already predicting those tokens).
- **Tier-2 pilot** (`s13_pilot/`, awaiting review): English transcribed
  as-sung in loan-kana (song 23: +660 morae; song 21: +281); no inventory
  change needed (two-mora expansions for missing うぃ/てぃ). Policy: no
  model-assisted drafting on test songs (contamination).
- Leak guard: pool/covers are YouTube-id-checked against the frozen test
  (the pool really did contain test song 89's source video).

## Model/training arc (frozen-test numbers; ref version noted)

| change | SER | timed-F1 | refs |
|---|---|---|---|
| two-stage baseline (paper-era) | 0.797 | 0.106 | v1 |
| semi-Markov segmental decoder (λ_d=2, λ_lm=0.3) | 0.744 | 0.203 | v1 |
| + corpus 66→93 retrain | 0.722 | 0.210 | v1 |
| CTC bootstrap (S9): HuBERT fine-tune, spike decode | 0.364 | 0.305 | v1 |
| champion 12→24 epochs | 0.339 | 0.330 | v1 |
| + pool self-training (9.6k pseudo-crops, conf ≥ 0.85, w 0.3) | 0.332 | 0.331 | v1 |
| + label-prior regularizer (+4 mixed epochs) | 0.316 | (0.314) | v1 |
| S12 ref corrections applied | 0.316 | 0.699 | v1+S12 |
| clean_v2 refs adopted | 0.263 | 0.710 | v2 |
| **+ covers self-training (current champion)** | **0.252** | **0.725** | v2 |

Training recipe (all local, ~1–3 h/run): `colab/ctc_bootstrap.ipynb` run via
`colab/run_nb.py` with per-run dirs under `artifacts/ctc_*`; crops ≤ 15 s cut
at silences, RAM-preloaded; per-item weighted CTC loss for pseudo data;
best-by-greedy-SER checkpointing; blank-collapse guard (`greedy_test_ser`
recorded; FA refused if collapsed).

Self-training machinery: `kashi loop ctc-harvest` (spike-decode the 267-song
local pool into transcript crops; confidence stats in the manifest) and
`kashi loop covers-harvest` (S15: 134 downloaded covers of 3 songs; a crop
survives only if its token 3-grams are reproduced by ≥ 2 other covers — the
n-grams are time-free, absorbing tempo/intro variation). Key controls:
labeled-only continuation REGRESSED end-to-end while pseudo-mixed improved;
pool round 2 and v2-labels-only continuation were rejected as ties —
**fresh voices singing verified lyrics is the lever that still works.**

## Closed directions (each with the killing measurement)

| direction | evidence |
|---|---|
| boundary snapping to unsupervised candidates (3 variants) | degrades v1 on gold (0.552→0.528); candidates near-dense |
| FA-extended frame targets | halves timed-F1 (0.114) |
| span-sum semi-Markov over peaky CTC | SER 0.994 (blank swamps spans) |
| beatrice phoneme-CTC warm start | never escapes blank collapse (12 ep) |
| weak-teacher frame pseudo-labeling (P5.1 v1) | frame-acc 0.396→0.394 |
| training-time label prior (de-peaking) | absorbed into lm_head bias; peakiness unchanged (blank 0.94, 1-frame runs) |
| decode-time prior + Viterbi over CTC | monotone in α but SER 0.78 ≫ 0.32; non-spike frames carry no token mass |
| onset shift −4 (test-tuned) | gold refutes: test refs were early, not model late |
| bucket hierarchy (coarse→fine) | only 7% of errors within-bucket at meaningful granularity |
| bigram beam rescoring (lyrics 46k / text 4.5M transitions) | +0.0015 / worse; domain mismatch beats scale |
| any substitution-only rescoring | oracle within top-8 lattice = 0.257 vs greedy 0.289 (~0.03 cap) |
| insertion decoding (held-vowel recovery) | gap frames are pure blank — continuations suppressed at spike level (training-time limitation) |
| karaoke lead-isolation on duets (S14) | SER 0.300 vs 0.170; mutes one duet voice; parked |

Residual error profile (test, v2 refs): 401 sub / 269 ins / 196 del; top
deletions う/い/ん (vowel continuations), top substitutions kernel-adjacent
(だ↔た, ね→み) — model-level, not decoder-level.

## Review tooling

`s12_review/` — video player: uncorrected vs corrected labels vs the model's
own prediction (3 lanes), per-song measurement figures, METHOD.md.
`s13_pilot/` — same player for Tier-2 drafts: source (excluded English) vs
current vs draft lanes. Both regenerable by script; media symlinked.

## Open

- S13-final: user reviews `s13_pilot/index.html` → "tier2 apply 23 21".
- Song 81: countdown + missing line need user's ear / official lyrics.
- Next training bundle: more cover playlists (user-provided) + SPEED_PERTURB
  augmentation + possibly pool+covers pseudo recombination.
- P5.4 cloud SSL pretraining (parked, user opt-in), custom separator training
  (parked), t1/ro 70 label-staged songs (need download OK).
- P6 research: HDP-HMM unit discovery report, TDA-vs-autocorr, calibration.
