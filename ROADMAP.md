# kashi — Specification & Roadmap

Textless syllable-level transcription of Japanese songs (v2 of the CS 541 project). Revised 2026-07-09 (rev. 3: no-forced-alignment constraint; segmental decoder core). **Status: planning — no implementation started.**

Mathematical companion: `docs/pipeline_specification.md` defines every model in full detail. This file covers architecture, data, phases, and acceptance criteria.

**Hard constraints (project identity):** (i) textless — no transcript exists or is used at inference; (ii) **no forced alignment anywhere**, including dataset cleaning — no stage infers timestamps by aligning a token sequence to audio; (iii) deep networks provide local evidence, explicit probabilistic structure (durations, transitions, boundaries, phonetic geometry) does the rest — less black box, not more.

---

## 0. Strategy and priorities

1. **Core inference = a semi-Markov segmental decoder** (spec §6): one exact Viterbi over all segmentations scoring each candidate segment by frame-level syllable posteriors (Model 2f) + duration prior (per-class negative binomial) + Model-1 boundary evidence + optional token bigram; N-best rescoring by the segment LSTM (Model 2). Textless by construction; the report's two-stage pipeline is a degeneration of it (λ_d = λ_ℓ = 0) and stays as the baseline configuration.
2. **Dataset cleaning without forced alignment** (spec §8): unsupervised boundary candidates (sticky HDP-HMM/HSMM posterior + spectral-flux onsets + voicing transitions) → monotone snapping of labeled boundaries (≤100 ms local moves, order-preserving event matching — no token sequence involved) → breath/`<noise>` tagging via voicing+energy → classifier-agreement QA flags → `clean_v2`.
3. **Import the unused labeled data** (~2–3× more songs already parsed in the dataset repo) before further modeling work.
4. **Registry + config swapping** for every component; **stage DAG** so `kashi run <target>` rebuilds only what changed.

| # | Goal | Mechanism | Phase |
|---|---|---|---|
| G1 | Clean command set | `kashi` CLI + one package + one venv | P0 |
| G2 | Swap models in/out | registry + `--set pipeline.separator=demucs` | P0 |
| G3 | Clean dataset (timing + breaths) | boundary discovery + snapping + noise tagging + auto-QA, measured on gold | P1–P2 |
| G4 | Web app mp4 → subtitles | FastAPI + job worker over the same pipeline | P4 |
| G5 | Use unsupervised data | pseudo-label loop (decoder confidence) ▸ temporal contrastive ▸ covers ▸ (cloud) pretraining | P5 |
| G6 | Statistical methods | semi-Markov decoder; sticky HDP-HMM/HSMM; phonetic kernel ×3; TDA voicing; latent-offset loss | P2, P3, P6 |
| — | Full automation | stage DAG, one-command scrape→train, auto-QA quarantine, eval-gated promotion | P0, P2, P2b, P3, P5 |

| Phase | Deliverable | Effort | Depends |
|---|---|---|---|
| P0 | rebuilt `kashi` package + DAG core; day-1 `kashi transcribe` (energy segmenter + legacy LSTM) | 1.5–2 d | — |
| P1 | eval harness + gold subset; first honest end-to-end baseline | 1 d + ~3 h listening | P0 |
| P2 | HDP-HMM/HSMM + boundary sources + snapping `realign` → `clean_v2` + `<noise>` + QA quarantine | 2–3 d | P1 |
| P2b | corpus import + Python VTT parser + `kashi dataset scrape`: 66 → ~150–240 songs | 1.5–2 d | P0 |
| P3 | segmental decoder + frame classifier + retrained M1 (latent-offset loss) / M2 (phonetic loss) on v2; ablations; promotion gate | 2–3 d | P2 |
| P4 | web app | 2–3 d | P0 (better after P3) |
| P5 | `kashi loop unsup`: scrape → decode → confidence-filter → retrain → promote-if-better; contrastive; covers | 3–5 d incr. | P2, P2b, P3 |
| P6 | research: unit discovery vs the 110 inventory, HSMM durations, TDA-vs-autocorr, calibration | open, parallel | P0, P1 |

Core product path P0→P4 ≈ 1.5–2 focused weeks.

---

## 1. Current state (verified 2026-07-09)

### 1.1 Paper results

66 labeled songs; vocal isolation → wav2vec2 (20 ms, 768-d) → Model 1 (per-frame break, BCE on soft-padded labels) → Model 2 (LSTM per segment, 110 classes). Model 1 **F1 0.408** (exact-frame, jittery labels); Model 2 **acc 0.536 / micro-F1 0.535** with ground-truth segments. Flaws: ±50 ms jitter, unlabeled breaths, slurred/skipped syllables, structure-blind BCE (ka/ga, wo/o).

### 1.2 Code reality (repo + `google drive/`)

- **Paper-era Model 1 recovered** in `google drive/Code/Transformer_Wave2Vec_AAAAA.ipynb` (soft labels `[0.2,.35,1,1,.8,.25]`, thr 0.45, Adam 1e-4, 10 epochs, batch = 1 song, split `[0:80]/[80:93]`; eval ckpt `model_9_final`). Variants: `Transformer_Wave2Vec.ipynb` (hard ±2 dilation), `Transformer_Spectrogram{,_v2}.ipynb` (32-mel; v2 + Conv1d front-end).
- **Three implementation bugs in the paper's Model 1:** (1) no `batch_first=True` on `[1,N,768]` → attention over length-1 sequences → zero temporal context (a per-frame MLP in effect); (2) no positional encoding; (3) `expand_ones` negative-index wraparound bleeds labels across song edges. The "transformer > LSTM" comparison is void; F1 0.408 is per-frame-MLP performance; per-frame framing was never fairly tested. Fixed port + bug-cost ablation in P1/P3.
- **Model 1 checkpoints still Drive-only** (`pretrained/Transformer_Wave2Vec_Models/`, `pretrained/Transformer_Spectrogram_Models/`) — requested; not blocking.
- **Model 2 survives**: `models/predict_syllables/pretrained/model_20ms_drop_0.5_144_0.5363_test` (LSTM 768→144×2, dropout 0.5; attrs `lstm`/`fc`). Drive `Code/{main,load,model,train,syllables,my_helper}` = Colab copies (path deltas; `main` reproduces micro-F1 0.53516). `Segment Break Code/main.ipynb` = unused template.
- Repo `best_model_*.pth` = later CNN-BiLSTM dead ends (constant predictors ~0.494–0.496).
- Encoder was English `facebook/wav2vec2-base`, re-instantiated per call. Keep: separated vocals (93), cleaning chain, `audio-separator` wrapper. End-to-end accuracy never measured.
- **Verified 2026-07-09: the local 11 GB tensor cache (`models/tensors/songs_20ms/`) is misnamed — it holds the 10 ms resample-trick variant** (frame count × 10 ms = audio duration on every checked song; the 20 ms checkpoint scores 3% on it vs the 52% of its `pred` column). True-20 ms tensors existed only on Colab/Drive. Consequence: `--from-legacy` adopts into the 10 ms cache; the 20 ms cache is re-encoded locally (`kashi encode` replicates legacy semantics — full-song Wav2Vec2Processor normalization + one full-song forward, fp16+SDPA on 6 GB).
- Hygiene debt (P0): mixed sec/ms units, 20 ms constant ×7 files, vocab ×4, cwd-dependent paths, dead inference CLI, destructive prep scripts, abandoned `paper_based_approach/`, 3 req-specs + 4 venvs, ~100 MB vendored UVR5/demucs. Map: Appendix A.
- Prior refactor deleted same day; design recovered from bytecode (Appendix B).

### 1.3 Data assets

`../karaoke_subtitle_dataset` (MIT; yt-dlp + Rust VTT color-diff parser + Python token pipeline):

| Set | Songs | Labels | Status |
|---|---|---|---|
| jp_t2 | 93 | kanji/kana per-mora | 66 cleaned & used; +27 uncleaned; audio on disk |
| jp_t1 | 59 | kanji/kana per-mora | unused; disjoint from T2 |
| roumanji | 151 | romaji per-mora | unused; 57∩T1 |

≈246 unique videos → **~150–220 usable** after dedupe/QA. Romaji ≈1:1 onto the 110 tokens; the 57 dual-track songs = free mapping/parser test corpus. Scraper reusable (needs color-coded karaoke subs). Partial gold set exists: songs {0,6,16,19} with `<sil>/<bre>` hand labels.

### 1.4 Compute

RTX A3000 Laptop 6 GB VRAM, 16 GB RAM, 16 cores → base-size frozen backbones, fp16, grad-accum; HDP-HMM Gibbs runs on CPU; no local SSL pretraining (cloud-optional P5.4); memmap caches; Python 3.12.

---

## 2. Method choices

### 2.1 Structural diagnosis

Per-frame independent break classification manufactures sparse positives and ignores that segments persist; hard 20 ms targets train on ±50 ms label noise (jitter, eq. (1) of the spec); flat CE ignores phonetic geometry. §1.2 adds: the per-frame model also never saw temporal context (bugs), so its ceiling is unknown — but the framing is corrected regardless, by making segmentation a joint decode rather than thresholded per-frame guesses.

### 2.2 Core: textless segmental decoding (spec §6)

Semi-Markov Viterbi over all (segmentation × labeling) hypotheses; per-segment score = λ_c·(frame-posterior sum, Model 2f) + λ_d·log NB-duration + λ_b·(Model-1 boundary logit at segment start) + λ_ℓ·log bigram (optional, default off). O(T·D_max·K²) with top-K shortlists; exact DP via cumulative sums; N-best rescored by the segment LSTM. Confidences from semi-Markov forward–backward. Every term is inspectable per decoded segment; setting λ's recovers the paper's two-stage pipeline (kept as baseline config).

Jitter handling at training time (no alignment): Model 1 trained with the **latent-offset marginalized loss** (spec eq. (17)) — the labeled break's true position is a local latent variable in ±3 frames, marginalized exactly; the model is never penalized inside the tolerance window. Model 2/2f trained with phonetic soft targets on `clean_v2` spans; `<noise>`-tagged frames excluded.

| Considered for timing/cleaning | Verdict | Reason |
|---|---|---|
| Semi-Markov decoder + snapping-based cleaning | **adopted** | textless; no forced alignment; interpretable score decomposition |
| CTC/neural forced alignment of label sequences | **rejected** | is forced alignment — violates the project constraint (and off-script singing breaks its premise) |
| MFA / MMS / WhisperX | rejected | forced aligners; also inventory/domain mismatch |
| CTC-style sequence-marginalized *training* of the frame posterior | optional, off by default | not forced alignment (no timestamps produced/consumed; see spec §9.5) — considered only if latent-offset loss + v2 labels prove insufficient on gold |
| Sticky HDP-HMM/HSMM (unsupervised) | **adopted** | boundary candidates + uncertainty for cleaning; unit discovery; no labels touched |

### 2.3 Encoder

P2 picks `rinna/japanese-hubert-base` vs `rinna/japanese-wav2vec2-base` vs legacy English by a frozen-feature linear syllable probe (½ day). Legacy 11 GB cache stays valid for baselines (`encode --from-legacy`).

### 2.4 Statistical-methods triage

| Idea | Verdict | Landing |
|---|---|---|
| Boundaries as latent (never per-frame Bernoulli targets) | adopted: semi-Markov decoder (supervised side) + sticky HDP-HMM/HSMM (unsupervised side) + latent-offset loss (training side) | P2/P3 |
| Phonetic articulatory kernel | adopted ×3: partial-credit metric; soft-target loss; contrastive soft negatives (explicit parameterization in spec §5.3: k(ka,ga)=0.91, k(ka,n)=0.10) | P1/P3/P5 |
| Contrastive g_w | adopted, below pseudo-labeling | P5 |
| TDA voicing | adopted narrow: v_t feature + noise tagging; autocorr default, ripser-H1 alternate benchmarked once | P2, P6 |
| GP time-warp | dropped — timing is handled by latent-offset loss + snapping | — |
| DPP negative mining | deferred | P5 stretch |
| Hawkes | dropped — durations are NB/HSMM, not self-exciting | — |

### 2.5 Unsupervised ranking

1. **Pseudo-labeling** via decoder segment-confidence c_k ≥ 0.9 (spec §9.4); 2. **temporal InfoNCE** head; 3. **covers** (audio↔audio DTW positives — signal-to-signal, no transcript); 4. **cloud SSL pretraining**.

### 2.6 Unchanged

16 kHz mono; 20 ms grid; 110-token inventory in legacy order; frozen paper split; separation stage 1 (mel-roformer default).

---

## 3. Architecture

### 3.1 Package layout

```
pyproject.toml                  # extras: [uvr] [demucs] [tda] [lm] [dev]
configs/default.toml
src/kashi/
  audio.py        # ffmpeg → 16 kHz mono f32; rms; content-hash cache keys
  tokens.py       # 110-token inventory (legacy order); kana helpers; romaji↔kana map
  phonetics.py    # decomposition, kernel_matrix (PSD-projected), soft_targets, partial_credit
  subtitles.py    # Segment (start,end,token,exclude,+meta); CSV/SRT/VTT/karaoke-ASS writers
  config.py       # TOML: defaults < --config < --set; dotted access
  registry.py     # @register(kind, name); create_from_config(cfg, kind)
  dag.py          # stage graph + input/config hashing + skip-if-fresh
  pipeline.py     # media → subtitles; topologies two_stage | segmental; progress + timings
  realign.py      # cleaning driver: candidates → snapping → noise tags → QA gates → clean_v2
  data/
    manifest.py   # layout, ids, frozen splits, multi-source import, QA status
    build.py      # port of clean_subtitles chain (pure, idempotent)
    scrape.py     # yt-dlp wrapper + VTT color-diff parser (Python port; Rust = test oracle)
    store.py      # feature cache artifacts/features/<encoder-id>/<frame_ms>ms/; legacy adoption
    datasets.py   # torch datasets; labels derived on the fly
  components/
    base.py       # Protocols: Separator, Encoder, Segmenter, Classifier, Decoder, BoundarySource
    separators.py # none | uvr | demucs
    encoders.py   # wav2vec2 (any HF ckpt; optional projection head) | mel
    segmenters.py # energy | transformer | hmm            (two_stage topology)
    classifiers.py# lstm (legacy-compatible) | silence_only
    decoders.py   # segmental (semi-Markov Viterbi + N-best LSTM rescoring)   [core]
    boundaries.py # hmm | onset (spectral flux) | voicing-delta   (candidate sources for realign)
  nn/
    segmenter.py  # Model 1 FIXED (batch_first, sinusoidal PE, windowed attention);
                  #   soft-kernel BCE + latent-offset marginalized loss; NMS; boundary-F1
    classifier.py # Model 2 LSTM (attrs lstm/fc) + Model 2f frame posterior + PhoneticCE
    contrastive.py# ProjectionHead + InfoNCE (temporal | supcon | covers)
  stats/
    hmm.py        # StickyHDPHMM weak-limit blocked Gibbs (FFBS) + HSMM (NB durations)
    durations.py  # per-class NB fits (MoM + shrinkage); silence mixture (EM)
    snapping.py   # monotone boundary matching DP (event sets; ≤100 ms moves)
    lm.py         # token bigram (add-k / Kneser–Ney)                    [lm extra]
    tda.py        # voicing: autocorr (default) | ripser-H1
  train/          # common.py + segmenter.py classifier.py frame.py encoder.py; --promote gate
  eval/
    metrics.py    # boundary F1@τ, SER, timed-token F1, partial credit, noise-span P/R
    gold.py       # gold export/import (Audacity labels, prefilled from current best decode)
  web/app.py      # FastAPI; single worker; in-memory jobs; static/index.html
  cli.py
tests/            # unit + 10 s CPU smoke fixture
docs/pipeline_specification.md   # the mathematical spec (source of truth for all models)
```

### 3.2 Component contracts

```python
class Separator(Protocol):      # none | uvr | demucs
    def separate(self, wav: Path, out_dir: Path) -> SeparationResult
class Encoder(Protocol):        # wav2vec2 | mel
    dim: int; frame_ms: int
    def encode(self, wave: np.ndarray, sr: int) -> np.ndarray            # [T, dim]
class Segmenter(Protocol):      # energy | transformer | hmm             (two_stage)
    def segment(self, feats: np.ndarray) -> list[Span]
class Classifier(Protocol):     # lstm | silence_only                    (two_stage)
    def classify(self, feats: np.ndarray, spans: list[Span]) -> list[tuple[str, float]]
class Decoder(Protocol):        # segmental                              (default topology)
    def decode(self, feats: np.ndarray, aux: FrameAux) -> list[Segment]  # aux: voicing, energy, M1 logits; Segments carry confidence
class BoundarySource(Protocol): # hmm | onset | voicing-delta            (realign only)
    def boundaries(self, wave, feats) -> list[Boundary]                  # (time_s, conf, std_ms)
```

New model = one class + one decorator + a `[<kind>.<name>]` config section; swap via `--set`.

### 3.3 Topologies

```
input ─ffmpeg→ 16k wav → separator → features ┬ segmental: Model2f+Model1+durations → semi-Markov decode ┐
                                              └ two_stage:  segmenter → classifier ───────────────────── ┴→ cleanup → SRT/VTT/ASS/CSV
```

`two_stage` (energy + legacy LSTM) runs with zero training — day-1 fallback and permanent baseline. `segmental` becomes default after P3.

### 3.4 Config (abridged; full defaults in spec §10 table)

```toml
[data]      frame_ms = 20; sample_rate = 16000; version = "clean"
[paths]     data_dir = "data"; artifacts_dir = "artifacts"; runs_dir = "runs"; dataset_repo = "../karaoke_subtitle_dataset"
[pipeline]  mode = "two_stage"          # flips to "segmental" via promotion after P3
            separator = "uvr"; encoder = "wav2vec2"; segmenter = "energy"; classifier = "lstm"; decoder = "segmental"
[encoder.wav2vec2]     checkpoint = "rinna/japanese-hubert-base"; projection_head = ""
[features]             voicing = "autocorr"   # autocorr | tda
[segmenter.transformer] checkpoint = ""; threshold = 0.45; nms_frames = 3; attn_window = 100
[classifier.lstm]      checkpoint = "models/predict_syllables/pretrained/model_20ms_drop_0.5_144_0.5363_test"
[decoder.segmental]    d_max = 60; top_k = 8; n_best = 4; lambda_c = 1.0; lambda_d = 0.5; lambda_b = 0.5; lambda_lm = 0.0
[boundaries]           sources = ["hmm", "onset", "voicing"]; hmm_p_min = 0.3
[hmm]                  L = 120; rho = 0.95; alpha = 4.0; gamma = 4.0; sweeps = 30; burnin = 10; pca_dim = 48
[realign]              out_version = "clean_v2"; delta_max_ms = 100; c_miss = 1.2; eta = 0.5
                       noise_min_ms = 60; noise_rms_db = -35; voicing_thresh = 0.35
[qa]                   max_mean_shift_ms = 60; max_flagged_frac = 0.15; min_candidate_recall = 0.6; quarantine = true
[train.segmenter]      loss = "latent_offset"   # latent_offset | soft_bce ; delta = 3
[train.classifier]     loss = "phonetic"; smooth_alpha = 0.1; kernel_power = 4
[train.frame]          head = "mlp256"
[train.encoder]        mode = "temporal"
[eval]                 tolerances_ms = [20, 50]; split = "paper"
[web]                  host = "127.0.0.1"; port = 8000; max_upload_mb = 500
```

### 3.5 CLI

| Command | Purpose |
|---|---|
| `kashi info` | config, components, device, dataset/cache/checkpoint/QA state, leaderboard |
| `kashi run <stage>` | DAG-resolve; execute only stale stages (§3.9) |
| `kashi dataset build` | raw subtitles → `clean` labels |
| `kashi dataset download` | yt-dlp audio from `index.tsv` |
| `kashi dataset import --sets t1,ro,t2-extra` | unify by YouTube ID; romaji→kana; QA filters |
| `kashi dataset scrape --playlist URL --lang ja\|en [--labels\|--audio-only]` | one-command ingest (labeled or unlabeled pool) |
| `kashi separate <files…>` | vocal isolation |
| `kashi encode [--songs …] [--unlabeled DIR] [--from-legacy] [--force]` | feature cache |
| `kashi train segmenter\|classifier\|frame\|encoder [--version clean_v2] [--init CKPT] [--promote]` | training; promotion gated on frozen-test eval |
| `kashi fit durations\|lm [--version clean_v2]` | closed-form/counting fits (NB durations, bigram) |
| `kashi eval segmenter\|classifier\|pipeline [--split paper] [--gold]` | metrics → `eval.json` + leaderboard |
| `kashi realign [--boundaries hmm,onset,voicing] [--out-version clean_v2]` | snap timings, tag `<noise>`, QA-quarantine, report |
| `kashi gold export\|import\|status [--window 90]` | gold round-trip (prefilled Audacity labels) |
| `kashi discover <audio>` | HDP-HMM unit discovery |
| `kashi transcribe <media> [--formats srt,vtt,ass,csv] [--romaji] [--no-separate]` | full pipeline |
| `kashi loop unsup [--rounds N]` | scrape → decode → confidence-filter → retrain → promote-if-better |
| `kashi serve` | web app |

### 3.6 Data layout & label versioning

```
data/
  raw/audio/<id>.mp3    raw/subtitles/{subtitle_files/<id>.csv, clips_to_exclude/, index.tsv}
  clean/audio/vocals/<id>.mp3    clean/subtitles/<id>.csv        # v1 (sec float: start,end,token,exclude)
  clean_v2/subtitles/<id>.csv    # + moved_ms, boundary_std_ms, conf, flags, <noise> spans
  gold/subtitles/<id>.csv        # hand-verified windows + extents
  unlabeled/audio/<key>.mp3
artifacts/features/<encoder-id>/<frame_ms>ms/<key>.npy    artifacts/<model>/current -> symlink
runs/<name>/{config.toml, ckpt/, eval.json, log}          runs/leaderboard.csv
```

Label versions append-only; times in float seconds; derived labels computed on the fly; `data.version` selects; paper split frozen in `manifest.py`.

### 3.7 Web app

FastAPI; one worker thread; in-memory jobs; artifacts `runs/web/<job_id>/`. `POST /jobs` (multipart; 413 over cap; queue ≤3) → `GET /jobs/<id>` (state/stage/frac; 1 s poll) → `/media`, `/files/{srt,vtt,ass,csv}`. UI: drop-zone → per-stage progress → `<video>` + VTT text track → downloads. ASS `{\k}` download-only.

### 3.8 Environment

One `pyproject.toml`, Python 3.12, torch-CUDA; extras `[uvr] [demucs] [tda] [lm] [dev]`. Retires Pipfile, both requirements.txt, 4 venvs. CI: unit tests + CPU smoke (mel encoder, none separator) + romaji↔kana dual-track differential test.

### 3.9 Automation layer

- **Stage DAG** (`dag.py`, ~150 lines, internal): stages declare inputs/outputs/config-subtree; fingerprint = hash(inputs + config + code version); `kashi run eval` after adding one song re-executes exactly that song's download→separate→encode→realign chain plus eval; `train` stages barrier-marked (`--allow-train`).
- **One-command ingest**: `dataset scrape` uses the Python VTT parser (differential-tested byte-identical against the Rust on all 93 T2 files).
- **Auto-QA quarantine**: `[qa]` thresholds exclude failing songs from training; humans see only quarantined songs, worst-first, flags attached. Gold annotation prefilled by current best decode.
- **Eval-gated promotion**: `train --promote` retargets `artifacts/<model>/current` only on frozen-test SER + timed-token-F1 wins; test-side gold is reporting-only.
- **Self-training loop**: `kashi loop unsup` (spec §9.4); stops after 2 non-improving rounds; resumable via DAG.
- **Free consistency oracle**: 57 kana+romaji dual-track songs gate the mapping and parser in CI.

---

## 4. Evaluation protocol (precedes improvements)

**Gold subset.** Extend golden songs {0,6,16,19} with 4 test-split songs; hand-verify one 90 s window per song via `kashi gold export` (prefilled) → correct → `import`. ≈8 windows ≈ 12 min ≈ 2–4 h once. Train-side gold measures realign; test-side gold is reporting-only.

**Metrics** (formal definitions: spec §10): boundary F1@{20,50} ms + mean|Δt|; **SER** (headline); timed-token F1 (token ∧ |Δstart| ≤ 50 ms); partial credit (phonetic kernel); noise-span P/R (IoU ≥ 0.3).

**P1 baselines — RECORDED 2026-07-09** (`runs/baselines/*.json`, frozen test split, re-encoded 20 ms features):

| Baseline | Headline | Detail |
|---|---|---|
| (a) legacy M2, GT segments | **acc 0.5319** (paper 0.5363, Δ0.4 pt — port validated) | partial credit 0.757 (first measurement) |
| (b) energy+LSTM end-to-end | **SER 0.797** · timed-token F1 0.106 | boundary F1@50ms 0.30 — the bar P3 must clear |
| (c) M1 verbatim (bugs incl.), 10 ep | boundary F1@60ms **0.646**, mean&#124;Δ&#124; 34.5 ms | paper-metric F1 0.351 (paper 0.408; different feature vintage/seed) |
| (d) M1 bug-fixed + soft-BCE, 20 ep | boundary F1@60ms 0.574 | paper-metric F1 0.348 |
| (d′) M1 bug-fixed + latent-offset, 10 ep | boundary F1@60ms **0.624**, mean&#124;Δ&#124; 32.6 ms | best fixed-arch trainer at half the epochs |

**Findings:** (1) the architecture "bug cost" is ≈ zero at equal budget — wav2vec2's own transformer already supplies temporal context, so Model-1's attention adds little; the lever is labels/decoding, not M1 architecture (consistent with §2.1). (2) The latent-offset loss beats soft-BCE (0.624 vs 0.574 at half the epochs) — P3 default confirmed. (3) End-to-end was never the sum of its stage metrics: SER 0.797 despite 53% classifier accuracy — segmentation is the bottleneck, exactly what the segmental decoder targets. Still open in P1: test-side gold windows need human listening (`kashi gold export/import`).

---

## 5. Phases

### P0 — Rebuild core (G1, G2) — 1.5–2 d
Package per §3.1 (recovered design, Appendix B): tokens/phonetics/subtitles/config/registry/audio/dag; manifest/build/store/datasets; separators, encoders, energy segmenter, legacy LSTM classifier; both topologies (decoder stubbed); CLI; tests; `encode --from-legacy`. Hygiene: `legacy/` moves (Appendix A), single source for frame_ms + tokens, gitignore caches, drop stray venvs.
**Accept:** fresh venv → `kashi info` clean; `kashi transcribe tests/fixtures/clip.mp4` → SRT/VTT/ASS; `kashi eval classifier --split paper` reproduces legacy acc ±1 pt; pytest green; unchanged `kashi run` is a no-op.

### P1 — Measurement — 1 d + listening
`eval/metrics.py`, `gold.py`, `kashi gold`; baselines (a)–(d); leaderboard. Gold windows annotated.
**Accept:** ≥8 gold windows imported; 4 baseline `eval.json`s; day-1 SER known.

### P2 — Data engine (G3) — 2–3 d
① Encoder probe (½ d): JA-hubert vs JA-wav2vec2 vs EN-base by frozen linear probe. ② `stats/hmm.py`: sticky HDP-HMM weak-limit blocked Gibbs (FFBS + NIG + CRT steps per spec §7); HSMM upgrade behind a flag. ③ `components/boundaries.py`: hmm posterior, spectral-flux onset, voicing-delta sources; union+merge. ④ `stats/snapping.py` monotone matching DP; `kashi realign`: snap (≤100 ms), `<noise>` tagging (voicing+RMS), `missed-vocal` flags, classifier-agreement flags, QA quarantine → `clean_v2` + report. ⑤ Retrain M1/M2 on v2 (P3 trainers pulled early if needed) and re-run once (v3 only if gold improves).
**Accept (gold train-side):** snapped labels mean|Δt| ≤ 30 ms and boundary F1@50 ≥ 0.80 (v1 baseline expected ≈ 0.6–0.7); candidate recall ≥ 0.85 within ±100 ms of gold boundaries; noise-span P ≥ 0.8 @ R ≥ 0.5; 100% of unmatched/large-shift rows flagged not silently moved; quarantine list plausible.

### P2b — Corpus expansion (parallel) — 1.5–2 d
Python VTT parser + differential test; `dataset import` (dedupe, romaji→kana with reject-log, QA filters from dataset repo's analysis_csvs); `dataset scrape`; download/separate/encode; realign before admission.
**Accept:** ≥140 labeled songs QA-passed; dual-track mismatch < 2%; frozen test split untouched.

### P3 — Models v2 + decoder — 2–3 d
Model 2f frame posterior; `fit durations` (NB per class, silence mixture) + optional `fit lm`; `decoders.segmental` (exact DP + cumulative sums + top-K + N-best LSTM rescoring + forward–backward confidences); retrain M1 with latent-offset loss and M2/2f with phonetic loss on clean_v2 + expanded corpus; voicing-append ablation. Grid: {v1/v2} × {EN/JA} × {two_stage/segmental} × {±bigram} × {±phonetic} × {±voicing} → `runs/ablations/`.
**Accept:** `mode=segmental` beats the P1 two-stage baseline on SER **and** timed-token F1 on frozen test; λ grid documented; default flipped by promotion, not by hand.

### P4 — Product (G4) — 2–3 d
Web app per §3.7 over pipeline progress callbacks.
**Accept:** fresh mp4 → preview + 4 formats ≤ ~5 min/song; survives 3 queued jobs; ASS `{\k}` correct in Aegisub.

### P5 — Unsupervised (G5) — 3–5 d incremental
① `loop unsup` (≥100 scraped songs; c_k ≥ 0.9; weak weight 0.3; stop after 2 dry rounds). ② temporal InfoNCE head; adopt if segmenter-F1 or decoder SER improves; then supcon with phonetic soft negatives. ③ covers via audio↔audio DTW positives. ④ cloud SSL pretraining if budgeted.
**Accept:** each experiment = run dir + keep/drop verdict; winners enter defaults via promotion only.

### P6 — Research track — parallel, open-ended
`discover` unit-inventory report (inferred units vs 110; majority-vote confusion); HSMM durations if HMM boundary residuals look non-geometric; autocorr vs ripser-H1 (one notebook, one table); confidence calibration (reliability diagram vs gold).
**Accept:** written findings; anything beating a P2/P3 component on gold promotes through the registry.

```
P0 ─→ P1 ─→ P2 ─→ P3 ─→ P4
  └─→ P2b ──↗  └─→ P5
  └─→ P6 (uses P1 metrics)
```

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| Snapping bounded to ±100 ms — cannot fix gross timing errors or wrong tokens | by design: flag + quarantine, never silently edit; wrong-token rows caught by classifier-agreement QA |
| HMM boundary recall low on legato/melisma singing | union with onset + voicing sources; recall AC in P2; HSMM upgrade path |
| Decoder over-segments melisma (one syllable, many pitch changes) | duration prior + boundary evidence penalize; melisma-specific gold windows in P1 selection |
| Singer skips/slurs labeled syllables | flags via unmatched boundaries + low classifier agreement; excluded from training |
| Separation bleed / retained breaths | separator swap = config line; `<noise>` frames excluded from losses |
| Romaji mapping edges (っ/ー/ゃゅょ) | 57-song dual-track differential test in CI |
| Python VTT parser diverges from Rust | byte-identical differential test on 93 T2 files |
| 6 GB VRAM / 16 GB RAM | frozen backbones, fp16, grad-accum; HMM on CPU; memmap caches |
| Promotion overfits eval | promotion on frozen test only; test-side gold reporting-only |
| Scraping ToS / takedowns | keep zips + download archives; volume is a user decision (§7) |
| Web app GPU contention | single worker + queue cap; CPU-fallback config |

## 7. Open items (user)

1. **From Google Drive:** add `pretrained/Transformer_Wave2Vec_Models/` (esp. `model_9_final`) and `pretrained/Transformer_Spectrogram_Models/`. Everything else the notebooks reference is in-repo or regenerable.
2. OK to move `paper_based_approach/`, vendored UVR5-GUI/demucs, vestigial scripts/EDA notebooks under `legacy/` in P0 (delete after P3)?
3. Scraping scope: additional channels/covers volume?
4. Cloud GPU budget for P5.4 — yes/no?
5. Romaji-only songs as first-class training data (recommended) or separate track?

---

## Appendix A — Port/drop map

| Source | Fate | Target |
|---|---|---|
| `data_processing/main_functions/clean_subtitles.py` + `utils/{character_filtering,time_ranges,row_filtering,silence_and_excluded,tokens}.py` | port (pure) | `kashi/data/build.py` |
| `data_processing/download_audio.sh` | port | `kashi dataset download` |
| `data_processing/prepare_segment_break_data.py`, `prepare_token_classification_data.py` | retire — on-the-fly labels | `kashi/data/datasets.py` |
| `main_functions/change_last_end_to_vid_length.py`, `utils/reduce_silence.py` | fold in | `kashi/data/build.py` |
| `models/wave2vec2/{wave,audio}.py` | port (cache model instance; keep resample trick; document `+320` pad) | `kashi/components/encoders.py` |
| `models/predict_syllables/*.py` | port; keep attr names + class order | `kashi/nn/classifier.py`, `train/classifier.py`, `tokens.py` |
| **`google drive/Code/Transformer_Wave2Vec_AAAAA.ipynb`** | **port as Model 1 reference, bugs fixed** (batch_first, PE, edge-guarded labels, windowed attention) | `kashi/nn/segmenter.py` |
| `google drive/Code/Transformer_Wave2Vec.ipynb` | archive (hard-dilation v1) | `legacy/` |
| `google drive/Code/Transformer_Spectrogram{,_v2}.ipynb` | archive (32-mel variants) | `legacy/` |
| `google drive/Code/{main,load,model,train,syllables,my_helper}.ipynb` | archive (Colab copies; `main` reproduces paper M2 numbers) | `legacy/` |
| `google drive/Segment Break Code/main.ipynb` | drop (unused template) | — |
| `models/segment_break_detection/` (CNN-BiLSTM ckpts, notebooks, dead CLI) | drop (non-discriminative dead ends) | — |
| `isolate_vocals/{inference,multi_inference}.py` | rewrap pip `audio-separator` | `kashi/components/separators.py` |
| `data_preparation/preparing_raw_data/` (UVR5-GUI + demucs vendor) | delete (pip replaces) | — |
| `models/paper_based_approach/` | archive (never functional) | `legacy/` |
| `data_processing/container_files/` | archive (SLURM-specific) | `legacy/` |
| `models/tensors/songs_20ms/*.pt` (11 GB; actually 10 ms variant) | keep; symlink-adopt into the 10 ms cache | `kashi encode --from-legacy` |
| golden CSVs (songs 0/6/16/19, `<sil>/<bre>`) | seed gold set | `kashi gold import` |
| `karaoke_subtitle_dataset/src/*.rs` (VTT parser) | port to Python; keep as differential oracle | `kashi/data/scrape.py` |
| root `out.csv`, `predictions/`, `data_preparation/out.csv` | delete | — |
| `Pipfile`, `requirements.txt`, 4 venvs | replace | `pyproject.toml` + one `.venv` |

## Appendix B — Decisions carried from the deleted 2026-07-09 scaffold

Recovered from bytecode: package name `kashi`; single entry point; layered TOML + `--set`; `@register`/`[<kind>.<name>]` registry; token inventory in legacy order (`<noise>` = annotation only, never a class); shared `Segment` schema; karaoke-ASS writer; realign carries `moved_ms` + boundary uncertainty; feature cache with legacy symlink adoption; "trust the sequence, re-infer the timings" — now implemented without forced alignment (snapping, spec §8); web app single-worker in-memory jobs.
Changes vs that scaffold: semi-Markov segmental decoder as the core inference (new); forced-alignment realign replaced by boundary snapping; `eval/` + gold gate added; `dataset import/scrape` added; DAG automation added; Python 3.12 pin; pseudo-labeling ranked first among unsupervised methods; Model 1 port must fix the three §1.2 bugs.
