# kashi — textless syllable-level transcription of Japanese songs

Input: a Japanese music video/audio file (no transcript). Output: per-syllable
timed hiragana subtitles (SRT/VTT/karaoke-ASS/CSV). No forced alignment
anywhere — segmentation and labeling are decoded jointly from acoustics by an
explicit semi-Markov model over neural frame evidence.

- **Plan / status:** `ROADMAP.md` (phases, acceptance criteria, recorded results)
- **Model math:** `docs/pipeline_specification.md` (every component, spec §1–§11)

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchaudio   # CUDA build on GPU boxes
uv pip install --python .venv/bin/python -e ".[web,dev]"
kashi info
```

Requires `ffmpeg` (and `yt-dlp` for dataset downloads). Optional extras:
`[uvr]` / `[demucs]` vocal separators, `[tda]` ripser voicing, `[lm]` beam LM.

## Commands

```bash
kashi transcribe song.mp4 --formats srt,vtt,ass   # media -> subtitles
kashi serve                                       # web app (upload, preview, download)
kashi info                                        # config / components / dataset state
kashi run <stage>                                 # DAG: run only stale stages
kashi dataset build|download|import|scrape        # labels & corpus expansion
kashi encode [--from-legacy]                      # feature cache
kashi train segmenter|classifier|frame            # models (Model 1 / 2 / 2f)
kashi fit durations|lm                            # closed-form decoder priors
kashi eval classifier|segmenter|pipeline|baseline # metrics on the frozen split
kashi realign [--vs-gold]                         # label cleaning (boundary snapping)
kashi gold seed|export|import|status              # hand-verified reference subset
kashi discover audio.wav                          # unsupervised HDP-HMM units
```

Every component (separator/encoder/segmenter/classifier/decoder) is a registry
entry selected in `configs/default.toml` — swap models with
`--set pipeline.separator=demucs` or a config overlay, never by editing code.

## Pipeline (default: `segmental`, promoted 2026-07-09)

```
media -> ffmpeg 16k mono -> [separator] -> wav2vec2 frames
      -> semi-Markov Viterbi over: frame posteriors (Model 2f)
         + NB duration priors + token bigram [+ Model-1 boundary logits]
      -> silence fill -> SRT/VTT/ASS/CSV
```

Frozen-test results and baselines: see `ROADMAP.md` §4 and `runs/leaderboard.csv`.
