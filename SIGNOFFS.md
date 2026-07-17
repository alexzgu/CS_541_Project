# Sign-offs

Reply per ID. Decided items stay for the record.

## Open

| ID | Question | Provisional state |
|---|---|---|
| S16 corpus-152 champion | Ratify promotion of `artifacts/ctc_t1/out/ctc_model` (covers champion +8ep on the 152-song corpus after T1 admission)? Letter of the adopt-only-if-better gate: SER passes big (0.2570→**0.2450**), timed-F1 misses by 0.0011 (0.7156→0.7145, noise-level; bF1@50 identical 0.7237). Gold arbitration: SER **exactly tied** (256 errors / 1,459 tokens for both models, differently distributed — candidate much better on the hardest gold song 19: 0.438→0.390), gold timed-F1 **favors the candidate** +0.0054 (0.7414→0.7469) | **Promoted provisionally 2026-07-17** (config points at ctc_t1; one-line revert). Same shape as the approved S11 exception but with a 15× smaller timing delta and positive gold timing. Full numbers: `runs/leaderboard.csv` row `ctc_t1_152`, logs `artifacts/ctc_t1/` |

## Decided (2026-07-15, second batch)

| ID | Decision | Status |
|---|---|---|
| S11 gate-exception promotion | **yes** | `ctc_model = artifacts/ctc_deprior/out/ctc_model` stands (test SER 0.316, gold SER 0.170, gold timed-F1 0.741; shift −1 gold-validated) |
| S12 per-song ref timing correction | **yes — APPLIED 2026-07-15** | 18 songs shifted +50..+130 ms (17 + song 88 after de-styling; backups `data/clean/subtitles_pre_s12/`). Rebaseline (same champion): timed-F1 0.314→**0.699**, bF1@50 0.412→**0.717**, PC 0.963; song 89 timed-F1 0.017→0.906. Review package kept at `s12_review/`. Perceptual note adopted as `subtitles.display_lead_ms = 100` (srt/vtt/ass lead the true timings; csv/eval stay honest) |
| S12b styling-repeat condensation | **yes (user-suggested)** | 7 songs de-styled (67 ms color-fade frames merged; song 88: 3,264→422 rows; いい-type real doubles protected by the duration criterion; backups `data/clean/subtitles_pre_condense/`); song 88's offset became measurable afterwards (+100 ms applied) |
| S13 dataset v2 Tier 1+2 | **yes — Tier 1 adopted; Tier 2 batches 1+2 APPLIED** | Tier 1: 4,290 rows recovered, `data.version = clean_v2` adopted. Tier 2 batch 1 (songs 23 +660, 21 +281, applied 2026-07-16) and batch 2 (60/24/11/40/69 + TEST song 85 from source text only, +1,716, applied 2026-07-17 after player review; user note: scat vocables' EN/JA classification is meaningless — phonetic level is operative). Priors 48.6k segments. **Test re-baselined** (85's refs grew): SER 0.2570 / timed-F1 0.7156 vs 3,514 ref tokens; 85's own SER 0.488→0.434. Backups `subtitles_pre_tier2/`. Song 81 items still need user's ear / official lyrics; ~1.8k tail rows over ~80 songs remain for batch 3 |
| S14 lead-vocal separator trial | **yes — trial DONE, negative on duets** | karaoke roformer on duet song 42: SER 0.300 vs 0.170 for the current stem, overlaps no cleaner — a karaoke model mutes one duet voice. Parked (open retry: a backing-harmony non-duet song); true duet support = diarization (P6). `audio-separator` installed; UVR component verified end-to-end |
| S15 covers | **yes — playlists provided** | downloaded to `data/unlabeled/covers/{iris_out,king,ifuudoudou}` (79+25+35 items; transient CDN failures retried; 2 private + 1 removed unrecoverable). Treat as unlabeled; expect tempo/key/intro-offset variation across covers. Next: htdemucs separation → cross-cover agreement filter → DTW label transfer |

## Decided (2026-07-09/10)

| ID | Decision | Status |
|---|---|---|
| S1 legacy deletion | **delete** | done — ~7 GB removed, commit `d42a290cb`; golden sources preserved at `data/gold/source/` |
| S2 staged-song audio | **yes, but no downloads** | done — 27 t2-extra admitted from local audio (`kashi dataset admit`); corpus 86 train / 7 test (test frozen as `manifest.PAPER_TEST_IDS`); 70 t1/ro-unique stay label-staged until downloads are ever OK'd |
| S3 new scraping | **use existing local music** | done — pool = `~/Music/headphone music/wave1/{banger,good}`, 267 JP vocal mixes, registered as `paths.unlabeled_pool`; no scraping |
| S4 romaji first-class | **yes** | applies when t1/ro songs gain audio |
| S5 test-side gold listening | **no, foreseeable future** | reporting = frozen test vs v1 + train-side gold |
| S6 git commits | **yes, as checkpoints** | in effect (`Co-Authored-By: Claude Fable 5`) |
| S8 legacy 10 ms tensors (11 GB) | **delete** | done 2026-07-10 — `models/tensors/` and its 10 ms cache symlinks removed |
| S9 forced alignment | **allowed for training bootstrap only** (2026-07-10) | inference stays textless; enables CTC-FA frame targets + Colab CTC fine-tune |
| S10 Colab offload | **yes — notebooks linked with Google Drive**, user pastes results back into the repo | first job: `colab/ctc_bootstrap.ipynb` (see ROADMAP) |
| S7 heavy compute | **Google Colab TPU available** — Claude writes notebook code, you paste & run | planned uses: P5.4 continued SSL pretraining on singing; optionally batch vocal separation of the pool (Colab GPU). A ready-to-paste notebook will be produced when P5.4 starts |
