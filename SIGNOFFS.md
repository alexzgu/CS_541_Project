# Sign-offs

Reply per ID. Decided items stay for the record.

## Open

*(none — S13/S14/S15 approved 2026-07-15, see below)*

## Decided (2026-07-15, second batch)

| ID | Decision | Status |
|---|---|---|
| S11 gate-exception promotion | **yes** | `ctc_model = artifacts/ctc_deprior/out/ctc_model` stands (test SER 0.316, gold SER 0.170, gold timed-F1 0.741; shift −1 gold-validated) |
| S12 per-song ref timing correction | **yes — APPLIED 2026-07-15** | 18 songs shifted +50..+130 ms (17 + song 88 after de-styling; backups `data/clean/subtitles_pre_s12/`). Rebaseline (same champion): timed-F1 0.314→**0.699**, bF1@50 0.412→**0.717**, PC 0.963; song 89 timed-F1 0.017→0.906. Review package kept at `s12_review/`. Perceptual note adopted as `subtitles.display_lead_ms = 100` (srt/vtt/ass lead the true timings; csv/eval stay honest) |
| S12b styling-repeat condensation | **yes (user-suggested)** | 7 songs de-styled (67 ms color-fade frames merged; song 88: 3,264→422 rows; いい-type real doubles protected by the duration criterion; backups `data/clean/subtitles_pre_condense/`); song 88's offset became measurable afterwards (+100 ms applied) |
| S13 dataset v2 Tier 1+2 | **yes** | Tier 1 DONE + **`data.version = clean_v2` adopted**: 4,290 rows recovered (728 sokuon merges, 1,441 chunk splits; `kashi.data.normalize`); champion on v2 test refs: **SER 0.263** / timed-F1 0.710 (recovered tokens were already predicted). `gold_v2` built the same way (103 rows). Tier 2 pilot queued on songs 81 (multi-singer), 23 (English-heavy, 300 rows), 21 (`<gap>`/stylized) |
| S14 lead-vocal separator trial | **yes** | off-the-shelf UVR karaoke-model trial queued (needs `audio-separator` install; component slot exists); custom separator training stays parked until user provides pair data |
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
