# Approaches by abstraction level

The task is textless syllable-level transcription of sung Japanese into timed hiragana (no transcript at inference; forced alignment permitted as a training bootstrap only, S9). Sung audio breaks most speech-pipeline assumptions — held vowels, melisma, ±50 ms label jitter, no word boundaries — so no single abstraction level suffices: raw-signal methods settle timing and segmentation questions that no classifier can see past, phonetics defines what a "correct" token even is for as-sung material, and language-level priors are the only source of context once acoustics run out. This document catalogs every approach tried or considered, placed on the signal → phonetics → language axis, with measured results and tradeoffs. All numbers are frozen-test (`PAPER_TEST_IDS`, 7 songs) unless noted; SER rebaselines at the clean_v2 ref adoption (v1/v2 refs marked where it matters). Sources: `runs/leaderboard.csv` (append-only record), `docs/project_summary.md`, `ROADMAP.md`, `SIGNOFFS.md`.

## Abstraction spectrum

| approach | level | status | evidence (one line) |
|---|---|---|---|
| energy/RMS segmentation + legacy LSTM | signal | superseded baseline | SER 0.797 / timed-F1 0.106 (day-1 end-to-end) |
| spectral-flux onset audit (S12) | signal | promoted (as instrument) | timed-F1 0.314→0.699 with zero model change |
| voicing (autocorr) + RMS frame channels | signal | adopted narrow | noise-tag P 0.2–0.4 / R 0.06–0.17 — below target |
| TDA (ripser H1) voicing | signal | parked (P6) | never benchmarked vs autocorr; same quantity, heavier dependency |
| speed-perturb augmentation | signal | live, unattributed | only tried inside rejected `ctc_bundle` (fails gates by ~0.008) |
| frozen wav2vec2/HuBERT features + heads | signal→symbol | superseded | frame-acc 0.382/110; SER floor 0.722 (frame era) |
| HuBERT CTC fine-tune + spike decode | signal→symbol | **promoted — champion path** | SER 0.722→0.364 at bootstrap; 0.245 today |
| de-peaking: training-time label prior | signal→symbol | rejected as de-peak; kept as regularizer | prior absorbed into lm_head bias; SER 0.332→0.316 |
| de-peaking: decode-time prior + Viterbi | signal→symbol | rejected | SER 0.78 vs 0.32 spike; non-spike frames carry no token mass |
| insertion decoding (held-vowel recovery) | signal→symbol | rejected | gap frames pure blank; +0.007 SER of junk at permissive θ |
| forced alignment as training bootstrap (S9) | signal→symbol | adopted, bootstrap only | enables CTC; FA-extended frame targets halve timed-F1 (0.114) |
| beatrice phoneme-CTC warm start | signal→symbol | rejected | never escapes blank collapse in 12 ep (blank-convention clash) |
| training curriculum: warm-lineage vs from-scratch | signal→symbol | measured — lineage holds | cold-start 0.2108 vs warm 0.1808 greedy, identical corpus-152 |
| temporal-contrastive projection head | signal→symbol | rejected | frame-acc 0.396→0.351 |
| articulatory kernel k(·,·), three uses | phonetic | adopted | PC 0.963; residual subs are kernel-adjacent (だ↔た) |
| bucket hierarchy (coarse→fine classifier) | phonetic | rejected by measurement | 7% of errors within-bucket at 62-bucket granularity |
| sokuon merge + chunk split (clean_v2) | phonetic | promoted | SER 0.316→0.263, zero model change |
| as-sung kana for English/vocables (Tier 2) | phonetic | live (8 songs applied) | +2,657 morae; test refs enriched via song 85 (SER re-baselined 0.2570) |
| T1 furigana corpus admission (corpus-152) | phonetic | **provisional champion (S16)** | +59 labeled songs (train +66%); SER 0.257→0.245, gold-arbitrated |
| romaji parallel-track QA (ro_dual) | phonetic (orthography) | direction closed (measurement) | 98.81% agreement / 70,289 morae; 0 label errors surfaced |
| lyrics bigram beam rescoring | language | rejected | best λ=0.25 gains 0.0015 SER |
| text bigram (Tatoeba, 4.5M transitions) | language | rejected, direction closed | worse than lyrics bigram at every λ |
| any substitution-only rescoring | language | closed | lattice oracle 0.2572 vs greedy 0.2894 — ~0.03 cap |
| semi-Markov decoder (durations + bigram) | language | promoted (frame era) → superseded | SER 0.797→0.744; SER 0.994 when run over CTC emissions |
| sticky HDP-HMM/HSMM unit discovery | language (proto) | rejected for realign; P6 open | snapped gold F1@50 0.528 < v1's 0.552 |
| pool self-training, round 1 | data-as-model | promoted | SER 0.339→0.332; equal-compute labeled-only control regressed |
| pool self-training, round 2 | data-as-model | rejected | SER 0.3547 regresses; greedy identical — one round holds the value |
| cross-cover 3-gram consensus | data-as-model | promoted — prior champion (S16 base) | SER 0.263→0.252, timed-F1 0.710→0.725, 5/7 songs |
| label repair (S12 + de-styling + clean_v2) | data-as-model | promoted | ~half the total timing gain; SER 0.316→0.263 |
| karaoke lead-isolation separator (S14) | data-as-model | parked | duet SER 0.300 vs 0.170 — mutes one duet voice |

## Signal level

**Energy/RMS segmentation.**
- What: threshold segmentation on log-RMS, feeding the legacy per-segment LSTM. The day-1 end-to-end system and permanent zero-training fallback topology.
- Code: `src/kashi/components/segmenters.py` (`energy`).
- Result: SER 0.797 / timed-F1 0.106 — despite the classifier scoring 0.532 accuracy on ground-truth segments. Segmentation, not classification, was the bottleneck.
- Buys: zero training, fully interpretable, runs anywhere. Costs: blind to everything but loudness; can never segment legato/melisma.

**Spectral-flux onset audit (S12).**
- What: model-free timing instrument. STFT spectral-flux envelope (32 ms Hann, 10 ms hop) cross-correlated with a Gaussian bump train at label onsets, lag search ±200 ms; gold songs calibrate the instrument's inherent −40 ms bias.
- Code: `src/kashi/data/offset_audit.py`; output `runs/ref_offset_audit.txt`.
- Result: 18 songs measured 30–130 ms EARLY, corroborated independently by the decoder onset-shift sweep (test timed-F1 peaked at shift −4 while gold peaked at −1). Corrections applied: timed-F1 0.314→0.699 with SER unchanged; song 89 went 0.017→0.906 — the model had been right, the labels were 130 ms off.
- Buys: no learned parameters, hence no circularity with the model under audit. Costs: measures only per-song constant offsets, not per-row jitter; needs a gold subset to calibrate.

**Voicing/autocorr + RMS frame channels.**
- What: per-frame periodicity v_t = max normalized autocorrelation over lags for F0 ∈ [70, 500] Hz, plus e_t = log RMS; appended to frame features, reused for `<noise>` tagging and as a boundary-candidate source. Targets the voicing confusion axis (ka/ga) and breath detection.
- Code: `src/kashi/stats/tda.py` (`autocorr_periodicity`), `src/kashi/components/boundaries.py`.
- Result: breath/noise tagging below target — P 0.2–0.4 / R 0.06–0.17 at −48 dBFS; breaths are quieter and shorter than the RMS gate.
- Buys: nearly free, interpretable scalar evidence. Costs: too coarse for the phenomena it was aimed at.

**TDA (ripser) voicing.**
- What: interchangeable estimator of the same v_t — Takens delay embedding → Vietoris–Rips filtration → longest normalized H1 loop lifetime (periodic ⇔ persistent loop).
- Code: `src/kashi/stats/tda.py` behind the `[tda]` extra.
- Result: none yet — parked as a P6 one-notebook-one-table benchmark vs autocorr.
- Buys: principled periodicity detection. Costs: heavy dependency for a quantity autocorrelation already estimates; only justified if the benchmark shows a difference.

**Speed-perturb augmentation.**
- What: resample each training crop by 1±U(0, s) before CTC training.
- Code: `SPEED_PERTURB` in `colab/ctc_bootstrap.ipynb`.
- Result: tried once at s=0.1, only inside the rejected `ctc_bundle` (bundled with Tier-2 rows + covers pseudo; fails both gates by ~0.008/0.003). Never tested in isolation — status is unattributed, queued for the next, larger bundle.
- Buys: free tempo diversity, no labels touched. Costs: the resample trick shifts pitch and tempo jointly; attribution requires a bigger data delta.

## Signal-to-symbol level

**Frozen wav2vec2/HuBERT frame features.**
- What: pretrained speech-SSL backbone (legacy: English `facebook/wav2vec2-base`; current: `yky-h/japanese-hubert-base` mirror), 20 ms stride, 768-d, frozen in the frame era with MLP/LSTM heads on top.
- Code: `src/kashi/components/encoders.py`.
- Result: frame era ceiling — frame-acc 0.382/110, end-to-end SER 0.722 with the segmental decoder over the heads.
- Buys: speech pretraining transfers to singing well enough to carry the whole project; features cache once. Costs: 6 GB VRAM caps at base-size backbones; frozen features leave all sequence structure to the decoder — which proved the wrong division of labor.

**CTC fine-tune + spike decoding — the champion path.**
- What: fine-tune the HuBERT backbone with CTC on (audio, token-sequence) pairs — timestamps never enter the loss — then decode greedily: each non-blank argmax spike emits a token; onset = spike − 1 frame (gold-validated); extent capped at next onset / 0.6 s; silence gap-fill.
- Code: training `colab/ctc_bootstrap.ipynb` via `colab/run_nb.py`; decoding `SegmentalDecoder._spike_decode` / `_ctc_log_probs` in `src/kashi/components/decoders.py`.
- Result: SER 0.722→0.364 at the S9 bootstrap (12 ep, greedy 0.198), 0.339 at 24 epochs, 0.245 today after data-level gains.
- Buys: sequence supervision is immune to the label timing jitter that twice masqueraded as model error; segmentation learned implicitly. Costs: no per-term score decomposition (interpretability lost vs the segmental decoder); the peaky solution concentrates all information at onsets — the root of every failure below.
- Related rejection: warm-start from a beatrice phoneme-CTC checkpoint never escapes blank collapse in 12 epochs (conflicting blank conventions; the mirror escapes at epoch 6).

**CTC peakiness and the de-peaking failures.**
- What: peaky CTC puts blank mass 0.94 on non-spike frames, token runs median 1 frame. Three attempts to work around it, all measured dead:
- Training-time label prior (`LABEL_PRIOR_ALPHA=0.3`): a per-class log-prior offset is absorbable into the `lm_head` bias — peakiness probe unchanged. Survived only as a regularizer (`ctc_deprior`: SER 0.332→0.316, gold SER 0.195→0.170).
- Decode-time prior (MMS-FA style) + semi-Markov Viterbi over the emissions: monotone in α (SER 0.96/0.91/0.78 at α 0.3/0.6/1.0) but structurally hopeless vs 0.32 spike — non-spike frames carry no token mass, so span scores carry no evidence.
- Insertion decoding for held-vowel deletions (う/い/ん = top deletions; `propose_continuations` in `decoders.py`, tuned by `src/kashi/eval/tune_insert.py`): gap frames are pure blank — permissive θ inserts 104 junk tokens (+0.007 SER), strict θ finds nothing.
- Consequence: token extents must come from acoustics, not the DP; vowel-continuation deletions are a training-time limitation; the decode program is closed.

**Forced alignment as training bootstrap only (S9).**
- What: inference stays textless; FA of known training lyrics is allowed to seed training. Guarded by a blank-collapse check (`greedy_test_ser` recorded; FA refused if collapsed).
- Result: the naive use — FA-extended frame targets (stretch labels to aligned extents) — halves timed-F1 to 0.114, rejected. The productive use is CTC training itself.
- Buys: exploits the one thing training data has that inference doesn't (the transcript) without touching project identity. Costs: FA-derived extents import alignment errors directly into targets.

**Temporal-contrastive projection head (P5.2).**
- What: 128-d InfoNCE head over frozen features, nearby frames as positives.
- Result: rejected — frame-acc 0.396→0.351, SER 0.786. Unsupervised invariance objectives can discard exactly the local detail a 110-way frame classifier needs. (Sibling: weak-teacher frame pseudo-labeling, P5.1 v1, also rejected — frame-acc 0.396→0.394.)

**Cold-start collapse, the training curriculum, and cloud offload.**
- What: can corpus-152 train from scratch (vs the warm bootstrap→pseudo→covers→corpus-152 lineage), and can cloud (Kaggle T4) stand in for the local Ampere card?
- Code: `colab/kaggle_push.py` / `colab/kaggle_watch.py` (push-button offload); artifacts `artifacts/ctc_scratch152`; leaderboard `ctc_scratch152_local`, `ctc_cloud_warm4`.
- Result (environment, not data): from-scratch on T4 collapsed twice (12 ep, loss pinned at the ~4.33 all-blank plateau, test SER 1.0) under LABEL_PRIOR_ALPHA both 0.3 and 0.0 — regularizer exonerated. The identical recipe run LOCALLY succeeds — collapse breaks at epoch 4, greedy test SER 0.2108 by ep 12 — so the blocker is the cloud environment (unpinned latest `transformers`, and/or T4 fp16 autocast vs the Ampere card), not the T1 data or the prior. The S9 blank-collapse FA guard caught it both times. Mitigation: the canonical notebook now pins `transformers==5.13.0` — and the pinned re-test (v7) CONFIRMED it: collapse breaks at ep 10 on T4 (greedy 0.2302 by ep 12, still descending), so the unpinned library was the blocker and T4 fp16 only delays the escape (~6 extra plateau epochs). Cloud cold-starts are viable with the pin plus an epoch surcharge.
- Infra: production cloud jobs are WARM continuations, so the collapse is operationally moot — a warm +4 ep from the uploaded champion (Kaggle dataset `kashi-ckpt`) ran the full loop on T4 (guard passed, FA'd all 152 songs, packaged `ctc_out.zip`, auto-fetched by `kaggle_watch.py`), greedy 0.1842 vs champion 0.1808 = the known same-data continuation oscillation. Cloud offload for larger/longer models is now push-button, zero manual uploads.
- Curriculum: cold-start 0.2108 vs warm-lineage 0.1808 on identical data — the incremental path holds a ~0.03 greedy edge over from-scratch. Training curriculum is load-bearing, not just data volume.

## Phonetic level

**Articulatory kernel k(·,·).**
- What: each token decomposes as (consonant, glide, vowel); similarity = 0.45·consonant (voice/place/manner, weights 0.2/0.4/0.4) + 0.45·vowel (height/backness L1) + 0.10·glide; homophones (を/お, ぢ/じ, づ/ず) pinned at 0.95; 110×110 matrix PSD-projected. Worked values: k(ka,ga)=0.91, k(ka,ta)=0.82, k(ka,ki)=0.66, k(ka,n)=0.10.
- Code: `src/kashi/phonetics.py`; spec §5.3. Three uses: soft classification targets (α=0.1, power 4), soft contrastive negatives, partial-credit metric.
- Result: champion PC 0.963. The residual error profile validates the geometry: top substitutions are kernel-adjacent (だ↔た, ね→み).
- Buys: domain knowledge at zero data cost; graded credit where flat CE sees only right/wrong. Costs: hand-fixed weights, not learned; as a metric it can flatter near-miss models.

**Bucket hierarchy (coarse→fine).**
- What: user-proposed — classify into articulatory buckets first, refine within. Measured on the spike lattice before building anything.
- Result: at meaningful granularity (62 buckets, voicing/nasal merges) only 7% of errors are within-bucket — errors cross exactly the boundaries a hierarchy would freeze. Rejected by measurement (`ctc_beam_bigram` leaderboard note); the kernel's soft targets already capture the recoverable structure.

**Sokuon/chōon policy (clean_v2 normalization).**
- What: っ has no token — a mora of closure folds into the host mora's row (728 merges); rows spanning decomposable multi-mora chunks split evenly (ない→な+い; 1,441 splits); 4,290 of 6,131 excluded rows recovered. Long vowels transcribed as sung (Tier-2 policy) — held vowels are real morae, which is also why they dominate deletions.
- Code/plan: `docs/dataset_v2_plan.md` Tier 1; output `data/clean_v2`.
- Result: champion SER 0.316→0.263 on the fairer refs with zero model change — it was already predicting those tokens.
- Buys: refs describe what is sung, not what is written. Costs: even span division is imperfect (auditable by the S12 machinery); SER comparability breaks across the rebaseline (marked in the leaderboard).

**As-sung kana for English/vocables (Tier 2).**
- What: English and vocables transcribed as sung in loan-kana, inside the existing 110-mora inventory (missing うぃ/てぃ handled as two-mora expansions; no inventory change). Policy: no model-assisted drafting on test songs (contamination).
- Result: batch 1 (songs 23 +660, 21 +281) and batch 2 (60/24/11/40/69 + TEST song 85 from source text only, +1,716) applied after user review of `s13_pilot/`; test refs grew 3,291→3,514 tokens, re-baselining the champion at SER 0.2570 / timed-F1 0.7156 — song 85's own SER improved 0.488→0.434 because the refs now credit the English spans the model already sings in kana. Batch 1's +973 train tokens were unattributable inside `ctc_bundle`; ~1.8k tail rows over ~80 songs remain for batch 3.
- The song-40 point (user): for scat vocables like "ri lu la" an English-vs-Japanese classification is meaningless — the phonetic level is the right abstraction, and the mora inventory covers them regardless of language.
- Buys: recovers signal a language-level label schema had excluded. Costs: listening and judgment, O(10–20 songs/session).

**T1 furigana corpus admission (corpus-152 retrain, S16).**
- What: 59 songs (ids 93–151) admitted from the T1 dataset — a kanji+furigana half-density subtitle format. Furigana readings are extracted (漢字(かな)→かな) and run through the standard clean_v2 normalizer — no new policy, the phonetics are read straight off the annotations rather than judged by ear. Corpus now 152 labeled / 145 train / 7 frozen test; train segments +~66% (bigram priors now 80,561 segments / 78,529 transitions). The retrain is the covers champion checkpoint + 8 ep on the full corpus, 3,049 crops (was 1,571 pre-admission), LABEL_PRIOR_ALPHA=0.3, ~90 s/ep on an RTX A3000.
- Result: greedy no-LM test SER 0.2045→0.1808 by epoch (best ep 7) against the covers champion probed at ~0.20 on the same crops — the new labels are worth ~0.02 greedy SER. End-to-end gate (spike decode): SER 0.2570→0.2450 (−0.012, passes); timed-F1 0.7156→0.7145 (−0.0011, timing noise — misses the letter of the F1 gate); boundary@50 F1 identical 0.7237. Gold arbitration breaks the tie the F1 gate leaves: pooled gold SER is EXACTLY tied (both models make 256 edits on 1,459 gold-window tokens, distributed differently — the candidate is clearly better on the hardest gold song 19, 0.438→0.390, slightly worse on song 0, 0.138→0.171), and gold timed-F1 favors the candidate +0.0054 (0.7414→0.7469). Promoted provisionally under the S11 gold-arbitration precedent — here with a 15× smaller timing delta — as leaderboard `ctc_t1_152`; SIGNOFFS.md S16, pending user ratification.
- Buys: honest labeled-phonetic scaling — furigana is a ready-made phonetic transcript, so admission is near-free (none of Tier-2's per-song listening) and it delivered ~5× the gain of the last decode-side attempts. This is where the champion crown moved off a data-as-model trick and onto plain labeled-data scale, reinforcing that with the decode program closed (greedy within ~0.03 of the lattice oracle) residual gains live in model/data. Costs: furigana accuracy is inherited from the source dataset, not audited song-by-song; the win clears SER but sits inside timed-F1 noise, hence provisional until ratified.

**Romaji parallel-track label QA (ro_dual).**
- What: 138/152 songs ship a parallel ROMAJI phonetic track (dual-subtitle uploads) — a second orthographic rendering of the same morae — aligned homophone-aware (は=わ, を=お, へ=え, じ=ぢ, ず=づ) against the hiragana labels as a redundancy cross-check.
- Output: `runs/ro_dual_qa_report.md`; leaderboard `ro_dual_qa`.
- Result: 98.81% agreement over 70,289 aligned morae, 131/138 songs ≥ 0.95. Only 82 clean isolated substitutions across train songs, and every interpretable one is a ROMAJI-side artifact — kana-ized English proper nouns (マリン vs `marine`), song 35's glitched track prepending a phantom `d`, dropped consonants — ZERO genuine hiragana-label errors. Falsified the prior belief that romaji tracks carry no English (77/138 do, 1,887 Latin tokens), but the labels correctly exclude those spans. Kept only as a per-song sanity metric — not an S12-class lever, not a Tier-2 source.
- Buys: a free orthographic redundancy channel straight from upload metadata, zero listening. Costs: it measured CLEAN — confirming label quality is no longer the low-hanging fruit of the S12/clean_v2 era; residual gains live in model/data, so this closes a lever rather than opening one.

**Phonetic (as-sung) label space — clean_v3 (S17, APPLIED 2026-07-17).**
- What: user decree that labels serve karaoke — every mora labeled by its SUNG sound, never its spelling (particle は→わ, を→お, へ→え; rendaku づ→ず, ぢ→じ). The ro_dual alignment (above) flipped from sanity check to relabel oracle: its homophone folds located each instance. Census first: ~half of all は/へ are genuine [ha]/[he] words → blanket rules only for を/づ/ぢ (を is 99.7% [o]); は/へ relabeled instance-level from the 90%-flanked worklist. Plus T1 structural repairs (610 composite ー-rows decomposed into real morae, 145 adjacent overlaps midpointed, 47 junk rows excluded) and the S17c Kotone harvest blocklist.
- Output: `kashi.data.phonetic`, manifest `data/clean_v3/CHANGES.tsv` (2,670 changes ≈ 1.7% of tokens); leaderboard `REBASELINE_S17`.
- Result: +8 ep warm retrain on v3 → NEW BASELINE SER 0.2516 / timed-F1 0.7200 / bF1@50 0.7312 (v3 refs). Test SER exactly ties the old champion — the Option-A seam: fixes on relabeled 81/83/85 offset by still-orthographic 89–92 (ear wave pending). GOLD, fully phonetic, shows the real effect: **SER 0.1885→0.1583 (−0.030), timed-F1 0.7365→0.7672 (+0.031)** — the largest single gold jump since the decode-program era.
- Buys: the model stops spending capacity on orthography (a pure linguistics tax with zero karaoke value); output is directly singable. Costs: one-time re-baseline breaks comparability with every pre-S17 number; the frozen test needs a user's-ear wave on 89–92 (~39 tokens) before its SER reads true; 14 non-ro songs' residual は/へ (~248 candidates) await review waves.

## Language level

**Bigram priors.**
- What: add-k syllable bigram over training lyrics (45.8k transitions) vs a text-trained bigram (Tatoeba 249k sentences kana-ized via pykakasi, 4.5M transitions).
- Code: `src/kashi/stats/lm.py`.
- Result: in the frame era the lyrics bigram earned its promoted slot (λ_lm=0.3). Over the CTC spike lattice: best λ=0.25 gains 0.0015 SER; the text bigram is worse than the lyrics one at every λ — domain mismatch beats a ~100× scale advantage. Both rejected.
- Buys: priors pay when acoustics are weak. Costs: noise once spikes are ~0.9-confident; sung-lyrics word order is not prose word order, so more text is not more signal.

**Beam rescoring and the oracle ceiling.**
- What: exact DP over per-spike top-K candidates with a bigram; ceiling measured before further LM investment.
- Code: `beam_pick` in `src/kashi/components/decoders.py`.
- Result: lattice oracle SER 0.2645 (top-3) / 0.2572 (top-8) vs greedy 0.2894 — any substitution-only rescorer is capped at ~0.03 SER, and the actual error mass is deletions/insertions. Direction closed on the measurement, not on a failed model.

**Semi-Markov segmental decoder.**
- What: exact Viterbi over all (segmentation × labeling); per-span score = frame-posterior sum + NB duration prior + Model-1 boundary logit + optional bigram; O(T·D_max·K²) with cumulative sums; N-best LSTM rescoring pass.
- Code: `SegmentalDecoder` in `src/kashi/components/decoders.py`; durations `src/kashi/stats/durations.py`; spec §6.
- Result: promoted in the frame era — SER 0.797→0.744, timed-F1 0.106→0.203 (λ_d=2, λ_lm=0.3); the naive λ_d=0.5 config over-segmented to SER 1.51, i.e. the priors are what tamed it. Superseded by the CTC path; run over peaky CTC emissions it collapses (span-sum SER 0.994 — blank swamps spans).
- Buys: every term inspectable per decoded segment; priors substitute for scarce data; the two-stage baseline is its λ=0 degeneration. Costs: needs emissions with mass on all frames — structurally incompatible with peaky CTC.

**Sticky HDP-HMM/HSMM unit discovery.**
- What: nonparametric acoustic-unit discovery (weak-limit blocked Gibbs FFBS, L=120, ρ=0.95 ≈ 400 ms mora-scale dwell); the fully language-free end of the axis — it invents its own inventory.
- Code: `src/kashi/stats/hmm.py`; snapping `src/kashi/stats/snapping.py`.
- Result: as an unsupervised boundary source for label realignment, rejected after three variants — candidates are near-dense (recall 0.99+ at ±100 ms), so snapping is indiscriminate jitter; best snapped gold F1@50 0.528 vs v1's 0.552; every unsupervised boundary source measured is a worse clock than the v1 labels. As a unit-discovery report vs the 110-token inventory (P6): open, not yet run.
- Buys: touches no labels or sequences. Costs: its boundaries answer "where does the signal change", not "where does the mora start".

## Data-as-model level

**Self-training on the pool.**
- What: spike-decode the 267-song local pool (266 after the leak guard caught test song 89's source video) into pseudo-transcript crops; conf ≥ 0.85, loss weight 0.3.
- Code: `kashi loop ctc-harvest` → `src/kashi/train/pseudo.py`.
- Result: round 1 promoted — +4 epochs on labeled + 9,605 crops, SER 0.339→0.332, while the equal-compute labeled-only control regressed to 0.347 (gain attributable to the unlabeled data). Round 2 rejected — SER 0.3547, greedy identical (0.1737): one round is where the value is.
- Buys: free data. Costs: confirmation bias — the confidence filter passes the model's own errors back in, which is what saturates it after one round.

**Cross-cover 3-gram consensus — current champion.**
- What: 134 covers of 3 songs (htdemucs-separated); spike-decode each; a crop survives only if ≥ 50% of its token 3-grams are reproduced by ≥ 2 other covers, then conf ≥ 0.85. N-grams are time-free, absorbing tempo/key/intro variation; per-cover hallucinations have no cross-cover support.
- Code: `kashi loop covers-harvest` → `src/kashi/train/covers.py`.
- Result: 2,095 crops, +8 epochs → SER 0.263→0.252, timed-F1 0.710→0.725, better on 5/7 test songs — where labeled-only continuation, pool round 2, and v2-labels-only continuation had all saturated. Fresh voices singing verified lyrics is the lever that still works.
- Buys: breaks self-training's confirmation-bias loop with an external consistency check no single model can fake. Costs: requires songs with cover ecosystems; download/curation overhead.

**Label repair as model improvement.**
- What: three interventions with zero model change — S12 per-song offset correction (+50..130 ms on 18 songs), de-styling of karaoke color-fade frames (song 88: 3,264→422 rows, which made its offset measurable), clean_v2 normalization.
- Result: timed-F1 0.314→0.699 (S12) and SER 0.316→0.263 (v2). Roughly half the project's total timing gain came from the labels, not the model — measured, not assumed.
- Buys: cheapest SER/F1 per hour of anything in this document. Costs: bounded (each repair is one-shot); demands instruments (S12 audit, gold subset) that separate label fault from model fault — without them repair tunes on noise (the onset-shift −4 near-miss).

**Covers/multi-singer and separator choices.**
- What: separators are registry components — mel-roformer/UVR default on the labeled corpus, htdemucs on the covers pool. S14 trialed UVR karaoke-roformer lead-isolation to make duet labels well-defined.
- Code: `src/kashi/components/separators.py`.
- Result: on duet song 42 the lead-isolated stem scores SER 0.300 vs 0.170 for the current stem (840 vs 980 tokens; overlap spans no cleaner, 95 vs 97) — in a duet both voices are lead, and the model mutes one. Parked; possible retry on a backing-harmony non-duet song; true duet support = diarization (P6).
- Buys: separation quality is a config swap, cheap to trial. Costs: "lead vocal" is not a signal-level concept when two singers alternate — the failure is definitional, not acoustic.

## Patterns

1. **Peaky CTC concentrates all sequence information at onsets.** Three decode-side programs died on the same fact: span-sum semi-Markov (SER 0.994), decode-time prior + Viterbi (0.78), held-vowel insertion (gaps are pure blank). Extents and deletions are model/training problems, not decoder problems.
2. **Label quality masqueraded as model error twice** — S12 timing (timed-F1 0.314→0.699; song 89 0.017→0.906) and clean_v2 refs (SER 0.316→0.263), both with zero model change — and once nearly caused a test-tuned regression (onset shift −4 was fitting label noise; gold caught it).
3. **New data sources beat continued optimization.** Labeled-only continuation regressed (0.347), pool round 2 regressed, v2-labels-only tied; fresh voices via covers delivered the 0.252 champion, and admitting 59 more labeled songs (T1 furigana, corpus-152) delivered the current one (0.245, provisional) — both wins were new data, not new decoding.
4. **Measure the ceiling before building.** The bucket hierarchy (7% within-bucket) and all LM rescoring (oracle 0.2572 vs greedy 0.2894) were closed by cheap measurements, not by failed training runs.
5. **Every level contributed, at its own moment.** Signal fixed the refs (S12); signal-to-symbol delivered the step change (SER 0.722→0.364); phonetics defined correctness for as-sung material (0.316→0.263), the error geometry, and the latest gain via labeled-corpus scale (0.257→0.245, S16); language priors carried the weak-acoustics frame era (0.797→0.744), then stopped paying against ~0.9-confident spikes; data-as-model owns the gains between (0.339→0.252).
6. **The training curriculum is load-bearing, not just the data.** Identical corpus-152 reaches greedy 0.2108 trained from scratch vs 0.1808 down the incremental warm lineage (bootstrap→pseudo→covers→corpus-152) — ~0.03 from path alone. Consistent with the beatrice warm-start clash and the cloud cold-start collapse: CTC's cold-start alignment discovery is fragile, so how you reach a checkpoint matters alongside the labels in it.

## Maintenance

Update this document at every promotion/rejection, in the same change as the `runs/leaderboard.csv` row.
