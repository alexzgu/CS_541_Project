# Dataset v2 — recovering the excluded signal (plan, awaiting S13–S15)

**Problem.** 6,131 lyric rows across the 93-song corpus are `exclude=true` — invisible
to training and eval. Census of the top offenders:

| excluded rows | what they are | example (song 81) |
|---|---|---|
| 666 × `っ` | sokuon has no token in the 110-mora inventory | ずっと → `っ` at 33.2 s dropped |
| ~1,500 multi-mora chunks (`って` 71, `ない` 65, `だん` 55, `よう` 54, `だっ` 48, `かっ` 48, …) | source rows spanning 2+ morae; the importer only maps 1-row = 1-token | 見えなかった → `かっ` at 19.3 s dropped |
| ~400 English fragments (`la` 100, `the` 57, `a` 52, `ing` 52, `i` 46, …) | English lyrics were out of scope | — |
| 191 × `<gap>` | known-missing content markers | stays excluded |
| rest | chorus/overlap exclusions, stylization oddities | — |

The fixes split into three tiers by how much judgment they need.

## Tier 1 — mechanical normalization (deterministic script, no listening) → S13

1. **Sokuon merge**: `っ` is a mora of closure belonging to the preceding syllable.
   Merge each `っ` row (and the `っ` inside chunks like `かっ`, `だっ`, `って`)
   into the preceding token's row, extending its `end`. No inventory change; the
   textless decoder never needs to emit a silent token.
2. **Chunk split**: rows whose text decomposes into inventory morae (`ない` → `な`+`い`,
   `よう` → `よ`+`う`, `だん` → `だ`+`ん`) are split, dividing the row's span evenly.
   Even division is imperfect but far better than exclusion — and the S12 audit
   machinery can verify the result acoustically per song.
3. Output goes to a **new version, `data/clean_v2`** (the config's `data.version`
   slot already exists) — `clean` stays untouched, so all current baselines remain
   reproducible. Token sequences change ⇒ SER re-baselines when v2 is adopted;
   adoption goes through the usual gate: champion re-evaluated on v2 refs, gold
   subset regenerated/checked first.

*Cost: a day. Recovers roughly 2,000+ rows of training/eval signal.*

## Tier 2 — Claude-assisted standardization (listening + judgment) → S13 pilot

For what a script cannot decide: English sung as katakana-Japanese (`the` → ザ,
`la` → ラ …, transcribed **as sung**, staying inside the kana inventory), morae the
original subtitler skipped, stylized spellings, and chorus markers. Workflow per song:

1. Assemble three sources: current rows, the champion's spike decode (with
   confidences), and the audio.
2. Claude drafts standardized rows under a fixed policy table (は/を/へ phonetic —
   already homophone-aware; long vowels as sung; English as-sung kana; chorus
   spans tagged, see Tier 3); every changed row carries a reason code.
3. Review with the existing s12_review player (it already does before/after);
   user spot-checks, approves per song.

**Pilot: 3 songs** — 81 (missing-morae case), one English-heavy song, one
multi-singer song — then decide whether to scale to the corpus.

*Cost: pilot ~a session; full corpus is O(10–20 songs per session).*

## Tier 3 — chorus & multiple singers → S14

Current state: chorus labeling is inconsistent across songs; overlapping main
singers were excluded outright.

1. **Near-term (recommended first)**: try off-the-shelf **lead-vocal isolation**
   models (UVR "karaoke" family separates lead from backing). The separator is
   already a registry component — this is a config swap plus an eval. If the lead
   stem is clean, main-singer labels become well-defined even under overlap, and
   chorus rows can be tagged `chorus=true` rather than excluded.
2. **Later (cloud, opt-in)**: train our own separator on creator-posted
   with/without-vocals pairs from YouTube (paired data exists; needs download
   sign-off and a Kaggle/Colab budget).
3. **Research (P6)**: architectures that estimate singer/chorus count on the fly
   (joint diarization + transcription). Filed, not scheduled.

## Tier 4 — covers → S15

User has playlists of many covers per song. Three concrete uses, cheapest first:

1. **Cross-cover agreement filter for self-training**: spike-decode every cover;
   pseudo-transcript spans that agree across covers of the same song are
   high-precision training crops (extends the existing `kashi loop ctc-harvest`).
2. **Label transfer**: DTW-align a labeled original to each cover → timed labels
   for free on new audio (voices differ, lyrics don't).
3. **Robustness eval**: same reference lyrics, different singers = a natural
   generalization test set.

Needs: the playlist URLs and a download OK (S2 previously said no downloads —
this would be a scoped exception), or local files if they already exist.

## Sign-offs requested

| ID | Ask |
|---|---|
| S13 | Tier 1 mechanical normalization into `data/clean_v2` + Tier 2 pilot on 3 songs |
| S14 | Tier 3.1 lead-vocal separator trial (local, off-the-shelf); 3.2 stays parked until you opt in |
| S15 | Tier 4: provide cover-playlist URLs + scoped download OK (or point at local copies) |
