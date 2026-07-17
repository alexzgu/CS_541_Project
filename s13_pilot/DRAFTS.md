# Tier-2 drafts (S13)

## Batch 2 — songs 60, 24, 11, 40, 69, 85 — FOR REVIEW (in the player)

+1,716 morae: 60 愛Dee (+423; spelled letters えるつざゆー…, てぃ→ち per
ロマンチック folding, ゔ→ぶ), 24 (+411, straight English), 11 曇天羊 (+311,
incl. Calliope's rap verse), 40 (+127, constructed scat language mapped
syllable-for-syllable), 69 (+221), **85 (TEST song, +223 — drafted from
source text ONLY, no model assistance; its ambiguous "tease y e a h" block
at 116.9 s stays excluded)**. Note: applying 85 changes frozen-test
references (more ref tokens; SER/timed-F1 re-baseline) — approve it
explicitly or say "apply all but 85".

## Batch 1 — songs 23, 21 — APPLIED 2026-07-16 (user-approved)

*(original pilot notes below)*

Nothing in `data/clean_v2` is changed. Drafts live in this folder as
`{23,21}_draft.csv`; regenerate with `python s13_pilot/make_drafts.py`
(the per-line kana readings are authored in that file — edit there).

## What was done

**Song 23 (English-heavy original):** all 12 excluded lyric lines transcribed
AS SUNG in standard Japanese loan pronunciation — 296 word-fragment rows became
660 mora rows, `exclude=False`, morae distributed over each fragment's time
span (mora-timed). Example: "lets see the world searching for" →
れつしいざわあるどさあちんぐふぉお (っ folded, ー expanded, real ふぉ token).

**Song 21 (sea shanty, interjections):** all 45 blocks — English (しゅりんぷ,
らぶりーおーしゃん, ないすぼーと), stylized kana (みぃ→みい, さぁ×8→ささ…さあ,
いぇい→いえい), and screams (wahhh→わああ). 132 rows → 281 mora rows.

**Song 81 (TEST song): drafted NOTHING — new policy.** Model-assisted drafting
on a frozen-test song would leak the model's own output into its references.
Its two items need non-model sources:
- 51.8–53.3s: a "3 2 1" countdown (heard as neither さんにいち nor すりーつーわん
  cleanly) — needs your ear or the source video.
- 71.9–75.2s: a `<gap>` (subtitler skipped a line) — recoverable only from
  official lyrics text, which I don't have. If you paste the line, I'll time it.

## Findings that shape the full Tier-2 pass

1. **No retrain needed for English.** The inventory already has ふぁ/ふぃ/ふぇ/ふぉ/でぃ;
   the missing extended morae (うぃ/うぇ/うぉ/てぃ/ゔ…) take standard two-mora
   expansions (うぃず→ういず, てぃ→てい) — and the model's own decode of these
   spans *confirms singers do this* (スイート was sung すいーと, not すうぃーと).
2. **Judgment calls to bless (or override):** to→つ and walk→わーく (no とぅ/うぉ
   tokens); いぇい→いえい; the→ざ/ずぃ by position; "TAKO over"→たこおーばー (the pun).
3. **Test songs need a text-only channel** (official lyrics) for Tier-2 fixes —
   worth deciding before the full pass so test refs stay uncontaminated.

## How to review

The drafts reuse the source rows' timing, so a text diff is the fastest check:
compare any block in `{sid}_draft.csv` against `data/clean_v2/subtitles/{sid}.csv`.
If you want them in the video player before deciding, say so and I'll add a
draft lane for these two songs. On approval ("tier2 apply 23 21"), the drafts
replace the rows in `data/clean_v2` (backups kept) and the champion re-evals.
