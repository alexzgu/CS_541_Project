# Tier-2 batch 3 review

Open `index.html` (same player as the pilot: SOURCE / CURRENT / DRAFT lanes).
125 songs, sorted by drafted-mora volume descending — reviewing the top ~20
covers most of the mass. Titles carry `[!N low-conf]` where N blocks were
drafted at low confidence (full list: `s13_batch3/report.md`).

- 2,422 blocks / 13,452 morae drafted (subagent, 2026-07-17); all morae
  validated against the 110-token inventory; test songs (81,83,85,89-92) untouched.
- Ids 93+ have audio-only media (black video canvas — review by ear).
- Approve like the pilot: name song ids (or "all but ..."), then
  `kashi tier2 apply` equivalents run from `s13_batch3/{sid}_draft.csv`.
- Skipped on purpose: 78 credit/metadata crawls, 15 unmappable, 3 symbol-only
  blocks (they stay excluded); see the report for the lists.
