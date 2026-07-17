# S17: phonetic label space (clean_v3) + corpus repairs

Principle (user, 2026-07-17): labels serve karaoke — every mora labeled by its **sung sound**
(particle は → わ, likewise を/へ), no orthography learning; linguistics only where English
makes it unavoidable. Full audits: `runs/s17_phonetic_census.md`, `runs/s17_t1_audio_qa.md`,
worklist `runs/s17_relabel_worklist.csv`.

## Census (what's wrong today)

| class | relabels found | method safety |
|---|---|---|
| は→わ | 633 | **instance-level only** — 50% of は are genuine [ha] words (はやく) |
| を→お | 612 | **blanket-safe** (99.7% sung [o]) |
| づ→ず | 91 | blanket-safe (always merged) |
| へ→え | 67 | instance-level only — 55% genuine [he] (へいき) |
| ぢ→じ | 2 | blanket-safe |

1,405 instances on the 138 ro-covered songs; **1,269 (90%) "flanked"** by exact alignment
agreement = auto-applicable. 136 near-desync → review wave. 14 songs have no romaji track
(15,19,24,55,56,57,58,59,127,132 + test 89–92): 248 orthographic-form tokens, ~163 expected
relabels, not machine-guidable. Gold is orthographic too (~22 window tokens change).
Corpus-wide: 1.73% of tokens.

## clean_v3 build steps (scripted except the review waves)

1. **Blanket rules, all 152 songs**: を→お, づ→ず, ぢ→じ (covers non-ro songs' largest class).
2. **Worklist flanked subset** (1,269 rows): apply by (song, start, token) match.
3. **T1 repairs** (from the GIGO audit):
   - 661 non-inventory `exclude=False` rows (296 bare ー, rest small kana): convert to the
     proper vowel-extension / folded mora; harden `admit.reading_of` (gate currently admits
     ー and small vowels as standalone readings) so future admissions can't reproduce this.
   - 249 overlapping lyric rows across 19 T1 songs: truncate at midpoint of the overlap.
   - id 52: two zero-duration `<silence>` rows dropped.
4. **gold_v3**: 21 ro-confirmed relabels (songs 0/6/16) + blanket rules.
5. **Review waves** (s13-style player, small):
   - Wave A: 136 non-flanked worklist rows.
   - Wave B: ~248 は/へ candidates on the 12 non-ro TRAIN songs.
6. **TEST refs — user decision** (frozen; S12/batch-2 precedent for approved re-baselines):
   - Option A (recommended): apply the 22 ro-verified relabels on 81/83/85 (21 flanked) now;
     songs 89–92 (~39 expected, no ro track) get a user's-ear wave like song 81's Tier-2 items.
   - Option B: defer all test relabeling one cycle; test stays orthographic (known-inconsistent
     with train — model trained on v3 will read ~0.63% worse on test than it really is).
7. **Adoption mechanics**: output to `data/clean_v3/` (never in place), backups implicit;
   `data.version = clean_v3`; priors refit; champion warm retrain (+8 ep, ~12 min local);
   full re-baseline (SER + timed-F1 + gold); leaderboard + approaches.md entries.

Batch-3 drafts are already as-sung — they apply on top of v3 unchanged (blanket rules also
run over them for consistency).

## Eval impact

Refs-relabel with today's (orthographic) champion: test SER 0.2450 → ~0.251 upper bound —
the model writes は where v3 says わ until retrained; retraining on v3 erases this and frees
the capacity currently spent on spelling. Timing metrics unaffected.

## Singer-contamination policy (GIGO finding #1)

Test 89–92 = one singer (Kotone). The unlabeled pool holds **6 more Kotone recordings with
different ytids** that bypass the exact-ytid leak filter (and test-89's exact audio, which is
index-filtered only). Actions:
1. Add the 6 recordings (list in `runs/s17_t1_audio_qa.md`) to a singer-level harvest
   blocklist alongside the existing ytid check.
2. Reporting caveat (standing): pseudo-label gains measured on 89–92 partially reflect
   same-singer familiarity, not unseen-singer generalization; 81/83/85 (different singers)
   are the cleaner generalization readout.
3. No change to the frozen test itself.

Also documented, no action: ids 39/58/64 are the same song in three languages (all train);
T1 tranche exclusion rate 15.2% vs 3.5–5.9% elsewhere (batch 3 addresses it).

## Rollback

clean_v2 untouched; one-line `data.version` revert; worklist CSV makes every relabel
reversible row-by-row.

## Sign-off needed

- S17a: approve clean_v3 build steps 1–5 + 7 (train-side).
- S17b: test-ref option A or B (step 6).
- S17c: singer-blocklist + reporting caveat.
