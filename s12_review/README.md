# S12 review package — label timing corrections, before/after (+ model)

**Open `index.html` in a browser.** Pick a song in the sidebar; the video plays
with the subtitle tracks side by side — blue = pre-correction labels (from the
backups), aqua = the applied corrected labels (now live in `data/clean`), and on
the 7 test songs violet = **the champion model's own prediction** (textless; no
reference used). Tips: switch source to *vocals only* and speed to 0.5× to judge
±50–130 ms by ear; click any token to jump to it; ←/→ seek 2 s.

If your browser refuses to play the video from `file://`:
`python -m http.server` in this folder, then open <http://localhost:8000>.

- `METHOD.md` — the measurement, every step, with parameters and results.
- `figures/<id>.png` — per-song: the correlation curve + an excerpt overlay.
- `pred/<id>.json` — cached champion predictions for the test songs.
- `media/` — symlinks into the dataset repo (videos) and `data/clean` (vocals);
  they break if this folder is moved out of the repo. Regenerate everything with
  `.venv/bin/python s12_review/make_review.py`.

Corrections are APPLIED (S12, 2026-07-15): originals live in
`data/clean/subtitles_pre_s12/` (and `subtitles_pre_condense/` for the
de-styled songs). Songs included: all 7 frozen-test songs (4 corrected, 3
no-change controls), 2 corrected train examples (13, 70), and song 88
(de-styled, then corrected +100 ms once its measurement became sharp).
