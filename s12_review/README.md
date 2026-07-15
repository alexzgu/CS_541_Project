# S12 review package — label timing corrections, before/after

**Open `index.html` in a browser.** Pick a song in the sidebar; the video plays
with both subtitle tracks at once — blue = current labels, aqua = proposed
correction. Tips: switch source to *vocals only* and speed to 0.5× to judge the
±50–130 ms shifts by ear; click any token to jump to it; ←/→ seek 2 s.

If your browser refuses to play the video from `file://`:
`python -m http.server` in this folder, then open <http://localhost:8000>.

- `METHOD.md` — the measurement, every step, with parameters and results.
- `figures/<id>.png` — per-song: the correlation curve + an excerpt overlay.
- `subtitles_corrected/<id>.csv` — the staged corrected references
  (**`data/clean` is untouched** until "S12 apply").
- `media/` — symlinks into the dataset repo (videos) and `data/clean` (vocals);
  they break if this folder is moved out of the repo. Regenerate everything with
  `.venv/bin/python s12_review/make_review.py`.

Songs included: all 7 frozen-test songs (4 corrected, 3 no-change controls),
2 corrected train examples (13, 70), and the one rejected measurement (88) for
transparency.
