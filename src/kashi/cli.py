"""kashi command-line interface.

    kashi info                          config, components, device, dataset state
    kashi run <stage>                   DAG-resolve; execute only stale stages
    kashi dataset build|download        rebuild labels / fetch audio
    kashi separate <files...>           vocal separation (swappable model)
    kashi encode [--from-legacy ...]    cache features for dataset/unlabeled audio
    kashi train segmenter|classifier    training (P3+: frame|encoder)
    kashi eval segmenter|classifier     metrics on the frozen split
    kashi transcribe <media>            full pipeline: file -> subtitles
    kashi realign / gold / discover / fit / loop / serve   (land in P1-P5)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Config


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None, help="TOML config overlaying configs/default.toml")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE", help="override a config value (repeatable)")


def _cfg(args) -> Config:
    return Config.load(args.config, args.overrides)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_info(args) -> None:
    cfg = _cfg(args)
    from . import registry
    from .data import manifest
    from .data.store import FeatureStore, encoder_cache_id

    print("kashi configuration")
    print(f"  data dir      : {cfg.data_dir}")
    print(f"  artifacts dir : {cfg.artifacts_dir}")
    print(f"  runs dir      : {cfg.runs_dir}")
    try:
        import torch

        dev = f"cuda:{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "cpu"
    except Exception:
        dev = "torch unavailable"
    print(f"  device        : {dev}")
    print("pipeline components")
    for kind in ("separator", "encoder", "segmenter", "classifier", "decoder"):
        chosen = cfg.get(f"pipeline.{kind}")
        print(f"  {kind:<11}: {chosen}   (available: {', '.join(registry.names(kind)) or '-'})")
    train, test = manifest.split_ids(cfg)
    print(f"dataset: {len(manifest.song_ids(cfg))} songs "
          f"({len(train)} train / {len(test)} test labeled, version={cfg['data.version']})")
    store = FeatureStore(cfg)
    print(f"feature cache : {len(store.keys())} entries under {store.dir}")
    print(f"  encoder id  : {encoder_cache_id(cfg)}")
    lb = cfg.runs_dir / "leaderboard.csv"
    if lb.is_file():
        print(f"leaderboard   : {lb}")


def cmd_dataset(args) -> None:
    cfg = _cfg(args)
    from .data import build

    if args.dataset_cmd == "build":
        build.build_dataset(cfg, out_version=args.out_version, force=args.force,
                            trim=not args.no_trim)
    elif args.dataset_cmd == "download":
        build.download_audio(cfg)
    elif args.dataset_cmd == "import":
        import json

        from .data import import_ds

        sets = (args.sets or "t1,ro,t2-extra").split(",")
        import_ds.import_sets(cfg, [s.strip() for s in sets])
        if "ro" in sets or "t1" in sets:
            report = import_ds.dual_track_report(cfg)
            out = cfg.runs_dir / "dual_track_report.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=2, default=float))
            print(f"[dual-track] full report -> {out}")
    elif args.dataset_cmd == "scrape":
        from .data import scrape

        if not args.playlist:
            raise SystemExit("--playlist URL required")
        raw = scrape.scrape_playlist(cfg, args.playlist, lang=args.lang,
                                     audio_only=args.audio_only)
        print(f"[scrape] staged under {raw}; index/parse with `kashi dataset import`")


def cmd_separate(args) -> None:
    cfg = _cfg(args)
    from .components.separators import separate_files

    out = Path(args.out) if args.out else cfg.data_dir / "clean" / "audio" / "vocals"
    separate_files(cfg, [Path(p) for p in args.inputs], out, name=args.separator)


def cmd_encode(args) -> None:
    cfg = _cfg(args)
    from .data.encode import adopt_legacy, encode_targets

    if args.from_legacy is not None:
        adopt_legacy(cfg, args.from_legacy or None)
        return
    songs = [int(s) for s in args.songs] if args.songs else None
    encode_targets(cfg, songs=songs, unlabeled=args.unlabeled, force=args.force)


def cmd_train(args) -> None:
    cfg = _cfg(args)
    if args.model == "segmenter":
        from .train import segmenter as mod

        mod.train(cfg, version=args.version, init=args.init, name=args.name)
    elif args.model == "classifier":
        from .train import classifier as mod

        mod.train(cfg, version=args.version, init=args.init, name=args.name)
    elif args.model == "frame":
        from .train import frame as mod

        mod.train(cfg, version=args.version, name=args.name)
    else:
        raise SystemExit(f"`kashi train {args.model}` lands in a later phase — see ROADMAP.md")


def cmd_fit(args) -> None:
    cfg = _cfg(args)
    if args.what == "durations":
        from .stats.durations import fit_durations

        fit_durations(cfg, version=args.version)
    elif args.what == "lm":
        from .stats.lm import fit_bigram

        fit_bigram(cfg, version=args.version)


def cmd_eval(args) -> None:
    import json

    cfg = _cfg(args)
    if args.model == "classifier":
        from .train.classifier import evaluate

        out = evaluate(cfg, checkpoint=args.checkpoint, split=args.split,
                       legacy_index=args.legacy_index)
    elif args.model == "segmenter":
        from .train.segmenter import evaluate

        if not args.checkpoint:
            raise SystemExit("--checkpoint required for segmenter eval")
        out = evaluate(cfg, checkpoint=args.checkpoint, split=args.split)
    elif args.model == "pipeline":
        from .eval.baselines import evaluate_pipeline

        out = evaluate_pipeline(cfg, split=args.split, gold_only=args.gold)
    elif args.model == "baseline":
        from .eval import baselines

        out = {}
        for which in (args.which or "a,b").split(","):
            fn = getattr(baselines, f"baseline_{which.strip()}")
            out[which.strip()] = fn(cfg)
    else:
        raise SystemExit(f"`kashi eval {args.model}` lands in a later phase")
    print(json.dumps(out, indent=2, default=float))


def cmd_gold(args) -> None:
    cfg = _cfg(args)
    from .eval import gold

    if args.gold_cmd == "seed":
        gold.seed_golden(cfg)
    elif args.gold_cmd == "export":
        gold.export(cfg, int(args.song), window_s=args.window, at=args.at)
    elif args.gold_cmd == "import":
        if args.window_start is None or args.window_end is None:
            raise SystemExit("--window-start/--window-end required for import")
        gold.import_labels(cfg, int(args.song), args.path,
                           window_start=args.window_start, window_end=args.window_end)
    else:
        gold.status(cfg)


def cmd_transcribe(args) -> None:
    cfg = _cfg(args)
    from .pipeline import transcribe

    def progress(stage: str, frac: float) -> None:
        print(f"[{frac:>4.0%}] {stage}")

    result = transcribe(
        cfg,
        args.input,
        out_dir=args.out,
        formats=[f.strip() for f in args.formats.split(",") if f.strip()],
        romaji=args.romaji,
        separate=False if args.no_separate else None,
        progress=progress,
    )
    n_lyric = sum(1 for s in result.segments if not s.is_silence)
    print(f"{n_lyric} syllables across {len(result.segments)} segments "
          f"(stage timings: {result.timings})")
    for fmt, path in result.out_files.items():
        print(f"  {fmt}: {path}")


def cmd_realign(args) -> None:
    import json

    cfg = _cfg(args)
    from .realign import realign_dataset, report_vs_gold

    if args.vs_gold:
        print(json.dumps(report_vs_gold(cfg, version=args.out_version), indent=2, default=float))
        return
    songs = [int(s) for s in args.songs] if args.songs else None
    realign_dataset(cfg, song_ids=songs, out_version=args.out_version)


def cmd_discover(args) -> None:
    cfg = _cfg(args)
    from pathlib import Path

    import numpy as np

    from . import audio as audio_mod
    from .registry import create
    from .stats.hmm import PCA, StickyHDPHMM

    wave = audio_mod.load_audio(args.input, sr=cfg.sample_rate)
    feats = create(cfg, "encoder").encode(wave, cfg.sample_rate)
    X = PCA(int(cfg["segmenter.hmm.pca_dim"])).fit(feats).transform(feats)
    res = StickyHDPHMM(
        L=int(cfg["segmenter.hmm.L"]), alpha=float(cfg["segmenter.hmm.alpha"]),
        gamma=float(cfg["segmenter.hmm.gamma"]), rho=float(cfg["segmenter.hmm.rho"]),
        sweeps=int(cfg["segmenter.hmm.sweeps"]), burnin=int(cfg["segmenter.hmm.burnin"]),
    ).fit(X)
    out = Path(args.out) if args.out else Path(args.input).with_suffix(".units.csv")
    frame_s = cfg.frame_ms / 1000.0
    z = res.last_path
    with open(out, "w") as f:
        f.write("start,end,unit\n")
        s = 0
        for t in range(1, len(z) + 1):
            if t == len(z) or z[t] != z[s]:
                f.write(f"{s*frame_s:.3f},{t*frame_s:.3f},{z[s]}\n")
                s = t
    print(f"{res.n_active_states} active units, {len(res.boundaries)} boundaries -> {out}")


def cmd_serve(args) -> None:
    cfg = _cfg(args)
    from .web.app import serve

    print(f"kashi web app on http://{args.host or cfg['web.host']}:{args.port or cfg['web.port']} "
          f"(pipeline: {cfg['pipeline.mode']}, separator: {cfg['pipeline.separator']})")
    serve(cfg, host=args.host, port=args.port)


def cmd_run(args) -> None:
    cfg = _cfg(args)
    from . import dag, stages as _  # noqa: F401  (stage definitions register on import)

    ran = dag.run(cfg, args.stage, force=args.force, allow_train=args.allow_train)
    print(f"ran {len(ran)} stage(s): {ran}" if ran else "everything fresh — no-op")


def _planned(phase: str):
    def handler(args) -> None:
        raise SystemExit(f"this command lands in {phase} — see ROADMAP.md")

    return handler


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kashi", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("info", help="show config, components, dataset and device state")
    _add_common(sp)
    sp.set_defaults(fn=cmd_info)

    sp = sub.add_parser("dataset", help="dataset build / download / import / scrape")
    _add_common(sp)
    sp.add_argument("dataset_cmd", choices=["build", "download", "import", "scrape"])
    sp.add_argument("--out-version", default="clean")
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--no-trim", action="store_true", help="skip the audio silence-trim step")
    sp.add_argument("--sets", default=None, help="import: comma list of t1,ro,t2-extra")
    sp.add_argument("--playlist", default=None, help="scrape: playlist/channel URL")
    sp.add_argument("--lang", default="ja", help="scrape: subtitle language track")
    sp.add_argument("--audio-only", action="store_true", help="scrape: fetch audio (unlabeled pool)")
    sp.set_defaults(fn=cmd_dataset)

    sp = sub.add_parser("separate", help="vocal/instrumental separation")
    _add_common(sp)
    sp.add_argument("inputs", nargs="+")
    sp.add_argument("--out", default=None)
    sp.add_argument("--separator", default=None, help="override [pipeline].separator")
    sp.set_defaults(fn=cmd_separate)

    sp = sub.add_parser("encode", help="cache frame features")
    _add_common(sp)
    sp.add_argument("--songs", nargs="*", default=None, help="specific dataset song ids")
    sp.add_argument("--unlabeled", default=None,
                    help="directory of arbitrary audio to encode (for unsupervised training)")
    sp.add_argument("--from-legacy", nargs="?", const="", default=None,
                    help="adopt legacy tensors (default dir: models/tensors/songs_20ms)")
    sp.add_argument("--force", action="store_true", help="re-encode even if cached")
    sp.set_defaults(fn=cmd_encode)

    sp = sub.add_parser("train", help="train a model")
    _add_common(sp)
    sp.add_argument("model", choices=["segmenter", "classifier", "frame", "encoder"])
    sp.add_argument("--version", default=None, help="label version: clean | clean_v2")
    sp.add_argument("--init", default=None, help="warm-start checkpoint (legacy ok)")
    sp.add_argument("--name", default=None, help="run name (default: timestamp)")
    sp.add_argument("--promote", action="store_true", help="adopt as default if eval improves (P3)")
    sp.set_defaults(fn=cmd_train)

    sp = sub.add_parser("eval", help="evaluate a checkpoint / the pipeline / baselines")
    _add_common(sp)
    sp.add_argument("model", choices=["segmenter", "classifier", "pipeline", "baseline"])
    sp.add_argument("--checkpoint", default=None)
    sp.add_argument("--split", default="test", choices=["train", "test"])
    sp.add_argument("--legacy-index", action="store_true",
                    help="classifier: use the legacy segment_index.csv slicing (paper repro)")
    sp.add_argument("--gold", action="store_true", help="pipeline: score only inside gold windows")
    sp.add_argument("--which", default=None, help="baseline: comma list of a,b,c,d")
    sp.set_defaults(fn=cmd_eval)

    sp = sub.add_parser("gold", help="gold subset: seed | export | import | status")
    _add_common(sp)
    sp.add_argument("gold_cmd", choices=["seed", "export", "import", "status"])
    sp.add_argument("song", nargs="?", default=None, help="song id (export/import)")
    sp.add_argument("path", nargs="?", default=None, help="label track file (import)")
    sp.add_argument("--window", type=float, default=90.0, help="export window length (s)")
    sp.add_argument("--at", type=float, default=None, help="export window start (s)")
    sp.add_argument("--window-start", type=float, default=None)
    sp.add_argument("--window-end", type=float, default=None)
    sp.set_defaults(fn=cmd_gold)

    sp = sub.add_parser("transcribe", help="full pipeline: media file -> subtitles")
    _add_common(sp)
    sp.add_argument("input", help="mp4/mp3/wav/... file")
    sp.add_argument("--out", default=None, help="output directory (default: runs/transcribe/<name>)")
    sp.add_argument("--formats", default="srt,vtt,csv", help="comma list: srt,vtt,ass,csv")
    sp.add_argument("--romaji", action="store_true", help="add a romaji line to srt/vtt")
    sp.add_argument("--no-separate", action="store_true", help="input is already isolated vocals")
    sp.set_defaults(fn=cmd_transcribe)

    sp = sub.add_parser("run", help="execute stale DAG stages up to <stage>")
    _add_common(sp)
    sp.add_argument("stage")
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--allow-train", action="store_true")
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("realign", help="snap label timings to acoustics, tag <noise> (no forced alignment)")
    _add_common(sp)
    sp.add_argument("--songs", nargs="*", default=None)
    sp.add_argument("--out-version", default=None)
    sp.add_argument("--vs-gold", action="store_true", help="report v1/v2 vs the gold subset")
    sp.set_defaults(fn=cmd_realign)

    sp = sub.add_parser("discover", help="unsupervised HDP-HMM unit discovery on one audio file")
    _add_common(sp)
    sp.add_argument("input")
    sp.add_argument("--out", default=None, help="output CSV (default: <input>.units.csv)")
    sp.set_defaults(fn=cmd_discover)

    sp = sub.add_parser("fit", help="closed-form fits: durations | lm")
    _add_common(sp)
    sp.add_argument("what", choices=["durations", "lm"])
    sp.add_argument("--version", default=None)
    sp.set_defaults(fn=cmd_fit)

    sp = sub.add_parser("serve", help="run the web app")
    _add_common(sp)
    sp.add_argument("--host", default=None)
    sp.add_argument("--port", type=int, default=None)
    sp.set_defaults(fn=cmd_serve)

    for name, phase, help_ in [
        ("loop", "P5", "self-training loop over unlabeled data"),
    ]:
        sp = sub.add_parser(name, help=f"{help_} (lands in {phase})")
        _add_common(sp)
        sp.add_argument("args", nargs="*")
        sp.set_defaults(fn=_planned(phase))

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
