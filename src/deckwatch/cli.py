"""deckwatch command line: every command prints one JSON document to stdout."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from deckwatch.config import ConfigError, load_config
from deckwatch.pipeline import Pipeline, PipelineError
from deckwatch.store import Store
from deckwatch.timeutil import parse_timestamp, to_iso

TS_PATTERN = re.compile(r"(\d{8})_(\d{6})")
EXIT_OK, EXIT_FRAME_ERROR, EXIT_CONFIG_ERROR = 0, 2, 3


def parse_ts(path: str | Path, explicit: str | None, filename_tz: str) -> datetime:
    if explicit:
        try:
            return parse_timestamp(explicit, filename_tz)
        except ValueError as exc:
            raise PipelineError("invalid_timestamp", f"invalid --ts value: {explicit}", 422) from exc
    m = TS_PATTERN.search(Path(path).name)
    if m:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=ZoneInfo(filename_tz))
    return datetime.fromtimestamp(Path(path).stat().st_mtime, tz=timezone.utc)


def build_pipeline(config_path: str | Path) -> Pipeline:
    from deckwatch.detector import load_detector

    cfg = load_config(config_path)
    return Pipeline(cfg, load_detector(cfg.detector), Store(cfg.db_path))


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False))


def _analyze(args) -> int:
    frame = Path(args.frame)
    if not frame.is_file():
        raise PipelineError("unreadable_image", f"file not found: {frame}", 422)
    pipeline = build_pipeline(args.config)
    ts = parse_ts(frame, args.ts, pipeline.config.filename_tz)
    _print(pipeline.analyze(frame.read_bytes(), ts, path=str(frame), include_items=args.items))
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="deckwatch")
    parser.add_argument("--config", default="config/deck.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("analyze", help="analyze one frame and print the FrameResult JSON")
    a.add_argument("frame")
    a.add_argument("--ts", help="ISO 8601 timestamp; default: from filename, else file mtime")
    a.add_argument("--items", action="store_true", help="include per-item details")
    sub.add_parser("status", help="current state, latest frame and open operation")
    o = sub.add_parser("operations", help="list operations")
    o.add_argument("--since", help="ISO 8601; only operations with t1 >= since")
    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    try:
        if args.command == "analyze":
            return _analyze(args)
        cfg = load_config(args.config)
        if args.command == "status":
            _print(Store(cfg.db_path).status())
        elif args.command == "operations":
            since = to_iso(parse_ts("", args.since, cfg.filename_tz)) if args.since else None
            _print(Store(cfg.db_path).list_operations(since))
        elif args.command == "serve":
            from deckwatch.api import serve

            serve(args.config, args.host, args.port)
        return EXIT_OK
    except ConfigError as exc:
        _print({"error": "config_error", "message": str(exc)})
        return EXIT_CONFIG_ERROR
    except PipelineError as exc:
        _print(exc.to_dict())
        return EXIT_FRAME_ERROR
