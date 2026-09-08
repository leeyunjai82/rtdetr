#!/usr/bin/env python3
# Apache-2.0
"""Start the platform.

    python platform/run.py                                   # the web app
    python platform/run.py label --source images/ --names can,bottle

The second form registers a folder already on this machine and opens the
labelling page on it — the quick path when all you want is to draw boxes.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # the app's own modules, without shadowing stdlib


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", nargs="?", default="serve", choices=("serve", "label"))
    parser.add_argument("--source", help="label mode: folder of images to open")
    parser.add_argument("--names", help="label mode: comma-separated class names")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--data", help="where datasets, runs and the database live")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    if args.data:
        os.environ["RTDETR_PLATFORM_HOME"] = str(Path(args.data).expanduser())

    import app as application
    import uvicorn

    url = f"http://{args.host}:{args.port}"
    path = "/"
    if args.mode == "label":
        if not args.source:
            parser.error("label mode needs --source")
        application._start()  # so the folders and the worker exist before we register
        dataset = application.add_local_dataset(
            {
                "path": args.source,
                "name": Path(args.source).resolve().name,
                "names": [n.strip() for n in (args.names or "").split(",") if n.strip()] or None,
            }
        )
        path = f"/label/{dataset['id']}"
        print(f"[platform] {dataset['images']} images, {dataset['labelled']} already labelled")

    print(f"[platform] {url}{path}")
    if not args.no_browser:
        threading.Thread(target=lambda: webbrowser.open(url + path), daemon=True).start()
    uvicorn.run(application.app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
