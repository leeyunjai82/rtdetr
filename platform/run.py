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


def _serve_https(application, uvicorn, args, local: str, path: str) -> None:
    """The same app over HTTPS, beside the HTTP one, for the webcam.

    Runs in a thread: uvicorn only takes over signals on the main thread, so
    Ctrl+C still stops the process through the HTTP server.
    """
    from tls import ensure_certificate

    try:
        cert, key = ensure_certificate(application.DATA / "tls")
    except Exception as exc:  # no or broken cryptography: HTTP still works
        print(f"[platform] HTTPS off ({type(exc).__name__}: {exc}); "
              f"pip install -U cryptography serves it — the webcam needs it")
        return
    os.environ["EASYDETECT_HTTPS_PORT"] = str(args.https_port)
    config = uvicorn.Config(
        application.app, host=args.host, port=args.https_port, log_level="warning",
        ssl_certfile=str(cert), ssl_keyfile=str(key),
    )
    threading.Thread(target=uvicorn.Server(config).run, daemon=True, name="https").start()
    print(f"[platform] https://{local}:{args.https_port}{path}  (for the webcam; "
          f"the browser warns once about the certificate)")


#: The oldest easydetect this platform works with. Checked against the code that is
#: actually imported — an editable install's metadata can lag behind a git pull.
EASYDETECT_AT_LEAST = "0.2.0"


def _which_easydetect(file: str) -> str:
    """Say which easydetect a path belongs to: an installed package, or a source tree.

    The platform sits in the package's repository but runs on whatever
    easydetect pip installed, normally the PyPI release. Only an editable
    install (``pip install -e``) or a PYTHONPATH points it at the clone's
    own ``easydetect/`` folder; this line makes the difference visible.
    """
    where = Path(file).resolve().parent
    if {"site-packages", "dist-packages"} & set(where.parts):
        return f"installed package at {where}"
    return f"editable install, the source at {where}"


def _check_easydetect() -> str:
    try:
        import easydetect
    except ImportError:
        raise SystemExit('easydetect is not installed: pip install -r platform/requirements.txt '
                         '(or pip install "easydetect[train]")') from None
    have = tuple(int(x) for x in easydetect.__version__.split(".")[:3])
    need = tuple(int(x) for x in EASYDETECT_AT_LEAST.split("."))
    if have < need:
        raise SystemExit(
            f"the platform needs easydetect >= {EASYDETECT_AT_LEAST}, "
            f"found {easydetect.__version__} at "
            f"{Path(easydetect.__file__).parent}.\n"
            f'  pip install -U "easydetect[train]"      # or, in a clone: pip install -e ".[train]"'
        )
    return f"easydetect {easydetect.__version__}: {_which_easydetect(easydetect.__file__)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", nargs="?", default="serve", choices=("serve", "label"))
    parser.add_argument("--source", help="label mode: folder of images to open")
    parser.add_argument("--names", help="label mode: comma-separated class names")
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="0.0.0.0 (default) answers on every address; 127.0.0.1 keeps it to this machine",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--https-port", type=int, default=8443,
        help="also serve HTTPS here, which the webcam needs from other machines; 0: off",
    )
    parser.add_argument("--data", help="where datasets, runs and the database live")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    if args.data:
        os.environ["EASYDETECT_PLATFORM_HOME"] = str(Path(args.data).expanduser())

    print(f"[platform] {_check_easydetect()}")
    import app as application
    import uvicorn

    # 0.0.0.0 is where the server listens, not an address a browser can open
    local = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{local}:{args.port}"
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

    where = "  (other machines: this one's IP)" if local != args.host else ""
    print(f"[platform] {url}{path}{where}")
    if args.https_port:
        _serve_https(application, uvicorn, args, local, path)
    if not args.no_browser:
        threading.Thread(target=lambda: webbrowser.open(url + path), daemon=True).start()
    uvicorn.run(application.app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
