# Apache-2.0
"""rtdetr studio — label, train, watch, download, all from a browser.

    pip install "rtdetr[studio]"
    rtdetr studio               # http://127.0.0.1:8080

Everything lives in one process and one SQLite file, under ``rtdetr-studio/``
in the directory you start it from. That is deliberate — a
single box (a workstation, a mini PC on a factory floor) is the case this is
built for, and it needs no infrastructure to stand up. Multi-tenancy, auth and a
real queue belong to whatever this grows into, not here.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import yaml
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from .db import Database
from .worker import Worker

ROOT = Path(__file__).resolve().parent
IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

#: Uploads, runs and the database live beside your work, never inside the
#: installed package. $RTDETR_STUDIO_HOME moves them.
DATA = Path(os.environ.get("RTDETR_STUDIO_HOME", Path.cwd() / "rtdetr-studio")).expanduser()
DATASETS, RUNS = DATA / "datasets", DATA / "runs"

db = Database(DATA / "studio.db")
worker = Worker(db, RUNS)
app = FastAPI(title="rtdetr studio")


@app.on_event("startup")
def _start() -> None:
    for folder in (DATASETS, RUNS):
        folder.mkdir(parents=True, exist_ok=True)
    if not worker.is_alive():
        worker.start()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")


# ----------------------------------------------------------------- datasets


@app.post("/api/datasets")
async def upload_dataset(name: str = Form(...), archive: UploadFile = None):
    """Take a zip of images (+ labels, + data.yaml if you have one)."""
    if archive is None:
        raise HTTPException(400, "no archive uploaded")
    target = DATASETS / f"{int(time.time())}_{_slug(name)}"
    target.mkdir(parents=True)
    try:
        with zipfile.ZipFile(io.BytesIO(await archive.read())) as zf:
            _safe_extract(zf, target)
    except zipfile.BadZipFile as exc:
        shutil.rmtree(target, ignore_errors=True)
        raise HTTPException(400, "that file is not a zip") from exc

    summary = inspect_dataset(target)
    if not summary["images"]:
        shutil.rmtree(target, ignore_errors=True)
        raise HTTPException(400, "no images found in the archive")
    dataset_id = db.add_dataset(
        name, target, summary["images"], summary["labelled"], summary["classes"]
    )
    return {"id": dataset_id, **summary}


@app.get("/api/datasets")
def list_datasets():
    rows = db.query("SELECT * FROM datasets ORDER BY id DESC")
    for row in rows:
        row["classes"] = json.loads(row["classes"])
    return rows


def inspect_dataset(root: Path) -> dict:
    """Find the images, the labels and the classes, whatever shape the zip was.

    A zip made by the labelling tool already has images/ and labels/ and a
    data.yaml; one exported from elsewhere might not. Both should work, so this
    normalises the folder and writes the data.yaml when it is missing.
    """
    images = [p for p in root.rglob("*") if p.suffix.lower() in IMG_SUFFIXES]
    images_root = _common_parent(images) if images else root
    labels_root = _labels_beside(images_root)

    labelled = 0
    seen_classes: set[int] = set()
    for image in images:
        label = (labels_root / image.relative_to(images_root)).with_suffix(".txt")
        if not label.exists():
            continue
        labelled += 1
        for line in label.read_text(encoding="utf-8").splitlines():
            if line.split():
                seen_classes.add(int(float(line.split()[0])))

    existing = next(iter(root.rglob("data.yaml")), None)
    if existing:
        cfg = yaml.safe_load(existing.read_text(encoding="utf-8")) or {}
        names = cfg.get("names") or {}
        if isinstance(names, list):
            names = dict(enumerate(names))
        else:
            names = {int(k): v for k, v in names.items()}
    else:
        names = {i: f"class_{i}" for i in sorted(seen_classes)} or {0: "class_0"}
        (root / "data.yaml").write_text(
            yaml.safe_dump(
                {
                    "path": str(root.resolve()),
                    "train": str(images_root.relative_to(root)),
                    "val": str(images_root.relative_to(root)),
                    "names": names,
                },
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
    return {"images": len(images), "labelled": labelled, "classes": list(names.values())}


# --------------------------------------------------------------------- jobs


@app.post("/api/jobs")
def create_job(payload: dict):
    dataset = db.one("SELECT * FROM datasets WHERE id = ?", (payload.get("dataset_id"),))
    if dataset is None:
        raise HTTPException(404, "no such dataset")
    if not dataset["labelled"]:
        raise HTTPException(400, "that dataset has no labels yet")
    job_id = db.add_job(
        dataset_id=dataset["id"],
        model=str(payload.get("model", "rtdetr-r18")),
        epochs=int(payload.get("epochs", 50)),
        imgsz=int(payload.get("imgsz", 640)),
        batch=int(payload.get("batch", 4)),
        freeze=payload.get("freeze") or None,
        device=payload.get("device") or None,
    )
    return {"id": job_id}


@app.get("/api/jobs")
def list_jobs():
    return db.query(
        "SELECT j.*, d.name AS dataset FROM jobs j"
        " JOIN datasets d ON d.id = j.dataset_id ORDER BY j.id DESC"
    )


@app.get("/api/jobs/{job_id}")
def get_job(job_id: int):
    job = db.one("SELECT * FROM jobs WHERE id = ?", (job_id,))
    if job is None:
        raise HTTPException(404, "no such job")
    job["epochs_done"] = db.query(
        "SELECT epoch, loss, map50_95, seconds FROM epochs WHERE job_id = ? ORDER BY epoch",
        (job_id,),
    )
    job["artifacts"] = _artifacts(job)
    return job


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: int):
    job = db.one("SELECT * FROM jobs WHERE id = ?", (job_id,))
    if job is None:
        raise HTTPException(404, "no such job")
    if job["status"] == "queued":
        db.update_job(job_id, status="cancelled", finished=time.time())
    elif job["status"] == "running":
        worker.cancel(job_id)  # stops at the next epoch boundary
    return {"status": "cancelling"}


@app.get("/api/jobs/{job_id}/stream")
def stream_job(job_id: int):
    """Server-sent events: one message per epoch, then a final status."""

    def events():
        sent = 0
        while True:
            job = db.one("SELECT * FROM jobs WHERE id = ?", (job_id,))
            if job is None:
                return
            rows = db.query(
                "SELECT epoch, loss, map50_95 FROM epochs WHERE job_id = ? AND epoch > ?"
                " ORDER BY epoch",
                (job_id, sent),
            )
            for row in rows:
                sent = row["epoch"]
                yield f"data: {json.dumps({'type': 'epoch', **row})}\n\n"
            if job["status"] in ("done", "failed", "cancelled"):
                yield f"data: {json.dumps({'type': 'end', 'status': job['status']})}\n\n"
                return
            yield ": keepalive\n\n"
            time.sleep(1.0)

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/api/jobs/{job_id}/download/{kind}")
def download(job_id: int, kind: str):
    job = db.one("SELECT * FROM jobs WHERE id = ?", (job_id,))
    if job is None or not job["run_dir"]:
        raise HTTPException(404, "nothing to download yet")
    run_dir = Path(job["run_dir"])
    if kind == "weights":
        path = run_dir / "weights" / "best.pt"
        if not path.exists():
            raise HTTPException(404, "no weights")
        return FileResponse(path, filename=f"job{job_id}-best.pt")
    if kind == "openvino":
        folder = run_dir / "openvino"
        if not folder.is_dir():
            raise HTTPException(404, "no exported IR")
        archive = shutil.make_archive(str(run_dir / f"job{job_id}-openvino"), "zip", folder)
        return FileResponse(archive, filename=f"job{job_id}-openvino.zip")
    if kind == "results":
        path = run_dir / "results.csv"
        if not path.exists():
            raise HTTPException(404, "no results yet")
        return FileResponse(path, filename=f"job{job_id}-results.csv")
    raise HTTPException(404, "unknown artefact")


@app.post("/api/jobs/{job_id}/predict")
async def predict(job_id: int, image: UploadFile = None, conf: float = Form(0.25)):
    """Try the trained model on one image; returns the annotated JPEG."""
    import cv2
    import numpy as np

    from rtdetr import RTDETR

    job = db.one("SELECT * FROM jobs WHERE id = ?", (job_id,))
    if job is None or not job["run_dir"]:
        raise HTTPException(404, "that job has no weights")
    if image is None:
        raise HTTPException(400, "no image uploaded")
    frame = cv2.imdecode(np.frombuffer(await image.read(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "could not read that image")

    xml = next(Path(job["run_dir"]).glob("openvino/*.xml"), None)
    weights = xml or Path(job["run_dir"]) / "weights" / "best.pt"
    result = RTDETR(str(weights), verbose=False)(frame, conf=conf)[0]
    ok, buffer = cv2.imencode(".jpg", result.plot())
    if not ok:
        raise HTTPException(500, "could not encode the result")
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg",
        headers={"X-Detections": json.dumps(result.summary(), ensure_ascii=False)},
    )


@app.get("/api/status")
def status():
    return {
        "running_job": worker.current,
        "queued": len(db.query("SELECT id FROM jobs WHERE status = 'queued'")),
        "datasets": len(db.query("SELECT id FROM datasets")),
    }


@app.exception_handler(HTTPException)
def http_error(_request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


# ------------------------------------------------------------------ helpers


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in name).strip("-")[:40] or "set"


def _safe_extract(zf: zipfile.ZipFile, target: Path) -> None:
    """Refuse entries that would escape the dataset folder."""
    for member in zf.infolist():
        destination = (target / member.filename).resolve()
        if not str(destination).startswith(str(target.resolve())):
            raise HTTPException(400, f"unsafe path in archive: {member.filename}")
    zf.extractall(target)


def _common_parent(paths: list[Path]) -> Path:
    parents = {p.parent for p in paths}
    if len(parents) == 1:
        return parents.pop()
    common = Path(*Path(paths[0]).parts[: min(len(p.parts) for p in paths) - 1])
    return common


def _labels_beside(images_root: Path) -> Path:
    parts = list(images_root.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    return images_root.parent / "labels"


def _artifacts(job: dict) -> list[str]:
    if not job.get("run_dir"):
        return []
    run_dir = Path(job["run_dir"])
    found = []
    if (run_dir / "weights" / "best.pt").exists():
        found.append("weights")
    if (run_dir / "openvino").is_dir():
        found.append("openvino")
    if (run_dir / "results.csv").exists():
        found.append("results")
    return found
