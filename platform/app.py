# Apache-2.0
"""rtdetr platform — label, train, watch, download, in a browser.

    pip install -r platform/requirements.txt
    python platform/run.py                 # http://127.0.0.1:8080

One process, one SQLite file, one folder (``rtdetr-platform/`` where you start
it; ``--data`` or ``$RTDETR_PLATFORM_HOME`` moves it). It is built for a single
box — a workstation, a mini PC beside a line — where the data must not leave the
machine and there is nobody to run a queue broker. Users, permissions and
schedulers are deliberately absent; see the README for what to do when you need
them.
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
from db import Database
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from labeling import (
    label_path,
    labels_beside,
    list_images,
    predict_boxes,
    read_labels,
    write_labels,
)
from worker import Worker

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("RTDETR_PLATFORM_HOME", Path.cwd() / "rtdetr-platform")).expanduser()
DATASETS, RUNS = DATA / "datasets", DATA / "runs"

db = Database(DATA / "platform.db")
worker = Worker(db, RUNS)
app = FastAPI(title="rtdetr platform")
_models: dict[str, object] = {}


@app.on_event("startup")
def _start() -> None:
    for folder in (DATASETS, RUNS):
        folder.mkdir(parents=True, exist_ok=True)
    reaped = worker.reap_stale()
    if reaped:
        print(f"[platform] {reaped} job(s) were left running by a previous process")
    if not worker.is_alive():
        worker.start()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/label/{dataset_id}", response_class=HTMLResponse)
def label_page(dataset_id: int) -> str:
    _dataset(dataset_id)
    return (ROOT / "static" / "label.html").read_text(encoding="utf-8")


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
    return _register(name, target)


@app.post("/api/datasets/local")
def add_local_dataset(payload: dict):
    """Register a folder already on this machine, without copying it."""
    path = Path(str(payload.get("path", ""))).expanduser()
    if not path.is_dir():
        raise HTTPException(400, f"{path} is not a folder")
    names = payload.get("names") or None
    return _register(payload.get("name") or path.name, path, names=names)


@app.get("/api/datasets")
def list_datasets():
    rows = db.query("SELECT * FROM datasets ORDER BY id DESC")
    for row in rows:
        row["classes"] = json.loads(row["classes"])
    return rows


@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: int):
    dataset = _dataset(dataset_id)
    images_root, labels_root = Path(dataset["images_dir"]), Path(dataset["labels_dir"])
    images = list_images(images_root)
    dataset["classes"] = json.loads(dataset["classes"])
    dataset["files"] = [
        {
            "name": str(p.relative_to(images_root)),
            "labelled": label_path(p, images_root, labels_root).exists(),
        }
        for p in images
    ]
    return dataset


@app.get("/api/datasets/{dataset_id}/image/{index}")
def dataset_image(dataset_id: int, index: int):
    dataset = _dataset(dataset_id)
    images = list_images(Path(dataset["images_dir"]))
    if not 0 <= index < len(images):
        raise HTTPException(404, "no such image")
    return FileResponse(images[index])


@app.get("/api/datasets/{dataset_id}/labels/{index}")
def get_labels(dataset_id: int, index: int):
    return {"boxes": read_labels(_label_file(dataset_id, index))}


@app.post("/api/datasets/{dataset_id}/labels/{index}")
def save_labels(dataset_id: int, index: int, payload: dict):
    write_labels(_label_file(dataset_id, index), payload.get("boxes", []))
    worker.refresh_counts(dataset_id)
    return {"saved": True}


@app.post("/api/datasets/{dataset_id}/autolabel/{index}")
def autolabel_one(dataset_id: int, index: int, payload: dict):
    """Boxes for the image on screen, from a model — the first draft to correct."""
    dataset = _dataset(dataset_id)
    images = list_images(Path(dataset["images_dir"]))
    model_name = str(payload.get("model") or "rtdetr-r18")
    try:
        model = _model(model_name)
        boxes = predict_boxes(
            model, images[index], json.loads(dataset["classes"]), float(payload.get("conf", 0.35))
        )
    except Exception as exc:  # a missing model must not kill the page
        raise HTTPException(503, str(exc)) from exc
    return {"boxes": boxes}


@app.post("/api/datasets/{dataset_id}/autolabel")
def autolabel_all(dataset_id: int, payload: dict):
    """Queue a pass over every unlabelled image in the dataset."""
    _dataset(dataset_id)
    job_id = db.add_job(
        kind="autolabel",
        dataset_id=dataset_id,
        model=str(payload.get("model") or "rtdetr-r18"),
        conf=float(payload.get("conf", 0.35)),
    )
    return {"id": job_id}


@app.get("/api/datasets/{dataset_id}/export")
def export_dataset(dataset_id: int):
    """The whole dataset as a zip: images, labels, data.yaml — ready to re-import."""
    dataset = _dataset(dataset_id)
    images_root, labels_root = Path(dataset["images_dir"]), Path(dataset["labels_dir"])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for image in list_images(images_root):
            relative = image.relative_to(images_root)
            zf.write(image, f"images/{relative}")
            label = label_path(image, images_root, labels_root)
            if label.exists():
                zf.write(label, f"labels/{relative.with_suffix('.txt')}")
        zf.writestr(
            "data.yaml",
            yaml.safe_dump(
                {
                    "train": "images",
                    "val": "images",
                    "names": dict(enumerate(json.loads(dataset["classes"]))),
                },
                sort_keys=False,
                allow_unicode=True,
            ),
        )
    buffer.seek(0)
    name = _slug(dataset["name"])
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}.zip"'},
    )


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: int):
    """Forget a dataset. Uploaded copies are removed; registered folders are left alone."""
    dataset = _dataset(dataset_id)
    running = db.one(
        "SELECT id FROM jobs WHERE dataset_id = ? AND status IN ('queued', 'running')",
        (dataset_id,),
    )
    if running:
        raise HTTPException(400, f"job #{running['id']} is still using it")
    path = Path(dataset["path"])
    removed = path.parent == DATASETS and path.is_dir()
    if removed:
        shutil.rmtree(path, ignore_errors=True)
    db.execute("DELETE FROM epochs WHERE job_id IN (SELECT id FROM jobs WHERE dataset_id = ?)",
               (dataset_id,))
    db.execute("DELETE FROM jobs WHERE dataset_id = ?", (dataset_id,))
    db.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
    return {"deleted": True, "files_removed": removed}


@app.post("/api/datasets/{dataset_id}/classes")
def set_classes(dataset_id: int, payload: dict):
    """Rename or add classes; the data.yaml follows."""
    dataset = _dataset(dataset_id)
    names = [str(n).strip() for n in payload.get("names", []) if str(n).strip()]
    if not names:
        raise HTTPException(400, "give at least one class name")
    db.update_dataset(dataset_id, classes=json.dumps(names, ensure_ascii=False))
    _write_data_yaml(Path(dataset["path"]), Path(dataset["images_dir"]), names)
    return {"names": names}


# --------------------------------------------------------------------- jobs


@app.post("/api/jobs")
def create_job(payload: dict):
    dataset = _dataset(payload.get("dataset_id"))
    if not dataset["labelled"]:
        raise HTTPException(400, "that dataset has no labels yet")
    note = _ensure_split(dataset, float(payload.get("val_ratio", 0.2)))
    job_id = db.add_job(
        kind="train",
        detail=note,
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
    elif job["status"] in ("running", "exporting"):
        worker.cancel(job_id)  # stops at the next epoch or image
    return {"status": "cancelling"}


@app.get("/api/jobs/{job_id}/stream")
def stream_job(job_id: int):
    """Server-sent events: progress as it happens, then a final status."""

    def events():
        sent, last_progress = 0, -1.0
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
            if job["progress"] != last_progress:
                last_progress = job["progress"]
                yield f"data: {json.dumps({'type': 'progress', 'value': last_progress})}\n\n"
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


    job = db.one("SELECT * FROM jobs WHERE id = ?", (job_id,))
    if job is None or not job["run_dir"]:
        raise HTTPException(404, "that job has no weights")
    if image is None:
        raise HTTPException(400, "no image uploaded")
    frame = cv2.imdecode(np.frombuffer(await image.read(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "could not read that image")

    xml = next(Path(job["run_dir"]).glob("openvino/*.xml"), None)
    weights = str(xml or Path(job["run_dir"]) / "weights" / "best.pt")
    result = _model(weights)(frame, conf=conf)[0]
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


def _dataset(dataset_id) -> dict:
    dataset = db.one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    if dataset is None:
        raise HTTPException(404, "no such dataset")
    return dataset


def _label_file(dataset_id: int, index: int) -> Path:
    dataset = _dataset(dataset_id)
    images_root = Path(dataset["images_dir"])
    images = list_images(images_root)
    if not 0 <= index < len(images):
        raise HTTPException(404, "no such image")
    return label_path(images[index], images_root, Path(dataset["labels_dir"]))


def _model(name: str):
    """Compiled models are expensive; keep one per name for the session."""
    from rtdetr import RTDETR

    if name not in _models:
        _models[name] = RTDETR(name, verbose=False)
    return _models[name]


def _register(name: str, root: Path, names: list[str] | None = None) -> dict:
    """Work out what a folder holds, write a data.yaml if it lacks one, record it."""
    images = list_images(root)
    if not images:
        if root.parent == DATASETS:
            shutil.rmtree(root, ignore_errors=True)
        raise HTTPException(400, "no images found")
    images_root = _common_parent(images)
    labels_root = labels_beside(images_root)

    labelled = 0
    seen: set[int] = set()
    for image in images:
        label = label_path(image, images_root, labels_root)
        if not label.exists():
            continue
        labelled += 1
        for line in label.read_text(encoding="utf-8").splitlines():
            if line.split():
                seen.add(int(float(line.split()[0])))

    existing = next(iter(root.rglob("data.yaml")), None)
    if names:
        class_names = list(names)
        _write_data_yaml(root, images_root, class_names)
    elif existing:
        cfg = yaml.safe_load(existing.read_text(encoding="utf-8")) or {}
        raw = cfg.get("names") or {}
        if isinstance(raw, list):
            table = dict(enumerate(raw))
        else:
            table = {int(k): v for k, v in raw.items()}
        class_names = [table[k] for k in sorted(table)]
    else:
        class_names = [f"class_{i}" for i in range(max(seen) + 1)] if seen else ["class_0"]
        _write_data_yaml(root, images_root, class_names)

    dataset_id = db.add_dataset(
        name=name,
        path=str(root),
        images_dir=str(images_root),
        labels_dir=str(labels_root),
        images=len(images),
        labelled=labelled,
        classes=class_names,
    )
    return {
        "id": dataset_id,
        "images": len(images),
        "labelled": labelled,
        "classes": class_names,
    }


def _ensure_split(dataset: dict, val_ratio: float) -> str | None:
    """Hold images out for validation, unless the dataset already has its own split.

    Validating on the training images reports a number that only ever flatters
    the run, which is worse than no number at all. Every k-th image (sorted, so
    the choice is stable across runs) becomes the val set, and data.yaml points
    at the two lists.
    """
    root = Path(dataset["path"])
    images_root, labels_root = Path(dataset["images_dir"]), Path(dataset["labels_dir"])
    cfg = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8")) or {}
    if cfg.get("train") != cfg.get("val"):
        return None  # the dataset came with its own split; leave it alone

    images = [
        image
        for image in list_images(images_root)
        if label_path(image, images_root, labels_root).exists()
    ]
    if len(images) < 4:
        return f"only {len(images)} labelled images — validating on the same ones"

    step = max(int(round(1 / max(min(val_ratio, 0.5), 0.05))), 2)
    val = images[::step]
    train = [i for i in images if i not in set(val)]
    (root / "train.txt").write_text("\n".join(str(p.resolve()) for p in train), encoding="utf-8")
    (root / "val.txt").write_text("\n".join(str(p.resolve()) for p in val), encoding="utf-8")

    cfg.update({"path": str(root.resolve()), "train": "train.txt", "val": "val.txt"})
    (root / "data.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return f"{len(train)} train / {len(val)} val images"


def _write_data_yaml(root: Path, images_root: Path, names: list[str]) -> Path:
    path = root / "data.yaml"
    split = images_root.relative_to(root) if images_root != root else "."
    path.write_text(
        yaml.safe_dump(
            {
                "path": str(root.resolve()),
                "train": str(split),
                "val": str(split),
                "names": dict(enumerate(names)),
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return path


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
    return Path(*Path(paths[0]).parts[: min(len(p.parts) for p in paths) - 1])


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
