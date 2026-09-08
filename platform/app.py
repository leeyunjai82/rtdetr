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
    IMG_SUFFIXES,
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


@app.post("/api/datasets/images")
async def upload_images(
    name: str = Form(...), dataset_id: int = Form(None), files: list[UploadFile] = None
):
    """Take image files straight from the file picker — no zip to make first."""
    if not files:
        raise HTTPException(400, "no files uploaded")
    target, dataset = _collection_target(name, dataset_id)
    written = 0
    for upload in files:
        if Path(upload.filename or "").suffix.lower() not in IMG_SUFFIXES:
            continue
        destination = _unique(target / Path(upload.filename).name)
        destination.write_bytes(await upload.read())
        written += 1
    if not written:
        _discard(target, dataset)
        raise HTTPException(400, "none of those files are images")
    return _finish_collection(name, target, dataset, added=written)


@app.post("/api/datasets/video")
async def upload_video(
    name: str = Form(...),
    dataset_id: int = Form(None),
    every: int = Form(30),
    max_frames: int = Form(300),
    video: UploadFile = None,
):
    """Sample frames out of a video into a dataset.

    Collection usually starts with a recording, not a folder of stills. One
    frame every ``every`` frames, up to ``max_frames`` — consecutive frames are
    nearly identical and only cost labelling time.
    """
    import cv2

    if video is None:
        raise HTTPException(400, "no video uploaded")
    target, dataset = _collection_target(name, dataset_id)
    scratch = target / f".{int(time.time())}_{Path(video.filename or 'clip').name}"
    scratch.write_bytes(await video.read())

    capture = cv2.VideoCapture(str(scratch))
    if not capture.isOpened():
        scratch.unlink(missing_ok=True)
        _discard(target, dataset)
        raise HTTPException(400, "could not read that video")

    stem = _slug(Path(video.filename or "clip").stem)
    written = index = 0
    try:
        while written < max(int(max_frames), 1):
            ok, frame = capture.read()
            if not ok:
                break
            if index % max(int(every), 1) == 0:
                cv2.imwrite(str(_unique(target / f"{stem}_{index:06d}.jpg")), frame)
                written += 1
            index += 1
    finally:
        capture.release()
        scratch.unlink(missing_ok=True)

    if not written:
        _discard(target, dataset)
        raise HTTPException(400, "that video had no frames")
    return _finish_collection(name, target, dataset, added=written, scanned=index)


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
    model_name = _resolve_model(payload.get("model") or "rtdetr-r18")
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
        model=_resolve_model(payload.get("model") or "rtdetr-r18"),
        conf=float(payload.get("conf", 0.35)),
    )
    return {"id": job_id}


@app.get("/api/datasets/{dataset_id}/stats")
def dataset_stats(dataset_id: int):
    """What is actually in there — the numbers you check before training."""
    dataset = _dataset(dataset_id)
    images_root, labels_root = Path(dataset["images_dir"]), Path(dataset["labels_dir"])
    names = json.loads(dataset["classes"])

    per_class = dict.fromkeys(range(len(names)), 0)
    unknown_class = 0
    boxes = tiny = 0
    unlabelled: list[str] = []
    empty: list[str] = []
    for image in list_images(images_root):
        label = label_path(image, images_root, labels_root)
        if not label.exists():
            unlabelled.append(str(image.relative_to(images_root)))
            continue
        rows = read_labels(label)
        if not rows:
            empty.append(str(image.relative_to(images_root)))
        for row in rows:
            boxes += 1
            if row["cls"] in per_class:
                per_class[row["cls"]] += 1
            else:
                unknown_class += 1
            if row["w"] * row["h"] < 0.001:  # under 0.1% of the frame
                tiny += 1
    return {
        "images": len(list_images(images_root)),
        "boxes": boxes,
        "per_class": [{"name": n, "boxes": per_class.get(i, 0)} for i, n in enumerate(names)],
        "unknown_class_boxes": unknown_class,
        "unlabelled": unlabelled,
        "empty_labels": empty,
        "tiny_boxes": tiny,
    }


@app.delete("/api/datasets/{dataset_id}/images/{index}")
def delete_image(dataset_id: int, index: int):
    """Drop one image and its label — blurred frames are not worth labelling."""
    dataset = _dataset(dataset_id)
    images_root = Path(dataset["images_dir"])
    images = list_images(images_root)
    if not 0 <= index < len(images):
        raise HTTPException(404, "no such image")
    label = label_path(images[index], images_root, Path(dataset["labels_dir"]))
    images[index].unlink(missing_ok=True)
    label.unlink(missing_ok=True)
    worker.refresh_counts(dataset_id)
    return {"deleted": True, "images": len(list_images(images_root))}


@app.post("/api/datasets/{dataset_id}/duplicate")
def duplicate_dataset(dataset_id: int, payload: dict = None):
    """A copy to experiment on — relabelling in place is a one-way door."""
    dataset = _dataset(dataset_id)
    name = (payload or {}).get("name") or f"{dataset['name']}-copy"
    return _combine([dataset], name)


@app.post("/api/datasets/merge")
def merge_datasets(payload: dict):
    """One dataset out of several, with class indices remapped onto a union."""
    # sorted, so the class union comes out the same whatever order they were picked in
    ids = sorted({int(i) for i in payload.get("ids") or []})
    if len(ids) < 2:
        raise HTTPException(400, "give at least two datasets to merge")
    return _combine([_dataset(i) for i in ids], payload.get("name") or "merged")


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


@app.get("/api/models")
def list_models():
    """Everything that can be used as a model: the registry plus the mirror names."""
    rows = db.query("SELECT * FROM models ORDER BY id DESC")
    for row in rows:
        row["classes"] = json.loads(row["classes"])
    from rtdetr.downloads import MODEL_NAMES

    return {"models": rows, "builtin": list(MODEL_NAMES)}


@app.post("/api/models")
def register_model(payload: dict):
    """Give a trained run a name, so it can be picked like any other model."""
    job = db.one("SELECT * FROM jobs WHERE id = ?", (payload.get("job_id"),))
    if job is None or not job["run_dir"]:
        raise HTTPException(404, "that job has no weights")
    xml = next(Path(job["run_dir"]).glob("openvino/*.xml"), None)
    weights = xml or Path(job["run_dir"]) / "weights" / "best.pt"
    if not Path(weights).exists():
        raise HTTPException(404, "no weights on disk")
    dataset = db.one("SELECT * FROM datasets WHERE id = ?", (job["dataset_id"],))
    model_id = db.add_model(
        name=str(payload.get("name") or f"job{job['id']}"),
        path=str(weights),
        kind="openvino" if xml else "torch",
        job_id=job["id"],
        classes=json.loads(dataset["classes"]) if dataset else [],
        note=payload.get("note"),
    )
    return {"id": model_id}


@app.post("/api/models/upload")
async def upload_model(name: str = Form(...), file: UploadFile = None, classes: str = Form("")):
    """Bring a model in from outside — a .pt from a colleague, an IR from a mirror."""
    if file is None:
        raise HTTPException(400, "no file uploaded")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".pt", ".xml", ".onnx"):
        raise HTTPException(400, "expected a .pt, .xml or .onnx file")
    folder = DATA / "models" / f"{int(time.time())}_{_slug(name)}"
    folder.mkdir(parents=True)
    path = folder / Path(file.filename).name
    path.write_bytes(await file.read())
    model_id = db.add_model(
        name=name,
        path=str(path),
        kind="openvino" if suffix == ".xml" else "torch",
        classes=[c.strip() for c in classes.split(",") if c.strip()],
        note="업로드",
    )
    return {"id": model_id, "path": str(path), "needs_bin": suffix == ".xml"}


@app.delete("/api/models/{model_id}")
def delete_model(model_id: int):
    model = db.one("SELECT * FROM models WHERE id = ?", (model_id,))
    if model is None:
        raise HTTPException(404, "no such model")
    db.execute("DELETE FROM models WHERE id = ?", (model_id,))
    return {"deleted": True}


@app.post("/api/predict")
async def batch_predict(
    model: str = Form(...),
    conf: float = Form(0.25),
    dataset_id: int = Form(None),
    path: str = Form(None),
    video: UploadFile = None,
):
    """Queue a run over a dataset, a folder on this machine, or an uploaded video."""
    if video is not None:
        folder = DATA / "predict-input" / str(int(time.time()))
        folder.mkdir(parents=True)
        source = folder / Path(video.filename or "clip.mp4").name
        source.write_bytes(await video.read())
    elif dataset_id:
        source = Path(_dataset(dataset_id)["images_dir"])
    elif path:
        source = Path(path).expanduser()
        if not source.exists():
            raise HTTPException(400, f"{source} does not exist")
    else:
        raise HTTPException(400, "give a dataset, a path, or a video")
    job_id = db.add_job(
        kind="predict",
        dataset_id=dataset_id or 0,
        model=_resolve_model(model),
        conf=conf,
        source=str(source),
    )
    return {"id": job_id}


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
        model=_resolve_model(payload.get("model", "rtdetr-r18")),
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


@app.post("/api/jobs/{job_id}/resume")
def resume_job(job_id: int, payload: dict = None):
    """Pick a run back up where it left off, and give it more epochs to use.

    Resuming with the schedule it already finished would do nothing, so the
    default is to extend it.
    """
    job = db.one("SELECT * FROM jobs WHERE id = ?", (job_id,))
    if job is None or job["kind"] != "train":
        raise HTTPException(404, "no such training job")
    if job["status"] in ("queued", "running", "exporting"):
        raise HTTPException(400, "that job is still going")
    root = job["resume_of"] or job_id
    if not (RUNS / f"job{root}" / "weights" / "last.pt").exists():
        raise HTTPException(400, "no last.pt to resume from")
    add = int((payload or {}).get("add_epochs", 10))
    new_id = db.add_job(
        kind="train",
        resume_of=root,
        dataset_id=job["dataset_id"],
        model=job["model"],
        epochs=job["epochs"] + max(add, 1),
        imgsz=job["imgsz"],
        batch=job["batch"],
        freeze=job["freeze"],
        device=job["device"],
        detail=f"#{root} 이어서 +{max(add, 1)}에폭",
    )
    return {"id": new_id}


@app.post("/api/jobs/{job_id}/evaluate")
def evaluate_job(job_id: int, payload: dict = None):
    """Queue a scoring pass over the validation split, with pictures."""
    job = db.one("SELECT * FROM jobs WHERE id = ?", (job_id,))
    if job is None or not job["run_dir"]:
        raise HTTPException(404, "that job has no weights")
    new_id = db.add_job(
        kind="evaluate",
        resume_of=job_id,
        dataset_id=job["dataset_id"],
        model=job["model"],
        conf=float((payload or {}).get("conf", 0.25)),
    )
    return {"id": new_id}


@app.get("/api/jobs/{job_id}/report")
def job_report(job_id: int):
    """The evaluation report, if this run has one."""
    job = db.one("SELECT * FROM jobs WHERE id = ?", (job_id,))
    if job is None or not job["run_dir"]:
        raise HTTPException(404, "no such job")
    report = Path(job["run_dir"]) / "eval" / "report.json"
    if not report.exists():
        raise HTTPException(404, "not evaluated yet")
    return json.loads(report.read_text(encoding="utf-8"))


@app.get("/api/jobs/{job_id}/eval/{name}")
def eval_image(job_id: int, name: str):
    job = db.one("SELECT * FROM jobs WHERE id = ?", (job_id,))
    if job is None or not job["run_dir"]:
        raise HTTPException(404, "no such job")
    path = (Path(job["run_dir"]) / "eval" / name).resolve()
    if not path.is_file() or not str(path).startswith(str(Path(job["run_dir"]).resolve())):
        raise HTTPException(404, "no such image")
    return FileResponse(path)


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
    if kind == "predictions":
        folder = run_dir / "images"
        if not folder.is_dir():
            raise HTTPException(404, "no predictions")
        shutil.copy2(run_dir / "results.json", folder / "results.json")
        archive = shutil.make_archive(str(run_dir / f"job{job_id}-predictions"), "zip", folder)
        return FileResponse(archive, filename=f"job{job_id}-predictions.zip")
    if kind == "video":
        path = run_dir / "annotated.mp4"
        if not path.exists():
            raise HTTPException(404, "no annotated video")
        return FileResponse(path, filename=f"job{job_id}-annotated.mp4")
    if kind == "log":
        path = run_dir / "train.log"
        if not path.exists():
            raise HTTPException(404, "no log")
        return FileResponse(path, filename=f"job{job_id}-train.log", media_type="text/plain")
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


@app.post("/api/preview")
async def preview(model: str = Form(...), conf: float = Form(0.35), image: UploadFile = None):
    """One frame in, one annotated frame out — what the webcam preview posts to."""
    import cv2
    import numpy as np

    if image is None:
        raise HTTPException(400, "no image uploaded")
    frame = cv2.imdecode(np.frombuffer(await image.read(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "could not read that image")
    try:
        result = _model(_resolve_model(model))(frame, conf=conf, verbose=False)[0]
    except Exception as exc:
        raise HTTPException(503, str(exc)) from exc
    ok, buffer = cv2.imencode(".jpg", result.plot(), [cv2.IMWRITE_JPEG_QUALITY, 80])
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


def _collection_target(name: str, dataset_id: int | None) -> tuple[Path, dict | None]:
    """Where new images go: into an existing dataset, or into a fresh folder."""
    if dataset_id:
        dataset = _dataset(dataset_id)
        return Path(dataset["images_dir"]), dataset
    target = DATASETS / f"{int(time.time())}_{_slug(name)}" / "images"
    target.mkdir(parents=True)
    return target, None


def _finish_collection(name: str, target: Path, dataset: dict | None, **counts) -> dict:
    """Register a new dataset, or refresh the counts of the one we added to."""
    if dataset is None:
        return {**_register(name, target.parent), **counts}
    worker.refresh_counts(dataset["id"])
    updated = _dataset(dataset["id"])
    return {
        "id": dataset["id"],
        "images": updated["images"],
        "labelled": updated["labelled"],
        "classes": json.loads(updated["classes"]),
        **counts,
    }


def _discard(target: Path, dataset: dict | None) -> None:
    """Undo a half-made dataset; never touch one that already existed."""
    if dataset is None and target.parent.parent == DATASETS:
        shutil.rmtree(target.parent, ignore_errors=True)


def _unique(path: Path) -> Path:
    """Keep a second upload of "frame.jpg" from overwriting the first."""
    if not path.exists():
        return path
    for i in range(2, 10000):
        candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise HTTPException(400, f"too many files named {path.name}")


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


def _resolve_model(name: str) -> str:
    """A registry id, a registry name, a mirror name, or a path — all usable."""
    text = str(name).strip()
    row = None
    if text.isdigit():
        row = db.one("SELECT * FROM models WHERE id = ?", (int(text),))
    if row is None:
        row = db.one("SELECT * FROM models WHERE name = ?", (text,))
    return row["path"] if row else text


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


def _combine(sources: list[dict], name: str) -> dict:
    """Copy several datasets into a new one, unioning their class lists.

    Class indices are per-dataset, so a straight file copy would silently turn
    every "car" in the second set into whatever index 1 means in the first.
    Names are matched instead, and labels rewritten onto the union.
    """
    names: list[str] = []
    for source in sources:
        for class_name in json.loads(source["classes"]):
            if class_name not in names:
                names.append(class_name)

    target = DATASETS / f"{int(time.time())}_{_slug(name)}"
    (target / "images").mkdir(parents=True)
    (target / "labels").mkdir(parents=True)
    copied = 0
    for source in sources:
        images_root, labels_root = Path(source["images_dir"]), Path(source["labels_dir"])
        remap = {i: names.index(n) for i, n in enumerate(json.loads(source["classes"]))}
        prefix = _slug(source["name"])
        for image in list_images(images_root):
            stem = f"{prefix}_{image.relative_to(images_root).as_posix().replace('/', '_')}"
            destination = _unique(target / "images" / stem)
            shutil.copy2(image, destination)
            copied += 1
            label = label_path(image, images_root, labels_root)
            if not label.exists():
                continue
            rows = [dict(row, cls=remap.get(row["cls"], row["cls"])) for row in read_labels(label)]
            write_labels((target / "labels" / destination.name).with_suffix(".txt"), rows)

    registered = _register(name, target, names=names)
    return {**registered, "added": copied, "sources": [s["id"] for s in sources]}


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
    if (run_dir / "train.log").exists():
        found.append("log")
    if (run_dir / "eval" / "report.json").exists():
        found.append("report")
    if (run_dir / "results.json").exists():
        found.append("predictions")
    if (run_dir / "annotated.mp4").exists():
        found.append("video")
    return found
