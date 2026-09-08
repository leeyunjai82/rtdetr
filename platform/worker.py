# Apache-2.0
"""One background thread that runs queued jobs — training, or a batch auto-label.

Work happens inside this process; a thread is enough because torch and OpenVINO
release the GIL. Cancelling sets a flag the job checks at its next natural
boundary (an epoch, or an image), so nothing is killed mid-write.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import tempfile
import threading
import time
import traceback
from pathlib import Path

from labeling import label_path, list_images, predict_boxes, read_labels, write_labels


class Cancelled(Exception):
    """Raised inside a job to stop it cleanly."""


def _tail(path: Path, lines: int = 8) -> str:
    """The end of a log, for a failure message that says something."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError:
        return ""
    return " / ".join(text[-lines:])[:600]


class Worker(threading.Thread):
    def __init__(self, db, runs_dir: Path, poll: float = 1.0) -> None:
        super().__init__(daemon=True)
        self.db = db
        self.runs_dir = Path(runs_dir)
        self.poll = poll
        self.cancelled: set[int] = set()
        self.current: int | None = None
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def cancel(self, job_id: int) -> None:
        self.cancelled.add(job_id)

    def reap_stale(self) -> int:
        """Jobs left 'running' by a previous process are dead, not running.

        Nothing survives a restart — say so instead of showing a spinner that
        will never finish.
        """
        stale = self.db.query("SELECT id FROM jobs WHERE status = 'running'")
        for job in stale:
            self.db.update_job(
                job["id"],
                status="failed",
                detail="interrupted by a restart",
                finished=time.time(),
            )
        return len(stale)

    def run(self) -> None:
        while not self._stop.is_set():
            job = self.db.one("SELECT * FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1")
            if job is None:
                time.sleep(self.poll)
                continue
            self.current = job["id"]
            try:
                if job["kind"] == "autolabel":
                    self._autolabel(job)
                elif job["kind"] == "evaluate":
                    self._evaluate(job)
                elif job["kind"] == "predict":
                    self._predict(job)
                else:
                    self._train(job)
            except Cancelled:
                self.db.update_job(job["id"], status="cancelled", finished=time.time())
            except Exception as exc:
                previous = self.db.one("SELECT detail FROM jobs WHERE id = ?", (job["id"],))
                context = (previous or {}).get("detail") or ""
                self.db.update_job(
                    job["id"],
                    status="failed",
                    detail=f"{type(exc).__name__}: {exc}" + (f" — {context}" if context else ""),
                    finished=time.time(),
                )
                traceback.print_exc()
            finally:
                self.cancelled.discard(job["id"])
                self.current = None

    # -- the two kinds of work ---------------------------------------------

    def _train(self, job: dict) -> None:
        from rtdetr import RTDETR

        job_id = job["id"]
        dataset = self.db.one("SELECT * FROM datasets WHERE id = ?", (job["dataset_id"],))
        self.db.update_job(job_id, status="running", started=time.time(), detail=None, progress=0)
        run_name = f"job{job['resume_of'] or job_id}"

        def on_epoch_end(row: dict) -> None:
            if job_id in self.cancelled:
                raise Cancelled()
            self.db.add_epoch(job_id, row)
            self.db.update_job(job_id, progress=row["epoch"] / max(job["epochs"], 1))

        model = RTDETR(job["model"], verbose=False)
        # The trainer picks its own run directory and skips one that already
        # exists, so the log cannot be written there until it comes back.
        handle, temporary = tempfile.mkstemp(prefix=f"job{job_id}_", suffix=".log")
        log = Path(temporary)
        try:
            with open(handle, "w", encoding="utf-8") as stream, contextlib.redirect_stdout(stream):
                best = model.train(
                    data=str(Path(dataset["path"]) / "data.yaml"),
                    epochs=job["epochs"],
                    imgsz=job["imgsz"],
                    batch=job["batch"],
                    freeze=job["freeze"] or None,
                    device=job["device"] or None,
                    project=str(self.runs_dir),
                    name=run_name,
                    resume=bool(job["resume_of"]),
                    workers=0,
                    on_epoch_end=on_epoch_end,
                )
        except Exception:
            self.db.update_job(job_id, detail=_tail(log))  # what it said before it died
            log.replace(self.runs_dir / f"job{job_id}-failed.log")
            raise
        run_dir = best.parent.parent
        shutil.move(str(log), run_dir / "train.log")
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        # export first: a job that says "done" must have everything it advertises
        self.db.update_job(job_id, status="exporting", progress=1.0, detail="exporting…")
        note = self._export(model, run_dir, job_id)
        self.db.update_job(
            job_id,
            status="done",
            detail=note,
            run_dir=str(run_dir),
            best_map=summary.get("best_map50_95"),
            finished=time.time(),
        )

    def _autolabel(self, job: dict) -> None:
        """Pre-label every unlabelled image in a dataset, so a person only corrects."""
        from rtdetr import RTDETR

        job_id = job["id"]
        dataset = self.db.one("SELECT * FROM datasets WHERE id = ?", (job["dataset_id"],))
        images_root, labels_root = Path(dataset["images_dir"]), Path(dataset["labels_dir"])
        names = json.loads(dataset["classes"])
        images = [
            image
            for image in list_images(images_root)
            if not label_path(image, images_root, labels_root).exists()
        ]
        self.db.update_job(job_id, status="running", started=time.time(), detail=None, progress=0)
        if not images:
            self.db.update_job(
                job_id, status="done", detail="every image already has labels",
                progress=1.0, finished=time.time(),
            )
            return

        model = RTDETR(job["model"], verbose=False)
        written = 0
        for i, image in enumerate(images, start=1):
            if job_id in self.cancelled:
                raise Cancelled()
            boxes = predict_boxes(model, image, names, conf=job["conf"] or 0.35)
            write_labels(label_path(image, images_root, labels_root), boxes)
            written += bool(boxes)
            self.db.update_job(job_id, progress=i / len(images))
        self.db.update_job(
            job_id,
            status="done",
            detail=f"{len(images)} images labelled, {written} with boxes — check them",
            progress=1.0,
            finished=time.time(),
        )
        self.refresh_counts(dataset["id"])

    def _evaluate(self, job: dict) -> None:
        """Score a trained run on its validation split, and draw what it got wrong.

        A single mAP tells you whether to keep going but never why. This writes
        one side-by-side image per validation frame — truth in green, prediction
        in red — plus per-class AP, which is where "the model never sees small
        cans" actually shows up.
        """
        import cv2
        import numpy as np

        from rtdetr import RTDETR
        from rtdetr.data.dataset import load_data_yaml
        from rtdetr.plotting import draw_boxes
        from rtdetr.results import Boxes
        from rtdetr.validator import validate_torch

        job_id = job["id"]
        parent = self.db.one("SELECT * FROM jobs WHERE id = ?", (job["resume_of"],))
        if parent is None or not parent["run_dir"]:
            raise RuntimeError("that training job has no weights to evaluate")
        dataset = self.db.one("SELECT * FROM datasets WHERE id = ?", (job["dataset_id"],))
        data_yaml = str(Path(dataset["path"]) / "data.yaml")
        weights = Path(parent["run_dir"]) / "weights" / "best.pt"

        self.db.update_job(job_id, status="running", started=time.time(), detail=None, progress=0)
        model = RTDETR(str(weights), verbose=False)
        conf = job["conf"] or 0.25

        metrics = validate_torch(
            model.net, data_yaml, imgsz=parent["imgsz"], batch=1, device="cpu", workers=0
        )
        names = load_data_yaml(data_yaml)["names"]

        out = Path(parent["run_dir"]) / "eval"
        out.mkdir(parents=True, exist_ok=True)
        from rtdetr.data.dataset import DetDataset

        val = DetDataset(data_yaml, "val", parent["imgsz"], augment=False)
        cards = []
        for i, image_path in enumerate(val.files):
            if job_id in self.cancelled:
                raise Cancelled()
            frame = cv2.imread(str(image_path))
            height, width = frame.shape[:2]
            truth = val._load_labels(image_path)             # cls cx cy w h, normalised
            corners = np.zeros((len(truth), 6), np.float32)
            if len(truth):
                cx, cy, w, h = (truth[:, 1] * width, truth[:, 2] * height,
                                truth[:, 3] * width, truth[:, 4] * height)
                corners[:, 0], corners[:, 1] = cx - w / 2, cy - h / 2
                corners[:, 2], corners[:, 3] = cx + w / 2, cy + h / 2
                corners[:, 4], corners[:, 5] = 1.0, truth[:, 0]
            painted = draw_boxes(
                frame,
                Boxes(corners, (height, width)),
                {k: f"참 {v}" for k, v in names.items()},
                conf=False,
                color=(110, 220, 60),  # truth is always green, whatever the class
            )
            result = model.predict(str(image_path), conf=conf, verbose=False)[0]
            painted = draw_boxes(painted, result.boxes, result.names)
            cv2.imwrite(str(out / f"{i:04d}_{image_path.stem}.jpg"), painted)
            cards.append(
                {
                    "file": f"{i:04d}_{image_path.stem}.jpg",
                    "truth": len(truth),
                    "found": len(result.boxes),
                }
            )
            self.db.update_job(job_id, progress=(i + 1) / max(len(val.files), 1))

        report = {
            "map50": metrics["map50"],
            "map50_95": metrics["map"],
            "conf": conf,
            "per_class": [
                {"name": names.get(k, str(k)), "ap50_95": v}
                for k, v in sorted(metrics.get("per_class", {}).items())
            ],
            "images": cards,
        }
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
        self.db.update_job(
            job_id,
            status="done",
            run_dir=parent["run_dir"],
            best_map=metrics["map"],
            progress=1.0,
            detail=f"mAP50 {metrics['map50']:.3f} · mAP50-95 {metrics['map']:.3f}"
                   f" · {len(cards)}장 미리보기",
            finished=time.time(),
        )

    def _predict(self, job: dict) -> None:
        """Run a model over everything in a source and keep the results.

        One image at a time through the browser is fine for a look; a shift's
        worth of footage is a job. Annotated frames land next to a results.json
        that holds every box, and the pair zips up for download.
        """
        import cv2

        from rtdetr import RTDETR
        from rtdetr.sources import SourceLoader

        job_id = job["id"]
        out = self.runs_dir / f"job{job_id}"
        (out / "images").mkdir(parents=True, exist_ok=True)
        self.db.update_job(job_id, status="running", started=time.time(), detail=None, progress=0)

        model = RTDETR(job["model"], verbose=False)
        conf = job["conf"] or 0.25
        loader = SourceLoader(job["source"], vid_stride=1)
        total = max(len(loader), 1)
        records, found, video = [], 0, None
        try:
            for i, frame in enumerate(loader, start=1):
                if job_id in self.cancelled:
                    raise Cancelled()
                result = model.predict(frame.img, conf=conf, verbose=False)[0]
                painted = result.plot()
                found += len(result.boxes)
                name = f"{i:05d}_{Path(frame.path).stem}.jpg"
                cv2.imwrite(str(out / "images" / name), painted)
                if frame.kind != "image":
                    if video is None:
                        height, width = painted.shape[:2]
                        video = cv2.VideoWriter(
                            str(out / "annotated.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (width, height),
                        )
                    video.write(painted)
                records.append({"file": name, "source": frame.path, "boxes": result.summary()})
                self.db.update_job(
                    job_id, progress=min(i / total, 0.99) if frame.kind == "image" else 0.5
                )
        finally:
            if video is not None:
                video.release()

        (out / "results.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.db.update_job(
            job_id,
            status="done",
            run_dir=str(out),
            progress=1.0,
            detail=f"{len(records)}장 · 상자 {found}개",
            finished=time.time(),
        )

    def refresh_counts(self, dataset_id: int) -> None:
        dataset = self.db.one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        images_root, labels_root = Path(dataset["images_dir"]), Path(dataset["labels_dir"])
        images = list_images(images_root)
        labelled = sum(label_path(i, images_root, labels_root).exists() for i in images)
        boxes = sum(
            len(read_labels(label_path(i, images_root, labels_root))) for i in images
        )
        self.db.update_dataset(dataset_id, images=len(images), labelled=labelled)
        return boxes

    def _export(self, model, run_dir: Path, job_id: int) -> str | None:
        """Deployable artefacts, so a finished job is downloadable straight away."""
        try:
            model.export(format="openvino", out_dir=run_dir / "openvino", verbose=False)
            return None
        except Exception as exc:  # a model that trained is still worth keeping
            return f"export failed: {exc}"
