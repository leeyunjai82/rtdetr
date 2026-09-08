# Apache-2.0
"""One background thread that runs queued jobs — training, or a batch auto-label.

Work happens inside this process; a thread is enough because torch and OpenVINO
release the GIL. Cancelling sets a flag the job checks at its next natural
boundary (an epoch, or an image), so nothing is killed mid-write.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path

from labeling import label_path, list_images, predict_boxes, read_labels, write_labels


class Cancelled(Exception):
    """Raised inside a job to stop it cleanly."""


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
                else:
                    self._train(job)
            except Cancelled:
                self.db.update_job(job["id"], status="cancelled", finished=time.time())
            except Exception as exc:
                self.db.update_job(
                    job["id"],
                    status="failed",
                    detail=f"{type(exc).__name__}: {exc}",
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

        def on_epoch_end(row: dict) -> None:
            if job_id in self.cancelled:
                raise Cancelled()
            self.db.add_epoch(job_id, row)
            self.db.update_job(job_id, progress=row["epoch"] / max(job["epochs"], 1))

        model = RTDETR(job["model"], verbose=False)
        best = model.train(
            data=str(Path(dataset["path"]) / "data.yaml"),
            epochs=job["epochs"],
            imgsz=job["imgsz"],
            batch=job["batch"],
            freeze=job["freeze"] or None,
            device=job["device"] or None,
            project=str(self.runs_dir),
            name=f"job{job_id}",
            workers=0,
            on_epoch_end=on_epoch_end,
        )
        run_dir = best.parent.parent
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        self.db.update_job(
            job_id,
            status="done",
            run_dir=str(run_dir),
            best_map=summary.get("best_map50_95"),
            progress=1.0,
            finished=time.time(),
        )
        self._export(model, run_dir, job_id)

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

    def _export(self, model, run_dir: Path, job_id: int) -> None:
        """Deployable artefacts, so a finished job is downloadable straight away."""
        try:
            model.export(format="openvino", out_dir=run_dir / "openvino", verbose=False)
        except Exception as exc:  # a model that trained is still worth keeping
            self.db.update_job(job_id, detail=f"export failed: {exc}")
