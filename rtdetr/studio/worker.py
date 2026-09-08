# Apache-2.0
"""One background thread that runs queued training jobs, one at a time.

Training is CPU- or GPU-bound work inside this process; a thread is enough
because torch releases the GIL. Cancelling sets a flag that the per-epoch
callback raises on, so a job stops at the next epoch boundary rather than
being killed mid-write.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path


class Cancelled(Exception):
    """Raised inside the epoch callback to stop a run cleanly."""


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

    def run(self) -> None:
        while not self._stop.is_set():
            job = self.db.one("SELECT * FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1")
            if job is None:
                time.sleep(self.poll)
                continue
            self._run_job(job)

    def _run_job(self, job: dict) -> None:
        from rtdetr import RTDETR

        job_id = job["id"]
        self.current = job_id
        dataset = self.db.one("SELECT * FROM datasets WHERE id = ?", (job["dataset_id"],))
        data_yaml = Path(dataset["path"]) / "data.yaml"
        self.db.update_job(job_id, status="running", started=time.time(), detail=None)

        def on_epoch_end(row: dict) -> None:
            if job_id in self.cancelled:
                raise Cancelled()
            self.db.add_epoch(job_id, row)

        try:
            model = RTDETR(job["model"], verbose=False)
            best = model.train(
                data=str(data_yaml),
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
                finished=time.time(),
            )
            self._export(model, run_dir, job_id)
        except Cancelled:
            self.db.update_job(job_id, status="cancelled", finished=time.time())
        except Exception as exc:
            self.db.update_job(
                job_id,
                status="failed",
                detail=f"{type(exc).__name__}: {exc}",
                finished=time.time(),
            )
            traceback.print_exc()
        finally:
            self.cancelled.discard(job_id)
            self.current = None

    def _export(self, model, run_dir: Path, job_id: int) -> None:
        """Deployable artefacts, so a finished job is downloadable straight away."""
        try:
            export_dir = run_dir / "openvino"
            model.export(format="openvino", out_dir=export_dir, verbose=False)
        except Exception as exc:  # a model that trained is still worth keeping
            self.db.update_job(job_id, detail=f"export failed: {exc}")
