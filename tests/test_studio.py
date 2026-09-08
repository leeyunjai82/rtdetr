# Apache-2.0
"""The studio's API: uploading a dataset, queueing a job, guarding both."""

from __future__ import annotations

import importlib
import io
import zipfile

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def studio(tmp_path, monkeypatch):
    """A studio rooted in a temp folder, with the worker left asleep."""
    monkeypatch.setenv("RTDETR_STUDIO_HOME", str(tmp_path / "home"))
    from rtdetr.studio import app as module

    module = importlib.reload(module)
    for folder in (module.DATASETS, module.RUNS):
        folder.mkdir(parents=True, exist_ok=True)
    # no context manager: startup never runs, so no worker picks jobs up
    return TestClient(module.app), module


def make_zip(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buffer.getvalue()


def image_bytes(colour: int = 40) -> bytes:
    import cv2

    ok, buffer = cv2.imencode(".jpg", np.full((32, 48, 3), colour, np.uint8))
    assert ok
    return buffer.tobytes()


def upload(client, files, name="set"):
    return client.post(
        "/api/datasets",
        data={"name": name},
        files={"archive": ("d.zip", make_zip(files), "application/zip")},
    )


def test_the_page_is_served(studio):
    client, _ = studio
    body = client.get("/").text
    assert "rtdetr studio" in body and "<canvas" in body


def test_uploading_images_and_labels_counts_them(studio):
    client, _ = studio
    response = upload(
        client,
        {
            "images/a.jpg": image_bytes(),
            "images/b.jpg": image_bytes(60),
            "labels/a.txt": b"0 0.5 0.5 0.2 0.2\n",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["images"] == 2 and body["labelled"] == 1
    assert client.get("/api/datasets").json()[0]["name"] == "set"


def test_a_dataset_without_a_data_yaml_gets_one(studio):
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"2 0.5 0.5 0.2 0.2\n"})
    path = module.Path(client.get("/api/datasets").json()[0]["path"])
    assert (path / "data.yaml").exists()
    import yaml

    assert 2 in yaml.safe_load((path / "data.yaml").read_text())["names"]


def test_an_existing_data_yaml_keeps_its_class_names(studio):
    client, _ = studio
    upload(
        client,
        {
            "images/a.jpg": image_bytes(),
            "labels/a.txt": b"0 0.5 0.5 0.2 0.2\n",
            "data.yaml": b"train: images\nval: images\nnames:\n  0: can\n  1: bottle\n",
        },
    )
    assert client.get("/api/datasets").json()[0]["classes"] == ["can", "bottle"]


def test_rubbish_uploads_are_rejected_with_a_reason(studio):
    client, _ = studio
    assert upload(client, {"notes.txt": b"no images here"}).status_code == 400
    response = client.post(
        "/api/datasets",
        data={"name": "x"},
        files={"archive": ("d.zip", b"not a zip at all", "application/zip")},
    )
    assert response.status_code == 400 and "zip" in response.json()["error"]


def test_an_archive_cannot_write_outside_its_folder(studio):
    """Zip-slip: an entry climbing out of the dataset directory is refused."""
    client, _ = studio
    response = upload(client, {"../escaped.jpg": image_bytes()})
    assert response.status_code == 400 and "unsafe path" in response.json()["error"]


def test_a_job_needs_a_dataset_that_exists_and_has_labels(studio):
    client, _ = studio
    assert client.post("/api/jobs", json={"dataset_id": 999}).status_code == 404

    upload(client, {"images/a.jpg": image_bytes()}, name="unlabelled")
    dataset_id = client.get("/api/datasets").json()[0]["id"]
    response = client.post("/api/jobs", json={"dataset_id": dataset_id})
    assert response.status_code == 400 and "labels" in response.json()["error"]


def test_queueing_a_job_and_cancelling_it_before_it_runs(studio):
    client, _ = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 0.5 0.5 0.2 0.2\n"})
    dataset_id = client.get("/api/datasets").json()[0]["id"]

    job_id = client.post(
        "/api/jobs",
        json={"dataset_id": dataset_id, "model": "rtdetr-r18", "epochs": 2, "freeze": "backbone"},
    ).json()["id"]

    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "queued" and job["epochs_done"] == [] and job["artifacts"] == []
    assert client.get("/api/status").json()["queued"] == 1

    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 200
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "cancelled"


def test_downloads_and_predictions_wait_for_a_finished_run(studio):
    client, _ = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 0.5 0.5 0.2 0.2\n"})
    dataset_id = client.get("/api/datasets").json()[0]["id"]
    job_id = client.post("/api/jobs", json={"dataset_id": dataset_id}).json()["id"]

    assert client.get(f"/api/jobs/{job_id}/download/weights").status_code == 404
    assert client.post(f"/api/jobs/{job_id}/predict").status_code == 404
    assert client.get("/api/jobs/999").status_code == 404
