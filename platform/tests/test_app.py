# Apache-2.0
"""The platform API: datasets, labels, jobs — and the guards around them."""

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
    """A platform rooted in a temp folder, with the worker left asleep."""
    monkeypatch.setenv("RTDETR_PLATFORM_HOME", str(tmp_path / "home"))
    import app as module

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


def test_the_pages_are_served(studio):
    client, _ = studio
    assert "rtdetr platform" in client.get("/").text

    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    assert "<canvas" in client.get("/label/1").text
    assert client.get("/label/999").status_code == 404


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


def test_a_folder_already_on_this_machine_can_be_registered(studio, tmp_path):
    """The label-a-folder path: no copying, no zip."""
    import cv2

    client, _ = studio
    folder = tmp_path / "shots"
    folder.mkdir()
    cv2.imwrite(str(folder / "a.jpg"), np.full((20, 20, 3), 30, np.uint8))

    response = client.post(
        "/api/datasets/local", json={"path": str(folder), "names": ["can", "bottle"]}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["images"] == 1 and body["classes"] == ["can", "bottle"]
    assert (folder / "data.yaml").exists()

    missing = client.post("/api/datasets/local", json={"path": str(tmp_path / "nope")})
    assert missing.status_code == 400


def test_labels_can_be_read_and_written_through_the_api(studio):
    client, _ = studio
    upload(client, {"images/a.jpg": image_bytes(), "images/b.jpg": image_bytes(70)})
    assert client.get("/api/datasets/1/labels/0").json() == {"boxes": []}

    boxes = [{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}]
    assert client.post("/api/datasets/1/labels/0", json={"boxes": boxes}).json()["saved"] is True
    assert client.get("/api/datasets/1/labels/0").json()["boxes"] == boxes

    listing = client.get("/api/datasets/1").json()
    assert [f["labelled"] for f in listing["files"]] == [True, False]
    assert client.get("/api/datasets").json()[0]["labelled"] == 1

    assert client.get("/api/datasets/1/labels/99").status_code == 404


def test_classes_can_be_renamed_and_the_data_yaml_follows(studio):
    import yaml

    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    renamed = client.post("/api/datasets/1/classes", json={"names": ["can"]})
    assert renamed.json()["names"] == ["can"]

    path = module.Path(client.get("/api/datasets").json()[0]["path"]) / "data.yaml"
    assert yaml.safe_load(path.read_text())["names"] == {0: "can"}
    assert client.post("/api/datasets/1/classes", json={"names": []}).status_code == 400


def test_a_batch_autolabel_is_queued_like_any_other_job(studio):
    client, _ = studio
    upload(client, {"images/a.jpg": image_bytes()})
    job_id = client.post("/api/datasets/1/autolabel", json={"model": "rtdetr-r18"}).json()["id"]

    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["kind"] == "autolabel" and job["status"] == "queued"
    assert client.get("/api/jobs").json()[0]["kind"] == "autolabel"

    client.post(f"/api/jobs/{job_id}/cancel")
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "cancelled"


def test_jobs_left_running_by_a_dead_process_are_marked_failed(studio):
    """Nothing survives a restart; a spinner that never finishes is worse than a message."""
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    job_id = client.post("/api/jobs", json={"dataset_id": 1}).json()["id"]
    module.db.update_job(job_id, status="running")

    assert module.worker.reap_stale() == 1
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "failed" and "restart" in job["detail"]
