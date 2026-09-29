# Apache-2.0
"""The platform API: datasets, labels, jobs — and the guards around them."""

from __future__ import annotations

import importlib
import io
import json
import os
import zipfile
from pathlib import Path

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


def test_a_dataset_can_be_exported_and_imported_again(studio):
    """Round trip: what comes out of export goes back in and counts the same."""
    client, _ = studio
    upload(
        client,
        {
            "images/a.jpg": image_bytes(),
            "images/b.jpg": image_bytes(70),
            "labels/a.txt": b"0 0.5 0.5 0.2 0.2\n",
            "data.yaml": b"train: images\nval: images\nnames:\n  0: can\n",
        },
    )
    response = client.get("/api/datasets/1/export")
    assert response.status_code == 200
    assert response.headers["content-disposition"].endswith('.zip"')

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    assert sorted(archive.namelist()) == [
        "data.yaml", "images/a.jpg", "images/b.jpg", "labels/a.txt",
    ]

    again = client.post(
        "/api/datasets",
        data={"name": "round-trip"},
        files={"archive": ("d.zip", response.content, "application/zip")},
    ).json()
    assert again["images"] == 2 and again["labelled"] == 1 and again["classes"] == ["can"]


def test_training_holds_images_back_for_validation(studio):
    """Validating on the training images only ever flatters the run."""
    import yaml

    client, module = studio
    files = {}
    for i in range(8):
        files[f"images/{i}.jpg"] = image_bytes(10 * i)
        files[f"labels/{i}.txt"] = b"0 0.5 0.5 0.2 0.2\n"
    upload(client, files)

    job = client.post("/api/jobs", json={"dataset_id": 1, "val_ratio": 0.25}).json()
    detail = client.get(f"/api/jobs/{job['id']}").json()["detail"]
    assert detail == "6 train / 2 val images"

    root = module.Path(client.get("/api/datasets").json()[0]["path"])
    cfg = yaml.safe_load((root / "data.yaml").read_text())
    assert cfg["train"] == "train.txt" and cfg["val"] == "val.txt"
    train = (root / "train.txt").read_text().splitlines()
    val = (root / "val.txt").read_text().splitlines()
    assert len(train) == 6 and len(val) == 2
    assert not set(train) & set(val)  # nothing is in both


def test_a_dataset_too_small_to_split_says_so(studio):
    client, _ = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    job = client.post("/api/jobs", json={"dataset_id": 1}).json()
    assert "validating on the same ones" in client.get(f"/api/jobs/{job['id']}").json()["detail"]


def test_a_dataset_can_be_deleted_unless_a_job_is_using_it(studio):
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    path = module.Path(client.get("/api/datasets").json()[0]["path"])
    job_id = client.post("/api/jobs", json={"dataset_id": 1}).json()["id"]

    blocked = client.request("DELETE", "/api/datasets/1")
    assert blocked.status_code == 400 and "still using it" in blocked.json()["error"]

    client.post(f"/api/jobs/{job_id}/cancel")
    assert client.request("DELETE", "/api/datasets/1").json()["files_removed"] is True
    assert client.get("/api/datasets").json() == [] and not path.exists()


def test_images_can_be_uploaded_without_making_a_zip(studio):
    client, _ = studio
    response = client.post(
        "/api/datasets/images",
        data={"name": "picked"},
        files=[
            ("files", ("a.jpg", image_bytes(), "image/jpeg")),
            ("files", ("b.png", image_bytes(80), "image/png")),
            ("files", ("notes.txt", b"not an image", "text/plain")),
        ],
    )
    assert response.status_code == 200
    assert response.json()["added"] == 2 and response.json()["images"] == 2


def test_more_images_can_be_added_to_a_dataset_later(studio):
    """Collection happens over time, not in one go."""
    client, _ = studio
    first = client.post(
        "/api/datasets/images",
        data={"name": "line"},
        files=[("files", ("a.jpg", image_bytes(), "image/jpeg"))],
    ).json()

    again = client.post(
        "/api/datasets/images",
        data={"name": "line", "dataset_id": str(first["id"])},
        files=[("files", ("a.jpg", image_bytes(90), "image/jpeg"))],
    ).json()
    assert again["id"] == first["id"] and again["images"] == 2  # same name, not overwritten


def test_uploading_no_images_at_all_leaves_nothing_behind(studio, tmp_path):
    client, module = studio
    response = client.post(
        "/api/datasets/images",
        data={"name": "junk"},
        files=[("files", ("notes.txt", b"nope", "text/plain"))],
    )
    assert response.status_code == 400
    assert client.get("/api/datasets").json() == []
    assert list(module.DATASETS.iterdir()) == []


def test_a_video_becomes_a_dataset_of_frames(studio, tmp_path):
    import cv2
    import numpy as np

    client, _ = studio
    clip = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (48, 32))
    for i in range(25):
        writer.write(np.full((32, 48, 3), i * 8, np.uint8))
    writer.release()

    response = client.post(
        "/api/datasets/video",
        data={"name": "clip", "every": "10", "max_frames": "50"},
        files={"video": ("clip.mp4", clip.read_bytes(), "video/mp4")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["added"] == 3 and body["images"] == 3  # frames 0, 10, 20
    assert body["scanned"] == 25


def test_a_video_frame_cap_is_respected(studio, tmp_path):
    import cv2
    import numpy as np

    client, _ = studio
    clip = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (48, 32))
    for i in range(30):
        writer.write(np.full((32, 48, 3), i * 8, np.uint8))
    writer.release()

    body = client.post(
        "/api/datasets/video",
        data={"name": "capped", "every": "1", "max_frames": "5"},
        files={"video": ("clip.mp4", clip.read_bytes(), "video/mp4")},
    ).json()
    assert body["added"] == 5


def test_something_that_is_not_a_video_is_refused(studio):
    client, module = studio
    response = client.post(
        "/api/datasets/video",
        data={"name": "bad"},
        files={"video": ("clip.mp4", b"definitely not a video", "video/mp4")},
    )
    assert response.status_code == 400
    assert list(module.DATASETS.iterdir()) == []


def test_stats_say_what_is_in_a_dataset_and_what_looks_wrong(studio):
    client, _ = studio
    upload(
        client,
        {
            "images/a.jpg": image_bytes(),
            "images/b.jpg": image_bytes(60),
            "images/c.jpg": image_bytes(90),
            "labels/a.txt": b"0 0.5 0.5 0.3 0.3\n1 0.2 0.2 0.1 0.1\n",
            "labels/b.txt": b"",                       # labelled as "nothing here"
            "labels/c.txt": b"0 0.5 0.5 0.01 0.01\n",  # a speck
            "data.yaml": b"train: images\nval: images\nnames:\n  0: can\n  1: bottle\n",
        },
    )
    stats = client.get("/api/datasets/1/stats").json()
    assert stats["images"] == 3 and stats["boxes"] == 3
    assert stats["per_class"] == [{"name": "can", "boxes": 2}, {"name": "bottle", "boxes": 1}]
    assert stats["empty_labels"] == ["b.jpg"] and stats["unlabelled"] == []
    assert stats["tiny_boxes"] == 1


def test_an_image_can_be_dropped_with_its_label(studio):
    client, _ = studio
    upload(
        client,
        {
            "images/a.jpg": image_bytes(),
            "images/b.jpg": image_bytes(60),
            "labels/a.txt": b"0 0.5 0.5 0.2 0.2\n",
        },
    )
    assert client.request("DELETE", "/api/datasets/1/images/0").json()["images"] == 1
    listing = client.get("/api/datasets/1").json()
    assert [f["name"] for f in listing["files"]] == ["b.jpg"]
    assert client.get("/api/datasets").json()[0]["labelled"] == 0  # the label went too
    assert client.request("DELETE", "/api/datasets/1/images/9").status_code == 404


def test_a_dataset_can_be_duplicated_before_a_risky_relabel(studio):
    client, _ = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 0.5 0.5 0.2 0.2\n"})
    copy = client.post("/api/datasets/1/duplicate", json={"name": "safe"}).json()
    assert copy["id"] == 2 and copy["images"] == 1 and copy["labelled"] == 1

    # the copy is independent: dropping from it leaves the original alone
    client.request("DELETE", "/api/datasets/2/images/0")
    assert client.get("/api/datasets/1").json()["images"] == 1


def test_merging_remaps_class_indices_onto_the_union(studio):
    """The subtle one: index 0 means different things in different datasets."""
    client, module = studio
    upload(
        client,
        {
            "images/a.jpg": image_bytes(),
            "labels/a.txt": b"0 0.5 0.5 0.2 0.2\n",
            "data.yaml": b"train: images\nval: images\nnames:\n  0: can\n",
        },
        name="cans",
    )
    upload(
        client,
        {
            "images/b.jpg": image_bytes(60),
            "labels/b.txt": b"0 0.4 0.4 0.2 0.2\n1 0.6 0.6 0.2 0.2\n",
            "data.yaml": b"train: images\nval: images\nnames:\n  0: bottle\n  1: can\n",
        },
        name="bottles",
    )
    # picked in either order, the union comes out the same
    merged = client.post("/api/datasets/merge", json={"ids": [2, 1], "name": "both"}).json()
    assert merged["classes"] == ["can", "bottle"] and merged["images"] == 2

    root = module.Path(client.get("/api/datasets").json()[0]["path"])
    labels = sorted((root / "labels").glob("*.txt"))
    written = {p.name: p.read_text().split() for p in labels}
    from labeling import read_labels

    # "can" was 0 in the first set and 1 in the second; both must end up 0
    classes = sorted(row["cls"] for p in labels for row in read_labels(p))
    assert classes == [0, 0, 1] and len(written) == 2
    assert client.post("/api/datasets/merge", json={"ids": [1]}).status_code == 400


def test_a_trained_run_can_be_registered_as_a_named_model(studio, tmp_path):
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    job_id = client.post("/api/jobs", json={"dataset_id": 1}).json()["id"]

    # nothing to register until the run has weights
    assert client.post("/api/models", json={"job_id": job_id, "name": "v1"}).status_code == 404

    run = module.RUNS / f"job{job_id}" / "weights"
    run.mkdir(parents=True)
    (run / "best.pt").write_bytes(b"weights")
    module.db.update_job(job_id, run_dir=str(run.parent), status="done")

    assert client.post("/api/models", json={"job_id": job_id, "name": "v1"}).json()["id"] == 1
    registry = client.get("/api/models").json()
    assert registry["models"][0]["name"] == "v1"
    assert "rtdetr-r18" in registry["builtin"]

    # and a job can name it instead of a path
    assert module._resolve_model("v1") == str(run / "best.pt")
    assert module._resolve_model("1") == str(run / "best.pt")
    assert module._resolve_model("rtdetr-r34") == "rtdetr-r34"

    assert client.request("DELETE", "/api/models/1").json()["deleted"] is True
    assert client.get("/api/models").json()["models"] == []


def test_a_model_file_can_be_brought_in_from_outside(studio):
    client, _ = studio
    response = client.post(
        "/api/models/upload",
        data={"name": "from-a-colleague", "classes": "can, bottle"},
        files={"file": ("best.pt", b"weights", "application/octet-stream")},
    )
    assert response.status_code == 200 and response.json()["path"].endswith("best.pt")
    assert client.get("/api/models").json()["models"][0]["classes"] == ["can", "bottle"]

    bad = client.post(
        "/api/models/upload",
        data={"name": "nope"},
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert bad.status_code == 400


def test_batch_prediction_needs_something_to_run_on(studio, tmp_path):
    client, _ = studio
    assert client.post("/api/predict", data={"model": "rtdetr-r18"}).status_code == 400

    missing = client.post(
        "/api/predict", data={"model": "rtdetr-r18", "path": str(tmp_path / "nowhere")}
    )
    assert missing.status_code == 400 and "does not exist" in missing.json()["error"]


def test_batch_prediction_over_a_dataset_is_queued_as_a_job(studio):
    client, _ = studio
    upload(client, {"images/a.jpg": image_bytes()})
    job = client.post(
        "/api/predict", data={"model": "rtdetr-r18", "dataset_id": "1", "conf": "0.4"}
    ).json()

    row = client.get(f"/api/jobs/{job['id']}").json()
    assert row["kind"] == "predict" and row["status"] == "queued"
    assert row["conf"] == 0.4 and row["source"].endswith("images")


def test_the_preview_endpoint_refuses_what_it_cannot_read(studio):
    client, _ = studio
    assert client.post("/api/preview", data={"model": "rtdetr-r18"}).status_code == 400
    unreadable = client.post(
        "/api/preview",
        data={"model": "rtdetr-r18"},
        files={"image": ("x.jpg", b"not an image", "image/jpeg")},
    )
    assert unreadable.status_code == 400


def test_a_job_without_a_dataset_still_shows_in_the_list(studio, tmp_path):
    """Inference over a folder belongs to no dataset; the list must still have it."""
    client, _ = studio
    folder = tmp_path / "shift"
    folder.mkdir()
    (folder / "a.jpg").write_bytes(image_bytes())

    job = client.post("/api/predict", data={"model": "rtdetr-r18", "path": str(folder)}).json()
    listed = client.get("/api/jobs").json()
    assert [row["id"] for row in listed] == [job["id"]]
    assert listed[0]["dataset"] is None and listed[0]["source"] == str(folder)


def test_an_archive_cannot_write_next_to_its_dataset(studio, tmp_path):
    """A sibling folder shares the dataset folder's prefix — that is not "inside"."""
    from fastapi import HTTPException

    _, module = studio
    target = tmp_path / "1757_set"
    target.mkdir()
    escape = f"../{target.name}-evil/pwned.txt"
    with zipfile.ZipFile(io.BytesIO(make_zip({escape: b"x"}))) as zf:
        with pytest.raises(HTTPException) as raised:
            module._safe_extract(zf, target)
    assert "unsafe path" in raised.value.detail
    assert not (tmp_path / f"{target.name}-evil").exists()


def test_cancelling_an_export_says_it_is_too_late(studio):
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    job = client.post("/api/jobs", json={"dataset_id": 1, "epochs": 1}).json()["id"]
    module.db.update_job(job, status="exporting")

    body = client.post(f"/api/jobs/{job}/cancel").json()
    assert body["status"] == "exporting"
    assert module.db.one("SELECT status FROM jobs WHERE id = ?", (job,))["status"] == "exporting"


def test_a_restart_clears_jobs_left_exporting(studio):
    """An export killed mid-write is dead too, not a spinner to stare at."""
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    job = client.post("/api/jobs", json={"dataset_id": 1, "epochs": 1}).json()["id"]
    module.db.update_job(job, status="exporting")

    assert module.worker.reap_stale() == 1
    row = module.db.one("SELECT * FROM jobs WHERE id = ?", (job,))
    assert row["status"] == "failed" and "restart" in row["detail"]


def test_autolabelling_an_image_that_is_not_there(studio):
    client, _ = studio
    upload(client, {"images/a.jpg": image_bytes()})
    assert client.post("/api/datasets/1/autolabel/9", json={}).status_code == 404


def test_the_worker_does_not_shadow_the_thread_machinery(studio):
    """threading.Thread has a private _stop(); overwriting it breaks fork handling."""
    import threading

    _, module = studio
    assert callable(threading.Thread._stop)
    assert callable(module.worker._stop)  # still the method, not an Event

    module.worker.stop()
    assert module.worker._stopping.is_set()


def test_an_ir_uploads_as_a_pair_and_the_bin_takes_the_xml_stem(studio):
    """An OpenVINO IR is .xml + .bin under one stem; half of it is useless."""
    client, module = studio
    alone = client.post(
        "/api/models/upload",
        data={"name": "half"},
        files=[("files", ("net.xml", b"<net/>", "text/xml"))],
    )
    assert alone.status_code == 400 and ".bin" in alone.json()["error"]

    pair = client.post(
        "/api/models/upload",
        data={"name": "line v2", "classes": "can,bottle"},
        files=[
            ("files", ("best.xml", b"<net/>", "text/xml")),
            ("files", ("weights-export.bin", b"\x00\x01", "application/octet-stream")),
        ],
    )
    assert pair.status_code == 200
    path = Path(pair.json()["path"])
    assert path.name == "best.xml" and path.with_suffix(".bin").read_bytes() == b"\x00\x01"
    row = module.db.one("SELECT * FROM models WHERE id = ?", (pair.json()["id"],))
    assert row["kind"] == "openvino" and json.loads(row["classes"]) == ["can", "bottle"]
    assert module._resolve_model("line v2") == str(path)


def test_a_class_that_labels_still_use_cannot_be_dropped(studio):
    client, _ = studio
    upload(client, {
        "images/a.jpg": image_bytes(),
        "labels/a.txt": b"2 .5 .5 .2 .2\n",       # class index 2 is in use
        "data.yaml": b"names: {0: can, 1: bottle, 2: cap}\n",
    })
    refused = client.post("/api/datasets/1/classes", json={"names": ["can", "bottle"]})
    assert refused.status_code == 400 and "class 2" in refused.json()["error"]

    renamed = client.post("/api/datasets/1/classes", json={"names": ["can", "bottle", "lid"]})
    assert renamed.status_code == 200 and renamed.json()["names"][2] == "lid"


def test_saving_a_label_moves_the_count_by_one_without_a_rescan(studio, monkeypatch):
    """One save is one file; it must not walk the whole dataset again."""
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "images/b.jpg": image_bytes(70)})
    assert client.get("/api/datasets/1").json()["labelled"] == 0

    walks = []
    monkeypatch.setattr(module.worker, "refresh_counts", lambda *a, **k: walks.append(a))
    box = [{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}]
    client.post("/api/datasets/1/labels/0", json={"boxes": box})
    assert client.get("/api/datasets/1").json()["labelled"] == 1
    client.post("/api/datasets/1/labels/0", json={"boxes": box + box})   # same image again
    client.post("/api/datasets/1/labels/1", json={"boxes": []})          # empty label counts
    assert client.get("/api/datasets/1").json()["labelled"] == 2
    assert walks == []


def test_a_dataset_split_into_train_and_val_folders_registers(studio):
    """The standard layout — images/train + images/val — used to crash registration:
    the common folder was taken to be images/train, and val images fell outside it."""
    client, module = studio
    response = upload(client, {
        "images/train/a.jpg": image_bytes(),
        "images/train/b.jpg": image_bytes(50),
        "images/val/c.jpg": image_bytes(90),
        "labels/train/a.txt": b"0 .5 .5 .2 .2\n",
        "labels/train/b.txt": b"1 .5 .5 .2 .2\n",
        "labels/val/c.txt": b"0 .5 .5 .2 .2\n",
        "data.yaml": b"train: images/train\nval: images/val\nnames: [square, circle]\n",
    })
    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["images"] == 3 and body["labelled"] == 3 and body["classes"] == ["square", "circle"]

    listing = client.get("/api/datasets/1").json()
    names = sorted(f["name"] for f in listing["files"])
    assert names == ["train/a.jpg", "train/b.jpg", "val/c.jpg"]
    assert module.Path(listing["labels_dir"]).name == "labels"
    # the dataset brought its own split, so queuing a run must not rewrite it
    job = client.post("/api/jobs", json={"dataset_id": 1, "epochs": 1}).json()
    assert client.get(f"/api/jobs/{job['id']}").json()["detail"] is None


def roboflow_zip(wrap: str = "") -> bytes:
    """What a YOLO export from a labelling service looks like, polygons and all."""
    files = {
        "README.roboflow.txt": b"exported for testing\n",
        "data.yaml": b"train: ../train/images\nval: ../valid/images\ntest: ../test/images\n"
                     b"nc: 2\nnames: ['can', 'bottle']\n",
        "train/images/a.jpg": image_bytes(),
        "train/images/b.jpg": image_bytes(60),
        "train/labels/a.txt": b"0 .5 .5 .2 .2\n",
        "train/labels/b.txt": b"1 0.2 0.3 0.6 0.3 0.4 0.7\n",   # a polygon
        "valid/images/c.jpg": image_bytes(80),
        "valid/labels/c.txt": b"1 .5 .5 .3 .3\n",
        "test/images/d.jpg": image_bytes(100),
    }
    return make_zip({f"{wrap}{name}": data for name, data in files.items()})


@pytest.mark.parametrize("wrap", ["", "cans.v3i.yolov8/"])
def test_a_labelling_service_export_uploads_and_is_ready_to_train(studio, wrap):
    client, module = studio
    response = client.post(
        "/api/datasets",
        data={"name": "cans"},
        files={"archive": ("cans.zip", roboflow_zip(wrap), "application/zip")},
    )
    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["classes"] == ["can", "bottle"]
    assert body["images"] == 4 and body["labelled"] == 3       # test/ has no labels

    # the polygon shows up in the labelling tool as its bounding box
    files = [f["name"] for f in client.get("/api/datasets/1").json()["files"]]
    polygon = client.get(f"/api/datasets/1/labels/{files.index('train/images/b.jpg')}").json()
    assert polygon["boxes"][0]["cx"] == pytest.approx(0.4)

    # queuing a run keeps the export's own train/valid split untouched
    job = client.post("/api/jobs", json={"dataset_id": 1, "epochs": 1}).json()
    assert client.get(f"/api/jobs/{job['id']}").json()["detail"] is None

    # and the trainer reads exactly those two splits, labels included
    pytest.importorskip("torch")
    from rtdetr.data.dataset import DetDataset

    root = module.Path(client.get("/api/datasets/1").json()["path"])
    train = DetDataset(root / "data.yaml", "train", imgsz=64, augment=False)
    val = DetDataset(root / "data.yaml", "val", imgsz=64, augment=False)
    assert sorted(f.name for f in train.files) == ["a.jpg", "b.jpg"]
    assert [f.name for f in val.files] == ["c.jpg"]
    assert len(val._load_labels(val.files[0])) == 1


def test_sibling_folders_keep_their_labels_apart(studio, tmp_path):
    """Registered side by side, two folders used to share one ../labels/ and
    overwrite each other — while training looked for neither."""
    import cv2

    client, _ = studio
    for folder in ("shots_a", "shots_b"):
        (tmp_path / folder).mkdir()
        cv2.imwrite(str(tmp_path / folder / "0000.jpg"), np.full((20, 20, 3), 50, np.uint8))
        client.post("/api/datasets/local", json={"path": str(tmp_path / folder), "names": ["can"]})

    one = [{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}]
    client.post("/api/datasets/1/labels/0", json={"boxes": one})
    client.post("/api/datasets/2/labels/0", json={"boxes": one * 2})
    assert len(client.get("/api/datasets/1/labels/0").json()["boxes"]) == 1
    assert (tmp_path / "shots_a" / "0000.txt").exists()      # where the trainer reads it
    assert not (tmp_path / "labels").exists()                 # nothing outside the folder


def test_an_export_comes_back_with_its_layout_labels_and_split(studio):
    client, _ = studio
    client.post(
        "/api/datasets",
        data={"name": "cans"},
        files={"archive": ("cans.zip", roboflow_zip(), "application/zip")},
    )
    exported = client.get("/api/datasets/1/export").content
    with zipfile.ZipFile(io.BytesIO(exported)) as zf:
        names = set(zf.namelist())
        cfg = zf.read("data.yaml").decode()
        val_list = zf.read("val.txt").decode().split()
    assert "images/train/labels/a.txt" in names and "images/valid/labels/c.txt" in names
    assert "train: train.txt" in cfg and val_list == ["images/valid/images/c.jpg"]

    again = client.post(
        "/api/datasets",
        data={"name": "again"},
        files={"archive": ("again.zip", exported, "application/zip")},
    ).json()
    assert again["images"] == 4 and again["labelled"] == 3 and again["classes"] == ["can", "bottle"]
    job = client.post("/api/jobs", json={"dataset_id": again["id"], "epochs": 1}).json()
    assert client.get(f"/api/jobs/{job['id']}").json()["detail"] is None   # split survived


def test_an_uploaded_copy_trains_on_itself_not_on_the_folder_it_came_from(studio, tmp_path):
    """A zip made on this machine carries path: pointing at the original folder."""
    pytest.importorskip("torch")
    from rtdetr.data.dataset import DetDataset

    client, module = studio
    original = tmp_path / "original"
    (original / "images").mkdir(parents=True)
    (original / "images" / "a.jpg").write_bytes(image_bytes())
    archive = make_zip({
        "images/a.jpg": image_bytes(),
        "labels/a.txt": b"0 .5 .5 .2 .2\n",
        "data.yaml": f"path: {original}\ntrain: images\nval: images\nnames: [can]\n".encode(),
    })
    client.post("/api/datasets", data={"name": "copy"},
                files={"archive": ("c.zip", archive, "application/zip")})
    root = module.Path(client.get("/api/datasets/1").json()["path"])
    ds = DetDataset(root / "data.yaml", "train", imgsz=64, augment=False)
    assert ds.files[0].is_relative_to(root) and len(ds._load_labels(ds.files[0])) == 1


def test_the_file_list_tells_boxes_from_background_frames(studio):
    """An empty label is a background frame; it must not look like one with boxes."""
    client, _ = studio
    upload(client, {
        "images/a.jpg": image_bytes(),
        "images/b.jpg": image_bytes(60),
        "images/c.jpg": image_bytes(90),
        "labels/a.txt": b"0 .5 .5 .2 .2\n1 .3 .3 .1 .1\n",
        "labels/b.txt": b"",
    })
    files = {f["name"]: f for f in client.get("/api/datasets/1").json()["files"]}
    assert (files["a.jpg"]["labelled"], files["a.jpg"]["boxes"]) == (True, 2)
    assert (files["b.jpg"]["labelled"], files["b.jpg"]["boxes"]) == (True, 0)
    assert (files["c.jpg"]["labelled"], files["c.jpg"]["boxes"]) == (False, 0)


def test_the_stream_carries_the_progress_text(studio):
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    job = client.post("/api/jobs", json={"dataset_id": 1, "epochs": 1}).json()["id"]
    module.db.update_job(job, status="done", progress=1.0, detail="에폭 1/1 · 검증 중")
    body = client.get(f"/api/jobs/{job}/stream").text
    events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
    assert events[0]["detail"] == "에폭 1/1 · 검증 중" and events[-1]["type"] == "end"


def test_status_says_whether_there_is_a_gpu(studio):
    client, _ = studio
    gpu = client.get("/api/status").json()["gpu"]
    assert gpu is None or {"name", "memory_gb"} <= set(gpu)


def test_loader_workers_and_time_left_read_sensibly(monkeypatch):
    import worker

    monkeypatch.setenv("RTDETR_WORKERS", "0")
    assert worker._loader_workers() == 0
    monkeypatch.delenv("RTDETR_WORKERS")
    assert 1 <= worker._loader_workers() <= 8
    assert [worker._duration(s) for s in (42, 125, 3725)] == ["42초", "2분", "1시간 2분"]


def test_a_finished_run_can_be_deleted_with_its_files(studio):
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    job = client.post("/api/jobs", json={"dataset_id": 1, "epochs": 1}).json()["id"]
    run = module.RUNS / f"job{job}"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"w")
    module.db.update_job(job, status="done", run_dir=str(run))
    module.db.add_epoch(job, {"epoch": 1, "loss": 1.0, "map50_95": 0.1, "seconds": 1})

    assert client.delete(f"/api/jobs/{job}").json() == {"deleted": True, "files_removed": True}
    assert not run.exists() and client.get(f"/api/jobs/{job}").status_code == 404
    assert module.db.query("SELECT * FROM epochs WHERE job_id = ?", (job,)) == []


def test_a_run_something_still_needs_is_kept(studio):
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    job = client.post("/api/jobs", json={"dataset_id": 1, "epochs": 1}).json()["id"]
    run = module.RUNS / f"job{job}"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"w")

    module.db.update_job(job, status="running", run_dir=str(run))
    assert "stop it first" in client.delete(f"/api/jobs/{job}").json()["error"]

    module.db.update_job(job, status="done")
    evaluation = client.post(f"/api/jobs/{job}/evaluate", json={}).json()["id"]
    assert f"#{evaluation}" in client.delete(f"/api/jobs/{job}").json()["error"]
    assert client.delete(f"/api/jobs/{evaluation}").json()["files_removed"] is False
    assert run.exists()                          # an evaluation never takes the run with it

    client.post("/api/models", json={"job_id": job, "name": "v1"})
    assert "'v1'" in client.delete(f"/api/jobs/{job}").json()["error"]
    client.delete("/api/models/1")
    assert client.delete(f"/api/jobs/{job}").json()["files_removed"] is True


def test_startup_runs_once_per_process(studio):
    """HTTP and HTTPS servers both fire startup; the second must not reap the
    job the worker has just started."""
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    first = client.post("/api/jobs", json={"dataset_id": 1, "epochs": 1}).json()["id"]
    module.db.update_job(first, status="running")          # left over from a dead process
    module._start()
    assert module.db.one("SELECT status FROM jobs WHERE id = ?", (first,))["status"] == "failed"

    second = client.post("/api/jobs", json={"dataset_id": 1, "epochs": 1}).json()["id"]
    module.db.update_job(second, status="running")         # the worker's own, just now
    module._start()
    module.worker.stop()
    assert module.db.one("SELECT status FROM jobs WHERE id = ?", (second,))["status"] == "running"


def test_a_certificate_is_made_once_and_reused(tmp_path):
    pytest.importorskip("cryptography")
    from cryptography import x509
    from tls import ensure_certificate

    cert, key = ensure_certificate(tmp_path / "tls")
    stamp = cert.read_bytes()
    assert ensure_certificate(tmp_path / "tls") == (cert, key) and cert.read_bytes() == stamp
    parsed = x509.load_pem_x509_certificate(stamp)
    names = parsed.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert "localhost" in names.get_values_for_type(x509.DNSName)
    assert oct(key.stat().st_mode)[-3:] == "600"


def test_auto_labelling_drafts_with_the_datasets_own_run(studio):
    """A COCO model knows nothing called Paper; once a dataset is trained on,
    that run is what pre-labels it unless someone picks otherwise."""
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes()})
    assert module._default_model(1) == "rtdetr-r18"

    job = client.post("/api/jobs", json={"dataset_id": 1, "epochs": 1})
    assert job.status_code == 400                      # unlabelled: nothing to train yet
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    run_id = client.post("/api/jobs", json={"dataset_id": 2, "epochs": 1}).json()["id"]
    run = module.RUNS / f"job{run_id}"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"w")
    module.db.update_job(run_id, status="done", run_dir=str(run))

    queued = client.post("/api/datasets/2/autolabel", json={}).json()["id"]
    assert client.get(f"/api/jobs/{queued}").json()["model"] == str(run / "weights" / "best.pt")
    assert client.get(f"/api/jobs/{queued}").json()["dataset"] == "set"


def _fake_smi(folder, lines):
    script = folder / "nvidia-smi"
    body = "\n".join(f"echo '{line}'" for line in lines)
    script.write_text(f"#!/bin/sh\n{body}\n")
    script.chmod(0o755)
    return folder


def test_the_gpu_readout_comes_from_nvidia_smi(studio, tmp_path, monkeypatch):
    client, module = studio
    _fake_smi(tmp_path, ["0, NVIDIA GeForce RTX 5090, 87, 12345, 32607, 64, 402.51, 575.00"])
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    module._gpu_cache.update(at=0.0)
    gpu = client.get("/api/gpu").json()["gpus"][0]
    assert gpu["name"] == "NVIDIA GeForce RTX 5090" and gpu["util"] == 87
    assert (gpu["mem_used_mb"], gpu["mem_total_mb"], gpu["temp_c"]) == (12345, 32607, 64)
    assert round(gpu["power_w"]) == 403 and gpu["power_limit_w"] == 575


def test_a_card_without_power_readings_and_a_box_without_a_gpu(studio, tmp_path, monkeypatch):
    client, module = studio
    _fake_smi(tmp_path, ["0, Some GPU, 5, 100, 8000, 40, [N/A], [N/A]"])
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    module._gpu_cache.update(at=0.0)
    gpu = client.get("/api/gpu").json()["gpus"][0]
    assert gpu["power_w"] is None and gpu["util"] == 5

    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    module._gpu_cache.update(at=0.0)
    assert client.get("/api/gpu").json() == {"gpus": None}


def test_a_finished_run_says_where_its_model_files_are(studio):
    client, module = studio
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"})
    job = client.post("/api/jobs", json={"dataset_id": 1, "epochs": 1}).json()["id"]
    assert client.get(f"/api/jobs/{job}").json()["files"] == {}

    run = module.RUNS / f"job{job}"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"w")
    (run / "openvino").mkdir()
    (run / "openvino" / "best.xml").write_text("<net/>")
    module.db.update_job(job, status="done", run_dir=str(run))

    files = client.get(f"/api/jobs/{job}").json()["files"]
    assert files == {"weights": str((run / "weights" / "best.pt").resolve()),
                     "ir": str((run / "openvino" / "best.xml").resolve())}


def _finished_run(client, module):
    """A training job whose run folder looks like a real one: IR, labels, record."""
    upload(client, {"images/a.jpg": image_bytes(), "labels/a.txt": b"0 .5 .5 .2 .2\n"},
           name="Rock Paper Scissors")
    job = client.post("/api/jobs", json={"dataset_id": 1, "epochs": 3}).json()["id"]
    run = module.RUNS / f"job{job}"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"w")
    (run / "openvino").mkdir()
    for name, body in {"best.xml": "<net/>", "best.bin": "b", "labels.txt": "rock\n",
                       "best.names.json": "{}", "best.onnx": "big"}.items():
        (run / "openvino" / name).write_text(body)
    setup = {"variant": "r34", "names": {"0": "rock"}, "start": {"kind": "coco"},
             "params": 31_000_000, "trainable_params": 20_000_000, "freeze": "backbone",
             "optimizer": {"name": "AdamW", "lr": 1e-4, "lr_backbone": 1e-5, "weight_decay": 1e-4,
                           "warmup_epochs": 1, "grad_clip": 0.1},
             "amp": True, "augment": ["horizontal flip (p=0.5)"], "patience": 50, "seed": 0,
             "data": {
                 "train": {"images": 8, "boxes": 12, "background_images": 1, "per_class": [12]},
                 "val": {"images": 2, "boxes": 3, "background_images": 0, "per_class": [3]}},
             "device": "cuda:0", "gpu": "NVIDIA GeForce RTX 5090",
             "versions": {"rtdetr": "0.6.9", "torch": "2.9.0", "cuda": "12.8", "python": "3.12.3"}}
    (run / "run.json").write_text(json.dumps(setup))
    (run / "summary.json").write_text(json.dumps({
        "best_map50_95": 0.762, "epochs_run": 3, "imgsz": 640, "names": {"0": "rock"},
        "run": setup, "best_epoch": 2, "stopped_early": False, "epoch_seconds": 98.0,
        "final": {"loss": 4.2, "vfl": 0.3, "l1": 0.1, "giou": 0.2}}))
    module.db.update_job(job, status="done", run_dir=str(run), started=1000.0, finished=1300.0)
    for epoch, score in [(1, 0.5), (2, 0.762), (3, 0.75)]:
        module.db.add_epoch(
            job, {"epoch": epoch, "loss": 5.0 - epoch, "map50_95": score, "seconds": 98})
    return job, run


def test_a_trained_run_writes_its_own_model_card(studio):
    client, module = studio
    job, _ = _finished_run(client, module)

    card = client.get(f"/api/jobs/{job}/modelcard").json()
    assert card["folder"] == "models/rock-paper-scissors" and "/" in card["repo"]
    readme = card["readme"]
    assert readme.startswith("---\nlicense: apache-2.0")
    for fact in ("RT-DETR r34", "PResNet-34", "**rock**", "**0.762**", "best at epoch 2",
                 "COCO-pretrained", "8 training images (12 boxes), 2 validation images (3 boxes)",
                 "| 0 | rock | 12 | 3 |", "RTX 5090", "AdamW", "20.0M of 31.0M", "on (CUDA AMP)",
                 'allow_patterns="models/rock-paper-scissors/*"',
                 "models/rock-paper-scissors/best.xml"):
        assert fact in readme, fact
    assert "best.onnx" not in readme                      # the upload leaves it behind

    # the running job's panel reads the same record
    assert client.get(f"/api/jobs/{job}").json()["run"]["gpu"] == "NVIDIA GeForce RTX 5090"


def test_the_hugging_face_bundle_is_the_folder_to_upload(studio):
    client, module = studio
    job, _ = _finished_run(client, module)

    response = client.get(f"/api/jobs/{job}/download/huggingface",
                          params={"repo": "me/models", "folder": "models/rps", "pt": True})
    names = sorted(zipfile.ZipFile(io.BytesIO(response.content)).namelist())
    assert names == ["rps/README.md", "rps/best.bin", "rps/best.names.json", "rps/best.pt",
                     "rps/best.xml", "rps/labels.txt"]
    readme = zipfile.ZipFile(io.BytesIO(response.content)).read("rps/README.md").decode()
    assert 'snapshot_download("me/models", allow_patterns="models/rps/*")' in readme
    assert "`best.pt`" in readme and "model.train(" in readme

    # the plain IR download carries the card too
    ir = zipfile.ZipFile(io.BytesIO(client.get(f"/api/jobs/{job}/download/openvino").content))
    assert "README.md" in ir.namelist()
