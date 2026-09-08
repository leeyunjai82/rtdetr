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
    assert response.status_code == 200 and response.json()["needs_bin"] is False
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
