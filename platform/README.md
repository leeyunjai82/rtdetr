# rtdetr platform

Label images, train on them, watch the run, take the model away — in a browser,
on one machine. It is a separate application that uses the `rtdetr` package; the
package itself stays a library with a CLI and no web dependencies.

```bash
pip install -r platform/requirements.txt
python platform/run.py                 # http://127.0.0.1:8080
```

![the platform](../docs/assets/platform.jpg)

[PLAN.md](PLAN.md) is what this is meant to become, and what is missing today.

## The loop

1. **Collect** — four ways into a dataset, and all of them can add to one that
   already exists: pick image files, upload a video and sample frames from it
   (interval and cap are yours), upload a zip (images, plus labels and a
   data.yaml if you have them), or register a folder already on this machine
   (`python platform/run.py label --source images/ --names can,bottle`).
2. **Label** — *라벨링* on the dataset card. Drag to draw, drag inside a box to
   move it, drag a corner to resize, <kbd>1</kbd>–<kbd>9</kbd> to set the class,
   <kbd>Del</kbd> to remove, <kbd>Ctrl</kbd>+<kbd>Z</kbd> to undo. Saving is
   automatic. The wheel zooms and the middle button (or <kbd>Space</kbd>) pans,
   for boxes too small to place at fit-to-window. <kbd>C</kbd> copies the
   previous image's boxes — a fixed camera repeats itself. Classes can be added
   from the sidebar and show how many boxes each has, and the file list can hide
   what is already labelled.
3. **Let the model do the first pass** — *자동 라벨* fills the current image;
   *전체 자동 라벨* queues a job over every unlabelled image in the dataset.
   Correcting boxes is far quicker than drawing them, and once you have a
   `best.pt` you can point the auto-labeller at it.
4. **Train** — model, epochs, image size, batch, whether to freeze the backbone,
   which device. Jobs queue and run one at a time; the loss and mAP curve updates
   per epoch; *중지* stops at the next epoch boundary.
5. **Take it away** — `best.pt`, the OpenVINO IR as a zip (exported when a run
   finishes), `results.csv`. *이 모델로 추론* tries the trained model on an image.

![labelling](../docs/assets/labeling.jpg)

## Where things live

Everything sits under `rtdetr-platform/` in the directory you start from —
`--data /some/path` or `$RTDETR_PLATFORM_HOME` moves it:

```
rtdetr-platform/
  platform.db          datasets, jobs, per-epoch numbers
  datasets/<name>/     uploaded datasets (registered folders stay where they are)
  runs/job<id>/        weights/, openvino/, results.csv, summary.json
```

## API

The pages are only clients of these, so a script can do anything the UI does:

| endpoint | |
| --- | --- |
| `POST /api/datasets` | multipart: `name`, `archive` (zip) |
| `POST /api/datasets/images` | multipart: `name`, `files` (images), optional `dataset_id` |
| `POST /api/datasets/video` | multipart: `name`, `video`, `every`, `max_frames` |
| `GET /api/datasets/{id}/export` | the dataset as a zip |
| `DELETE /api/datasets/{id}` | |
| `POST /api/datasets/local` | `{path, name, names}` — register a folder in place |
| `GET /api/datasets` · `GET /api/datasets/{id}` | the second lists the images |
| `GET`/`POST /api/datasets/{id}/labels/{index}` | read and write one image's boxes |
| `POST /api/datasets/{id}/autolabel/{index}` | boxes for one image, from a model |
| `POST /api/datasets/{id}/autolabel` | queue a pass over the whole dataset |
| `POST /api/datasets/{id}/classes` | rename or add classes; data.yaml follows |
| `POST /api/jobs` | `{dataset_id, model, epochs, imgsz, batch, freeze, device}` |
| `GET /api/jobs` · `GET /api/jobs/{id}` | the second includes per-epoch rows |
| `GET /api/jobs/{id}/stream` | server-sent events: progress, then a status |
| `POST /api/jobs/{id}/cancel` | |
| `GET /api/jobs/{id}/download/{weights,openvino,results}` | |
| `POST /api/jobs/{id}/predict` | multipart: `image`, `conf` → annotated JPEG |

## What it is not

One process, one SQLite file, one folder. No accounts, no permissions, no
quotas, no scheduler, and one job at a time. That is the point: a workstation or
a mini PC beside a line needs no infrastructure, and the data never leaves it.

When that stops being true — several people, several GPUs, work that must
survive a restart — the things to change are the worker (onto a real queue) and
the storage (onto a service), not the package underneath.

Other limits worth knowing:

* Boxes only. Polygons, keypoints, review workflows and team assignment are out
  of scope; [CVAT](https://github.com/cvat-ai/cvat) (MIT) and
  [Label Studio](https://github.com/HumanSignal/label-studio) (Apache-2.0) export
  the same layout, so you can label there and train here.
* Training runs in this process, so restarting the server ends a run. Jobs left
  behind are marked failed on the next start rather than spinning forever.
* It binds `127.0.0.1` by default and has no authentication. Behind `--host
  0.0.0.0` it is open to whoever can reach the port.

## Tests

```bash
pytest platform/tests
```
