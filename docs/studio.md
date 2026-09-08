# Studio — the web app

Back to the [README](../README.md).

```bash
pip install "rtdetr[studio]"
rtdetr studio                  # http://127.0.0.1:8080
```

Upload a dataset, start a training run, watch the curve, download the weights and
the OpenVINO IR, and try the result on an image — in a browser, on one machine.

![the studio](assets/studio.jpg)

## What it is, and is not

One process, one SQLite file, one folder (`rtdetr-studio/` where you started it;
`RTDETR_STUDIO_HOME` or `rtdetr studio data=/path` moves it). Jobs run one at a
time in a background thread inside the same process.

That is the whole design, and it is deliberate: a workstation or a mini PC on a
factory floor needs no queue broker, no object store and no database server to
be useful, and the data never leaves the machine. What it does **not** have is
users, permissions, quotas or a scheduler — if several people need to share it
over a network, put it behind a reverse proxy with auth, or move the worker onto
a real queue.

## The flow

1. **Label** — `rtdetr label source=images/ names=can,bottle` ([labelling](labeling.md)),
   then zip the `images/` and `labels/` folders.
2. **Upload** — name it, drop the zip. Images-only archives work too; the studio
   writes a `data.yaml` from what it finds, and a dataset with no labels cannot
   start a job.
3. **Train** — pick model, epochs, size, batch, whether to freeze the backbone,
   and the device (blank picks a GPU when there is one). Queued jobs run in order.
4. **Watch** — the loss and mAP curve updates per epoch over server-sent events;
   *중지* stops a run at the next epoch boundary.
5. **Take it away** — `best.pt`, the OpenVINO IR as a zip (exported automatically
   when a run finishes), and `results.csv`. *이 모델로 추론* runs the trained model
   on an image you pick, at a confidence you choose.

## API

The UI is only a client of these, so anything it does, a script can do:

| endpoint | |
| --- | --- |
| `POST /api/datasets` | multipart: `name`, `archive` (zip) |
| `GET /api/datasets` | |
| `POST /api/jobs` | `{dataset_id, model, epochs, imgsz, batch, freeze, device}` |
| `GET /api/jobs` · `GET /api/jobs/{id}` | the second includes per-epoch rows |
| `GET /api/jobs/{id}/stream` | server-sent events, one per epoch |
| `POST /api/jobs/{id}/cancel` | |
| `GET /api/jobs/{id}/download/{weights,openvino,results}` | |
| `POST /api/jobs/{id}/predict` | multipart: `image`, `conf` → annotated JPEG |
| `GET /api/status` | queue depth, running job |

## Notes

* Uploaded archives are checked for entries that would write outside the dataset
  folder; anything else is refused.
* Training in-process means restarting the server stops a run. Jobs left as
  `running` after a restart are stale — start them again.
* The studio is bound to `127.0.0.1` unless you pass `host=0.0.0.0`. There is no
  authentication, so do not expose it to a network you do not trust.
