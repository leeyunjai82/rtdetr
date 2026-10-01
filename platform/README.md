# easydetect platform

Label images, train on them, watch the run, take the model away — in a browser,
on one machine. It is a separate application that uses the `easydetect` package; the
package itself stays a library with a CLI and no web dependencies.

```bash
pip install -r platform/requirements.txt   # in a clone, also: pip install -e ".[train]"
python platform/run.py                 # http://<this machine>:8080, from any machine
```

![the platform](../docs/assets/platform.jpg)

[PLAN.md](PLAN.md) is what this is meant to become, and what is missing today.

The screens follow [themaker-ui](https://github.com/themakerrobot/themaker-ui),
the design kit shared by themakerrobot's education tools: its stylesheet and the
Pretendard font are copied unchanged into `static/css/` and `static/assets/`
(CI checks the copy against the kit's v1.0.0 tag), and what is particular to
this app sits in `static/css/app.css`. The **EN / 한** button at the top right
switches between Korean and English; the choice is remembered in the browser,
and a first visit follows the browser's language. Strings live in
`static/js/i18n.js` — Korean is the source, with an English entry for each.
Button names below are the English ones.

## The loop

1. **Collect** — four ways into a dataset, and all of them can add to one that
   already exists: pick image files, upload a video and sample frames from it
   (interval and cap are yours), upload a zip, or register a folder already on
   this machine (`python platform/run.py label --source photos/ --names can,bottle`).
   A YOLO-format export from a labelling service goes in as the zip it
   downloaded as — Roboflow's *YOLOv8* / *YOLOv11* TXT, say — with its classes,
   labels and train/valid split kept; *Statistics* and *Draw boxes* show it straight away.
   Labels are stored where training reads them: in `labels/` beside an
   `images/` folder, otherwise next to each picture, inside the folder.
2. **Label** — *Draw boxes* on the dataset card. Drag to draw, drag inside a box to
   move it, drag a corner to resize, <kbd>1</kbd>–<kbd>9</kbd> to set the class,
   <kbd>Del</kbd> to remove, <kbd>Ctrl</kbd>+<kbd>Z</kbd> to undo. Saving is
   automatic. The wheel zooms and the middle button (or <kbd>Space</kbd>) pans,
   for boxes too small to place at fit-to-window. <kbd>C</kbd> copies the
   previous image's boxes — a fixed camera repeats itself. Classes can be added
   from the sidebar and show how many boxes each has, and the file list can hide
   what is already labelled. A dataset spread over folders (`train/images`,
   `valid/images`, …) can be browsed one folder at a time, and ←/→ stay in it;
   background frames — a label with no boxes — are marked apart from pictures
   that have boxes.
3. **Keep the set honest** — *Statistics* counts boxes per class and flags what usually
   bites: images with no label, labels with no boxes, specks under 0.1% of the
   frame. *Duplicate* takes a snapshot before a risky relabel, *Merge selected* combines sets and
   remaps class indices by name (index 0 rarely means the same thing in two
   datasets), and an image can be dropped with its label from the labelling page.
4. **Let the model do the first pass** — *Auto-label* fills the current image;
   *Auto-label all* queues a job over every unlabelled image in the dataset.
   Correcting boxes is far quicker than drawing them, and once you have a
   `best.pt` you can point the auto-labeller at it.
5. **Train** — a goal (빠르게 / 균형 / 정확하게) or your own model, epochs,
   batch, backbone freezing, "stop when it stops improving" (patience) and
   zoom/crop augmentation; image size and device under 고급 설정. A goal fills
   only the fields you have not changed yourself. The header shows each NVIDIA GPU live —
   utilisation, memory, temperature, power, read from `nvidia-smi` every two
   seconds — and warns when the driver sees a GPU that PyTorch cannot use (a CPU
   build of torch), since training would then run on the CPU. On a GPU the
   presets batch four times larger. Jobs queue and run one
   at a time; inside an epoch the job shows its batch, what is left and when it
   is validating, and the loss and mAP curve updates per epoch; images are
   decoded by loader processes (half the cores, at most eight —
   `EASYDETECT_WORKERS` overrides); *Stop* stops within a batch;
   *Train more* continues a finished run with more epochs in the same directory;
   every run keeps a `train.log`, and a failure carries the end of it.
   *Delete* removes a finished job with its weights, IR and log, and the runs
   continued from it and the evaluations of it; it is refused while any of them
   runs, or while a registered model uses those files. A run cut short — stopped,
   crashed or interrupted by a restart — keeps its best epoch for evaluating,
   downloading and registering.
   *Run details* on the job shows what the run was set up with from its first
   batch — starting weights, train/validation images and boxes per class,
   trainable parameters, optimizer and schedule, augmentation, device, versions
   — and, once it ends, the best epoch, time per epoch and final losses. The
   trainer writes this to `run.json` in the run folder (`summary.json` repeats
   it with the outcome), so it is there for scripts too.
6. **See what it learned** — *Evaluate* scores the run on its validation split and
   draws every one of those frames with the truth in green and the prediction in
   its class colour, next to per-class AP. A single mAP says whether to keep
   going; this says what to fix.
7. **Run it on everything else** — *Predict* takes a dataset, a folder path on this
   machine, or an uploaded video, and queues it like a training job. While it
   works the job says how far it is ("이미지 54/300 · 상자 5개 · 남은 시간 약
   43초") and shows frames as they are drawn; when done, the boxes per class and
   the first 60 drawn frames are right there, with a zip of all of them (and
   `results.json`) and, for a video, `annotated.mp4` to download. The model
   picker lists your finished runs first — no need to register one to use it —
   and auto-labelling drafts with the dataset's own latest run when it has one. *Webcam* opens the browser's camera and
   posts a frame every 400 ms, so you see the model on live video even when the
   server is somewhere else.
8. **Take it away** — `best.pt`, the OpenVINO IR as a zip (exported when a run
   finishes), `results.csv`. *Register model* gives the run a name (the IR if it has
   one, otherwise the weights) and it then appears wherever a model is chosen —
   auto-labelling, batch inference, the webcam. A `.pt`, `.onnx` or IR (its
   `.xml` and `.bin` picked together) from elsewhere goes into the same list.
   *Use it in code* on a finished run gives Python, command-line and "another
   PC" snippets with this run's paths filled in, ready to copy.
9. **Share it** — the *Hugging Face* tab builds the folder to upload: the IR,
   `labels.txt`, optionally `best.pt`, and a `README.md` model card written
   from the run itself — classes with their train/val box counts (and AP if
   you evaluated), mAP, backbone, starting weights, the training settings and
   hardware, and download-and-run code pointing at the repo and folder you
   chose (by default the repo your weights come from, `models/<dataset>`).
   It shows the `hf upload` command to run next. The plain IR zip carries the
   same card.

![labelling](../docs/assets/labeling.jpg)

## Where things live

Everything sits under `easydetect-platform/` in the directory you start from —
`--data /some/path` or `$EASYDETECT_PLATFORM_HOME` moves it:

```
easydetect-platform/
  platform.db          datasets, jobs, per-epoch numbers
  datasets/<name>/     uploaded datasets (registered folders stay where they are)
  runs/job<id>/        weights/, openvino/, results.csv, summary.json
                       (inference jobs: images/, results.json, annotated.mp4)
  models/<name>/       models uploaded from outside
```

## API

The pages are only clients of these, so a script can do anything the UI does:

| endpoint | |
| --- | --- |
| `POST /api/datasets` | multipart: `name`, `archive` (zip) |
| `POST /api/datasets/images` | multipart: `name`, `files` (images), optional `dataset_id` |
| `POST /api/datasets/video` | multipart: `name`, `video`, `every`, `max_frames` |
| `GET /api/datasets/{id}/export` | the dataset as a zip |
| `GET /api/datasets/{id}/stats` | boxes per class, and what looks wrong |
| `DELETE /api/datasets/{id}/images/{index}` | drop one image and its label |
| `POST /api/datasets/{id}/duplicate` | a copy to experiment on |
| `POST /api/datasets/merge` | `{ids, name}` — one set, class indices remapped |
| `DELETE /api/datasets/{id}` | |
| `POST /api/datasets/local` | `{path, name, names}` — register a folder in place |
| `GET /api/datasets` · `GET /api/datasets/{id}` | the second lists the images |
| `GET`/`POST /api/datasets/{id}/labels/{index}` | read and write one image's boxes |
| `POST /api/datasets/{id}/autolabel/{index}` | boxes for one image, from a model |
| `POST /api/datasets/{id}/autolabel` | queue a pass over the whole dataset |
| `POST /api/datasets/{id}/classes` | rename or add classes; data.yaml follows |
| `GET /api/models` | the registry, plus the names that download themselves |
| `POST /api/models` | `{job_id, name, note}` — register a finished run |
| `POST /api/models/upload` | multipart: `name`, `files` (a .pt or .onnx, or an IR's .xml + .bin together), `classes` |
| `DELETE /api/models/{id}` | forget it (files stay) |
| `POST /api/predict` | multipart: `model`, `conf`, and one of `dataset_id`, `path`, `video` |
| `POST /api/preview` | multipart: `model`, `conf`, `image` → annotated JPEG, one frame |
| `POST /api/jobs` | `{dataset_id, model, epochs, imgsz, batch, freeze, device}` |
| `GET /api/jobs` · `GET /api/jobs/{id}` | the second includes per-epoch rows |
| `GET /api/gpu` | utilisation, memory, temperature and power per GPU, from nvidia-smi |
| `GET /api/jobs/{id}/stream` | server-sent events: progress, then a status |
| `POST /api/jobs/{id}/cancel` | |
| `DELETE /api/jobs/{id}` | the job and its files, unless something still uses them |
| `POST /api/jobs/{id}/resume` | `{add_epochs}` — continue a finished run |
| `POST /api/jobs/{id}/evaluate` | `{conf}` — queue a scoring pass with pictures |
| `GET /api/jobs/{id}/report` · `/eval/{name}` | the numbers, and the pictures |
| `GET /api/jobs/{id}/download/{weights,openvino,results,log,predictions,video}` | |
| `GET /api/jobs/{id}/download/huggingface` | `?repo=&folder=&pt=` — the upload folder, zipped, with its README |
| `GET /api/jobs/{id}/modelcard` | the README.md that bundle would carry, and where it would go |
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
* It listens on every address (`0.0.0.0`) by default, so other machines can
  open it — and it has no login: anyone who can reach the port can use it.
  `--host 127.0.0.1` keeps it to this machine. Browsers only allow the webcam
  on `https://` or `localhost`, so the same app is also served over HTTPS on
  8443 (`--https-port`, 0 turns it off) with a certificate made on first start
  and kept in `easydetect-platform/tls/`. The browser warns about it once; the
  webcam tab opened over `http://` links to the HTTPS address.

## Tests

```bash
pytest platform/tests
```
