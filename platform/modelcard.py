# Apache-2.0
"""A README.md for a trained run, written to sit beside the model on Hugging Face.

Everything in it comes from the run itself — the classes, the scores, what it
started from, how long it took — so the card cannot drift from the weights.
"""

from __future__ import annotations

import re

#: What each size is made of, for people who have not read the paper.
VARIANTS = {
    "n": {"backbone": "HGNetv2-B0", "decoder": 3, "params": "4M", "coco": "42.8"},
    "s": {"backbone": "HGNetv2-B0", "decoder": 3, "params": "10M", "coco": "48.5"},
    "m": {"backbone": "HGNetv2-B2", "decoder": 4, "params": "19M", "coco": "52.3"},
    "l": {"backbone": "HGNetv2-B4", "decoder": 6, "params": "31M", "coco": "54.0"},
    "x": {"backbone": "HGNetv2-B5", "decoder": 6, "params": "62M", "coco": "55.8"},
}


def slug(name: str, fallback: str) -> str:
    """A folder name Hugging Face and every shell take without quoting."""
    text = re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")
    return text or fallback


def _duration(seconds: float | None) -> str:
    if not seconds:
        return "—"
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600} h {s % 3600 // 60} min"
    return f"{s // 60} min {s % 60} s" if s >= 60 else f"{s} s"


def _score(value) -> str:
    return "—" if value is None else f"{value:.3f}"


def _millions(n) -> str:
    return f"{n / 1e6:.1f}M"


def _details(facts: dict) -> list[tuple[str, str]]:
    """The settings behind the numbers — what someone reproducing it would ask."""
    run = facts.get("run") or {}
    if not run:
        return []
    rows = []
    opt = run.get("optimizer") or {}
    if opt:
        rows.append(("Optimizer", f"{opt.get('name', 'AdamW')}, lr {opt['lr']:g} "
                     f"(backbone {opt['lr_backbone']:g}), weight decay {opt['weight_decay']:g}"))
        rows.append(("Schedule", f"{opt.get('warmup_epochs', 1)} warmup epoch(s), "
                     "then cosine decay to 1% · gradient clip " + f"{opt.get('grad_clip', 0.1):g}"))
    rows.append(("Mixed precision", "on (CUDA AMP)" if run.get("amp") else "off"))
    if run.get("augment"):
        rows.append(("Augmentation", ", ".join(run["augment"])))
    if run.get("params"):
        frozen = f" — {run['freeze']} frozen" if run.get("freeze") else ""
        rows.append(("Trainable parameters", f"{_millions(run['trainable_params'])} of "
                     f"{_millions(run['params'])}{frozen}"))
    if run.get("patience"):
        stop = (f"stopped early at epoch {facts.get('epochs_run')}" if facts.get("stopped_early")
                else "ran the full schedule")
        rows.append(("Early stopping", f"patience {run['patience']} — {stop}"))
    rows.append(("Seed", str(run.get("seed", 0))))
    final = facts.get("final") or {}
    if final:
        rows.append(("Final training loss", " · ".join(f"{k} {v:.3f}" for k, v in final.items())))
    if facts.get("epoch_seconds"):
        rows.append(("Time per epoch", _duration(facts["epoch_seconds"])))
    hardware = run.get("gpu") or run.get("device")
    if hardware:
        rows.append(("Hardware", hardware + (f" · CPU {run['cpu']}" if run.get("cpu") else "")))
    versions = run.get("versions") or {}
    if versions:
        names = {"easydetect": "easydetect", "torch": "PyTorch", "cuda": "CUDA", "python": "Python"}
        rows.append(("Software", " · ".join(f"{names[k]} {v}" for k, v in versions.items() if v)))
    if run.get("resumed_at_epoch"):
        rows.append(("Resumed", f"continued from epoch {run['resumed_at_epoch']}"))
    return rows


def model_card(facts: dict) -> str:
    """Render the card. ``facts`` keys:

    title, repo, folder, variant, names (list), imgsz, epochs, epochs_run,
    best_epoch, batch, freeze, start ("coco" | "imagenet" | a path), device,
    train_images, dataset_images, map50_95, map50 (optional), per_class_ap
    (name -> AP, optional), boxes (name -> count), seconds, date, version,
    files (list of file names in the folder), curve (list of epoch rows).
    """
    names = list(facts.get("names") or [])
    variant = facts.get("variant") or "s"
    spec = VARIANTS.get(variant, {"backbone": variant, "decoder": "?", "params": "?", "coco": "—"})
    repo, folder = facts["repo"], facts["folder"].strip("/")
    files = facts.get("files") or []
    xml = next((f for f in files if f.endswith(".xml")), "best.xml")
    stem = xml[: -len(".xml")]

    start = facts.get("start")
    started_from = {
        "coco": "COCO-pretrained D-FINE weights (official release, Apache-2.0)",
        "imagenet": "ImageNet backbone only — the COCO weights were not reachable when it trained",
        "scratch": "random initialisation",
    }.get(start, f"an earlier run (`{start}`)" if start else "COCO-pretrained D-FINE weights")
    run = facts.get("run") or {}
    split = run.get("data") or {}

    freeze = "frozen" if facts.get("freeze") else "trained"
    epochs = facts.get("epochs_run") or facts.get("epochs")
    best = f" (best at epoch {facts['best_epoch']})" if facts.get("best_epoch") else ""
    rows = [
        ("Task", f"object detection — {len(names)} class{'es' if len(names) != 1 else ''}"),
        ("Architecture", f"D-FINE-{variant.upper()}: {spec['backbone']} backbone → hybrid "
                         f"encoder → {spec['decoder']}-layer decoder with fine-grained "
                         "distribution refinement, no NMS"),
        ("Parameters", _millions(run["params"]) if run.get("params") else spec["params"]),
        ("Started from", started_from),
        ("Input size", f"{facts.get('imgsz', 640)} × {facts.get('imgsz', 640)}"),
        ("Epochs", f"{epochs}{best}"),
        ("Batch / backbone", f"{facts.get('batch', '—')} / {freeze}"),
    ]
    if split.get("train"):
        val = split.get("val") or {}
        train = split["train"]
        rows.append(("Data", f"{train['images']} training images ({train['boxes']} boxes)"
                     + (f", {val['images']} validation images ({val['boxes']} boxes)"
                        if val else "")))
    elif facts.get("train_images"):
        total = facts.get("dataset_images")
        rows.append(("Training images", f"{facts['train_images']}"
                     + (f" of {total} (the rest held out for validation)" if total else "")))
    rows.append(("Validation mAP50-95", f"**{_score(facts.get('map50_95'))}**"))
    if facts.get("map50") is not None:
        rows.append(("Validation mAP50", _score(facts["map50"])))
    if facts.get("seconds"):
        rows.append(("Training time", _duration(facts["seconds"])
                     + (f" on {facts['device']}" if facts.get("device") else "")))
    trained = f"{facts.get('date', '')} with easydetect {facts.get('version', '')}"
    rows.append(("Trained", trained.strip()))

    out = [
        "---",
        "license: apache-2.0",
        "library_name: easydetect",
        "pipeline_tag: object-detection",
        "tags:",
        "  - object-detection",
        "  - d-fine",
        "  - openvino",
        "---",
        "",
        f"# {facts['title']} — D-FINE-{variant.upper()}",
        "",
        "Finds " + ", ".join(f"**{n}**" for n in names) + " in images and video." if names else "",
        "Trained with [easydetect](https://github.com/themakerrobot/easydetect); runs on OpenVINO "
        "(CPU, Intel GPU, NPU) with no PyTorch needed at inference.",
        "",
        "| | |",
        "| --- | --- |",
        *[f"| {k} | {v} |" for k, v in rows],
        "",
        "## Classes",
        "",
    ]
    per_ap = facts.get("per_class_ap") or {}
    boxes = facts.get("boxes") or {}
    ap_col = bool(per_ap)
    train_c = (split.get("train") or {}).get("per_class")
    val_c = (split.get("val") or {}).get("per_class")
    if train_c:                                       # what it actually learned from
        head = "| id | name | train boxes |" + (" val boxes |" if val_c else "")
    else:
        head = "| id | name | boxes in the dataset |"
    out.append(head + (" AP50-95 (val) |" if ap_col else ""))
    out.append("| --- " * (head.count("|") - 1 + ap_col) + "|")
    for i, name in enumerate(names):
        if train_c:
            line = f"| {i} | {name} | {train_c[i] if i < len(train_c) else '—'} |"
            if val_c:
                line += f" {val_c[i] if i < len(val_c) else '—'} |"
        else:
            line = f"| {i} | {name} | {boxes.get(name, '—')} |"
        if ap_col:
            line += f" {_score(per_ap.get(name))} |"
        out.append(line)
    background = [(k, v["background_images"]) for k, v in split.items()
                  if v.get("background_images")]
    if background:
        out += ["", "Background images (no objects, they teach it what to ignore): "
                + ", ".join(f"{n} {k}" for k, n in background) + "."]

    details = _details(facts)
    if details:
        out += ["", "## Training details", "", "| | |", "| --- | --- |",
                *[f"| {k} | {v} |" for k, v in details]]

    what = {
        f"{stem}.xml": "OpenVINO IR — the network",
        f"{stem}.bin": "OpenVINO IR — the weights",
        "labels.txt": "class names, one per line (line number = class id)",
        f"{stem}.names.json": "the same names, keyed by id",
        "best.pt": "PyTorch checkpoint — to train further or export again",
    }
    listed = [f for f in what if f in files]          # network, weights, names, checkpoint
    if listed:
        out += ["", "## Files", "", "| file | what it is |", "| --- | --- |",
                *[f"| `{f}` | {what[f]} |" for f in listed]]

    pattern = f"{folder}/*" if folder else "*"
    path = f"{folder}/{xml}" if folder else xml
    out += [
        "",
        "## Use it",
        "",
        "```bash",
        "pip install easydetect huggingface_hub",
        "```",
        "",
        "```python",
        "from huggingface_hub import snapshot_download",
        "from easydetect import Detector",
        "",
        f'root = snapshot_download("{repo}", allow_patterns="{pattern}")',
        f'model = Detector(f"{{root}}/{path}")      # device="CPU" / "GPU" / "NPU", default AUTO',
        "",
        'for r in model("photo.jpg", conf=0.25):',
        "    for box, score, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):",
        '        print(r.names[int(cls)], f"{score:.2f}", box.astype(int).tolist())',
        '    r.save("result.jpg")                  # the picture with boxes drawn',
        "",
        "# video, webcam (0) or an RTSP stream, one frame at a time",
        "for r in model.predict(0, stream=True, show=True):",
        "    pass",
        "```",
        "",
        "From the command line, once downloaded:",
        "",
        "```bash",
        f"easydetect predict model=<download folder>/{path} source=photo.jpg conf=0.25",
        "```",
    ]
    if "best.pt" in files:
        out += [
            "",
            "Train further on more data (needs `pip install \"easydetect[train]\"`):",
            "",
            "```python",
            f'model = Detector(f"{{root}}/{folder + "/" if folder else ""}best.pt")',
            'model.train(data="data.yaml", epochs=30)',
            "```",
        ]

    curve = facts.get("curve") or []
    if curve:
        step = max(1, len(curve) // 10)
        shown = curve[::step]
        if shown[-1] is not curve[-1]:
            shown.append(curve[-1])
        out += ["", "## Training curve", "", "| epoch | loss | mAP50-95 |", "| --- | --- | --- |",
                *[f"| {r['epoch']} | {r['loss']:.3f} | {_score(r.get('map50_95'))} |"
                  for r in shown]]

    out += [
        "",
        "## License",
        "",
        "The model code and the COCO starting weights are Apache-2.0, and so are these weights.",
        "The training images keep their own license — say here where they came from "
        "(a Roboflow Universe dataset, for instance, is often CC BY 4.0, which asks for credit).",
        "",
    ]
    return "\n".join(line for line in out if line is not None)
