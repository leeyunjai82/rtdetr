# Pretrained weights and the mirror

Back to the [README](../README.md).

## Pretrained weights

`RTDETR("rtdetr-r18")` downloads the IR for that name on first use and caches it
in `~/.rtdetr/` (`$RTDETR_HOME` to move it, `$RTDETR_ASSETS_URL` to point at an
internal mirror — handy for air-gapped sites). Known names: `rtdetr-r18`,
`rtdetr-r34`, `rtdetr-r50`; each mirror entry is `<name>/<name>.xml`, `.bin`,
`.pt` and `labels.txt`.

The weights are the original RT-DETR COCO checkpoints, which their authors
release under Apache-2.0. This package's network matches that reference layout
module for module, so they load with `strict=True` — no remapping and no
re-training. Building the whole mirror takes a few CPU-minutes:

```bash
python tools/build_mirror.py --out mirror   # r18 + r34 + r50, .pt + IR + labels

pip install -U "huggingface_hub[cli]"       # ships the `hf` command
hf auth login                               # a token with write access
hf upload leeyunjai/rtdetr mirror . --repo-type=model
```

(`huggingface-cli` is the old name for `hf` and still works if you have it.
No CLI on PATH? `python -m huggingface_hub.cli.hf upload …` does the same,
and `HfApi().upload_folder(folder_path="mirror", repo_id=…)` does it from
Python.)

> Until that upload happens, `RTDETR("rtdetr-r18")` raises a `ModelNotFoundError`
> naming the ways forward — it never silently falls back to an untrained network.
> Point `$RTDETR_ASSETS_URL` at any host serving the same layout to use it now.

For a single checkpoint without the IR:

```bash
python tools/convert_official.py --variant r18 --out rtdetr-r18.pt
```

