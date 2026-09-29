# Apache-2.0
"""Training loop (PyTorch). Kept deliberately small and readable.

AdamW with a lower LR on the backbone, linear warmup into a cosine decay, AMP
on CUDA, ``last.pt``/``best.pt`` after every epoch, early stop on ``patience``,
and ``resume=True`` to pick a killed run back up where it stopped.
"""

from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.dataset import DetDataset, load_data_yaml
from .utils.loss import SetCriterion


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


#: Shorthands accepted by ``freeze``.
FREEZE_GROUPS = {
    "backbone": ("backbone",),
    "encoder": ("encoder",),
    "backbone+encoder": ("backbone", "encoder"),
}


def freeze_modules(net, freeze):
    """Stop training the named parts. Returns the modules to keep in eval mode.

    ``freeze="backbone"`` is the useful one: on a small dataset it trains
    roughly twice as fast and overfits less, because the features it starts
    with are already good.
    """
    if not freeze:
        return []
    if freeze is True:
        freeze = "backbone"
    prefixes = FREEZE_GROUPS.get(freeze, (freeze,) if isinstance(freeze, str) else tuple(freeze))
    unknown = [p for p in prefixes if not hasattr(net, p.split(".")[0])]
    if unknown:
        raise ValueError(f"nothing called {unknown} to freeze; try {sorted(FREEZE_GROUPS)}")

    frozen_params = 0
    for name, param in net.named_parameters():
        if name.startswith(prefixes):
            param.requires_grad = False
            frozen_params += param.numel()
    modules = [getattr(net, p.split(".")[0]) for p in prefixes]
    print(f"[rtdetr] frozen: {', '.join(prefixes)} ({frozen_params / 1e6:.1f}M parameters)")
    return modules


def _split_counts(ds, num_classes: int) -> dict:
    """Images, boxes and boxes per class in one split — label files only, no pixels."""
    per_class = [0] * num_classes
    boxes = empty = 0
    for f in ds.files:
        labels = ds._load_labels(f)
        boxes += len(labels)
        empty += len(labels) == 0
        for c in labels[:, 0].astype(int) if len(labels) else ():
            if 0 <= c < num_classes:
                per_class[c] += 1
    return {"images": len(ds.files), "boxes": boxes, "background_images": empty,
            "per_class": per_class}


def _one_thread_per_worker(_worker_id: int) -> None:
    import cv2

    cv2.setNumThreads(0)
    torch.set_num_threads(1)


class Trainer:
    def __init__(
        self,
        net,
        data,
        epochs=100,
        imgsz=640,
        batch=8,
        lr=1e-4,
        lr_backbone_mult=0.1,
        weight_decay=1e-4,
        warmup_epochs=1,
        device=None,
        workers=4,
        project="runs",
        name="train",
        amp=True,
        resume=False,
        patience=50,
        seed=0,
        freeze=None,
        on_epoch_end=None,
        on_progress=None,
        origin=None,
    ):
        seed_everything(seed)
        self.seed, self.freeze, self.origin = seed, freeze, origin
        self.lr, self.lr_backbone_mult, self.weight_decay = lr, lr_backbone_mult, weight_decay
        self.net = net
        self.frozen = freeze_modules(net, freeze)
        self.on_epoch_end = on_epoch_end
        self.on_progress = on_progress
        self.data_yaml = data
        self.cfg = load_data_yaml(data)
        self.epochs, self.imgsz, self.batch = epochs, imgsz, batch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.workers = workers
        self.amp = amp and self.device.type == "cuda"
        self.patience = patience
        self.start_epoch = 0
        self.best = -1.0

        run_root = Path(project) / name
        self.save_dir = self._resume_dir(run_root) if resume else self._unique_dir(run_root)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "weights").mkdir(exist_ok=True)

        # on the device before the optimizer exists: loading a resumed optimizer
        # state puts each tensor beside its parameter, and a parameter still on
        # the CPU left Adam's moments there — "cuda:0 and cpu" on the first step
        self.net = net = net.to(self.device)

        backbone_params, other_params = [], []
        for n, p in net.named_parameters():
            if not p.requires_grad:
                continue
            (backbone_params if n.startswith("backbone") else other_params).append(p)
        groups = [{"params": other_params, "lr": lr}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr * lr_backbone_mult})
        self.opt = torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)
        self.criterion = SetCriterion(net.num_classes)
        self.warmup_epochs = warmup_epochs
        self.base_lrs = [g["lr"] for g in self.opt.param_groups]
        self.scaler = torch.amp.GradScaler(enabled=self.amp)
        if resume:
            self._load_resume_state()

    @staticmethod
    def _unique_dir(p: Path) -> Path:
        if not p.exists():
            return p
        i = 2
        while (q := p.with_name(f"{p.name}{i}")).exists():
            i += 1
        return q

    @staticmethod
    def _resume_dir(p: Path) -> Path:
        """The most recent ``name``/``name2``/... directory that has a last.pt."""
        candidates = [
            d
            for d in sorted(p.parent.glob(f"{p.name}*"))
            if (d / "weights" / "last.pt").exists()
        ]
        return candidates[-1] if candidates else p

    def _load_resume_state(self) -> None:
        last = self.save_dir / "weights" / "last.pt"
        if not last.exists():
            print(f"[rtdetr] resume: nothing to resume from in {self.save_dir}, starting fresh")
            return
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        self.net.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer"):
            self.opt.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") and self.amp:
            self.scaler.load_state_dict(ckpt["scaler"])
        self.start_epoch = int(ckpt.get("epoch", -1)) + 1
        self.best = float(ckpt.get("best", -1.0))
        print(f"[rtdetr] resuming {self.save_dir} at epoch {self.start_epoch + 1}/{self.epochs}")

    # ------------------------------------------------------------- the record

    def _write_setup(self, ds) -> dict:
        """``run.json``: what this run is, written before the first batch.

        Everything someone reading the result later would ask — where it
        started, on what data, with which settings, on what machine — so a
        model card or a report never has to guess.
        """
        import platform
        import sys

        from . import __version__

        previous = self.save_dir / "run.json"
        old = json.loads(previous.read_text(encoding="utf-8")) if previous.exists() else {}
        params = sum(p.numel() for p in self.net.parameters())
        trainable = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        freeze = self.freeze if not isinstance(self.freeze, (list, tuple)) else list(self.freeze)
        setup = {
            "variant": self.net.variant,
            "num_classes": self.net.num_classes,
            "names": self.cfg["names"],
            # a resumed run started where its first attempt did
            "start": old.get("start") or self.origin,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "epochs": self.epochs,
            "patience": self.patience,
            "seed": self.seed,
            "workers": self.workers,
            "freeze": "backbone" if freeze is True else freeze or None,
            "params": params,
            "trainable_params": trainable,
            "optimizer": {
                "name": "AdamW",
                "lr": self.lr,
                "lr_backbone": self.lr * self.lr_backbone_mult,
                "weight_decay": self.weight_decay,
                "schedule": "linear warmup, then cosine decay to 1%",
                "warmup_epochs": self.warmup_epochs,
                "grad_clip": 0.1,
            },
            "amp": self.amp,
            "augment": ["horizontal flip (p=0.5)", "HSV jitter (hue ±8, sat ±30, value ±30)",
                        f"resize to {self.imgsz}×{self.imgsz}"],
            "data": {"train": _split_counts(ds, self.net.num_classes)},
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "cpu": platform.processor() or platform.machine(),
            "versions": {
                "rtdetr": __version__,
                "torch": torch.__version__,
                "cuda": torch.version.cuda if self.device.type == "cuda" else None,
                "python": sys.version.split()[0],
            },
            "started": old.get("started") or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "resumed_at_epoch": self.start_epoch + 1 if self.start_epoch else None,
        }
        try:
            val = DetDataset(self.data_yaml, "val", self.imgsz, augment=False)
            setup["data"]["val"] = _split_counts(val, self.net.num_classes)
        except (ValueError, FileNotFoundError, KeyError):
            pass                                     # no val split: nothing to count
        previous.write_text(json.dumps(setup, ensure_ascii=False, indent=2), encoding="utf-8")
        return setup

    def _outcome(self, setup: dict, started: float, stopped_early: bool) -> dict:
        """How it went, read back from results.csv so a resumed run counts every epoch."""
        rows = []
        path = self.save_dir / "results.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        scored = [r for r in rows if r.get("map50_95")]
        best = max(scored, key=lambda r: float(r["map50_95"])) if scored else None
        last = rows[-1] if rows else {}
        seconds = sum(float(r.get("seconds") or 0) for r in rows)
        return {
            "run": setup,
            "best_epoch": int(best["epoch"]) if best else None,
            "stopped_early": stopped_early,
            "train_seconds": round(seconds, 1),
            "epoch_seconds": round(seconds / len(rows), 1) if rows else None,
            "final": {k: float(last[k]) for k in ("loss", "vfl", "l1", "giou") if last.get(k)},
            "wall_seconds": round(time.time() - started, 1),
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def _set_lr(self, epoch, step, steps_per_epoch):
        t = epoch + step / max(steps_per_epoch, 1)
        if t < self.warmup_epochs:
            scale = t / max(self.warmup_epochs, 1e-8)
        else:
            prog = (t - self.warmup_epochs) / max(self.epochs - self.warmup_epochs, 1e-8)
            scale = 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
            scale = max(scale, 0.01)
        for g, base in zip(self.opt.param_groups, self.base_lrs, strict=False):
            g["lr"] = base * scale

    def train(self, val_fn=None):
        ds = DetDataset(self.data_yaml, "train", self.imgsz, augment=True)
        dl = DataLoader(
            ds,
            batch_size=self.batch,
            shuffle=True,
            drop_last=len(ds) > self.batch,
            num_workers=self.workers,
            collate_fn=DetDataset.collate,
            pin_memory=self.device.type == "cuda",
            # workers live for the whole run instead of being forked again for
            # every epoch, and each decodes on one thread: N workers each using
            # every core would fight over the CPU that feeds the GPU
            persistent_workers=self.workers > 0,
            worker_init_fn=_one_thread_per_worker if self.workers > 0 else None,
        )
        net = self.net.to(self.device)
        since_improved = 0
        if self.start_epoch >= self.epochs:
            # resuming a run that already finished its schedule: there is nothing
            # to do until someone asks for more epochs
            print(
                f"[rtdetr] already trained {self.start_epoch} epochs; "
                f"raise epochs above {self.epochs} to continue"
            )
            return self.save_dir / "weights" / "best.pt"
        print(
            f"[rtdetr] training on {self.device}, {len(ds)} images, "
            f"{self.epochs} epochs -> {self.save_dir}"
        )
        setup = self._write_setup(ds)
        started, stopped_early = time.time(), False

        logs = {"vfl": 0.0, "l1": 0.0, "giou": 0.0}
        for epoch in range(self.start_epoch, self.epochs):
            net.train()
            for module in self.frozen:  # frozen batch norms must not keep adapting
                module.eval()
            t0, running, reported = time.time(), 0.0, 0.0
            for step, (imgs, targets) in enumerate(dl):
                self._set_lr(epoch, step, len(dl))
                imgs = imgs.to(self.device, non_blocking=True)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                with torch.amp.autocast(self.device.type, enabled=self.amp):
                    out = net(imgs)
                    loss, logs = self.criterion(out, targets)

                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
                self.scaler.step(self.opt)
                self.scaler.update()
                running += float(loss.detach())
                if self.on_progress is not None and (
                    time.time() - reported >= 1.0 or step + 1 == len(dl)
                ):
                    reported = time.time()  # about once a second, not every batch
                    self.on_progress({
                        "phase": "train", "epoch": epoch + 1, "epochs": self.epochs,
                        "step": step + 1, "steps": len(dl), "seconds": reported - t0,
                    })

            avg = running / max(len(dl), 1)
            msg = (
                f"epoch {epoch + 1}/{self.epochs}  loss {avg:.3f}  "
                f"vfl {logs['vfl']:.3f} l1 {logs['l1']:.3f} giou {logs['giou']:.3f}  "
                f"{time.time() - t0:.1f}s"
            )

            metric = None
            if val_fn is not None:
                if self.on_progress is not None:
                    self.on_progress({
                        "phase": "val", "epoch": epoch + 1, "epochs": self.epochs,
                        "seconds": time.time() - t0,
                    })
                metric = val_fn(net)
                msg += f"  mAP50-95 {metric:.4f}"
            print("[rtdetr] " + msg)

            row = {
                "epoch": epoch + 1,
                "loss": round(avg, 5),
                "vfl": round(float(logs["vfl"]), 5),
                "l1": round(float(logs["l1"]), 5),
                "giou": round(float(logs["giou"]), 5),
                "map50_95": round(metric, 5) if metric is not None else "",
                "lr": self.opt.param_groups[0]["lr"],
                "seconds": round(time.time() - t0, 2),
            }
            self._append_results(row)

            improved = metric is None or metric > self.best
            if improved:
                self.best = metric if metric is not None else self.best
                since_improved = 0
            else:
                since_improved += 1
            self._save(net, epoch, "last.pt")
            if improved:
                self._save(net, epoch, "best.pt")
            # after the checkpoints: a callback that stops the run (a cancel
            # button) must not throw away the epoch that just finished
            if self.on_epoch_end is not None:
                self.on_epoch_end(dict(row, epochs=self.epochs, save_dir=str(self.save_dir)))
            if val_fn is not None and self.patience and since_improved >= self.patience:
                print(
                    f"[rtdetr] early stop: no mAP improvement for {self.patience} epochs "
                    f"(best {self.best:.4f})"
                )
                stopped_early = True
                break

        best_path = self.save_dir / "weights" / "best.pt"
        (self.save_dir / "summary.json").write_text(
            json.dumps(
                {
                    "best_map50_95": self.best if self.best >= 0 else None,
                    "epochs_run": epoch + 1,
                    "weights": str(best_path),
                    "imgsz": self.imgsz,
                    "names": self.cfg["names"],
                    **self._outcome(setup, started, stopped_early),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[rtdetr] done. weights: {best_path}")
        return best_path

    def _append_results(self, row):
        """One CSV line per epoch, so a dashboard can just tail the file."""
        path = self.save_dir / "results.csv"
        write_header = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _save(self, net, epoch, fname):
        torch.save(
            {
                "model": net.state_dict(),
                "optimizer": self.opt.state_dict(),
                "scaler": self.scaler.state_dict() if self.amp else None,
                "variant": net.variant,
                "num_classes": net.num_classes,
                "names": self.cfg["names"],
                "epoch": epoch,
                "best": self.best,
                "imgsz": self.imgsz,
            },
            self.save_dir / "weights" / fname,
        )
