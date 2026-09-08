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
    ):
        seed_everything(seed)
        self.net = net
        self.frozen = freeze_modules(net, freeze)
        self.on_epoch_end = on_epoch_end
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
        )
        net = self.net.to(self.device)
        since_improved = 0
        print(
            f"[rtdetr] training on {self.device}, {len(ds)} images, "
            f"{self.epochs} epochs -> {self.save_dir}"
        )

        logs = {"vfl": 0.0, "l1": 0.0, "giou": 0.0}
        for epoch in range(self.start_epoch, self.epochs):
            net.train()
            for module in self.frozen:  # frozen batch norms must not keep adapting
                module.eval()
            t0, running = time.time(), 0.0
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

            avg = running / max(len(dl), 1)
            msg = (
                f"epoch {epoch + 1}/{self.epochs}  loss {avg:.3f}  "
                f"vfl {logs['vfl']:.3f} l1 {logs['l1']:.3f} giou {logs['giou']:.3f}  "
                f"{time.time() - t0:.1f}s"
            )

            metric = None
            if val_fn is not None:
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
            if self.on_epoch_end is not None:
                self.on_epoch_end(dict(row, epochs=self.epochs, save_dir=str(self.save_dir)))

            improved = metric is None or metric > self.best
            if improved:
                self.best = metric if metric is not None else self.best
                since_improved = 0
            else:
                since_improved += 1
            self._save(net, epoch, "last.pt")
            if improved:
                self._save(net, epoch, "best.pt")
            if val_fn is not None and self.patience and since_improved >= self.patience:
                print(
                    f"[rtdetr] early stop: no mAP improvement for {self.patience} epochs "
                    f"(best {self.best:.4f})"
                )
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
