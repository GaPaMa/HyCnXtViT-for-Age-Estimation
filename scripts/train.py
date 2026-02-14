#!/usr/bin/env python3
"""
Unified training script for ConvNeXt, ViT and hybrid age‑estimation models.

This script serves as a thin wrapper around the core library contained in
``cnxtvit``.  It makes no assumptions about your filesystem other than
requiring a dataset root and a YAML configuration.  Hyper‑parameters
defined in the YAML can be overridden on the command line using
``KEY VALUE`` pairs in YACS notation.

Example usage (hybrid model on MORPH2):

```
python scripts/train.py \
    --config configs/morph2_hybrid.yaml \
    --data-root /path/to/DATA_ROOT \
    --dataset MORPH2 \
    --outdir runs/morph2_hybrid
```

The script automatically writes the best checkpoint under
``<outdir>/checkpoints`` and evaluates it on the test set once
training finishes.
"""

import argparse
import os
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader

from cnxtvit import MODEL
from cnxtvit.data import DATASET, load_transforms
from cnxtvit.utils_cfg import load_cfg, override_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ConvNeXt/VIT/Hybrid models for age estimation")
    parser.add_argument(
        "--config",
        required=True,
        help="YAML configuration file defining the model and training hyper‑parameters",
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Root directory containing dataset folders and their .pkl index files",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (e.g. MORPH2, CACD, AFAD). Must match the .pkl filename",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory where logs, checkpoints and results will be saved",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "opts",
        nargs=argparse.REMAINDER,
        default=None,
        help="Optional KEY VALUE pairs to override config values (YACS syntax)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    cfg = load_cfg(args.config)

    # Apply key/value overrides
    if args.opts:
        if len(args.opts) % 2 != 0:
            raise SystemExit("Override options must be supplied as KEY VALUE pairs")
        overrides = {args.opts[i]: args.opts[i + 1] for i in range(0, len(args.opts), 2)}
        # Cast numeric values to int or float
        for k, v in list(overrides.items()):
            try:
                overrides[k] = int(v)
            except ValueError:
                try:
                    overrides[k] = float(v)
                except ValueError:
                    overrides[k] = v
        cfg = override_cfg(cfg, overrides)

    outdir = Path(args.outdir)
    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Data transformations
    train_tf, val_tf = load_transforms()

    # Datasets
    train_ds = DATASET(args.data_root, args.dataset, subset="Train", transform=train_tf)
    val_ds = DATASET(args.data_root, args.dataset, subset="Val", transform=val_tf)

    # Data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.Trainer.batch_size),
        shuffle=True,
        num_workers=int(cfg.Trainer.num_workers),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.Trainer.batch_size),
        shuffle=False,
        num_workers=int(cfg.Trainer.num_workers),
        pin_memory=True,
    )

    # Model
    model = MODEL(cfg)

    # Callbacks
    monitor_metric = "MAE/Val" if model.regression else "Loss/Val"
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor=monitor_metric,
        mode="min",
        save_top_k=1,
        filename="epoch={epoch:03d}-{metric:.4f}",
        auto_insert_metric_name=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Trainer
    trainer = pl.Trainer(
        accelerator=str(cfg.Trainer.accelerator),
        devices=cfg.Trainer.devices,
        max_epochs=int(cfg.Trainer.num_epochs),
        callbacks=[checkpoint_cb, lr_monitor],
        default_root_dir=str(outdir),
        log_every_n_steps=50,
        deterministic=True,
    )

    # Fit
    trainer.fit(model, train_loader, val_loader)

    # Evaluate on test split using the best checkpoint
    if checkpoint_cb.best_model_path:
        print(f"Best checkpoint saved at: {checkpoint_cb.best_model_path}")
        test_ds = DATASET(args.data_root, args.dataset, subset="Test", transform=val_tf)
        test_loader = DataLoader(
            test_ds,
            batch_size=int(cfg.Trainer.batch_size),
            shuffle=False,
            num_workers=int(cfg.Trainer.num_workers),
            pin_memory=True,
        )
        best_model = MODEL.load_from_checkpoint(checkpoint_cb.best_model_path, cfg=cfg)
        trainer.test(best_model, test_loader)


if __name__ == "__main__":
    main()