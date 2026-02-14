import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from cnxtvit import MODEL
from cnxtvit.data import DATASET, load_transforms
from cnxtvit.utils_cfg import load_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='Path to Lightning checkpoint (.ckpt)')
    ap.add_argument('--config', required=True, help='YAML config used to build the model')
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--split', default='Test', choices=['Train', 'Val', 'Test'])
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    _, tf_eval = load_transforms()
    ds = DATASET(args.data_root, args.dataset, args.split, tf_eval)
    loader = DataLoader(ds, batch_size=cfg.Trainer.batch_size, shuffle=False, num_workers=int(cfg.Trainer.num_workers), pin_memory=True)

    model = MODEL.load_from_checkpoint(args.ckpt, cfg=cfg)

    trainer = pl.Trainer(accelerator=str(cfg.Trainer.accelerator), devices=cfg.Trainer.devices, logger=False)
    out = trainer.test(model, dataloaders=loader)
    print(out)


if __name__ == '__main__':
    main()
