# ConvNeXt + Transformer Hybrid for Age Estimation

This repository is a **research / educational** release of the code structure used in the paper:

**"[Integrating ConvNeXt and Transformers for Age Estimation](https://www.sciencedirect.com/science/article/pii/S1077314225002656?via%3Dihub)"**

It provides:
- ConvNeXt backbone
- ViT backbone (via `timm`)
- Hybrid fusion model (ConvNeXt + ViT)
- PyTorch Lightning training loop
- Dataset loader based on a simple `{dataset}.pkl` index file

## Important notes (read this once)

- **No datasets are redistributed here.** Some face-age datasets are research-only.
- **Do not upload / publish the dataset itself or derived subject identifiers.**
- Sharing **model weights** is usually fine, but you should check your dataset license and co-author constraints.

## Repository layout

- `src/cnxtvit/` : main library (models + dataset)
- `scripts/` : training / evaluation / utilities
- `configs/` : example YAML configs
- `legacy/` : original scripts preserved (paths were hardcoded; kept for reference)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Data format

The loader expects this structure:

```
DATA_ROOT/
  MORPH2/
    <image files...>
  MORPH2.pkl
```

Where `MORPH2.pkl` is a pandas dataframe saved with columns:
- `Image` : relative path inside `DATA_ROOT/MORPH2/` (string)
- `Age` : numeric age
- `Fold` : one of `Train`, `Val`, `Test`

### Create the `.pkl`

If you have a CSV with those columns you can build the pickle:

```bash
python scripts/build_index_pickle.py \
  --csv /path/to/MORPH2_splits.csv \
  --out /path/to/DATA_ROOT/MORPH2.pkl
```

## Train (hybrid model)

```bash
python scripts/train.py \
  --config configs/morph2_hybrid.yaml \
  --data-root /path/to/DATA_ROOT \
  --dataset MORPH2 \
  --outdir runs/morph2_hybrid
```

The best checkpoint is saved under `runs/.../checkpoints/`.

## Evaluate

```bash
python scripts/eval.py \
  --ckpt runs/morph2_hybrid/checkpoints/best.ckpt \
  --config configs/morph2_hybrid.yaml \
  --data-root /path/to/DATA_ROOT \
  --dataset MORPH2 \
  --split Test
```

## Sharing weights

Recommended:
- Put weights in **GitHub Releases** or **Hugging Face Hub**.
- Provide checksum (SHA256) and exact config used.

## Citation

If you use this repo, cite the paper and link to this repository.

## License

MIT for this repository. ConvNeXt code includes upstream license headers where applicable.
