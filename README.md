# ConvNeXt + Transformer Hybrid for Age Estimation

This repository provides a **reproducible** implementation of the ConvNeXt–Vision Transformer hybrid framework presented across two related publications:

- **Original article (Elsevier / Computer Vision and Image Understanding, 2025):** **“[Integrating ConvNeXt and vision transformers for enhancing facial age estimation](https://www.sciencedirect.com/science/article/pii/S1077314225002656?via%3Dihub)”**
- **Extended open-access article (MDPI / Applied Sciences, 2026):** **“[ConvNeXt Meets Vision Transformers: A Powerful Hybrid Framework for Facial Age Estimation](https://www.mdpi.com/2076-3417/16/7/3281)”**

The MDPI version extends the experimental analysis with the UTKFace benchmark, cumulative-score reporting (CS@5), computational-efficiency measurements, and additional ablation and interpretability analyses.

It is intended for research and educational use only. The code enables you to train and evaluate three architectures for age estimation:

- **ConvNeXt**
- **Vision Transformer (ViT)**
- **Hybrid ConvNeXt + ViT** (our proposed model)

<img width="1743" height="1003" alt="download" src="https://github.com/user-attachments/assets/d7c46e45-6b1f-4d91-83e8-51e2752f3509" />

## Reported results

The table below summarizes the architecture-level mean absolute error (MAE, in years) reported in the extended MDPI article. The MORPH II, CACD, and AFAD hybrid results are also reported in the original CVIU article; UTKFace is added in the MDPI extension. **Lower is better.**

| Dataset | ConvNeXt | Vision Transformer (ViT) | Hybrid ConvNeXt–Transformer |
|---|---:|---:|---:|
| MORPH II | 2.29 | 2.47 | **2.26** |
| CACD | 4.40 | 4.71 | **4.35** |
| AFAD | 3.12 | 3.43 | **3.09** |
| UTKFace | 4.65 | 4.96 | **4.47** |

<details>
<summary><strong>Additional MDPI results: CS@5 and computational efficiency</strong></summary>

### Cumulative score within a five-year error margin (CS@5)

**Higher is better.**

| Dataset | ConvNeXt | Vision Transformer (ViT) | Hybrid ConvNeXt–Transformer |
|---|---:|---:|---:|
| MORPH II | **90.10%** | 86.33% | 78.65% |
| CACD | 71.20% | 66.30% | **72.00%** |
| UTKFace | 64.84% | 61.50% | **86.77%** |

### Computational efficiency on MORPH II

Inference time is the average reported per image.

| Architecture | MAE | Parameters | Model size | Inference time |
|---|---:|---:|---:|---:|
| Vision Transformer (ViT) | 2.47 | 5.55 M | 21.17 MB | **1.20 ms** |
| ConvNeXt | 2.29 | 28.02 M | 106.88 MB | 2.39 ms |
| Hybrid ConvNeXt–Transformer | **2.26** | 33.42 M | 127.48 MB | 3.22 ms |

</details>

By default the scripts are configured for the MORPH II dataset, but you can run them on other datasets (e.g. CACD, AFAD, UTKFace) by pointing to your own data.

> **Important licensing notice**
> 
> Some datasets used in the paper (MORPH II, CACD, AFAD) carry research‑only licences.  **We do not distribute any data.**  You must obtain these datasets from their official sources and comply with their terms.  Do **not** redistribute the raw images or derived metadata (e.g. `.pkl` files) in any public repository.

---

## Repository structure

The key folders in this project are:

- `src/cnxtvit/` – core library: model definitions, dataset loading and utilities.
- `scripts/` – entrypoints for training (`train.py`), evaluation (`eval.py`) and building dataset indices (`build_index_pickle.py`).
- `configs/` – example YAML configuration files.
- `legacy/` – original scripts preserved for reference; they contain hard‑coded paths and are not recommended for reproduction.
- `weights/` – **empty by default**; place downloaded checkpoints here when evaluating or fine‑tuning.  This folder is ignored by git via `.gitignore`.

---

## Installation

This project requires Python 3.9 or later.  We recommend using a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

Alternatively, install the package in editable mode using:

```bash
pip install -e .
```

---

## Data format

The data loader expects each dataset to have a corresponding **index file** (`<DATASET>.pkl`) and a folder of images.  The folder layout is:

```
DATA_ROOT/
  MORPH2/
    <image files...>
  MORPH2.pkl
```

The `.pkl` file should be a pandas DataFrame with columns:

- `Image` – relative path of each image within `DATA_ROOT/<DATASET>/`.
- `Age` – numeric age label.
- `Fold` – one of `Train`, `Val`, `Test`.

If you do not already have a pickle file, use the provided script to build one from a CSV:

```bash
python scripts/build_index_pickle.py \
    --csv /path/to/MORPH2_splits.csv \
    --out /path/to/DATA_ROOT/MORPH2.pkl
```

The CSV must contain the same columns (`Image`, `Age`, `Fold`).  The training scripts will automatically detect and use the pickle if present; otherwise they scan the dataset on the fly.

---

## Training

Use the unified training script in `scripts/train.py`.  You must provide a YAML configuration and dataset parameters.  A ready‑to‑use example for the hybrid model is provided in `configs/morph2_hybrid.yaml`.

### Example: hybrid model on MORPH II

```bash
python scripts/train.py \
    --config configs/morph2_hybrid.yaml \
    --data-root /path/to/DATA_ROOT \
    --dataset MORPH2 \
    --outdir runs/morph2_hybrid
```

This will:

1. Load the YAML config and apply any command‑line overrides.
2. Build training/validation datasets using `DATA_ROOT` and `DATASET`.
3. Train a model for the number of epochs specified in the config.
4. Save the best checkpoint into `outdir/checkpoints/`.
5. Automatically evaluate the best checkpoint on the test split.

You can override any hyper‑parameter defined in the config using key–value pairs at the end of the command.  For example, to change the batch size and learning rate:

```bash
python scripts/train.py ... TRAIN.batch_size 128 TRAIN.lr 3e-4
```

---

## Evaluation

To evaluate a trained checkpoint, use `scripts/eval.py`.  You must provide the checkpoint (`.ckpt`), the same YAML configuration used for training, the dataset root and name, and the split to evaluate.

```bash
python scripts/eval.py \
    --ckpt runs/morph2_hybrid/checkpoints/epoch=079-val_mae=2.2600.ckpt \
    --config configs/morph2_hybrid.yaml \
    --data-root /path/to/DATA_ROOT \
    --dataset MORPH2 \
    --split Test
```

The script will print the mean absolute error (MAE) on the specified split.

---

## Pretrained weights

We provide pretrained checkpoints corresponding to the MORPH II results reported in the papers.  These files are large and **are not tracked by git**.  To use them:

1. Download the desired `.ckpt` file from the project’s release page.
2. Place the file into the `weights/` folder at the repository root.  Do not commit it.
3. Use the checkpoint when evaluating or fine‑tuning by specifying the path in the appropriate argument or config entry.

The available checkpoints for MORPH II are:

| Filename | Architecture | Dataset | MAE |
|---------|-------------|---------|----|
| `MORPH2_Transformer_FC2_128_batch256_epoch_499_mae_2.47.ckpt` | Transformer (ViT) | MORPH II | 2.47 |
| `MORPH2_ConvNeXT_FC2_256_epoch_499_mae_2.29.ckpt` | ConvNeXt | MORPH II | 2.29 |
| `MORPH2_CNNXT_VIT_preT_MORPH2_FC2_batch128_lossfunc_Adaptive_epoch_499_mae_2.26.ckpt` | **Hybrid** (ConvNeXt+ViT) | MORPH II | 2.26 |

Place the files in `weights/` like so:

```text
weights/
  MORPH2_Transformer_FC2_128_batch256_epoch_499_mae_2.47.ckpt
  MORPH2_ConvNeXT_FC2_256_epoch_499_mae_2.29.ckpt
  MORPH2_CNNXT_VIT_preT_MORPH2_FC2_batch128_lossfunc_Adaptive_epoch_499_mae_2.26.ckpt
```

and then reference them via the `--ckpt` flag in `eval.py` or set `Model.pretrained` in your YAML if you wish to fine‑tune.

---

## Citation

If you use this code, models, or weights in your work, please cite the publication relevant to your use (or both when using the extended evaluation).

### Original CVIU article

```bibtex
@article{Maroun2025HyCnXtViT,
  title   = {Integrating ConvNeXt and vision transformers for enhancing facial age estimation},
  author  = {Maroun, G. and Bekhouche, S.E. and Charafeddine, J. and Dornaika, F.},
  journal = {Computer Vision and Image Understanding},
  volume  = {262},
  pages   = {104542},
  year    = {2025},
  month   = dec,
  doi     = {10.1016/j.cviu.2025.104542}
}
```

### Extended MDPI article

```bibtex
@article{Maroun2026ConvNeXtMeetsViT,
  title   = {ConvNeXt Meets Vision Transformers: A Powerful Hybrid Framework for Facial Age Estimation},
  author  = {Maroun, Gaby and Bekhouche, Salah Eddine and Dornaika, Fadi},
  journal = {Applied Sciences},
  volume  = {16},
  number  = {7},
  pages   = {3281},
  year    = {2026},
  month   = mar,
  doi     = {10.3390/app16073281},
  url     = {https://www.mdpi.com/2076-3417/16/7/3281}
}
```

---

## License

This repository is released under the MIT License.  Portions of the code originate from the official ConvNeXt and Vision Transformer implementations and retain their respective licence headers.  Please respect the licences of any datasets you use in conjunction with this code.
