# White Blood Cell Classification: Hyperparameter Optimization of a Custom ResNeXt-50 Network

A controlled comparison of **manual tuning vs. Optuna vs. Keras Tuner** for nine-class white blood cell (WBC) classification, using a custom **ResNeXt-50-style CNN trained entirely from scratch** (no pretrained weights) in TensorFlow/Keras.

> Research paper: *Comparative Analysis of Hyperparameter Optimization Techniques for White Blood Cell Classification Using a Custom ResNeXt-50 Network* — KIET Group of Institutions, Ghaziabad. Status: **<in preparation / submitted / under review>**

---

## Highlights

- Custom ResNeXt-50-style architecture (3-4-6-3 bottleneck pattern, cardinality 32) built and trained from scratch.
- Four public WBC dataset partitions merged via **label harmonization** into a unified 9-class taxonomy.
- Identical data pipeline, augmentation, loss, optimizer and training schedule across all experiments, so differences come only from hyperparameters.
- Optuna (TPE + Median Pruner) and Keras Tuner (Hyperband) both improved accuracy by about **29 percentage points** over the manual baseline.

## Results

Evaluated on an independent held-out test set (9,535 images) that was never used for training or tuning.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Macro AUC |
|---|---|---|---|---|---|
| Baseline (manual) | 0.5787 | 0.5068 | 0.5231 | 0.4273 | 0.9180 |
| **Optuna** | **0.8729** | 0.7702 | **0.8682** | **0.7991** | 0.9865 |
| Keras Tuner | 0.8712 | **0.7788** | 0.8510 | 0.7948 | **0.9866** |

### Selected hyperparameters

| Hyperparameter | Baseline | Optuna | Keras Tuner |
|---|---|---|---|
| Initial learning rate | 1e-3 | 2.369e-4 | 1.870e-4 |
| Dropout rate | 0.30 | 0.3901 | 0.40 |
| Dense layer units | 512 | 256 | 256 |

Both optimizers converged to a similar region: lower learning rate, stronger dropout, smaller classification head.

## Dataset

| Source | Host |
|---|---|
| KU-Optofil Peripheral Blood Cell (PBC) dataset | Zenodo |
| Raabin-WBC (training and test sets) | Raabin-WBC |
| Peripheral Blood Cell dataset (Acevedo et al.) | Mendeley Data |

**Unified classes (9):** Basophil, Blast, Eosinophil, Erythroblast, Giant Platelet, Lymphocyte, Monocyte, Neutrophil, Immature Granulocyte (IG).

**Label harmonization**

| Original label | Unified label |
|---|---|
| Band Neutrophil, Segmented Neutrophil | Neutrophil |
| Reactive Lymphocyte, Lymphocyte | Lymphocyte |
| Platelet Cluster, Giant Platelet | Giant Platelet |

**Splits:** 45,399 train / 5,157 validation / 9,535 test images. Validation and test sets are left unchanged.

**Training subset:** to reduce imbalance and compute cost during HPO, up to 1,000 images per class were sampled deterministically (`SEED = 42`), giving **8,571 training images** (571 Blast, 1,000 for every other class). The sampled manifest is saved for reproducibility.

## Method

**Preprocessing:** resize to 224×224×3, normalize to [0, 1]. Training-only augmentation: rotation ≤20°, translation ≤10%, zoom ≤15%, horizontal flip, brightness 0.85–1.15.

**Class imbalance:** inverse-frequency class weights, `w_i = N / (C × n_i)` (0.9523 for most classes, 1.6678 for Blast), used in a weighted categorical cross-entropy with 0.05 label smoothing.

**Architecture:** 7×7 conv stem, then four residual stages (3, 4, 6, 3 bottleneck blocks; filters 64/128/256/512) with 32-group convolutions, then Global Average Pooling, BatchNorm, FC-512, FC-256 (with dropout), and a 9-way softmax. He-normal initialization, L2 regularization (1e-4).

**Training:** Adam with a three-stage learning-rate schedule (1e-3, 3e-4, 1e-4 for 3, 3, 4 epochs; max 10 epochs), batch size 8, EarlyStopping and ReduceLROnPlateau, checkpoint on best validation AUC.

**Hyperparameter optimization** (objective: validation macro AUC)

| | Optuna | Keras Tuner |
|---|---|---|
| Algorithm | TPE with Median Pruner | Hyperband (successive halving) |
| Learning rate | Log-uniform 1e-4 to 1e-3 | Log-uniform 1e-4 to 1e-3 |
| Dropout | Uniform 0.20–0.40 | {0.20, 0.30, 0.40} |
| Dense units | {256, 512} | {256, 512} |

The best configuration from each tuner was used to retrain a fresh model from scratch, so reported numbers are for fully retrained models.

**Evaluation:** Accuracy, macro precision/recall/F1, macro one-vs-rest AUC, ROC and Precision–Recall curves, confusion matrices.

## Repository Structure

> Update this section to match your repo.

```
.
├── data/                 # dataset manifests / label mapping (raw data not included)
├── notebooks/            # baseline, Optuna and Keras Tuner experiments
├── src/                  # model, data pipeline, training and evaluation code
├── results/              # curves, confusion matrices, metrics
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <repo-url>
cd <repo-name>
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Environment used:** Python 3.12, TensorFlow 2.18.0, Optuna 4.3.0, Keras Tuner 1.4.7. Trained on an Apple M-series GPU via the TensorFlow Metal backend.

## Usage

> Adjust the commands to your actual scripts or notebooks.

```bash
# 1. Prepare the harmonized dataset and the deterministic 1,000-per-class subset
python src/prepare_data.py --seed 42

# 2. Train the manual baseline
python src/train.py --config baseline

# 3. Run hyperparameter search
python src/tune_optuna.py
python src/tune_keras_tuner.py

# 4. Retrain with best configs and evaluate on the test set
python src/train.py --config optuna
python src/train.py --config keras_tuner
python src/evaluate.py
```

Datasets must be downloaded separately from their original sources (see Dataset) and placed under `data/`.

## Reproducibility

Fixed seed (42), deterministic subset sampling with a saved manifest, identical splits and training protocol for all three configurations, and a test set that stays unseen until final evaluation.

## Limitations and Future Work

- Search space covers only learning rate, dropout and first dense-layer size.
- Training subset capped at 1,000 images per class; Blast remains a minority class.
- No cross-institutional validation yet.
- Planned: more HPO strategies, Vision Transformers and hybrid CNN–Transformer models, self-supervised learning, explainability (e.g., Grad-CAM) and uncertainty estimation.

## Authors

<Add author names as in the paper>, Dept. of Computer Science and Engineering (Artificial Intelligence), KIET Group of Institutions, Ghaziabad.

## Citation

```bibtex
@unpublished{wbc_hpo_resnext,
  title  = {Comparative Analysis of Hyperparameter Optimization Techniques for White Blood Cell Classification Using a Custom ResNeXt-50 Network},
  author = {<authors>},
  year   = {2026},
  note   = {<status>}
}
```

## Acknowledgements

Datasets: KU-Optofil PBC (Zenodo), Raabin-WBC, and the Acevedo et al. Peripheral Blood Cell dataset (Mendeley Data).

## License

<Choose a license, e.g. MIT>