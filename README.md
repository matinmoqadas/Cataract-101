# Cataract-101 · Clip-Based Surgical Phase Classifier

A **minimal, self-contained demo** for recognising surgical phases in Cataract-101 videos with an MS-TCN trained on short-clip features.

---

## 1 Overview

| Component    | Description                                                            |
| ------------ | ---------------------------------------------------------------------- |
| **Input**    | `videos.csv`, `annotations.csv`, `phases.csv` from Cataract-101        |
| **Features** | Pre-computed ResNet-50 or ViT embeddings (per frame / 2 s clip)        |
| **Model**    | Multi-Stage TCN — *N* stages × 10 dilated 1-D conv layers              |
| **Loss**     | Cross-entropy (frame labels) + temporal smoothing (MSE or MS-TCN loss) |
| **Output**   | Per-frame predictions, metrics, and the best checkpoint                |

---

## 2 Repository Layout

```text
.
├── Dataset/
│   ├── videos.csv
│   ├── phases.csv
│   ├── annotations.csv
│   └── about.md
└── Cataract101_clip_classifier.ipynb
└── Cataract101_MS-TCN_PhaseSegmentation.ipynb
---

## 3 Quick Start

```bash
# 1 · Clone (Python ≥ 3.10)
git clone https://github.com/matinmoqadas/eda.git
cd eda

# 2 · Add the data (≈ 9 GB)
#    https://ftp.itec.aau.at/datasets/ovid/cat-101/downloads/cataract-101.zip

# 3 · Run the notebook (Jupyter / Colab / Kaggle)
open Cataract101_clip_classifier.ipynb
open Cataract101_MS-TCN_PhaseSegmentation.ipynb

---

## 4 Notebook Workflow

1. **Build** a tiny PyTorch dataset of ≈ 2 s clips
2. **Train** a lightweight 3D-ResNet *or* load `model.pt` if present
3. **Predict** per-frame phases and compute accuracy / mAP
4. **Export** the best MS-TCN weights (`best_clip_model.pt`)

---

## 5 Outputs

| Artifact                 | Contents                                                          |
| ------------------------ | ----------------------------------------------------------------- |
| **Per-epoch logs**       | Training / validation loss, accuracy, mAP                         |
| **`best_clip_model.pt`** | Checkpoint with highest validation accuracy                       |
| *(optional)* CSV / txt   | Predicted phase timelines per video (for F1@10/25/50, edit score) |

### 5.1 Loading the Checkpoint

```python
from model import MSTCN
import torch

model = MSTCN(num_stages=4, num_layers=10, num_classes=10)
state = torch.load("best_clip_model.pt")
model.load_state_dict(state["state_dict"])
model.eval()
```

---

## 6 Dataset

* **Kaggle mirror:** [https://www.kaggle.com/datasets/matinmo/cataract-101](https://www.kaggle.com/datasets/matinmo/cataract-101)
* See `Dataset/about.md` for a concise description of the CSV schema.

---

## 7 Pre-trained Model

* **12 h MS-TCN:** [https://www.kaggle.com/models/matinmo/clip-based-model](https://www.kaggle.com/models/matinmo/clip-based-model)

---

## 8 License

| Asset    | License                                                          |
| -------- | ---------------------------------------------------------------- |
| **Code** | MIT                                                              |
| **Data** | Original Cataract-101 license — please respect the source terms. |
