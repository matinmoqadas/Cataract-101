# Cataract-101 Clip-Based Surgical Phase Classification

A minimal, self-contained demo for recognising surgical phases in Cataract-101 videos using clip-level features and a Multi-Stage Temporal Convolutional Network (MS-TCN).


---

## Overview

| Component    | Description                                                           |
| ------------ | --------------------------------------------------------------------- |
| Input        | `videos.csv`, `annotations.csv`, `phases.csv` from Cataract-101       |
| Features     | Pre-computed CNN embeddings per frame or per 2 s clip                 |
| Model        | MS-TCN: N stages x 10 dilated 1-D conv layers                          |
| Loss         | Cross-entropy (frame labels) + temporal smoothing (MSE/MS-TCN loss)   |
| Output       | Per-frame predictions, metrics, and the best checkpoint               |

---

## Repository Layout

```text
.
data/
  metadata/
    videos.csv
    phases.csv
    annotations.csv
  about.md
notebooks/
  01_clip_baseline.ipynb
  02_mstcn_temporal.ipynb
```

---

## 01_clip_baseline.ipynb
```text
**Here’s what the notebook is doing, in plain-English terms**

The notebook builds a *clip-level* video-classification baseline for the Cataract-101 surgical-phase dataset.  After importing the raw videos and accompanying metadata from Kaggle, it:

1. **Pre-processes the data**

   * Reads a CSV that maps each numeric phase ID to a human-readable phase name.
   * Generates a per-frame ground-truth timeline for every video, then slices each video into short, fixed-length clips (default = 16 frames, sampled every two frames) so that the model can treat surgery videos like a sequence of independent “mini-videos”.

2. **Builds the model**

   * Uses a ResNet-50 backbone (from `torchvision`) that is optionally partially frozen; its final convolutional features are average-pooled over time and passed through a small fully-connected head that predicts one of *N* surgical phases for each clip.
   * Trains with cross-entropy loss, AdamW, cosine LR decay, optional automatic-mixed precision, and tracks the best validation accuracy; that best checkpoint is saved as `best_clip_resnet50.pt`.

3. **Evaluates & visualises**

   * After every epoch it runs a full validation pass, accumulating predicted labels and targets so it can print a **`classification_report`** (per-class precision, recall, F1 and overall accuracy) plus a confusion matrix.
   * For one sample video it stitches the clip-level predictions back into a *frame-level* timeline and draws two Matplotlib charts:

     * the predicted vs. ground-truth phase label over time, and
     * the model’s confidence for each frame.

---

### What you’ll see when you run it

* **Console output**

  * A list of the detected phase names (e.g. “0 – Wound Incision, 1 – Capsulorrhexis, …”).
  * Epoch-by-epoch logs showing training/validation loss and accuracy.
  * At the end of training (or after loading an existing checkpoint) a neatly formatted **classification report**, e.g.

    ```
                 precision  recall  f1-score  support
       Phase 0       0.83    0.79      0.81     4,215
       Phase 1       0.77    0.74      0.75     3,608
       …                                   
       accuracy                       0.78    27,346
    ```
  * A text or table version of the confusion matrix so you can see where the model is mixing up neighbouring phases.

* **Plots**

  * **Timeline plot** – coloured band showing, second-by-second, which phase the ground truth (solid bar) and the model (dashed bar) assign to the sample video; mismatches jump out visually.
  * **Confidence plot** – a simple line chart of prediction probability over time, useful for spotting uncertain regions.

Together, these outputs give you a quick sense of both overall performance (via the metrics) and qualitative behaviour (via the timeline visualisation) without having to open the videos by hand.

```

## 02_mstcn_temporal.ipynb
```text
### 1  – What the notebook does (high-level)

The notebook implements **multi-stage temporal convolutional networks (MS-TCN)** to recognise cataract-surgery phases frame-by-frame.
In broad strokes it

| Pipeline step            | Key actions                                                                                                                                                                           |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data prep**            | Reads *videos.csv*, *phases.csv* and *annotations.csv* to build per-frame ground-truth labels; splits videos into train/val/test folds.                                               |
| **Feature extraction**   | Either <br>• loads pre-computed CNN clip features **or** <br>• extracts them on-the-fly with a 2D/3D backbone (e.g. ResNet + TSM) and stores `.npy` files to speed subsequent epochs. |
| **Model**                | Creates an **MS-TCN** with several stages of 1-D dilated convolutions, residual blocks and softmax output over the ten surgical phases.                                               |
| **Training loop**        | Runs epoch cycles with cross-entropy + temporal smoothing loss, mixed-precision AMP, gradient clipping, learning-rate scheduler and early-stopping.                                   |
| **Validation & logging** | Computes frame-wise accuracy, segmental F1@{10,25,50}, edit distance and plots loss/metric curves.                                                                                    |
| **Checkpointing**        | Saves the best validation model to *best_segmentation_model.pt* and optionally exports per-video CSV timelines.                                                                       |

These steps correspond to the **“Temporal Action Segmentation”** paradigm described in your project brief.

---


### 2  – One-paragraph summary

> *This notebook trains a multi-stage temporal convolutional network to label every frame of Cataract-101 surgery videos with its current operative phase. After parsing the annotation CSVs it extracts or loads clip-level visual features, feeds them to an MS-TCN whose stacked dilated 1-D filters learn long-range temporal patterns, and iteratively minimises a combination of cross-entropy and smoothing losses. Validation after each epoch reports segmental F1-scores and edit distance; the weights with the best overall F1@25 are stored for later inference. The end product is a model able to generate a precise phase timeline for an unseen surgery video, forming the segmentation half of the clip-vs-segmentation comparison outlined in the project plan.*

---

#### 2.1  Outputs produced by the notebook

* **`best_segmentation_model.pt`** – PyTorch `state_dict` for the MS-TCN achieving the highest validation metric.
* **Metrics log / plots** – loss curves and per-epoch tables (accuracy, F1@k, edit distance).
* **Optional CSV timelines** – one file per test video containing predicted phase labels per frame or per second.

Together these files let you reload the model for inference, reproduce results or visualise predictions on new videos.

---

#### 2.2  What is **`best_clip_model.pt`**?

Earlier in your workflow you trained a **clip-based classifier** (likely a CNN such as ResNet, R(2+1)D or SlowFast) that predicts a single phase for short 2-3 s snippets—this corresponds to the “Clip-based Video Classification” baseline in the proposal.
`best_clip_model.pt` therefore stores the weight dictionary (`model.state_dict()`) for that *clip* model.  You can reuse it by

you can download the model on kaggle: https://www.kaggle.com/models/matinmo/clip-based-model
```python
model = ClipModel()          # same architecture definition
model.load_state_dict(torch.load('best_clip_model.pt'))
model.eval()
```

Typical uses:

* **Feature extractor** – freeze the backbone and output penultimate-layer activations for the MS-TCN input.
* **Baseline inference** – slide a 3 s window over a full video, get clip predictions, then smooth or majority-vote to build a phase timeline.
* **Ensemble** – combine clip and segmentation probabilities to boost boundary accuracy.

Because it was trained for 12 h on Kaggle, expect it to already converge to >80 % clip accuracy and serve as a solid comparison point against the temporal segmentation network.
```


## Quick Start

```bash
# 1) Clone (Python >= 3.10 recommended)
git clone https://github.com/matinmoqadas/Cataract-101.git
cd Cataract-101

# 2) Add the data (~9 GB)
#    Place the CSVs under data/metadata/
#    videos.csv, annotations.csv, phases.csv

# 3) Launch Jupyter and open the notebooks
jupyter lab  # or: jupyter notebook
# Open: notebooks/01_clip_baseline.ipynb
#       notebooks/02_mstcn_temporal.ipynb
```

---

## Notebook Workflow

1. 01_clip_baseline.ipynb: build clip features and train a baseline classifier.
2. 02_mstcn_temporal.ipynb: train MS-TCN on features, produce per-frame predictions and metrics.

Adjust file paths inside notebooks to match your environment if needed.

---

## Outputs

Typical artifacts include:
- Training/validation logs (loss, accuracy, mAP)
- Best checkpoint (e.g., `best_clip_model.pt`)
- Optional CSV timelines for evaluation (F1@10/25/50, edit score)

Loading a checkpoint (example):

```python
import torch

state = torch.load("best_clip_model.pt", map_location="cpu")
# Restore into your model as done in the notebook
```

---

## Dataset

- Kaggle mirror: https://www.kaggle.com/datasets/matinmo/cataract-101
- See `data/about.md` for a concise description of the CSV schema.

---

## Pre-trained Model

- Example MS-TCN: https://www.kaggle.com/models/matinmo/clip-based-model

---

## License

| Asset | License |
| ----- | ------- |
| Code  | MIT     |
| Data  | CC BY-NC 4.0 |
