# Cataract-101: Clip-Based Surgical Phase Classifier (Minimal Repo)

This repository contains a very lightweight demo for recognizing surgical phases in cataract videos using a clip-based deep-learning model.

## Folder Layout

- `Dataset/` - CSV metadata and dataset description (`Dataset/about.md`)
- `cataract101_clip_classifier.ipynb` - single notebook covering training and inference
- Additional experimental notebooks: `ms_tcn_train.ipynb`, `ms_tcn_full.ipynb`, `day*_ms-tcn*.ipynb`

## Quick Start

1. Clone the repo (tested with Python 3.10):

   ```bash
   git clone https://github.com/matinmoqadas/eda.git
   cd eda
   ```

2. Obtain the Cataract-101 data and place the videos next to the CSV files inside `Dataset/`.
   - Official download: https://ftp.itec.aau.at/datasets/ovid/cat-101/downloads/cataract-101.zip

3. Open the notebook and run cells in Jupyter/Colab/Kaggle:
   - `cataract101_clip_classifier.ipynb`

### What the notebook does

- Builds a tiny PyTorch dataset of ~2-second clips
- Trains a small 3D-ResNet (or loads `model.pt` if already trained)
- Produces per-frame predictions and simple accuracy metrics

## Dataset

See `Dataset/about.md` for a concise description of the Cataract-101 dataset and the meaning of each CSV file (`videos.csv`, `phases.csv`, `annotations.csv`).

## License

- Code: MIT License
- Data: governed by the original Cataract-101 license - please respect the source terms.

