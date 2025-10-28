\# Cataract-101 — Clip-Based Phase Classifier (Minimal Repo)



This repository contains a \*\*very lightweight demo\*\* for recognising surgical phases in cataract videos with a \*clip-based\* deep-learning model.



\## 📂 Folder Layout



Dataset/ # raw CSV metadata and this dataset description

cataract101\_clip\_classifier.ipynb # training \& inference demo in a single notebook





\## 🚀 Quick Start



1\. \*\*Clone\*\* the repo and install basic dependencies (tested with Python 3.10):

&nbsp;  ```bash

&nbsp;  git clone https://github.com/your-username/cataract101-clip-classifier.git

&nbsp;  cd cataract101-clip-classifier

&nbsp;  pip install -r requirements.txt   # optional: only if you add it





Download videos from the official Cataract-101 release and place them next to the CSV files inside Dataset/.



Open cataract101\_clip\_classifier.ipynb in Jupyter / Colab / Kaggle and run the cells.

The notebook:



builds a tiny PyTorch dataset of 2-second clips,



trains a small 3D-ResNet (or loads model.pt if already trained),



outputs per-frame predictions and simple accuracy metrics.



📒 Dataset



See Dataset/about.md for a concise description of the Cataract-101 dataset and the meaning of each CSV file.



🔖 License



The code is released under the MIT License. The Cataract-101 data follow their own license—please respect the original terms.









