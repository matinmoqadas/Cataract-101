# Cataract-101 Dataset

Cataract-101 provides **101 anonymised cataract surgery videos** recorded in the operating theatre at **25 fps** with a resolution of **720×540 px**.

| Item | Value |
|------|-------|
| Total videos | 101 |
| Surgeons | 4 (2 residents, 2 experts) |
| Average duration | ~10 min |
| Resolution | 720 × 540 |
| Frame rate | 25 fps |
| Label schema | 7 surgical phases (frame-level) |

### Phase distribution (approx. share of frames)

| Phase | % of frames |
|-------|-------------|
| Phacoemulsification | **30 %** |
| Irrigation & Aspiration | 13 % |
| Viscoelastic Removal | 13 % |
| Capsulorhexis | 9 % |
| Others (remaining phases) | 35 % |

Each CSV file in this folder is **self-contained**:

* **`videos.csv`** – video file names, duration, surgeon ID.  
* **`annotations.csv`** – per-video frame‐level labels (`video_id, frame_idx, phase`).  
* **`phases.csv`** – lookup table with numeric ID, phase name, and colour code for plots.

> The dataset is shared under the original Cataract-101 license. Please cite the authors if you use it in your research.
