# ⚽ DataDriven Soccer Scouting

**Deep Player Embeddings: Dimensionality Reduction & Anomaly Detection in European Soccer**

Unsupervised machine learning for tactical scouting. This project compresses **~115 technical, tactical and physical metrics** for **~4,000 players** from the Top-5 European leagues into a compact *"Tactical DNA"*, then uses it to answer two scouting questions:

1. **Who plays like this player?** — find affordable "hidden gems" whose style mirrors an elite target.
2. **Who breaks the mould?** — surface tactical outliers who ignore the conventions of their nominal position.

Data comes from **FBref** (performance stats) enriched with **EA FC 24** (height, weight, preferred foot).

---

## Table of Contents

- [Motivation](#motivation)
- [The Two Tasks](#the-two-tasks)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Tech Stack](#tech-stack)
- [Acknowledgements](#acknowledgements)

---

## Motivation

Modern scouting still leans heavily on rigid positional labels ("he's a right-back", "she's a number 10"). Those labels hide as much as they reveal: two players sharing a label can have completely different jobs on the pitch, and the most valuable transfer-market opportunities are often players doing an elite job under the "wrong" label or in an under-watched league.

Instead of trusting labels, this project learns a numerical fingerprint of *how* a player actually plays — directly from the data — and uses distance in that fingerprint space to drive objective scouting.

---

## The Two Tasks

### Task 1 — Similarity Search ("The Hidden Gem Engine")

Compress every player into a low-dimensional latent vector, then rank all other players by **cosine similarity** to a chosen target. Because cosine similarity compares the *direction* of the vectors rather than their magnitude, a prospect playing 1,500 minutes in a smaller league can score as a near-perfect match for a superstar playing 3,500 minutes in the Champions League.

### Task 2 — Anomaly Detection ("The Anomaly Hunter")

Find players whose statistical profile structurally violates the template for their role — a centre-back who builds play like a deep-lying playmaker, a full-back with a winger's attacking footprint. Two independent approaches are compared:

- **Clustering view** — group players purely by playing style and flag positional minorities inside highly homogeneous clusters.
- **Reconstruction view** — train one autoencoder per position and flag the players it fails to reconstruct.

---

## Dataset

| | |
|---|---|
| **Records** | ~4,000 players (~4,800 player-season rows) |
| **Features** | ~115 technical, tactical & physical metrics |
| **Leagues** | Premier League, La Liga, Serie A, Bundesliga, Ligue 1 |
| **Seasons** | 2022/23, 2023/24, 2024/25 |
| **Positions** | Outfield players only (goalkeepers excluded) |

**Sources**

- **[FBref](https://fbref.com/)** (via the [`soccerdata`](https://github.com/probberechts/soccerdata) library) — per-90 and season-total performance data: shooting, passing, progression, defending, possession, aerials, etc.
- **[EA Sports FC 24 database](https://www.kaggle.com/)** (via [`kagglehub`](https://github.com/Kaggle/kagglehub)) — physical attributes: height, weight, preferred foot.

The two sources are joined by a fuzzy name/team match in [`dataset_geneator.ipynb`](dataset_geneator.ipynb), producing the committed [`merged_data.csv`](merged_data.csv). You do **not** need to regenerate the dataset to run the apps.

---

## Methodology

### Preprocessing

- Drop "status" features that leak playing time (`90s`, `Starts`, `MP`) and penalty metrics (`PK`, `PKatt`) so the models group players by *tactics*, not by *minutes*.
- `StandardScaler` (mean 0, std 1) on all features.
- **L2 normalization** of each player vector, removing "possession bias" — a player with 100 passes / 10 tackles lands next to one with 50 passes / 5 tackles (same 10:1 ratio).
- Macro-position mapping (e.g. `RB`/`LB` → *Fullback*, `RW`/`LW` → *Winger*) to group functionally equivalent roles.

### Task 1 — Latent spaces

| Model | What it does | Config |
|---|---|---|
| **PCA** *(baseline)* | Linear compression of 109 features | 27 components → **95%** cumulative explained variance |
| **Deep Autoencoder** *(advanced)* | Symmetric feed-forward network learning non-linear tactical structure | 109 → hidden → **16-node linear bottleneck** → hidden → 109; **Tanh** activations, **Huber** loss, EarlyStopping |

Four autoencoder variants were tested (ReLU / Tanh, with and without dropout); **pure Tanh without dropout** won on reconstruction quality and scouting sanity checks.

**Z-Score Weighted Ensemble.** Neither model is best alone: the autoencoder respects positional discipline, PCA better preserves raw statistical volume. Their cosine-similarity scores live on different scales, so each is converted to a **Z-score** and combined **70% autoencoder / 30% PCA** (the weighting is adjustable in the app).

### Task 2 — Anomaly detection

**A. K-Means clustering** ([`app_anomaly_det.py`](app_anomaly_det.py))
- `k = 20` clusters, `n_init = 50`, Euclidean distance on L2-normalized data (equivalent to cosine similarity).
- Each cluster is profiled with its top ±3 Z-score features, dominant role, and a human-readable scouting report.
- Anomalies = players sitting in a cluster dominated by a different position.

**B. Per-position Autoencoder reconstruction error** ([`app_anomaly_det_ae.py`](app_anomaly_det_ae.py))
- A **separate autoencoder per macro-position**, so a winger is never flagged just for not defending like a centre-back.
- `QuantileTransformer(output_distribution='normal')` fitted locally per position.
- Architecture scales with sample size (deeper network + 20% dropout for positions with >250 players).
- Robust anomaly score based on **Median Absolute Deviation (MAD)**; score **> 2.5** ⇒ "true anomaly".
- Feature deviations are inverse-transformed back to real-world units to explain *why* each player is an outlier.

---

## Key Findings

- **Full-backs are the most reinvented role** (~11.7% anomaly rate) — the purely defensive full-back is effectively obsolete; many now operate as inverted playmakers or auxiliary wingers.
- **Midfield is the tactical laboratory** — central and wide midfielders show both the highest anomaly frequency and the most extreme deviations.
- **Centre-backs and defensive midfielders are rigid** (CB ~3.4%) — coaches demand traditional profiles; experimentation is rare.
- **Full-back symmetry** — RB and LB cluster distributions correlate at **0.97**: the modern job is nearly flank-agnostic.
- **The striker is a lone wolf** — the ST profile is so distinct (high shots, high xG, low build-up involvement) that strikers almost never share a cluster with even the most advanced wingers or attacking midfielders.
- **League cultures differ** — the Premier League (~9.1%) and Bundesliga (~8.1%) reward positional fluidity; Ligue 1 (~5.4%) is the most template-driven.

---

## Repository Structure

```
DataDriven_Soccer_Scouting/
├── DataDriven_Soccer_Scouting.ipynb   # Main analysis: EDA, PCA, autoencoders, clustering, anomaly detection
├── dataset_geneator.ipynb             # Builds merged_data.csv from FBref + EA FC 24
├── cluster_functions.py               # Shared plotting / analysis helpers for the Streamlit apps
│
├── app_similarity_search.py           # Task 1 app — Hidden Gem Engine (PCA + AE ensemble)
├── app_anomaly_det.py                 # Task 2 app — K-Means clustering anomaly explorer
├── app_anomaly_det_ae.py              # Task 2 app — Deep per-position reconstruction-error explorer
│
├── merged_data.csv                    # Final joined dataset
├── resources/                         # Precomputed cluster profiles, anomaly tables, glossary
├── saved_models/                      # Trained Keras encoders + exported latent-space CSVs
└── ProjectOutline/                    # Original project brief
```

---

## Installation

Requires **Python 3.11+**.

```bash
git clone https://github.com/andreamurari/DataDriven_Soccer_Scouting.git
cd DataDriven_Soccer_Scouting

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> `tensorflow`, `soccerdata` and `kagglehub` are only needed to re-run the notebooks — see the comments in [`requirements.txt`](requirements.txt). The three Streamlit apps run on the committed CSVs and only need the core dependencies.

---

## Usage

### Run the scouting apps

```bash
streamlit run app_similarity_search.py   # find players with a similar tactical DNA
streamlit run app_anomaly_det.py         # K-Means clustering anomaly explorer
streamlit run app_anomaly_det_ae.py      # deep per-position anomaly explorer
```

Each app opens in the browser with an **Overview** section (methodology, walkthrough) and an interactive **explorer** with filters for position, league, age and season.

### Reproduce the analysis

1. *(optional)* Run [`dataset_geneator.ipynb`](dataset_geneator.ipynb) to rebuild `merged_data.csv` — needs a Kaggle account configured for `kagglehub`.
2. Run [`DataDriven_Soccer_Scouting.ipynb`](DataDriven_Soccer_Scouting.ipynb) end to end to regenerate everything in `resources/` and `saved_models/`.

---

## Tech Stack

**ML / data:** scikit-learn (PCA, K-Means, `QuantileTransformer`), TensorFlow / Keras (autoencoders), pandas, NumPy, SciPy
**Apps / viz:** Streamlit, Plotly, Matplotlib
**Data collection:** `soccerdata` (FBref), `kagglehub` (EA FC 24)

---

## Acknowledgements

- Performance data: [FBref](https://fbref.com/) / StatsBomb, accessed through [`soccerdata`](https://github.com/probberechts/soccerdata).
- Physical data: EA Sports FC 24 player database (Kaggle).

Educational project — data used for non-commercial analysis only.
