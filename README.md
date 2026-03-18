# march_madness

# March Madness Simulation Suite — Documentation

This document covers three main workflows: data ingestion, simulation execution, and visualization.

---

## Part 1: Data Ingestion Pipeline

The data ingestion pipeline transforms raw web data into trained machine learning models.

### Step 1: Extract Web Data → `extract_web_data.py`

**Purpose:** Scrape NCAA tournament data from the web and store it in a raw format.

**Input:** External data sources (URLs configured in the script)

**Output:** Raw CSV files in the `data/` directory

**Usage:**
```bash
python extract_web_data.py
```

**Key outputs:**
- `ncaam.csv` — Full NCAA men's basketball dataset
- Year-specific files (e.g., `data/full/cbb25.csv`)

---

### Step 2: Data Preparation → `data_prep.py`

**Purpose:** Clean and normalize raw data for use in model training.

**Input:** Raw CSV files from `extract_web_data.py`

**Output:** Cleaned datasets with standardized formats

**Usage:**
```bash
python data_prep.py --year 2025
```

**What it does:**
- Handles missing values
- Standardizes column names and data types
- Creates region assignments for tournament teams
- Generates training-ready datasets

**Key outputs:**
- `master_ncaa.csv` — Master dataset with all teams and their stats
- Files in `data/training/` directory for model training

---

### Step 3: Prepare Training Data → `prep_training.py`

**Purpose:** Split and format data specifically for model training.

**Input:** Cleaned data from `data_prep.py`

**Output:** Train/test splits and feature-engineered datasets

**Usage:**
```bash
python prep_training.py --year 2025
```

**What it does:**
- Splits data by tournament round (R64, R32, Sweet 16, Elite 8, Final Four)
- Creates feature sets: `big` (large upsets), `little` (mid-tier), `comp` (competitive)
- Scales features using standardization
- Saves scaler object for later use

**Key outputs:**
- `data/training/w1.csv`, `w2.csv`, `ff.csv` — Games by week
- `data/training/big.csv`, `little.csv`, `comp.csv` — Games by seed differential
- `models/my_scaler.pkl` — Fitted scaler for model predictions

---

### Step 4: Fit Models → `fit_models.py`

**Purpose:** Train machine learning models on the prepared data.

**Input:** Training datasets from `prep_training.py`

**Output:** Trained model objects ready for simulation

**Usage:**
```bash
python fit_models.py --year 2025
```

**What it trains:**
- **Master model** — Single model predicting all rounds
- **Weekly models** (w1, w2, ff) — Week-specific models for R64–R32, Sweet 16–Elite 8, Final Four
- **Seed differential models** (big, little, comp) — Models tuned to different seed gaps

**Supported model types:**
- `knn` — K-Nearest Neighbors
- `DT` — Decision Tree
- `forest` — Random Forest
- `mlp` — Multi-Layer Perceptron (Neural Network)
- `clf` — Classifier (Logistic Regression)
- `gnb` — Gaussian Naive Bayes
- `svc` — Support Vector Classifier

**Key outputs:**
- `models/{MODEL_TYPE}/{NAME}.pkl` — Trained model files
- Example: `models/knn/master.pkl`, `models/forest/w1.pkl`

---

## Part 2: Running Simulations

Once models are trained, you can run simulations to predict tournament outcomes. Three scripts support different use cases.

### Script 1: Full Tournament Simulation → `run_single_simulation.py`

**Purpose:** Simulate one complete NCAA tournament bracket from Round of 64 to Champion.

**Input:** Year and model selections

**Output:** JSON file with all rounds' results

**Usage:**
```bash
# Using master model
python run_single_simulation.py --year 2025 --model knn

# Using weekly models
python run_single_simulation.py --year 2025 --sim_type weeks --w1_model knn --w2_model forest --ff_model mlp

# Using seed differential models
python run_single_simulation.py --year 2025 --sim_type seed_diff --big_model knn --little_model forest --comp_model clf
```

**Arguments:**
- `--year` (default: 2025) — Tournament year
- `--sim_type` (default: master) — Simulation type: `master`, `weeks`, or `seed_diff`
- `--model` — Model for master mode (knn, DT, forest, mlp, clf, gnb, svc)
- `--w1_model`, `--w2_model`, `--ff_model` — Models for weeks mode
- `--big_model`, `--little_model`, `--comp_model` — Models for seed_diff mode

**Output location:**
- `Sims/{YEAR}/{FILENAME}_{TIMESTAMP}.json`

**JSON structure:**
```json
{
  "64": [ {"TEAM": "Duke", "SEED": 1, "region": "east"}, ... ],
  "32": [ {"TEAM": "Duke", "SEED": 1, "region": "east"}, ... ],
  "16": [ ... ],
  "8": [ ... ],
  "4": [ ... ],
  "2": [ ... ],
  "1": [ {"TEAM": "Duke", "SEED": 1, "region": "east"} ]
}
```

---

### Script 2: Single Game Prediction → `single_game_simulation.py`

**Purpose:** Predict the outcome of a single matchup between two teams.

**Input:** Two team names and optional model configuration

**Output:** Prediction with higher/lower seed designation

**Usage:**
```bash
# Basic usage (defaults to knn master model)
python single_game_simulation.py "Duke" "Kansas"

# With specific model
python single_game_simulation.py "Duke" "Kansas" --year 2025 --model forest

# Using different sim types
python single_game_simulation.py "Duke" "Kansas" --sim_type weeks --w1_model knn --w2_model forest --ff_model clf
```

**Output example:**
```
============================================================
SINGLE GAME SIMULATION
============================================================
Year: 2025
Sim Type: master
Model: knn
============================================================

1 (Higher Seed) Duke
vs.
16 (Lower Seed) Kansas

WINNER: 1 Duke
============================================================

✓ (Expected) Higher seed 1 Duke wins
```

**Arguments:**
- `team1`, `team2` — Team names (case-insensitive)
- `--year` (default: 2025) — Tournament year
- `--sim_type` (default: master) — `master`, `weeks`, or `seed_diff`
- Model selection arguments (same as `run_single_simulation.py`)

---

### Script 3: Batch Simulations → `batch_simulation.py`

**Purpose:** Run multiple tournament simulations across different model combinations.

**Input:** Sim type and optional model filters

**Output:** Multiple JSON files, one per model permutation

**Usage:**
```bash
# Run all master model permutations (7 models)
python batch_simulation.py master --year 2025

# Run all weeks model permutations (7^3 = 343 combinations)
python batch_simulation.py weeks --year 2025

# Run all seed_diff permutations
python batch_simulation.py seed_diff --year 2025

# Filter to specific models
python batch_simulation.py master --year 2025 --models knn forest
```

**Arguments:**
- `sim_type` — Required: `master`, `weeks`, or `seed_diff`
- `--year` (default: 2025) — Tournament year
- `--models` — Optional: space-separated list of models to use (default: all 7)

**Output location:**
- `Sims/{YEAR}/{FILENAME}_{TIMESTAMP}.json`
- Multiple files generated (one per combination)

**Example outputs:**
- `master_knn_1734567890.json`
- `weeks_knn_knn_forest_1734567891.json`
- `seed_diff_forest_knn_mlp_1734567892.json`

---

## Part 3: Visualization & Analysis

Two dashboards visualize the simulation results with different purposes.

### Dashboard 1: Pre-Tournament Analysis → `mm_dashboard.py`

**Purpose:** Evaluate simulation predictions **before** the tournament starts.

**Usage:**
```bash
streamlit run mm_dashboard.py
```

**Features:**

#### Overview Metrics
- Total simulations loaded
- Number of teams
- Most likely champion
- Champion win probability

#### Tab 1: By Region (R64 to Final Four)
- **Region filter** (sidebar, required) — Select one of four regions
- **Seed range filter** (sidebar) — Filter by seed (1–16)
- Shows which teams reach each round in your selected region
- **Data display:** Table with teams/seed counts by round
- **Color map:** Heatmap showing frequency of team advancement
- **Format:** "1. Duke", "2. Kansas", etc.
- **Ordering:** Sorted by seed ascending in visualization

#### Tab 2: Finals (Championship Game & Winner)
- Shows all teams appearing in Championship Game and Finals
- No region filter needed
- **Data display:** Table and heatmap
- Same formatting and sorting as Tab 1
- **Color scale:** Purple (to differentiate from Tab 1)

#### Sidebar Controls
- Region selector (one region required for Tab 1)
- Seed range slider (1–16)
- Simulation statistics

---

### Dashboard 2: Post-Tournament Evaluation → `WAE_dash.py`

**Purpose:** Analyze actual tournament results **after** the event concludes.

**Usage:**
```bash
streamlit run WAE_dash.py
```

**Features:**

#### Key Metrics
- Actual champion
- Model predictions vs. actual results
- Accuracy by round
- Upset statistics

#### Analysis Views
- Regional performance breakdown
- Seed-based predictions vs. outcomes
- Model comparison (which models were most accurate)
- Game-by-game prediction evaluation

#### Post-Tournament Evaluation
- Highlights correct predictions
- Identifies major upsets
- Compares different model types' accuracy
- Shows prediction confidence by round

---

## Quick Reference

### Full Workflow

```bash
# 1. Ingest and prepare data
python extract_web_data.py
python data_prep.py --year 2025
python prep_training.py --year 2025
python fit_models.py --year 2025

# 2. Run simulations
python run_single_simulation.py --year 2025 --model knn          # Full tournament
python single_game_simulation.py "Duke" "Kansas"                 # Single game
python batch_simulation.py master --year 2025                    # All models

# 3. Visualize before tournament
streamlit run mm_dashboard.py

# 4. After tournament, evaluate
streamlit run WAE_dash.py
```

### Model Selection Guide

| Use Case | Sim Type | Command |
|----------|----------|---------|
| Quick single tournament | master | `--model knn` |
| Pre-tournament analysis | master | `--model forest` (better for complex patterns) |
| Round-specific tuning | weeks | `--w1_model knn --w2_model forest --ff_model mlp` |
| Seed-based predictions | seed_diff | `--big_model forest --little_model knn --comp_model clf` |
| Compare all models | batch | `python batch_simulation.py master` |

### File Organization

```
.
├── extract_web_data.py          # (1) Web scraping
├── data_prep.py                 # (2) Data cleaning
├── prep_training.py             # (3) Train/test split
├── fit_models.py                # (4) Model training
├── run_single_simulation.py      # Simulation: full tournament
├── single_game_simulation.py     # Simulation: one game
├── batch_simulation.py           # Simulation: batch models
├── mm_dashboard.py              # Pre-tournament dashboard
├── WAE_dash.py                  # Post-tournament dashboard
├── data/                        # Data storage
│   ├── full/                    # Full datasets (cbb13.csv, cbb14.csv, ...)
│   ├── training/                # Training-ready data
│   └── ...
├── models/                      # Trained models
│   ├── knn/
│   ├── forest/
│   ├── mlp/
│   └── ...
└── Sims/{YEAR}/                 # Simulation outputs (JSON)
```

---

## Troubleshooting

**Q: Team name not found in single game simulation**
- A: Use the exact team name as it appears in the dataset. Names are case-insensitive, but spelling must match.

**Q: Which model should I use?**
- A: Start with `knn` for speed, `forest` for accuracy, or run `batch_simulation.py` to compare all models.

**Q: How do I know if my models are good?**
- A: Run simulations, then use `mm_dashboard.py` to evaluate the predictions before the tournament, and `WAE_dash.py` after to see actual accuracy.

**Q: Can I run multiple simulations at once?**
- A: Yes, use `batch_simulation.py`. It will generate separate JSON files for each model combination.

