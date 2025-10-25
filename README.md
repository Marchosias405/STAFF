# HS Code Classifier (Classification-Only, Structured Features)

Fast, explainable HS-code predictions for Canadian shippers using **structured inputs** (tags, price, weight, origin, dest, gift). The app returns the **most likely HS category** plus **restricted/gift flags**—without digging through customs tables.

---

## 🚀 Quick Start — How to Run

> Windows PowerShell examples shown; adjust paths as needed.

### 0) Create & activate a Python env
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 1) Prepare features (build train/valid/test arrays + preprocessor)
```powershell
python -m src.features
```
**Expect:** console lines like
```
[features] Saved arrays & artifacts for 'clean' with 84 features.
```
and files under `artifacts/embeddings/` plus `artifacts/preprocessor_clean.joblib`.

### 2) Train a model (KNN or Decision Tree)
```powershell
# KNN (k=3, cosine)
python -m src.pipeline --dataset clean --model knn --k 3 --metric cosine

# or Decision Tree (ID3)
python -m src.pipeline --dataset clean --model id3 --max_depth 12 --min_samples 5
```
**Expect:** metrics printed + files saved to `artifacts/models/`, `artifacts/metrics_*.json`, and `artifacts/confusion_*.png`.

### 3) Predict for a single item (CLI)
```powershell
# Example: wireless headphones (no flags)
python -m src.cli --model artifacts/models/clean_knn_k3_cosine.joblib ^
  --tags "electronics,headphones,wireless" --price 299.99 --weight 0.35 ^
  --origin CA --dest US --gift 0
```
**Expect:** top HS predictions with raw scores and (if matched) rule flags.

### 4) Try flags (Restricted / Gift)
```powershell
# Lithium battery pack (add a lithium rule to data/rules_restricted.csv first)
python -m src.cli --model artifacts/models/clean_knn_k3_cosine.joblib ^
  --tags "electronics,battery,lithium,pack" --price 89.99 --weight 0.80 ^
  --origin CN --dest US --gift 0

# Toy gift
python -m src.cli --model artifacts/models/clean_knn_k3_cosine.joblib ^
  --tags "toy,plush,gift" --price 19.99 --weight 0.40 ^
  --origin CN --dest US --gift 1
```

### 5) Explore the data (optional)
Open and **Run All**:
```
notebooks/01_EDA.ipynb
```
**Expect:** schema checks, class counts, top tags, price/weight histograms (saved to `notebooks/figs/`).

---

## 📁 Project Layout

```
project_root/
├─ requirements.txt
├─ data/
│   ├─ samples_clean.csv
│   ├─ samples_noisy.csv
│   ├─ hs_mapping.csv
│   ├─ rules_restricted.csv
│   └─ rules_gift.csv
├─ artifacts/
│   ├─ embeddings/            # prepared arrays + indices + meta
│   ├─ models/                # saved models (.joblib)
│   ├─ preprocessor_clean.joblib / preprocessor_noisy.joblib
│   ├─ metrics_*.json
│   └─ confusion_*.png
├─ notebooks/
│   └─ 01_EDA.ipynb
└─ src/
    ├─ features.py
    ├─ tokenizers.py
    ├─ knn.py
    ├─ id3.py
    ├─ rules.py
    ├─ pipeline.py
    └─ cli.py
```

---

# 🧱 File-by-File (what, how, who uses it)

## 1) `src/features.py` — “Feature Kitchen”
**What it does (plain English):**
- Cleans your CSV (fills missing values, fixes types).
- Buckets numbers:
  - `price` → `0–25`, `25–50`, `50–100`, `100–200`, `200+`
  - `weight` → `0–0.25`, `0.25–0.5`, `0.5–1`, `1–2`, `2+`
- Turns **tags** (e.g., `electronics,headphones,wireless`) into a **bag-of-words** vector.
- One-hot encodes **origin/dest/gift** (each value → its own yes/no column).  
  Common ISO codes: **CA** (Canada), **US** (United States), **CN** (China), **JP**, **IN**, **DE**, **FR**, **MX**, **GB**, **KR**.
- Splits into **train/valid/test** and saves arrays to `artifacts/embeddings/`.
- Saves dataset-specific **preprocessor** so the CLI can transform new rows **exactly like training**:
  - `artifacts/preprocessor_clean.joblib`
  - `artifacts/preprocessor_noisy.joblib`

**How others use it:**
- `src/pipeline.py` loads saved arrays to train/evaluate.
- `src/cli.py` loads the preprocessor to transform a single input.

**Tiny example (one row → numbers):**
```
Input:
  tags="electronics,headphones,wireless"
  price=299.99 → price_bin=200+
  weight=0.35  → weight_bin=0.25-0.5
  origin=CA, dest=US, gift=0

Output (partial 0/1 switches):
  tags:electronics=1, tags:headphones=1, tags:wireless=1,
  price_bin:200+=1, weight_bin:0.25-0.5=1,
  origin:CA=1, dest:US=1, gift:0=1, ...
```

**How to use it (steps & what to expect):**
```powershell
python -m src.features
```
Expect console messages and files under `artifacts/embeddings/` + `artifacts/preprocessor_clean.joblib`.

---

## 2) `src/tokenizers.py` — “Tag Splitter”
**Purpose:** Split comma-separated tags into clean tokens your vectorizer can count.

**Example:**
```python
split_commas("electronics, headphones, wireless")
# → ["electronics", "headphones", "wireless"]
```

**Why this file exists (pickling in one sentence):**  
When we save the preprocessor with `joblib.dump`, Python “pickles” it. Python can only **unpickle** functions it can find by name, so putting `split_commas` at module top-level makes it loadable later when `src/cli.py` restores the preprocessor.

**Who uses it:** `features.py` via `CountVectorizer(tokenizer=split_commas, ...)`.

---

## 3) `src/knn.py` — “K-Nearest Neighbors”
**What it does:**
- Stores **training feature vectors + HS labels**.
- For a new item, finds the **k most similar** training rows using:
  - `cosine` (angle similarity; **higher = closer**) or
  - `l2` (Euclidean/L2 distance; **lower = closer**).
- **Votes by label** (closer neighbors weigh more).
- Returns **Top-k labels + raw weights** (what the CLI prints).

**How others use it:**
- `src/pipeline.py` trains/evaluates KNN.
- `src/cli.py` loads the saved KNN to predict.

### 🎯 KNN step-by-step (toy demo)
**Feature columns** (simplified):
```
tags:electronics, tags:headphones, tags:wireless, tags:battery,
price_bin:200+, price_bin:50-100,
weight_bin:0.25-0.5, weight_bin:1-2,
origin:CA, origin:CN, dest:US, gift:0, gift:1
```

**Training items remembered:**
- **T1 → HS 8518** (headphones; CA→US; 200+; 0.25–0.5)  
  `[1,1,1,0,  1,0,  1,0,  1,0, 1,1,0]`
- **T2 → HS 8507** (battery; CN→US; 50–100; 1–2)  
  `[1,0,0,1,  0,1,  0,1,  0,1, 1,1,0]`
- **T3 → HS 8518** (headphones; CN→US; 50–100; 0.25–0.5)  
  `[1,1,0,0,  0,1,  1,0,  0,1, 1,1,0]`

**New item Q (your CLI input becomes):**
```
tags="electronics,headphones,wireless"
price=299.99 → price_bin=200+
weight=0.35  → weight_bin=0.25-0.5
origin=CA, dest=US, gift=0

Q = [1,1,1,0,  1,0,  1,0,  1,0, 1,1,0]
```

**Similarity (intuitive overlap for demo):**
- Q vs T1 → 10 matches
- Q vs T2 → 4 matches
- Q vs T3 → 7 matches

**k=3 vote (sum weights by label):**
- HS **8518**: 10 + 7 = **17**
- HS **8507**: 4

**Top predictions (CLI-style):**
```
HS 8518 (score=17), HS 8507 (score=4)
```
> In real runs, scores are cosine/distance-weighted, but the **voting logic is the same**.

---

## 4) `src/id3.py` — “Decision Tree (ID3)”
**What it does:**
- Builds a tree of **yes/no** questions using **entropy / information gain**.
- Splits until leaves are pure (or until depth/sample limits).
- Can **explain** the decision path for one row.

**How others use it:**
- `src/pipeline.py` trains/evaluates ID3.
- `src/cli.py` loads the saved tree; returns a single HS (Top-1).

### 🌳 ID3 step-by-step (beginner-friendly)
**Mini training table:**

| tags                     | price_bin | weight_bin | origin | dest | gift | HS |
|-------------------------|-----------|------------|--------|------|------|----|
| electronics, headphones | 200+      | 0.25–0.5   | CA     | US   | 0    | 8518 |
| electronics, battery    | 50–100    | 1–2        | CN     | US   | 0    | 8507 |
| toy, gift               | 0–25      | 0.5–1      | CN     | US   | 1    | 9503 |
| electronics, headphones | 100–200   | 0.25–0.5   | CN     | US   | 0    | 8518 |
| electronics, battery    | 25–50     | 1–2        | CN     | US   | 0    | 8507 |

**Possible splits learned:**
```
Root:
 ├─ (gift = 1) → HS 9503
 └─ (gift = 0)
       ├─ (tags:battery = 1) → HS 8507
       └─ (tags:headphones = 1) → HS 8518
```

**Predict a new headphones item (gift=0):**
- gift=1? → No  
- tags:battery=1? → No  
- tags:headphones=1? → Yes → **HS 8518**

---

## 5) `src/pipeline.py` — “Trainer & Evaluator”
**What it does:**
- Loads arrays from `artifacts/embeddings/` (made by `features.py`).
- Trains **KNN** or **ID3** with your parameters.
- Computes and saves:
  - **Model** → `artifacts/models/{dataset}_{model}.joblib`
  - **Metrics** → `artifacts/metrics_{dataset}_{model}.json`
  - **Confusion plots** → `artifacts/confusion_*.png`

**Metrics (plain English):**
- **Accuracy** — % of exact matches (e.g., 3/4 → **75%**).
- **Macro-F1** — average F1 across **each HS label** (treats rare labels fairly).
- **Accuracy@3** (KNN only) — % of cases where the **true HS** is in the **top-3** guesses.

**How to use it (steps & what to expect):**
```powershell
python -m src.pipeline --dataset clean --model knn --k 3 --metric cosine
```
Expect printed metrics + saved model/metrics/plots under `artifacts/`.

---

## 6) `src/cli.py` — “Predict for One Item”
**What it does:**
1. Reads your inputs: `--tags`, `--price`, `--weight`, `--origin`, `--dest`, `--gift`.
2. **Bins** numeric fields like training.
3. Auto-selects the matching **preprocessor** (e.g., `preprocessor_clean.joblib` if the model path starts with `clean_`).
4. Transforms the row → numeric vector (same layout as training).
5. Loads the **model** and predicts:
   - KNN → **Top-k** labels with **raw weights**.
   - ID3 → **Top-1** label.
6. Calls `rules.py` to add **Restricted/Gift** flags.

**How to use it (steps & what to expect):**
```powershell
# Headphones (no flags)
python -m src.cli --model artifacts/models/clean_knn_k3_cosine.joblib ^
  --tags "electronics,headphones,wireless" --price 299.99 --weight 0.35 ^
  --origin CA --dest US --gift 0
```
Expect: model & preprocessor names, **Top predictions** with HS codes + scores, and (if matched) **Flags**.

---

## 7) `src/rules.py` — “Restricted & Gift Flags”
**What it does:**
- Loads:
  - `data/rules_restricted.csv` (e.g., “lithium” → Dangerous Goods)
  - `data/rules_gift.csv` (e.g., “gift” → Exemption note)
- Checks **tags** (and optional `description`) for those keywords.
- Returns messages like:
  ```
  Restricted: Flammable Liquid (perfume) – Contains high alcohol content – treated as Class 3 flammable (hazmat)
  ```

**Rules CSV examples (headers required):**
```
# data/rules_restricted.csv
term,category,notes
lithium,Dangerous Goods,Special packing/labels for air
perfume,Flammable Liquid,Contains high alcohol content – treated as Class 3 flammable (hazmat)

# data/rules_gift.csv
term,category,notes
gift,Exemption,Check threshold/personal use criteria
present,Exemption,Synonym for gift; may imply non-commercial gift
```

---

## 8) `notebooks/01_EDA.ipynb` — “Explore the Data”
**What it does:**
- Checks schema/missing values.
- Shows class counts per `label_id`.
- Lists top tags.
- Plots price/weight distributions.
- (Optional) Joins `label_id` to human titles via `hs_mapping.csv`.

**How to use it (steps & what to expect):**
1. Open `notebooks/01_EDA.ipynb`.
2. **Run All**.
3. Expect printed checks + charts saved to `notebooks/figs/`.

---

## 9) `data/*.csv` — “Your Inputs”
- `samples_clean.csv` / `samples_noisy.csv` → training data with `label_id` (HS).
- `hs_mapping.csv` → optional mapping from label_id → HS titles (ensure proper quoting).
- `rules_restricted.csv` / `rules_gift.csv` → keyword rules for flags.

---

## 🔟 `artifacts/*` — “Outputs”
- `embeddings/` → saved feature arrays + split indexes + meta.
- `models/` → trained models (`*.joblib`).
- `preprocessor_clean.joblib`, `preprocessor_noisy.joblib` → how to transform new rows.
- `metrics_*.json`, `confusion_*.png` → evaluation reports.

---

## 🔄 End-to-End Flow

```
data/*.csv
   ↓
features.py   → cleans + buckets + one-hot + tag vectors
   ↓           saves arrays + preprocessor_{dataset}.joblib
pipeline.py  → trains KNN/ID3, saves model + metrics (Accuracy, Macro-F1, Accuracy@3) + plots
   ↓
cli.py       → loads preprocessor + model → predicts HS Top-k/Top-1
   ↓
rules.py     → adds Restricted/Gift notes (if tags match)
   ↓
Console      → HS codes + raw weights + compliance flags
```

---

## 🧰 Troubleshooting

- **PicklingError / can’t load preprocessor**  
  Make sure `split_commas` lives in `src/tokenizers.py` (top-level), and re-run:
  ```
  python -m src.features
  ```

- **“columns are missing: {'price_bin','weight_bin'}”**  
  Your CLI must **bin** fields exactly like training. Use the provided `cli.py`.

- **Shape mismatch (feature count differs)**  
  Rebuild features & retrain so preprocessor and model match:
  ```
  python -m src.features
  python -m src.pipeline --dataset clean --model knn --k 3 --metric cosine
  ```
