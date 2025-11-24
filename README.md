# HS Code Predictor

A small machine-learning + rule-based system that predicts HS (Harmonized System) tariff codes using:

- Product tags / description
- Price (USD)
- Weight (kg)
- Origin country
- Destination country
- Gift flag

It also applies **business rules** for gifts 

---

## 1. Project Structure

```text
Root/
├─ .venv/                     
├─ artifacts/
│  ├─ embeddings/             
│  ├─ models/
│  │  ├─ clean_knn_k3_cosine.joblib
│  │  ├─ clean_id3_d12_m5.joblib
│  │  └─ other variants
│  ├─ preprocessor_clean.joblib
│  ├─ preprocessor_noisy.joblib
│  ├─ preprocessor.joblib
│  ├─ metrics_*.json
│  └─ confusion_*.png
├─ config/
│  ├─ settings.py
│  ├─ urls.py
│  └─ wsgi.py
├─ data/
│  ├─ samples_clean.csv
│  ├─ samples_noisy.csv
│  ├─ cleanEmbeddings.csv
│  ├─ noisyEmbeddings.csv
│  ├─ hs_mapping.csv
│  ├─ rules_gift.csv
│  └─ rules_restricted.csv
├─ notebooks/
│  ├─ 01_EDA.ipynb
│  └─ 02_Evaluation.ipynb
├─ src/
│  ├─ cli.py
│  ├─ features.py
│  ├─ pipeline.py
│  ├─ knn.py
│  ├─ id3.py
│  ├─ rules.py
│  └─ tokenizers.py
├─ web/
│  ├─ app.py
│  ├─ forms.py
│  ├─ urls.py
│  ├─ views.py
│  └─ templates/
│     └─ predict.html
├─ manage.py
├─ requirements.txt
└─ README.md
```

## 2. Installation

### 2.1 Create & activate virtual environment (Windows PowerShell)
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2.2 Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Data & Features

### 3.1 Source data

- `samples_clean.csv`
- `samples_noisy.csv`
- `hs_mapping.csv`
- `rules_gift.csv`
- `rules_restricted.csv`

### 3.2 Price & weight binning
Used both in training and web app.

```python
_PRICE_BINS = [0, 25, 50, 100, 200, 10**9]
_PRICE_LABELS = ["0-25", "25-50", "50-100", "100-200", "200+"]

_WEIGHT_BINS = [0, 0.25, 0.5, 1, 2, 10**9]
_WEIGHT_LABELS = ["0-0.25", "0.25-0.5", "0.5-1", "1-2", "2+"]
```

## 4. Feature Generation

Run:
```bash
python -m src.features
```

This:
- Loads clean + noisy samples
- Generates text embeddings
- Adds price/weight bins
- Encodes origin, dest, gift
- Saves preprocessors + embeddings

## 5. Model Training

### 5.1 Train main KNN model
```bash
python -m src.pipeline --dataset clean --model knn --k 3 --metric cosine
```

Saves:
- `clean_knn_k3_cosine.joblib`
- metrics JSON
- confusion matrix images  

### 5.2 Optional experiments
```bash
python -m src.pipeline --dataset clean --model knn --k 10 --metric cosine
python -m src.pipeline --dataset clean --model id3 --max_depth 12 --min_samples 5
```

## 6. Running the Web App

```bash
python manage.py runserver
```



## 7. Demo Workflow

```bash
python -m src.features
python -m src.pipeline --dataset clean --model knn --k 3 --metric cosine
python manage.py runserver
```

Example input:
- makeup
- Price: 50
- Weight: 0.1
- Origin: CA
- Destination: US
- Gift: yes

Output:
- Ranked HS predictions with similarity
- Gift / restricted flags

## 8. How the System Works

- Text → embeddings  
- Structured → one-hot  
- KNN (cosine) → top-k labels  
- Map via `hs_mapping.csv`  
- Rule engine parses gift terms  

## 9. Limitations & Improvements
- Limited synthetic data
- No material composition detection
- No country-specific overrides  
- Top‑3 accuracy high on synthetic test  
- Rule engine effective for gifts & hazards  
- Future: transformer model integration  

## 10. Implementation Notes & File-Level Changes

### 10.1 web/views.py
- Added root resolver
- Added binning helpers
- Implemented `load_mapping()`
- Implemented `get_predictor()`

### 10.2 src/rules.py
- Loads rule CSVs
- Applies keyword-based tagging

### 10.3 src/features.py
- Ensured consistent bins + columns

### 10.4 src/pipeline.py
- Enforced model naming convention


## 11. Troubleshooting

### Missing model
Run:
```bash
python -m src.features
python -m src.pipeline --dataset clean --model knn --k 3 --metric cosine
```

### Missing preprocessor
```bash
python -m src.features
```

### ModuleNotFoundError
Run commands from project root (where manage.py is).

---

