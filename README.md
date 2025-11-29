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

# 12 Running The Web Interface

The web interface provides a user-friendly alternative to the command-line predictor, featuring visual confidence displays and real-time validation.

## 12.1 Launch the Web Server

From the project root directory:

```bash
python manage.py runserver
```

Expected output:

```
Performing system checks...

System check identified no issues (0 silenced).
Starting development server at http://127.0.0.1:8000/
Quit the server with CTRL-BREAK.
```

## 12.2 Access the Interface

Open your web browser and navigate to:

```
http://localhost:8000/
```

You will see a dark-themed interface with two panels:

- **Left panel**: Product information form
- **Right panel**: Prediction results (empty until first submission)

## 12.3 Demo Workflow: Basic Prediction

### Example 1: Electronics (Headphones)

**Scenario**: Predicting HS code for sweater shipped from China to the US.

1. Fill in the form:

   - **Product Tags**: `sweater`
   - **Price (USD)**: `80`
   - **Weight (kg)**: `0.8`
   - **Origin Country**: `CN`
   - **Destination Country**: `US`
   - **Is this a gift?**: Unchecked
   - **Number of Predictions**: `3`

2. Click **"Predict HS Code"**

3. **Expected Results** (right panel):
   - Top prediction highlighted in green with trophy badge
   - HS code displayed (e.g., HS 6109)
   - Description: "T-shirts and vests, knitted or crocheted"
   - Confidence score
   - Additional predictions ranked by confidence

### Example 2: Gift Item with Validation Flag

**Scenario**: Predicting HS code for iPhone that may trigger gift-related validation.

1. Fill in the form:

   - **Product Tags**: `iphone`
   - **Price (USD)**: `1000`
   - **Weight (kg)**: `0.40`
   - **Origin Country**: `CA`
   - **Destination Country**: `US`
   - **Is this a gift?**: Checked
   - **Number of Predictions**: `3`

2. Click **"Predict HS Code"**

3. **Expected Results**:
   - Top prediction: `HS 8517` (Telephone sets including cell phones)
   - **Validation Flags section** appears below predictions:
     - Yellow warning box with flag icon
     - Message: "Gift: Exemption (gift) – Indicates shipment is a personal gift which may qualify for duty or tax exemption under a value threshold “
       “Gift: Exemption (present) – Synonym for gift used in descriptions of personal shipments”
       “Gift: Exemption (birthday) – Marks shipment as a birthday gift in a non commercial context"

## 12.4 Understanding the Results

### Prediction Display Elements

1. **Top Prediction Badge**:

   - Green border
   - Indicates the most confident prediction

2. **HS Code**:

   - Large monospace font (e.g., `HS 8518`)
   - Color-coded: green for top, blue for others

3. **Description**:

   - Human-readable title from `data/hs_mapping.csv`
   - Falls back to label ID if mapping unavailable

4. **Confidence Score**:

   - Displayed as score
   - Higher scores indicate stronger match

5. **Validation Flags** (if triggered):
   - **Yellow boxes**: Warnings (gift declarations, special handling)

## 12.5 Stopping the Server

Press `CTRL+C` in the terminal running the server:

```
^C
Quit the server with CTRL-BREAK.
```

---

# Demo Examples

> Note: Due to dataset noise, these examples work best when using the clean (labelled) data, predictions are off for noisy data or data that's not in the training set. See the features pdf to see exactly why.

### 1. iPhone – Perfect Electronics Example

- **Product Tags:** electronics, phone, smartphone, brand_apple
- **Price:** 999
- **Origin:** CA
- **Weight:** 0.40
- **Destination:** US
- **Gift:** No
- **Expected:**
  - HS Code: **8517**
  - No flags

### 2. Sony WH-1000XM4 – Audio Category

- **Product Tags:** electronics, audio, headphones, wireless, noise_cancelling
- **Price:** 349
- **Origin:** CA
- **Weight:** 0.30
- **Destination:** US
- **Gift:** No
- **Expected:**
  - HS Code: **8518**

### 3. Power Bank – Restricted Lithium Battery

- **Product Tags:** electronics, battery, lithium, rechargeable, cell
- **Price:** 15
- **Origin:** CA
- **Weight:** 0.06
- **Destination:** US
- **Gift:** No
- **Expected:**
  - HS Code: **8507**
  - Restricted Flag: _Lithium battery (Dangerous Goods)_

### 4. Men's T-Shirt – Gift Exemption Example

- **Product Tags:** apparel, shirt, tshirt, cotton, men, clothing
- **Price:** 20
- **Origin:** CA
- **Weight:** 0.20
- **Destination:** US
- **Gift:** Yes
- **Expected:**
  - HS Code: **6109**
  - Gift exemption flag

### 5. Running Shoes – Footwear

- **Product Tags:** footwear, shoes, sneakers, running, textile_upper
- **Price:** 95
- **Origin:** CA
- **Weight:** 0.80
- **Destination:** US
- **Gift:** No
- **Expected:**
  - HS Code: **6404**

### 6. Chocolate Gift Box – Food + Gift

- **Product Tags:** food, chocolate, cocoa, snack, gift_item
- **Price:** 29
- **Origin:** CA
- **Weight:** 0.55
- **Destination:** US
- **Gift:** Yes
- **Expected:**
  - HS Code: **1806**
  - Gift exemption flag

### 7. Cordless Drill – Tools Category

- **Product Tags:** tools, power_tool, drill, cordless
- **Price:** 219
- **Origin:** CA
- **Weight:** 2.10
- **Destination:** US
- **Gift:** No
- **Expected:**
  - HS Code: **8467**

### 8. Rechargeable Flashlight – Restricted

- **Product Tags:** electronics, battery, flashlight, usb_c, rechargeable
- **Price:** 26
- **Origin:** CA
- **Weight:** 0.18
- **Destination:** US
- **Gift:** No
- **Expected:**
  - HS Code: **8507**
  - Restricted Flag: _Lithium battery_

### 9. Lip Gloss – Misfire Example (NLP Limitation)

- **Product Tags:** cosmetics, makeup, lip, gloss
- **Price:** 22
- **Origin:** CA
- **Weight:** 0.10
- **Destination:** US
- **Gift:** No
- **Expected:**
  - Often misclassified
  - Demonstrates need for a transformer-based model

### 10. Wooden Dining Table – Misfire Example

- **Product Tags:** furniture, table, wood, home, dining
- **Price:** 299
- **Origin:** CA
- **Weight:** 18.00
- **Destination:** US
- **Gift:** No
- **Expected:**
  - Sometimes misclassified
  - Shows sparse category + tag limitations

# Contributions

Kazi Boni Amin: 1/3,
Ertugrul Yurtseven: 1/3,
Brayden Yee: 1/3
