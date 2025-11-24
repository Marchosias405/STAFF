# HS Code Classifier

Machine learning classifier for predicting Harmonized System (HS) codes from structured product data. Supports both CLI and web interface.

## Features

- **K-Nearest Neighbors (KNN)** and **Decision Tree (ID3)** classifiers
- **Web interface** with visual confidence bars and real-time validation
- **CLI tool** for scripting and automation
- **Rule-based flagging** for restricted items and gifts
- **Top-k predictions** with confidence scores

## Quick Start

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd <repo-name>

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Train Model

```bash
# Generate features
python -m src.features

# Train KNN model
python -m src.pipeline --dataset clean --model knn --k 3 --metric cosine

# Or train Decision Tree
python -m src.pipeline --dataset clean --model id3 --max_depth 12 --min_samples 5
```

### Web Interface

```bash
python manage.py runserver
```

Open browser to `http://localhost:8000/`

### Command Line

```bash
python -m src.cli --model artifacts/models/clean_knn_k3_cosine.joblib `
  --tags "electronics,headphones,wireless" --price 299.99 --weight 0.35 `
  --origin CA --dest US --gift 0
```

## Project Structure

```
project_root/
├── data/                       # Training data and rules
│   ├── samples_clean.csv
│   ├── hs_mapping.csv
│   ├── rules_restricted.csv
│   └── rules_gift.csv
├── src/                        # ML pipeline
│   ├── features.py            # Feature engineering
│   ├── pipeline.py            # Model training
│   ├── knn.py                 # KNN implementation
│   ├── id3.py                 # Decision tree
│   ├── rules.py               # Compliance rules
│   └── cli.py                 # Command line interface
├── config/                     # Django settings
├── web/                        # Web interface
│   ├── views.py
│   ├── forms.py
│   └── templates/
└── artifacts/                  # Trained models and outputs
    ├── models/
    ├── embeddings/
    └── preprocessor_*.joblib
```

## How It Works

### Feature Engineering

- **Price binning**: `0-25`, `25-50`, `50-100`, `100-200`, `200+`
- **Weight binning**: `0-0.25`, `0.25-0.5`, `0.5-1`, `1-2`, `2+`
- **Tag vectorization**: Bag-of-words from comma-separated tags
- **One-hot encoding**: Origin, destination, gift flag

### Models

**K-Nearest Neighbors (KNN)**

- Finds k most similar training examples
- Supports cosine and L2 distance metrics
- Returns top-k predictions with confidence scores

**Decision Tree (ID3)**

- Information gain-based splitting
- Configurable depth and minimum samples
- Explainable decision paths

### Rule-Based Validation

- Checks tags against `rules_restricted.csv` for dangerous goods
- Checks tags against `rules_gift.csv` for gift declarations
- Flags displayed in both web and CLI interfaces

## Model Performance

Metrics computed on test set:

- **Accuracy**: Exact match rate
- **Macro-F1**: Average F1 across all HS codes
- **Accuracy@3**: True label in top-3 predictions (KNN only)

Results saved to:

- `artifacts/metrics_*.json`
- `artifacts/confusion_*.png`

## API Reference

### CLI Arguments

```
--model          Path to saved model (.joblib)
--preprocessor   Path to preprocessor (optional, auto-detected)
--tags           Comma-separated tags (e.g., "electronics,laptop")
--price          Price in USD (float)
--weight         Weight in kg (float)
--origin         Origin country code (e.g., "CA", "US", "CN")
--dest           Destination country code
--gift           Gift flag (0 or 1)
--topk           Number of predictions to return (default: 3)
```

### Web Form Fields

Same as CLI arguments, presented in a user-friendly form with:

- Input validation
- Helpful placeholders
- Real-time predictions
- Visual confidence displays

## Data Format

### Training Data (samples_clean.csv)

```csv
tags,price,weight,origin,dest,gift,description,label_id
"electronics,laptop",899.99,2.5,CN,US,0,"Dell XPS 13",8471
"toy,plush",19.99,0.4,CN,CA,1,"Teddy bear",9503
```

### HS Mapping (hs_mapping.csv)

```csv
label_id,hs_code,title
8471,8471,"Automatic data processing machines"
9503,9503,"Toys, games and sports equipment"
```

### Rules (rules_restricted.csv)

```csv
term,category,notes
lithium,Dangerous Goods,Special packing/labels required
perfume,Flammable Liquid,Class 3 hazmat
```

## Deployment

**Development:**

```bash
python manage.py runserver
```

**Production:**

- Use Railway, Render, or Fly.io (persistent containers)
- Avoid Vercel (serverless incompatible with large models)
- Consider model optimization (ONNX, quantization) for faster loading

## Troubleshooting

**Template not found**

- Ensure `web/templates/predict.html` exists
- Check `web/views.py` render path matches template location

**Model not found**

- Train models first: `python -m src.features` then `python -m src.pipeline`
- Check model path in CLI/web configuration

**PicklingError on preprocessor**

- Re-run `python -m src.features` to regenerate preprocessor
- Ensure `src/tokenizers.py` contains `split_commas` at top level

**Shape mismatch**

- Rebuild features and retrain model together
- Preprocessor and model must be from same training run

## Technology Stack

- **Python 3.8+**
- **scikit-learn** - Machine learning
- **pandas** - Data processing
- **Django 4.2** - Web framework
- **Bootstrap 5** - UI components
- **joblib** - Model serialization

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here]
