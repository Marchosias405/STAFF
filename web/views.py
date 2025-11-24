import sys
from pathlib import Path
import joblib
import pandas as pd
from django.shortcuts import render
from .forms import HSPredictionForm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rules import apply_rules

# Bins from src/cli.py
_PRICE_BINS = [0, 25, 50, 100, 200, 10**9]
_PRICE_LABELS = ["0-25", "25-50", "50-100", "100-200", "200+"]
_WEIGHT_BINS = [0, 0.25, 0.5, 1, 2, 10**9]
_WEIGHT_LABELS = ["0-0.25", "0.25-0.5", "0.5-1", "1-2", "2+"]

def bin_price(v: float) -> str:
    return pd.cut([v], bins=_PRICE_BINS, labels=_PRICE_LABELS, 
                  right=True, include_lowest=True)[0]

def bin_weight(v: float) -> str:
    return pd.cut([v], bins=_WEIGHT_BINS, labels=_WEIGHT_LABELS, 
                  right=True, include_lowest=True)[0]

def load_mapping() -> dict:
    """Load HS mapping"""
    path = PROJECT_ROOT / "data" / "hs_mapping.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        cols = {c.lower().strip(): c for c in df.columns}
        if "label_id" not in cols:
            return {}
        
        lid = cols["label_id"]
        hs = cols.get("hs_code")
        title = cols.get("title")
        
        out = {}
        for _, row in df.iterrows():
            try:
                key = int(row[lid])
                pieces = []
                if hs and str(row[hs]).strip():
                    pieces.append(str(row[hs]).strip())
                if title and str(row[title]).strip():
                    pieces.append(str(row[title]).strip())
                if pieces:
                    out[key] = " — ".join(pieces)
            except:
                continue
        return out
    except:
        return {}

# Load models once (singleton pattern)
_MODEL = None
_PREPROCESSOR = None
_MAPPING = None

def get_predictor():
    """Load model and preprocessor once"""
    global _MODEL, _PREPROCESSOR, _MAPPING
    
    if _MODEL is None:
        # Find model
        model_path = PROJECT_ROOT / "artifacts" / "models" / "knn_clean.joblib"
        if not model_path.exists():
            model_dir = PROJECT_ROOT / "artifacts" / "models"
            if model_dir.exists():
                models = list(model_dir.glob("*.joblib"))
                if models:
                    model_path = models[0]
        
        # Find preprocessor
        preproc_path = PROJECT_ROOT / "artifacts" / "preprocessor_clean.joblib"
        if not preproc_path.exists():
            preproc_path = PROJECT_ROOT / "artifacts" / "preprocessor.joblib"
        
        if not model_path.exists():
            raise FileNotFoundError("Model not found. Run: python -m src.pipeline")
        if not preproc_path.exists():
            raise FileNotFoundError("Preprocessor not found. Run: python -m src.features")
        
        _MODEL = joblib.load(model_path)
        _PREPROCESSOR = joblib.load(preproc_path)
        _MAPPING = load_mapping()
    
    return _MODEL, _PREPROCESSOR, _MAPPING

def predict_view(request):
    """Main prediction view"""
    results = None
    error_msg = None
    
    if request.method == 'POST':
        form = HSPredictionForm(request.POST)
        if form.is_valid():
            try:
                model, preprocessor, mapping = get_predictor()
                
                # Prepare input
                row = pd.DataFrame([{
                    "tags": form.cleaned_data['tags'],
                    "price_bin": bin_price(form.cleaned_data['price']),
                    "weight_bin": bin_weight(form.cleaned_data['weight']),
                    "origin": form.cleaned_data['origin'].upper(),
                    "dest": form.cleaned_data['dest'].upper(),
                    "gift": int(form.cleaned_data['gift']),
                }])
                
                # Transform and predict
                X = preprocessor.transform(row)
                
                topk = form.cleaned_data['topk']
                if hasattr(model, "predict_topk"):
                    ranked = model.predict_topk(X, k_labels=topk)[0]
                else:
                    pred = int(model.predict(X)[0])
                    ranked = [(pred, 1.0)]
                
                # Apply rules
                flags = apply_rules(
                    tags=form.cleaned_data['tags'],
                    gift=form.cleaned_data['gift'],
                    hs_pred=ranked[0][0],
                    description=None
                )
                
                # Format results
                predictions = []
                for label, score in ranked:
                    predictions.append({
                        'hs_code': label,
                        'description': mapping.get(label, ''),
                        'confidence': float(score)
                    })
                
                results = {
                    'predictions': predictions,
                    'flags': flags
                }
                
            except Exception as e:
                error_msg = str(e)
    else:
        form = HSPredictionForm()
    
    return render(request, 'predict.html', {
        'form': form,
        'results': results,
        'error_msg': error_msg
    })