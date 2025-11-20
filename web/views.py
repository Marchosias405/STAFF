import sys
from pathlib import Path

import joblib
import pandas as pd
from django.shortcuts import render

from .forms import HSPredictionForm

# ----------------------------------------------------------------------
# Project root (…/310 M2 V2/)
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rules import apply_rules

# ----------------------------------------------------------------------
# Binning helpers (same as CLI)
# ----------------------------------------------------------------------
_PRICE_BINS = [0, 25, 50, 100, 200, 10**9]
_PRICE_LABELS = ["0-25", "25-50", "50-100", "100-200", "200+"]

_WEIGHT_BINS = [0, 0.25, 0.5, 1, 2, 10**9]
_WEIGHT_LABELS = ["0-0.25", "0.25-0.5", "0.5-1", "1-2", "2+"]


def bin_price(v: float) -> str:
    return pd.cut(
        [v],
        bins=_PRICE_BINS,
        labels=_PRICE_LABELS,
        right=True,
        include_lowest=True,
    )[0]


def bin_weight(v: float) -> str:
    return pd.cut(
        [v],
        bins=_WEIGHT_BINS,
        labels=_WEIGHT_LABELS,
        right=True,
        include_lowest=True,
    )[0]


# ----------------------------------------------------------------------
# HS mapping
# ----------------------------------------------------------------------
def load_mapping() -> dict:
    """Load HS mapping from data/hs_mapping.csv."""
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
            except Exception:
                continue
        return out
    except Exception:
        return {}


# ----------------------------------------------------------------------
# Lazy-loaded model + preprocessor
# ----------------------------------------------------------------------
_MODEL = None
_PREPROCESSOR = None
_MAPPING = None


def get_predictor():
    """Load model and preprocessor once (singleton-style)."""
    global _MODEL, _PREPROCESSOR, _MAPPING

    if _MODEL is None:
        # Use your trained KNN model (k=3, cosine)
        model_path = PROJECT_ROOT / "artifacts" / "models" / "clean_knn_k3_cosine.joblib"
        # Correct "clean" preprocessor
        preproc_path = PROJECT_ROOT / "artifacts" / "preprocessor_clean.joblib"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run: python -m src.pipeline"
            )

        if not preproc_path.exists():
            raise FileNotFoundError(
                f"Preprocessor not found at {preproc_path}. Run: python -m src.features"
            )

        _MODEL = joblib.load(model_path)
        _PREPROCESSOR = joblib.load(preproc_path)
        _MAPPING = load_mapping()

    return _MODEL, _PREPROCESSOR, _MAPPING


# ----------------------------------------------------------------------
# Main prediction view
# ----------------------------------------------------------------------
def predict_view(request):
    """Main prediction view."""
    results = None
    error_msg = None

    if request.method == "POST":
        form = HSPredictionForm(request.POST)
        if form.is_valid():
            try:
                model, preprocessor, mapping = get_predictor()

                tags = form.cleaned_data["tags"]

                # Build single-row DataFrame input
                row = pd.DataFrame(
                    [
                        {
                            "tags": tags,
                            "description": tags,  # simple proxy so the column exists
                            "price_bin": bin_price(form.cleaned_data["price"]),
                            "weight_bin": bin_weight(form.cleaned_data["weight"]),
                            "origin": form.cleaned_data["origin"].upper(),
                            "dest": form.cleaned_data["dest"].upper(),
                            "gift": int(form.cleaned_data["gift"]),
                        }
                    ]
                )

                # Transform and predict
                X = preprocessor.transform(row)
                topk = form.cleaned_data["topk"]

                if hasattr(model, "predict_topk"):
                    ranked = model.predict_topk(X, k_labels=topk)[0]
                else:
                    pred = int(model.predict(X)[0])
                    ranked = [(pred, 1.0)]

                # Apply rules based on top prediction
                flags = apply_rules(
                    tags=form.cleaned_data["tags"],
                    gift=form.cleaned_data["gift"],
                    hs_pred=ranked[0][0],
                    description=None,
                )

                # Format predictions for template
                predictions = []
                for label, score in ranked:
                    full = mapping.get(
                        label, ""
                    )  # e.g. "3304 — Beauty or make-up preparations and skin care"

                    hs_code = ""
                    title = full
                    if full:
                        parts = [p.strip() for p in full.split("—", 1)]
                        if len(parts) == 2:
                            hs_code, title = parts
                        else:
                            hs_code = full

                    predictions.append(
                        {
                            "label_id": int(label),  # internal id (0, 7, 16…)
                            "hs_code": hs_code,  # real HS code ("3304")
                            "title": title,  # human-readable title
                            "full_text": full,  # full mapping line (optional)
                            "confidence": float(score),
                        }
                    )

                results = {"predictions": predictions, "flags": flags}

            except Exception as e:
                error_msg = str(e)
    else:
        form = HSPredictionForm()

    return render(
        request,
        "predict.html",
        {
            "form": form,
            "results": results,
            "error_msg": error_msg,
        },
    )
