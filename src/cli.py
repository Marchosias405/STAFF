# src/cli.py
# -----------------------------------------------------------------------------
# CLI predictor: structured inputs -> Top-3 HS predictions + rule flags
# -----------------------------------------------------------------------------
import argparse
from pathlib import Path
import re
import joblib
import pandas as pd

from .rules import apply_rules

ART_DIR = Path("artifacts")
MAP_PATH = Path("data/hs_mapping.csv")  # optional

# --- bins must match features.py exactly ---
_PRICE_BINS   = [0, 25, 50, 100, 200, 10**9]
_PRICE_LABELS = ["0-25", "25-50", "50-100", "100-200", "200+"]
_WEIGHT_BINS   = [0, 0.25, 0.5, 1, 2, 10**9]
_WEIGHT_LABELS = ["0-0.25", "0.25-0.5", "0.5-1", "1-2", "2+"]

def bin_price(v: float) -> str:
    return pd.cut([v], bins=_PRICE_BINS, labels=_PRICE_LABELS, right=True, include_lowest=True)[0]

def bin_weight(v: float) -> str:
    return pd.cut([v], bins=_WEIGHT_BINS, labels=_WEIGHT_LABELS, right=True, include_lowest=True)[0]

def infer_dataset_from_model(model_path: Path) -> str | None:
    m = re.search(r"(clean|noisy)_", model_path.name)
    return m.group(1) if m else None

def find_preprocessor(model_path: Path, override: Path | None) -> Path:
    # explicit override
    if override:
        return override
    # infer from model filename
    ds = infer_dataset_from_model(model_path)
    if ds:
        p = ART_DIR / f"preprocessor_{ds}.joblib"
        if p.exists():
            return p
    # fallback legacy
    p = ART_DIR / "preprocessor.joblib"
    if p.exists():
        return p
    raise FileNotFoundError("Could not find a matching preprocessor. "
                            "Pass --preprocessor, or re-run `python -m src.features`.")

def load_mapping(path: Path) -> dict[int, str]:
    """Load hs_mapping.csv tolerantly; return {label_id: display_str} or {} on failure."""
    if not path.exists():
        return {}
    try:
        # Try robust parse: allow quotes, bad lines skipped, Python engine tolerates oddities
        df = pd.read_csv(path, engine="python", on_bad_lines="skip", keep_default_na=False)
    except Exception:
        return {}
    # Normalize columns
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
        except Exception:
            continue
        pieces = []
        if hs and str(row[hs]).strip():
            pieces.append(str(row[hs]).strip())
        if title and str(row[title]).strip():
            pieces.append(str(row[title]).strip())
        if not pieces:
            continue
        out[key] = " — ".join(pieces)
    return out

def parse_args():
    p = argparse.ArgumentParser("Predict HS code from structured fields")
    p.add_argument("--model", required=True, help="Path to saved model .joblib")
    p.add_argument("--preprocessor", help="Path to matching preprocessor .joblib (optional)")
    p.add_argument("--tags", required=True, help="Comma-separated tags")
    p.add_argument("--price", type=float, required=True)
    p.add_argument("--weight", type=float, required=True)
    p.add_argument("--origin", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--gift", type=int, choices=[0, 1], default=0)
    p.add_argument("--topk", type=int, default=3)
    return p.parse_args()

def main():
    args = parse_args()
    model_path = Path(args.model)
    preproc_path = find_preprocessor(model_path, Path(args.preprocessor) if args.preprocessor else None)

    pre = joblib.load(preproc_path)
    model = joblib.load(model_path)

    row = pd.DataFrame([{
        "tags": args.tags,
        "price_bin": bin_price(args.price),
        "weight_bin": bin_weight(args.weight),
        "origin": args.origin,
        "dest": args.dest,
        "gift": int(args.gift),
        # extras (ignored by transformer)
        "price": args.price,
        "weight": args.weight,
        "description": "",
        "label_id": 0,
    }])

    X = pre.transform(row[["tags", "price_bin", "weight_bin", "origin", "dest", "gift"]])

    if hasattr(model, "predict_topk"):
        ranked = model.predict_topk(X, k_labels=args.topk)[0]
    else:
        pred = int(model.predict(X)[0]); ranked = [(pred, 1.0)]

    titles = load_mapping(MAP_PATH)  # tolerant; {} if mapping can’t be parsed

    flags = apply_rules(tags=args.tags, gift=args.gift, hs_pred=ranked[0][0], description=None)

    print(f"\n[using] model={model_path.name}  preprocessor={preproc_path.name}")
    print("\nTop predictions:")
    for lab, score in ranked:
        label_str = titles.get(lab, "")
        suffix = f" — {label_str}" if label_str else ""
        print(f"  - HS {lab}{suffix}  (score={score:.4f})")

    if flags:
        print("\nFlags:")
        for f in flags:
            print(f"  - {f}")

if __name__ == "__main__":
    main()
