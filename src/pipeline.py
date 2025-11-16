# src/pipeline.py
# -----------------------------------------------------------------------------
# Training & evaluation pipeline for HS-code classification (classification-only)
#
# - Loads structured features saved by features.py:
#     artifacts/embeddings/{prefix}_{train,valid,test}_{X,y}.npy
# - Trains either:
#     * KNN (from src/knn.py) – supports top-k predictions
#     * ID3 Decision Tree (from src/id3.py)
# - Evaluates on valid/test splits, saves:
#     * metrics JSON (accuracy, macro-F1; top-3 for KNN)
#     * confusion matrix PNG
# - Saves the fitted model with joblib under artifacts/models/
#
# Usage (from repo root):
#   python -m src.pipeline --dataset clean --model knn --k 5
#   python -m src.pipeline --dataset noisy --model id3 --max_depth 12
#
# Pre-req:
#   1) Run features.py once to create the numpy arrays & preprocessor:
#      python -m src.features
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

# Local models
from .knn import KNNClassifier
from .id3 import ID3Classifier

ART_EMB = Path("artifacts/embeddings")
ART_MOD = Path("artifacts/models")
ART_OUT = Path("artifacts")
ART_MOD.mkdir(parents=True, exist_ok=True)


# ----------------------------- data utilities ----------------------------- #

def load_split(prefix: str, split: str):
    """
    Load a pair of .npy files saved by features.py:
      artifacts/embeddings/{prefix}_{split}_X.npy
      artifacts/embeddings/{prefix}_{split}_y.npy
    """
    Xp = ART_EMB / f"{prefix}_{split}_X.npy"
    yp = ART_EMB / f"{prefix}_{split}_y.npy"
    if not Xp.exists() or not yp.exists():
        raise FileNotFoundError(
            f"Missing arrays for split '{split}' (prefix '{prefix}'). "
            f"Expected: {Xp} and {yp}. Run `python -m src.features` first."
        )
    X = np.load(Xp)
    y = np.load(yp)
    return X, y


def load_all(prefix: str):
    """
    Load train/valid/test arrays for a given dataset prefix ('clean' or 'noisy').
    """
    X_train, y_train = load_split(prefix, "train")
    X_valid, y_valid = load_split(prefix, "valid")
    X_test,  y_test  = load_split(prefix, "test")
    return X_train, y_train, X_valid, y_valid, X_test, y_test


# ----------------------------- eval utilities ----------------------------- #

def topk_accuracy_knn(model: KNNClassifier, X: np.ndarray, y: np.ndarray, k_labels: int = 3) -> float:
    """
    Compute Accuracy@k for KNN using its predict_topk() output.
    For models without top-k (like ID3), we simply skip this metric.
    """
    hits = 0
    topk = model.predict_topk(X, k_labels=k_labels)
    for i, cand in enumerate(topk):
        cand_labels = [lab for lab, _score in cand]
        if int(y[i]) in cand_labels:
            hits += 1
    return hits / len(y) if len(y) else 0.0


def save_confusion_png(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str):
    """
    Save a basic confusion matrix plot as a PNG.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


# ------------------------------ train & eval ------------------------------ #

def train_and_eval(
    dataset_prefix: str,
    model_type: str,
    k: int = 5,
    metric: str = "cosine",
    max_depth: int = 12,
    min_samples: int = 5,
    min_gain: float = 1e-6,
    max_features: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train the chosen model on train split and evaluate on valid + test.
    Returns a dict of metrics and file paths.
    """
    # Load arrays produced by features.py
    X_tr, y_tr, X_va, y_va, X_te, y_te = load_all(dataset_prefix)

    # Build model
    if model_type == "knn":
        model = KNNClassifier(k=k, metric=metric)
        model_name = f"knn_k{k}_{metric}"
    elif model_type == "id3":
        model = ID3Classifier(
            max_depth=max_depth,
            min_samples=min_samples,
            min_gain=min_gain,
            max_features=max_features,
            random_state=seed,
        )
        model_name = f"id3_d{max_depth}_m{min_samples}"
    else:
        raise ValueError("model_type must be 'knn' or 'id3'")

    # Fit
    model.fit(X_tr, y_tr)

    # Predict
    yhat_va = model.predict(X_va)
    yhat_te = model.predict(X_te)

    # Core metrics
    metrics: Dict[str, Any] = {
        "dataset": dataset_prefix,
        "model": model_name,
        "acc_valid": accuracy_score(y_va, yhat_va),
        "acc_test": accuracy_score(y_te, yhat_te),
        "macro_f1_valid": f1_score(y_va, yhat_va, average="macro"),
        "macro_f1_test": f1_score(y_te, yhat_te, average="macro"),
    }

    # Accuracy@3 (KNN only)
    if isinstance(model, KNNClassifier):
        metrics["acc3_valid"] = topk_accuracy_knn(model, X_va, y_va, k_labels=3)
        metrics["acc3_test"]  = topk_accuracy_knn(model, X_te, y_te, k_labels=3)

    # Save confusion matrices
    cm_va_path = ART_OUT / f"confusion_{dataset_prefix}_{model_name}_valid.png"
    cm_te_path = ART_OUT / f"confusion_{dataset_prefix}_{model_name}_test.png"
    save_confusion_png(y_va, yhat_va, cm_va_path, f"Confusion (valid) — {model_name}")
    save_confusion_png(y_te, yhat_te, cm_te_path, f"Confusion (test) — {model_name}")

    # Save metrics JSON
    metrics_path = ART_OUT / f"metrics_{dataset_prefix}_{model_name}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # Save model
    model_path = ART_MOD / f"{dataset_prefix}_{model_name}.joblib"
    joblib.dump(model, model_path)

    # Return a summary
    summary = {
        "metrics_path": str(metrics_path),
        "cm_valid_path": str(cm_va_path),
        "cm_test_path": str(cm_te_path),
        "model_path": str(model_path),
        "metrics": metrics,
    }
    return summary


# ---------------------------------- CLI ---------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Train & evaluate HS classifier (structured features).")
    p.add_argument("--dataset", choices=["clean", "noisy"], default="clean",
                   help="Which prepared dataset prefix to use (from features.py).")
    p.add_argument("--model", choices=["knn", "id3"], default="knn",
                   help="Model type to train.")
    # KNN params
    p.add_argument("--k", type=int, default=5, help="K for KNN.")
    p.add_argument("--metric", choices=["cosine", "l2"], default="cosine",
                   help="Distance metric for KNN.")
    # ID3 params
    p.add_argument("--max_depth", type=int, default=12, help="Max tree depth for ID3.")
    p.add_argument("--min_samples", type=int, default=5, help="Min samples to split for ID3.")
    p.add_argument("--min_gain", type=float, default=1e-6, help="Min information gain to split for ID3.")
    p.add_argument("--max_features", type=int, default=None, help="Optional feature subsampling per split.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


def main():
    args = parse_args()

    # Train + evaluate
    summary = train_and_eval(
        dataset_prefix=args.dataset,
        model_type=args.model,
        k=args.k,
        metric=args.metric,
        max_depth=args.max_depth,
        min_samples=args.min_samples,
        min_gain=args.min_gain,
        max_features=args.max_features,
        seed=args.seed,
    )

    # Console summary
    print("\n=== Training complete ===")
    for k, v in summary["metrics"].items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print("\nSaved:")
    print("  metrics:", summary["metrics_path"])
    print("  confusion (valid):", summary["cm_valid_path"])
    print("  confusion (test): ", summary["cm_test_path"])
    print("  model:", summary["model_path"])


if __name__ == "__main__":
    main()
