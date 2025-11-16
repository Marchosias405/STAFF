# src/features.py
# -----------------------------------------------------------------------------
# Build structured features for HS-code classification (no raw text).
# -----------------------------------------------------------------------------

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from .tokenizers import split_commas  # <-- moved here so joblib can import it

# --- paths ---
ART_DIR = Path("artifacts")
EMB_DIR = ART_DIR / "embeddings"
ART_DIR.mkdir(exist_ok=True)
EMB_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def bin_price(x: pd.Series) -> pd.Series:
    bins = [0, 25, 50, 100, 200, 10**9]
    labels = ["0-25", "25-50", "50-100", "100-200", "200+"]
    return pd.cut(x, bins=bins, labels=labels, right=True, include_lowest=True)

def bin_weight(x: pd.Series) -> pd.Series:
    bins = [0, 0.25, 0.5, 1, 2, 10**9]
    labels = ["0-0.25", "0.25-0.5", "0.5-1", "1-2", "2+"]
    return pd.cut(x, bins=bins, labels=labels, right=True, include_lowest=True)

def _clean_basic(df: pd.DataFrame) -> pd.DataFrame:
    req = ["id", "description", "tags", "price", "weight", "origin", "dest", "gift", "label_id"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["price"]  = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)

    for c in ["tags", "origin", "dest"]:
        df[c] = df[c].fillna("")

    df["gift"] = pd.to_numeric(df["gift"], errors="coerce").fillna(0).astype(int)
    df["label_id"] = pd.to_numeric(df["label_id"], errors="coerce").astype(int)

    df["tags"] = (
        df["tags"].astype(str).str.split(",")
          .apply(lambda toks: ",".join([t.strip() for t in toks if t.strip()]))
    )
    return df

def make_splits(df: pd.DataFrame, label_col: str = "label_id", seed: int = 42):
    vc = df[label_col].value_counts()
    can_stratify = (vc.min() >= 2)

    if can_stratify:
        train_df, temp = train_test_split(df, test_size=0.30, stratify=df[label_col], random_state=seed)
        valid_df, test_df = train_test_split(temp, test_size=0.50, stratify=temp[label_col], random_state=seed)
    else:
        print("[features] Warning: some classes have <2 samples; using non-stratified split.")
        train_df, temp = train_test_split(df, test_size=0.30, random_state=seed)
        valid_df, test_df = train_test_split(temp, test_size=0.50, random_state=seed)

    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)

# -----------------------------------------------------------------------------
# Main API
# -----------------------------------------------------------------------------

def fit_transform_and_cache(csv_path: str, out_prefix: str):
    df = pd.read_csv(csv_path)
    df = _clean_basic(df)

    # Derived columns (bins)
    df["price_bin"]  = bin_price(df["price"])
    df["weight_bin"] = bin_weight(df["weight"])

    # Inputs (no raw description)
    used_cols = ["tags", "price_bin", "weight_bin", "origin", "dest", "gift"]

    # Split
    train_df, valid_df, test_df = make_splits(df, label_col="label_id", seed=42)

    # Targets
    y_train = train_df["label_id"].astype(int).to_numpy()
    y_valid = valid_df["label_id"].astype(int).to_numpy()
    y_test  = test_df["label_id"].astype(int).to_numpy()

    # Vectorizers/encoders
    tag_vec = CountVectorizer(
        tokenizer=split_commas,   # defined in src/tokenizers.py (picklable)
        lowercase=False,
        token_pattern=None,       # silence warning since tokenizer is provided
    )
    cat_enc = OneHotEncoder(handle_unknown="ignore")

    pre = ColumnTransformer(
        transformers=[
            ("tags", tag_vec, "tags"),
            ("price_bin", cat_enc, ["price_bin"]),
            ("weight_bin", cat_enc, ["weight_bin"]),
            ("odg", cat_enc, ["origin", "dest", "gift"]),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    # Fit/transform
    X_train = pre.fit_transform(train_df[used_cols])
    X_valid = pre.transform(valid_df[used_cols])
    X_test  = pre.transform(test_df[used_cols])

    # Save arrays as proper .npy files
    def _to_dense(X):
        return X.toarray() if isinstance(X, csr_matrix) else np.asarray(X)

    np.save(EMB_DIR / f"{out_prefix}_train_X.npy", _to_dense(X_train).astype(np.float32))
    np.save(EMB_DIR / f"{out_prefix}_valid_X.npy", _to_dense(X_valid).astype(np.float32))
    np.save(EMB_DIR / f"{out_prefix}_test_X.npy",  _to_dense(X_test).astype(np.float32))

    np.save(EMB_DIR / f"{out_prefix}_train_y.npy", y_train)
    np.save(EMB_DIR / f"{out_prefix}_valid_y.npy", y_valid)
    np.save(EMB_DIR / f"{out_prefix}_test_y.npy",  y_test)

    # Indices + preprocessor
    train_df.to_csv(EMB_DIR / f"{out_prefix}_train_index.csv", index=False)
    valid_df.to_csv(EMB_DIR / f"{out_prefix}_valid_index.csv", index=False)
    test_df.to_csv(EMB_DIR / f"{out_prefix}_test_index.csv",  index=False)

    # save a dataset-specific preprocessor so it never mismatches
    joblib.dump(pre, ART_DIR / f"preprocessor_{out_prefix}.joblib")

    # (optional legacy file) only write the generic pointer for 'clean'
    # to avoid clobbering when you also build 'noisy'
    if out_prefix == "clean":
        joblib.dump(pre, ART_DIR / "preprocessor.joblib")


    # Meta
    meta = {
        "n_train": int(X_train.shape[0]),
        "n_valid": int(X_valid.shape[0]),
        "n_test":  int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "used_cols": used_cols,
    }
    (EMB_DIR / f"{out_prefix}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[features] Saved arrays & artifacts for '{out_prefix}' with {meta['n_features']} features.")

if __name__ == "__main__":
    fit_transform_and_cache("data/samples_clean.csv", "clean")
    fit_transform_and_cache("data/samples_noisy.csv", "noisy")
