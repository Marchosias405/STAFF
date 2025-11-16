# src/id3.py
# -----------------------------------------------------------------------------
# A simple, from-scratch Decision Tree classifier (ID3-style with entropy).
#
# Designed for *structured features* produced by your features.py:
# - One-hot / multi-hot categorical features (0/1)
# - Binned numeric features (represented as one-hot columns)
# - Dense NumPy arrays (shape: [n_samples, n_features])
#
# Notes
# -----
# • We use entropy + information gain to pick splits.
# • For numeric features we test thresholds between unique sorted values.
#   (For one-hot inputs, the best threshold typically becomes 0.5.)
# • Stopping criteria: max_depth, min_samples, zero gain, or pure node.
# • Prediction: descend the tree using feature-threshold tests.
# • Explainability: `explain_one(x)` returns the decision path for a row.
#
# This is intentionally minimal and readable for CMPT 310.
# -----------------------------------------------------------------------------

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict


# ----------------------------- utility functions ---------------------------- #

def _entropy(y: np.ndarray) -> float:
    """
    Shannon entropy of a label vector y (integers).
    """
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    # Avoid log2(0) by masking
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _majority_label(y: np.ndarray) -> int:
    """
    Returns the most frequent label in y (ties broken by smallest label id).
    """
    vals, counts = np.unique(y, return_counts=True)
    return int(vals[np.argmax(counts)])


# ------------------------------ tree structures ---------------------------- #

@dataclass
class _Node:
    """
    A tree node. If `is_leaf` is True, we use `prediction`.
    Otherwise, we hold a split: X[:, feature] <= threshold -> left, else right.
    """
    is_leaf: bool
    # Split parameters (for internal nodes)
    feature: Optional[int] = None
    threshold: Optional[float] = None
    # Children
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None
    # Leaf prediction
    prediction: Optional[int] = None
    # Debugging / explanation
    depth: int = 0
    n_samples: int = 0
    impurity: float = 0.0


# --------------------------------- classifier ------------------------------- #

class ID3Classifier:
    """
    Simple Decision Tree classifier (ID3-style with entropy & information gain).

    Parameters
    ----------
    max_depth : int
        Maximum tree depth. Depth=0 means a single leaf (root).
    min_samples : int
        Minimum number of samples required to split a node.
    min_gain : float
        Minimum information gain required to accept a split.
    max_features : Optional[int]
        If set, we will randomly sample up to max_features columns when
        searching for the best split (can speed up training on wide data).
    random_state : Optional[int]
        Seed for the feature subsampling.

    Attributes
    ----------
    root_ : _Node
        Root of the trained tree.
    n_features_ : int
        Number of features seen during fit.
    classes_ : np.ndarray
        Sorted unique class labels.
    """

    def __init__(
        self,
        max_depth: int = 12,
        min_samples: int = 5,
        min_gain: float = 1e-6,
        max_features: Optional[int] = None,
        random_state: Optional[int] = 42,
    ):
        self.max_depth = int(max_depth)
        self.min_samples = int(min_samples)
        self.min_gain = float(min_gain)
        self.max_features = max_features
        self.random_state = random_state

        self.root_: Optional[_Node] = None
        self.n_features_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(random_state)

    # ------------------------------- public API ---------------------------- #

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ID3Classifier":
        """
        Build the tree from training data.
        X: (n_samples, n_features) dense float array
        y: (n_samples,) int labels
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        assert X.ndim == 2 and y.ndim == 1 and X.shape[0] == y.shape[0]

        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.root_ = self._build_node(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for a batch of rows.
        """
        X = np.asarray(X, dtype=np.float32)
        preds = [self._predict_one(row, self.root_) for row in X]
        return np.array(preds, dtype=int)

    def explain_one(self, x: np.ndarray, feature_names: Optional[List[str]] = None) -> List[str]:
        """
        Return a human-readable path explanation for a single row x.
        """
        x = np.asarray(x, dtype=np.float32)
        path = []
        node = self.root_
        while node and not node.is_leaf:
            fname = f"f{node.feature}"
            if feature_names is not None and 0 <= node.feature < len(feature_names):
                fname = feature_names[node.feature]
            test_val = float(x[node.feature])
            cond = f"{fname} <= {node.threshold:.4f} (x={test_val:.4f})"
            path.append(cond)
            node = node.left if test_val <= node.threshold else node.right
        # Append leaf prediction
        if node and node.is_leaf:
            path.append(f"→ predict: {node.prediction}")
        return path

    # ------------------------------ building logic ------------------------ #

    def _build_node(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        """
        Recursively build a node (and its children).
        """
        node = _Node(
            is_leaf=False,
            depth=depth,
            n_samples=X.shape[0],
            impurity=_entropy(y),
        )

        # Stopping criteria: pure, size, or depth
        if node.impurity == 0.0 or X.shape[0] < self.min_samples or depth >= self.max_depth:
            node.is_leaf = True
            node.prediction = _majority_label(y)
            return node

        # Find the best split across features
        best_feature, best_thr, best_gain, (left_idx, right_idx) = self._best_split(X, y)

        # If no gain, make leaf
        if best_feature is None or best_gain < self.min_gain:
            node.is_leaf = True
            node.prediction = _majority_label(y)
            return node

        # Apply split
        node.feature = int(best_feature)
        node.threshold = float(best_thr)

        X_left, y_left = X[left_idx], y[left_idx]
        X_right, y_right = X[right_idx], y[right_idx]

        node.left = self._build_node(X_left, y_left, depth + 1)
        node.right = self._build_node(X_right, y_right, depth + 1)
        return node

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float, Tuple[np.ndarray, np.ndarray]]:
        """
        Search all (or a random subset) of features for the split with max information gain.
        Returns (best_feature, best_threshold, best_gain, (left_idx, right_idx)).
        If no valid split, returns (None, None, 0.0, (None, None)).
        """
        n_samples, n_features = X.shape
        parent_entropy = _entropy(y)

        # Feature subset (optional) to speed up on wide data
        if self.max_features is not None and self.max_features < n_features:
            feat_indices = self._rng.choice(n_features, size=self.max_features, replace=False)
        else:
            feat_indices = np.arange(n_features)

        best_gain = 0.0
        best_feature = None
        best_threshold = None
        best_left_idx = None
        best_right_idx = None

        # For each candidate feature, try splitting thresholds
        for f in feat_indices:
            col = X[:, f]
            # Unique sorted values
            uniq = np.unique(col)
            if uniq.size <= 1:
                continue

            # Candidate thresholds: midpoints between consecutive sorted unique values
            # This also works for {0,1} one-hot features (threshold becomes 0.5).
            thr_candidates = (uniq[:-1] + uniq[1:]) / 2.0

            for thr in thr_candidates:
                left_idx = col <= thr
                right_idx = ~left_idx

                # Ignore degenerate splits
                n_left = int(left_idx.sum())
                n_right = n_samples - n_left
                if n_left < self.min_samples or n_right < self.min_samples:
                    continue

                # Compute information gain
                gain = self._information_gain(y, left_idx, right_idx, parent_entropy)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = f
                    best_threshold = thr
                    best_left_idx = left_idx
                    best_right_idx = right_idx

        if best_feature is None:
            return None, None, 0.0, (None, None)
        return best_feature, best_threshold, float(best_gain), (best_left_idx, best_right_idx)

    def _information_gain(self, y: np.ndarray, left_idx: np.ndarray, right_idx: np.ndarray, parent_entropy: float) -> float:
        """
        Information gain = H(parent) - [ (nL/N) * H(left) + (nR/N) * H(right) ]
        """
        y_left, y_right = y[left_idx], y[right_idx]
        n = y.size
        nL, nR = y_left.size, y_right.size
        if nL == 0 or nR == 0:
            return 0.0
        wL = nL / n
        wR = nR / n
        gain = parent_entropy - (wL * _entropy(y_left) + wR * _entropy(y_right))
        return float(gain)

    # ------------------------------ inference ----------------------------- #

    def _predict_one(self, x: np.ndarray, node: _Node) -> int:
        """
        Traverse the tree for a single sample x and return a label.
        """
        while not node.is_leaf:
            # If feature or threshold missing (defensive), break
            if node.feature is None or node.threshold is None:
                break
            if float(x[node.feature]) <= node.threshold:
                node = node.left
            else:
                node = node.right
            if node is None:  # safety: fallback
                break
        # Fallbacks
        if node and node.is_leaf and node.prediction is not None:
            return node.prediction
        # In case of an unexpected structure, predict majority of training classes (first class)
        return int(self.classes_[0]) if self.classes_ is not None else 0


# ------------------------------- quick example ----------------------------- #
# (Remove or keep for reference; this won't run during import)
if __name__ == "__main__":
    # Tiny sanity test on a toy dataset (AND-like pattern)
    X_demo = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [1, 1],
    ], dtype=np.float32)
    y_demo = np.array([0, 0, 0, 1, 1], dtype=int)

    tree = ID3Classifier(max_depth=3, min_samples=1, min_gain=0.0)
    tree.fit(X_demo, y_demo)

    preds = tree.predict(X_demo)
    print("Preds:", preds)
    for i, x in enumerate(X_demo):
        print(f"Explain row {i}:", "  >  ".join(tree.explain_one(x)))
