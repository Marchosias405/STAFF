# src/knn.py
# -----------------------------------------------------------------------------
# From-scratch KNN classifier for dense (or CSR) feature matrices.
# Supports cosine similarity and L2 distance. Includes top-k label ranking.
# -----------------------------------------------------------------------------

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Iterable, Optional

class KNNClassifier:
    """
    Simple KNN for classification.

    Parameters
    ----------
    k : int
        Number of neighbors.
    metric : {"cosine", "l2"}
        Similarity/distance metric.
    batch_size : int
        Query batch size for memory-safe inference.
    eps : float
        Numerical stability.

    Notes
    -----
    - For cosine: we L2-normalize rows and use dot products as similarity.
    - For L2: squared distances via ||x||^2 + ||y||^2 - 2 x·y.
    - Predictions use **weighted voting**:
        * cosine -> weights = similarity (>=0)
        * l2     -> weights = 1 / (distance + eps)
      Ties break by higher total weight, then smaller label id.
    """

    def __init__(self, k: int = 5, metric: str = "cosine", batch_size: int = 2048, eps: float = 1e-8):
        assert metric in ("cosine", "l2"), "metric must be 'cosine' or 'l2'"
        self.k = int(k)
        self.metric = metric
        self.batch_size = int(batch_size)
        self.eps = float(eps)

        # Fitted attributes
        self.X_: Optional[np.ndarray] = None          # (n_train, n_features), possibly normalized
        self.y_: Optional[np.ndarray] = None          # (n_train,)
        self.train_norms_: Optional[np.ndarray] = None  # (n_train,), for L2
        self._normalized_: bool = False

    # ----------------------------- public API ----------------------------- #

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        X = self._to_dense_f32(X)
        y = np.asarray(y, dtype=int)
        if X.shape[0] == 0:
            raise ValueError("Empty training set.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

        if self.metric == "cosine":
            Xn = self._row_normalize(X)
            self.X_ = Xn
            self._normalized_ = True
            self.train_norms_ = None
        else:  # l2
            self.X_ = X
            self._normalized_ = False
            self.train_norms_ = (np.sum(X * X, axis=1)).astype(np.float32)

        self.y_ = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Use weighted voting among k nearest neighbors
        topk_neighbors = self._topk_neighbors(X, k_neighbors=max(1, min(self.k, self.X_.shape[0])))
        preds = []
        for neigh_idx, neigh_w in topk_neighbors:
            labels = self.y_[neigh_idx]
            pred = self._weighted_vote(labels, neigh_w)
            preds.append(pred)
        return np.asarray(preds, dtype=int)

    def predict_topk(self, X: np.ndarray, k_labels: int = 3) -> List[List[Tuple[int, float]]]:
        """
        Return, for each row, a ranked list of (label, score) pairs.
        Score is the summed weight of neighbors belonging to that label.
        """
        k_labels = int(k_labels)
        topk_neighbors = self._topk_neighbors(X, k_neighbors=max(1, min(self.k, self.X_.shape[0])))

        out: List[List[Tuple[int, float]]] = []
        for neigh_idx, neigh_w in topk_neighbors:
            labs = self.y_[neigh_idx]
            label_scores = self._aggregate_scores(labs, neigh_w)
            # Sort by score desc, then label asc for determinism
            ranked = sorted(label_scores.items(), key=lambda kv: (-kv[1], kv[0]))
            out.append(ranked[:k_labels])
        return out

    # ---------------------------- core utilities --------------------------- #

    def _topk_neighbors(self, X: np.ndarray, k_neighbors: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        For each query row, find indices of k nearest neighbors and their weights.
        Returns list of tuples [(idx_vec, weight_vec), ...] per query row.
        """
        if self.X_ is None or self.y_ is None:
            raise RuntimeError("Model not fitted.")

        Xq = self._to_dense_f32(X)
        n_train = self.X_.shape[0]
        k_neighbors = min(k_neighbors, n_train)

        results: List[Tuple[np.ndarray, np.ndarray]] = []
        for Xb, sl in self._iter_batches(Xq, self.batch_size):
            if self.metric == "cosine":
                # Normalize queries and compute cosine similarity
                Xb_norm = self._row_normalize(Xb)
                sims = Xb_norm @ self.X_.T  # (b, n_train)
                # Top-k by similarity (largest)
                idx_part = np.argpartition(sims, kth=n_train - k_neighbors, axis=1)[:, -k_neighbors:]
                # Gather sorted within each row
                batch_res = []
                for i in range(Xb.shape[0]):
                    cand_idx = idx_part[i]
                    cand_scores = sims[i, cand_idx]
                    order = np.argsort(-cand_scores, kind="stable")
                    top_idx = cand_idx[order]
                    top_scores = cand_scores[order]
                    # weights = similarity (clip to >= 0)
                    weights = np.maximum(top_scores, 0.0).astype(np.float32)
                    batch_res.append((top_idx, weights))
                results.extend(batch_res)

            else:  # l2
                # Use squared distances via norms
                # d2 = ||x||^2 + ||y||^2 - 2 x·y
                xb2 = np.sum(Xb * Xb, axis=1, keepdims=True)  # (b,1)
                gram = Xb @ self.X_.T                          # (b, n_train)
                d2 = xb2 + self.train_norms_[None, :] - 2.0 * gram
                np.maximum(d2, 0.0, out=d2)  # numerical clamp
                # Top-k by smallest distance
                idx_part = np.argpartition(d2, kth=k_neighbors - 1, axis=1)[:, :k_neighbors]
                batch_res = []
                for i in range(Xb.shape[0]):
                    cand_idx = idx_part[i]
                    cand_d2 = d2[i, cand_idx]
                    order = np.argsort(cand_d2, kind="stable")
                    top_idx = cand_idx[order]
                    top_d2 = cand_d2[order]
                    # weights = 1/(distance + eps)
                    weights = (1.0 / (np.sqrt(top_d2) + self.eps)).astype(np.float32)
                    batch_res.append((top_idx, weights))
                results.extend(batch_res)

        return results

    # --------------------------- voting & scoring -------------------------- #

    def _aggregate_scores(self, labels: np.ndarray, weights: np.ndarray) -> dict:
        scores = {}
        for lab, w in zip(labels, weights):
            scores[int(lab)] = scores.get(int(lab), 0.0) + float(w)
        return scores

    def _weighted_vote(self, labels: np.ndarray, weights: np.ndarray) -> int:
        scores = self._aggregate_scores(labels, weights)
        # winner: max score, tiebreak by smallest label id
        best_label, best_score = None, -np.inf
        for lab, sc in scores.items():
            if sc > best_score or (sc == best_score and (best_label is None or lab < best_label)):
                best_label, best_score = lab, sc
        return int(best_label)

    # -------------------------- batching utilities ------------------------- #

    @staticmethod
    def _iter_batches(X: np.ndarray, batch_size: int) -> Iterable[Tuple[np.ndarray, slice]]:
        n = X.shape[0]
        if n == 0:
            return
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            yield X[start:end], slice(start, end)

    @staticmethod
    def _to_dense_f32(X) -> np.ndarray:
        # Accept dense or scipy.sparse (CSR/CSC) without importing scipy explicitly
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32, order="C")
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features).")
        return X

    def _row_normalize(self, X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, self.eps)
        return X / norms


# ------------------------------- self-test --------------------------------- #
if __name__ == "__main__":
    # Tiny check: two classes separable along x
    rng = np.random.default_rng(0)
    X0 = rng.normal(loc=-1.0, scale=0.2, size=(20, 2))
    X1 = rng.normal(loc=+1.0, scale=0.2, size=(20, 2))
    X = np.vstack([X0, X1]).astype(np.float32)
    y = np.array([0]*20 + [1]*20, dtype=int)

    knn = KNNClassifier(k=5, metric="cosine").fit(X, y)
    preds = knn.predict(X)
    acc = (preds == y).mean()
    print("Train acc (cosine):", acc)

    top3 = knn.predict_topk(X[:3], k_labels=3)
    print("Top-3 for first 3 rows:", top3)
