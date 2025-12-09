# backend/rag_engine.py
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
from typing import List, Dict, Optional, Sequence
import numpy as np
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
INDEX_DIR = DATA_DIR / "index"


class RAGEngine:
    def __init__(self):
        emb_path = INDEX_DIR / "embeddings.npy"
        chunks_path = INDEX_DIR / "chunks.json"
        pca_path = INDEX_DIR / "pca_whitening.npz"

        if not (emb_path.exists() and chunks_path.exists() and pca_path.exists()):
            raise RuntimeError(
                "RAG index not found. Build the index by uploading at least one PDF."
            )

        self.embeddings_raw = np.load(emb_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        pca = np.load(pca_path)
        self.mean = pca["mean"]
        self.W = pca["W"]
        self.k = int(pca["k"][0])
        self.d = int(pca["d"][0])

        self.embeddings = self._project_and_normalize(self.embeddings_raw)

    def _project_and_normalize(self, X: np.ndarray) -> np.ndarray:
        Xc = X - self.mean
        Z = Xc @ self.W
        norms = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
        return Z / norms

    def embed_query_vec(self, v: np.ndarray) -> np.ndarray:
        v_centered = v - self.mean
        z = v_centered @ self.W
        z_norm = z / (np.linalg.norm(z) + 1e-8)
        return z_norm

    # ---------- math bits (cosine + Rocchio + MMR) ----------

    def cosine_similarities(self, q: np.ndarray) -> np.ndarray:
        return self.embeddings @ q

    def _rocchio_update(
        self,
        q_vec: np.ndarray,
        sims_subset: np.ndarray,
        doc_vectors_subset: np.ndarray,
        feedback_k: int = 5,
        alpha: float = 1.0,
        beta: float = 0.75,
    ) -> np.ndarray:
        """
        Rocchio update on a subset of docs (already filtered by allowed_sources).
        """
        n = sims_subset.shape[0]
        if n == 0:
            return q_vec

        k_fb = min(feedback_k, n)
        fb_idx = np.argpartition(-sims_subset, k_fb - 1)[:k_fb]
        rel_vecs = doc_vectors_subset[fb_idx]
        rel_mean = rel_vecs.mean(axis=0)

        q_prime = alpha * q_vec + beta * rel_mean
        q_prime /= np.linalg.norm(q_prime) + 1e-8
        return q_prime

    def _mmr(
        self,
        q: np.ndarray,
        doc_vectors: np.ndarray,
        candidate_indices: np.ndarray,
        top_k: int,
        lambda_param: float = 0.7,
    ) -> List[int]:
        selected: List[int] = []
        candidates = candidate_indices.tolist()

        while len(selected) < top_k and candidates:
            best_score = -1e9
            best_idx = None

            for c_idx in candidates:
                rel = doc_vectors[c_idx] @ q
                if selected:
                    selected_vecs = doc_vectors[selected]
                    sim_to_selected = selected_vecs @ doc_vectors[c_idx]
                    max_sim_to_selected = float(sim_to_selected.max())
                else:
                    max_sim_to_selected = 0.0

                mmr_score = (
                    lambda_param * rel
                    - (1 - lambda_param) * max_sim_to_selected
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = c_idx

            selected.append(best_idx)
            candidates.remove(best_idx)

        return selected

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 3,
        candidate_k: int = 20,
        use_rocchio: bool = True,
        feedback_k: int = 5,
        alpha: float = 1.0,
        beta: float = 0.75,
        allowed_sources: Optional[Sequence[str]] = None,
    ) -> List[Dict]:
        """
        Perform PCA-space cosine similarity + optional Rocchio + MMR.

        If allowed_sources is provided, restrict retrieval to chunks whose
        `source_pdf` is in that list.
        """
        sims_full = self.cosine_similarities(query_vec)  # (N,)
        N = sims_full.shape[0]

        # Build pool of indices we're allowed to search over
        indices_pool = np.arange(N, dtype=int)
        if allowed_sources:
            allowed_set = set(allowed_sources)
            mask = np.array(
                [c["source_pdf"] in allowed_set for c in self.chunks],
                dtype=bool,
            )
            indices_pool = np.where(mask)[0]
            if indices_pool.size == 0:
                # No chunks from the attached PDFs -> return empty
                return []

        doc_vectors_pool = self.embeddings[indices_pool]  # (M, k)
        sims_pool = sims_full[indices_pool]               # (M,)

        # Rocchio on this subset only
        if use_rocchio:
            query_vec = self._rocchio_update(
                q_vec=query_vec,
                sims_subset=sims_pool,
                doc_vectors_subset=doc_vectors_pool,
                feedback_k=feedback_k,
                alpha=alpha,
                beta=beta,
            )
            # Recompute sims with updated query
            sims_full = self.cosine_similarities(query_vec)
            sims_pool = sims_full[indices_pool]

        # Candidate pool within allowed docs
        if candidate_k > len(indices_pool):
            candidate_k = len(indices_pool)
        pool_candidate_local = np.argpartition(
            -sims_pool, candidate_k - 1
        )[:candidate_k]

        # MMR works in the pool's local index space
        selected_local = self._mmr(
            q=query_vec,
            doc_vectors=doc_vectors_pool,
            candidate_indices=pool_candidate_local,
            top_k=top_k,
            lambda_param=0.7,
        )

        # Map back to global indices
        selected_global = indices_pool[selected_local]
        # Sort by similarity
        selected_global = sorted(
            selected_global, key=lambda i: sims_full[i], reverse=True
        )

        results: List[Dict] = []
        for idx in selected_global:
            chunk = self.chunks[idx]
            results.append(
                {
                    "id": chunk["id"],
                    "score": float(sims_full[idx]),
                    "text": chunk["text"],
                    "source_pdf": chunk["source_pdf"],
                    "chunk_index": chunk["chunk_index"],
                }
            )
        return results
