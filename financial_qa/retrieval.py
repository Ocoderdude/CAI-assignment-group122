# Added comments for clarity of the retriever internals
from __future__ import annotations
import pickle
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .config import (
	EMBEDDING_MODEL_NAME,
	FAISS_INDEX_PATH,
	BM25_INDEX_PATH,
	CHUNKS_META_PATH,
)


@dataclass
class RetrievedChunk:
	# Result item returned by retriever combining text, score, and metadata
	chunk_id: str
	text: str
	score: float
	source: str  # "dense" or "sparse"
	metadata: Dict


class HybridRetriever:
	"""Hybrid dense+sparse retriever with adaptive chunking/merging (Group 122 feature)."""
	def __init__(self, embedding_model_name: str = EMBEDDING_MODEL_NAME):
		"""Load sentence embedding model; initialize holders for indexes and data."""
		self.embedding_model = SentenceTransformer(embedding_model_name)
		self.faiss_index: faiss.IndexFlatIP | None = None
		self.bm25: BM25Okapi | None = None
		self.chunks: List[Dict] = []
		self.embeddings: np.ndarray | None = None
	
	def fit(self, chunks: List[Dict]):
		"""Build dense FAISS index and sparse BM25 index from chunk texts."""
		self.chunks = chunks
		texts = [c["text"] for c in chunks]
		# Dense
		emb = self.embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
		self.embeddings = emb.astype("float32")
		index = faiss.IndexFlatIP(self.embeddings.shape[1])
		index.add(self.embeddings)
		self.faiss_index = index
		# Sparse BM25
		tokenized = [t.lower().split() for t in texts]
		self.bm25 = BM25Okapi(tokenized)
	
	def save(self):
		"""Persist FAISS index, BM25 model, and chunk metadata to disk."""
		assert self.faiss_index is not None and self.embeddings is not None and self.bm25 is not None
		faiss.write_index(self.faiss_index, str(FAISS_INDEX_PATH))
		with open(BM25_INDEX_PATH, "wb") as f:
			pickle.dump(self.bm25, f)
		with open(CHUNKS_META_PATH, "wb") as f:
			pickle.dump(self.chunks, f)
	
	def load(self):
		"""Load persisted FAISS, BM25, and chunk metadata from disk."""
		self.faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
		with open(BM25_INDEX_PATH, "rb") as f:
			self.bm25 = pickle.load(f)
		with open(CHUNKS_META_PATH, "rb") as f:
			self.chunks = pickle.load(f)
		# Embeddings are not needed after FAISS is persisted for search; optional to keep
	
	def _complexity_score(self, query: str) -> float:
		"""Heuristic score [0..1] for query complexity to drive adaptive retrieval params."""
		tokens = query.lower().split()
		length_score = min(1.0, len(tokens) / 20.0)
		keywords = {"compare", "trend", "year", "quarter", "growth", "drivers", "breakdown", "change", "variance", "why"}
		keyword_score = min(1.0, sum(1 for t in tokens if t in keywords) / 3.0)
		connector_score = 0.3 if any(k in tokens for k in ("and", "or")) else 0.0
		return max(0.0, min(1.0, 0.5 * length_score + 0.4 * keyword_score + connector_score))
	
	def _adaptive_params(self, query: str, default_dense: int, default_sparse: int, default_alpha: float) -> Tuple[str, int, int, float]:
		"""Map complexity score to chunk size preference and fusion parameters."""
		score = self._complexity_score(query)
		prefer = "small" if score < 0.45 else "large"
		top_k_dense = default_dense + (2 if score >= 0.6 else 0)
		top_k_sparse = default_sparse + (2 if score < 0.4 else 0)
		alpha = 0.6 if score >= 0.6 else (0.5 if score >= 0.4 else 0.4)
		return prefer, top_k_dense, top_k_sparse, alpha
	
	def _merge_adjacent_small(self, hits: List[RetrievedChunk], max_merge: int = 2) -> List[RetrievedChunk]:
		"""Merge adjacent 'small' chunks within the same document to form a richer context window."""
		# Consider only small chunks; merge adjacent by start offset within same doc
		small_hits = [h for h in hits if h.metadata.get("size") == "small"]
		others = [h for h in hits if h.metadata.get("size") != "small"]
		grouped: Dict[str, List[RetrievedChunk]] = {}
		for h in small_hits:
			key = h.metadata.get("doc_name", "")
			grouped.setdefault(key, []).append(h)
		merged_results: List[RetrievedChunk] = []
		for key, group in grouped.items():
			group_sorted = sorted(group, key=lambda x: x.metadata.get("start", 0))
			i = 0
			while i < len(group_sorted):
				curr = [group_sorted[i]]
				j = i + 1
				while j < len(group_sorted) and len(curr) < max_merge:
					prev = group_sorted[j - 1]
					next_c = group_sorted[j]
					if next_c.metadata.get("start", 0) <= prev.metadata.get("start", 0) + len(prev.text) + 120:
						curr.append(next_c)
						j += 1
					else:
						break
					
				# Merge curr into one chunk
				if len(curr) == 1:
					merged_results.append(curr[0])
				else:
					text = "\n".join([c.text for c in curr])
					score = max(c.score for c in curr) + 0.05 * (len(curr) - 1)
					meta = dict(curr[0].metadata)
					meta["merged"] = True
					meta["merged_count"] = len(curr)
					merged_results.append(RetrievedChunk(
						chunk_id=f"merged:{curr[0].chunk_id}",
						text=text,
						score=score,
						source="adaptive-merge",
						metadata=meta,
					))
				i = j
			
		# Keep top others (non-small) and merged smalls, dedup by chunk_id
		all_hits = merged_results + others
		# Sort by score descending
		all_hits = sorted(all_hits, key=lambda x: x.score, reverse=True)
		return all_hits
	
	def retrieve(self, query: str, top_k_dense: int = 5, top_k_sparse: int = 5, alpha: float = 0.5, adaptive: bool = True) -> Tuple[List[RetrievedChunk], float]:
		"""Retrieve top chunks using dense (FAISS) and sparse (BM25), fuse scores, and optionally adapt chunking."""
		assert self.faiss_index is not None and self.bm25 is not None
		start = time.time()
		if adaptive:
			prefer, top_k_dense, top_k_sparse, alpha = self._adaptive_params(query, top_k_dense, top_k_sparse, alpha)
		else:
			prefer = "none"
		q_emb = self.embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
		D, I = self.faiss_index.search(q_emb, top_k_dense)
		dense_hits: List[RetrievedChunk] = []
		for score, idx in zip(D[0].tolist(), I[0].tolist()):
			if idx == -1:
				continue
			c = self.chunks[idx]
			dense_hits.append(RetrievedChunk(chunk_id=c["chunk_id"], text=c["text"], score=float(score), source="dense", metadata=c))
		# BM25
		bm_scores = self.bm25.get_scores(query.lower().split())
		bm_top_idx = np.argsort(bm_scores)[-top_k_sparse:][::-1]
		sparse_hits: List[RetrievedChunk] = []
		for idx in bm_top_idx.tolist():
			c = self.chunks[idx]
			sparse_hits.append(RetrievedChunk(chunk_id=c["chunk_id"], text=c["text"], score=float(bm_scores[idx]), source="sparse", metadata=c))
		# Combine via score normalization and union
		def normalize(scores: List[float]) -> List[float]:
			"""Min-max normalize a list of scores to [0,1] safeguarding zero-range cases."""
			arr = np.array(scores, dtype="float32")
			if arr.size == 0:
				return []
			mn, mx = arr.min(), arr.max()
			if mx - mn < 1e-6:
				return [0.0 for _ in scores]
			return ((arr - mn) / (mx - mn)).tolist()
		
		d_norm = normalize([h.score for h in dense_hits])
		s_norm = normalize([h.score for h in sparse_hits])
		for i, h in enumerate(dense_hits):
			h.score = d_norm[i]
		for i, h in enumerate(sparse_hits):
			h.score = s_norm[i]
		merged: Dict[str, RetrievedChunk] = {}
		for h in dense_hits + sparse_hits:
			# Optional: prefer chosen chunk size by slightly boosting score
			if adaptive and prefer in ("small", "large") and h.metadata.get("size") == prefer:
				h.score += 0.05
			key = h.chunk_id
			if key not in merged:
				merged[key] = h
			else:
				merged[key].score = alpha * max(merged[key].score, h.score) + (1 - alpha) * min(merged[key].score, h.score)
		results = sorted(merged.values(), key=lambda x: x.score, reverse=True)
		# Adaptive post-processing: merge adjacent small chunks if appropriate
		if adaptive and prefer == "small":
			results = self._merge_adjacent_small(results)
		elapsed = time.time() - start
		return results, elapsed
