import os
from pathlib import Path

ROOT_DIR = Path(os.environ.get("FIN_QA_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"
MODELS_DIR = ROOT_DIR / "models"
QA_DATASET_PATH = DATA_DIR / "qa_pairs.jsonl"
MERGED_TXT_PATH = PROCESSED_DIR / "merged_financial_reports.txt"

for p in [DATA_DIR, RAW_DIR, PROCESSED_DIR, INDEX_DIR, MODELS_DIR]:
	p.mkdir(parents=True, exist_ok=True)

# Defaults - Use a more reliable model for better text generation
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GENERATION_MODEL_NAME = os.environ.get("GEN_MODEL", "google/flan-t5-small")
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index.bin"
BM25_INDEX_PATH = INDEX_DIR / "bm25_index.pkl"
CHUNKS_META_PATH = INDEX_DIR / "chunks_meta.pkl"

# Chunk sizes per assignment (100, 400 tokens approximated by chars)
SMALL_CHUNK_CHARS = 600
LARGE_CHUNK_CHARS = 2400
CHUNK_OVERLAP_CHARS = 100
