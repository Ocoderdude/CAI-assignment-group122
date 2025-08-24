import time
from pathlib import Path
import pickle
import streamlit as st
import pandas as pd
import re
import torch

from financial_qa.config import (
	INDEX_DIR, CHUNKS_META_PATH, FAISS_INDEX_PATH, BM25_INDEX_PATH,
)
from financial_qa.data_processing import build_chunk_corpus
from financial_qa.retrieval import HybridRetriever
from financial_qa.generate import T5Generator

st.set_page_config(page_title="Financial QA: RAG vs Fine-Tune", layout="wide")

@st.cache_resource(show_spinner=False)
def get_retriever() -> HybridRetriever:
	retriever = HybridRetriever()
	if CHUNKS_META_PATH.exists() and FAISS_INDEX_PATH.exists() and BM25_INDEX_PATH.exists():
		retriever.load()
	else:
		st.write("Building indexes from HTML reports...")
		chunks = build_chunk_corpus()
		retriever.fit(chunks)
		retriever.save()
	return retriever

@st.cache_resource(show_spinner=False)
def get_rag_generator() -> T5Generator:
	return T5Generator(use_adapter=False)

@st.cache_resource(show_spinner=False)
def get_ft_generator() -> T5Generator:
	return T5Generator(use_adapter=True)

# --- Helpers for correctness/expected ---
_DEF_NA = {"na", "not applicable", "not available", "data not in scope"}

def _norm(s: str) -> str:
	try:
		return re.sub(r"[\s,$]", "", s.lower())
	except Exception:
		return s.lower().strip()

def _join_context(results) -> str:
	texts = [r.text for r in results[:5]]
	return " \n".join(texts)

def _derive_expected_from_context(q: str, ctx: str) -> str:
	ql = q.lower()
	year = None
	m = re.search(r"(20[12][0-9])", q)
	if m:
		year = m.group(1)
	# Revenue
	if "revenue" in ql or "revenues" in ql:
		if not year:
			return "Not applicable"
		patterns = [
			rf'revenues?\s+were\s+\$([0-9,]+)\s+for\s+the\s+(?:six\s+months|year)\s+ended\s+[^,]*,?\s*{year}',
			rf'revenues?\s+for\s+the\s+(?:six\s+months|year)\s+ended\s+[^,]*,?\s*{year}[^$]*?to\s+\$([0-9,\.]+)\s*million',
			rf'revenues?\s+for\s+the\s+(?:six\s+months|year)\s+ended\s+[^,]*,?\s*{year}[^$]*?were\s+\$([0-9,]+)',
			rf'{year}[^$]*?revenues?[^$]*?\$([0-9,]+)'
		]
		for p in patterns:
			m2 = re.search(p, ctx, re.IGNORECASE | re.DOTALL)
			if m2:
				val = m2.group(1)
				# million conversion
				window = ctx[max(0, m2.start()-50):m2.end()+50].lower()
				if "million" in window:
					try:
						if "." in val:
							val = f"{int(float(val)*1000000):,}"
						else:
							val = f"{int(val)*1000000:,}"
					except Exception:
						pass
				return f"${val}"
		return "Not available"
	# Profit/Loss
	if any(k in ql for k in ("profit", "net", "loss", "income")):
		if not year:
			return "Not available"
		patterns = [
			rf'net\s+loss\s+for\s+the\s+period\s+\$\s*\(([0-9,]+)\)[^$]*?{year}',
			rf'net\s+loss\s+for\s+the\s+(?:six\s+months|year)\s+ended\s+[^,]*,?\s*{year}[^$]*?was\s+\$([0-9,]+)'
		]
		for p in patterns:
			m2 = re.search(p, ctx, re.IGNORECASE | re.DOTALL)
			if m2:
				val = m2.group(1)
				return f"${val}"
		return "Not available"
	# Other types default NA
	return "Not applicable"

st.title("Comparative Financial QA System: RAG vs Fine-Tuning")
mode = st.radio("Select Mode", ["RAG", "Fine-Tuned"], horizontal=True)
adaptive = st.checkbox("Use Advanced RAG: Chunk Merging & Adaptive Retrieval (Group 122)", value=True)
query = st.text_input("Ask a question about the company's financials:", "What was the total revenue in 2023?")

col1, col2 = st.columns([2,1])
with col1:
	if st.button("Get Answer", type="primary"):
		retriever = get_retriever()
		with st.spinner("Retrieving context..."):
			results, rt_retrieve = retriever.retrieve(query, top_k_dense=6, top_k_sparse=6, alpha=0.6, adaptive=adaptive)
			st.write(f"Retrieved {len(results)} chunks in {rt_retrieve:.2f}s")

		if mode == "RAG":
			generator = get_rag_generator()
			method_name = "RAG-T5 (Adaptive)" if adaptive else "RAG-T5"
		else:
			generator = get_ft_generator()
			method_name = "FT-T5 (LoRA)"

		with st.spinner("Generating answer..."):
			gen = generator.generate(query, results)

		st.subheader("Answer")
		st.markdown(f"**{gen.answer}**")
		st.markdown(f"Method: `{method_name}`  |  Confidence: `{gen.confidence:.2f}`  |  Time: `{gen.inference_time_sec:.2f}s`")

		with st.expander("Show top contexts"):
			for i, (score, text) in enumerate(gen.contexts, start=1):
				st.markdown(f"**{i}. score={score:.2f}**\n\n{text}")

with col2:
	st.markdown("### Index Status")
	if CHUNKS_META_PATH.exists():
		st.write("Chunks metadata present")
	else:
		st.warning("Chunks metadata missing")
	st.write(f"FAISS index: {'present' if FAISS_INDEX_PATH.exists() else 'missing'}")
	st.write(f"BM25 index: {'present' if BM25_INDEX_PATH.exists() else 'missing'}")

	if st.button("Rebuild Indexes"):
		st.info("Rebuilding...")
		retriever = HybridRetriever()
		chunks = build_chunk_corpus()
		retriever.fit(chunks)
		retriever.save()
		st.success("Rebuilt indexes.")

st.markdown("---")

# Batch comparison section
st.subheader("Batch Comparison: Questions Table (RAG vs Fine-Tuned)")
left, right = st.columns([1.2, 1])
with left:
	st.write("Provide questions (one per line) or upload a .txt file. We'll run both methods and compare.")
	default_questions_path = Path("questions/questions.txt")
	default_text = ""
	if default_questions_path.exists():
		try:
			default_text = default_questions_path.read_text(encoding="utf-8")
		except Exception:
			default_text = ""
	questions_text = st.text_area("Questions (one per line)", value=default_text, height=160)
	uploaded = st.file_uploader("Or upload questions.txt", type=["txt"]) 
	if uploaded is not None:
		try:
			questions_text = uploaded.read().decode("utf-8")
		except Exception:
			st.error("Could not read uploaded file. Ensure UTF-8 text.")

with right:
	st.write("We'll auto-derive expected answers from the retrieved context and mark correctness.")
	run_batch = st.button("Run Batch Comparison", type="primary")
	include_retrieval_time = st.checkbox("Include retrieval time in Time(s)", value=True)

if run_batch:
	qs = [q.strip() for q in questions_text.splitlines() if q.strip()]
	if not qs:
		st.warning("Please provide at least one question.")
	else:
		retriever = get_retriever()
		rows = []
		prog = st.progress(0)
		# Pass 1: RAG (no adapter)
		_rag = T5Generator(use_adapter=False)
		for i, q in enumerate(qs, start=1):
			results, rt_retrieve = retriever.retrieve(q, top_k_dense=6, top_k_sparse=6, alpha=0.6, adaptive=True)
			ctx = _join_context(results)
			exp = _derive_expected_from_context(q, ctx)
			rag_res = _rag.generate(q, results)
			time_rag = (rt_retrieve + rag_res.inference_time_sec) if include_retrieval_time else rag_res.inference_time_sec
			if exp.lower() in _DEF_NA:
				correct_rag = "Y" if any(k in rag_res.answer.lower() for k in _DEF_NA) else "N"
			elif exp == "Not available":
				correct_rag = "Y" if "not available" in rag_res.answer.lower() else "N"
			else:
				correct_rag = "Y" if _norm(exp) in _norm(rag_res.answer) else "N"
			rows.append({
				"Question": q,
				"Method": "RAG",
				"Answer": rag_res.answer,
				"Confidence": f"{rag_res.confidence:.2f}",
				"Time (s)": f"{time_rag:.2f}",
				"Expected": exp,
				"Correct (Y/N)": correct_rag
			})
			prog.progress(int(i / max(1, len(qs)) * 50))
		# Free RAG generator to lower peak memory
		del _rag
		try:
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
		except Exception:
			pass
		# Pass 2: Fine-Tuned (adapter) – instantiate after freeing RAG
		_ft = T5Generator(use_adapter=True)
		for j, q in enumerate(qs, start=1):
			results, rt_retrieve = retriever.retrieve(q, top_k_dense=6, top_k_sparse=6, alpha=0.6, adaptive=True)
			ctx = _join_context(results)
			exp = _derive_expected_from_context(q, ctx)
			ft_res = _ft.generate(q, results)
			time_ft = (rt_retrieve + ft_res.inference_time_sec) if include_retrieval_time else ft_res.inference_time_sec
			if exp.lower() in _DEF_NA:
				correct_ft = "Y" if any(k in ft_res.answer.lower() for k in _DEF_NA) else "N"
			elif exp == "Not available":
				correct_ft = "Y" if "not available" in ft_res.answer.lower() else "N"
			else:
				correct_ft = "Y" if _norm(exp) in _norm(ft_res.answer) else "N"
			rows.append({
				"Question": q,
				"Method": "Fine-Tuned",
				"Answer": ft_res.answer,
				"Confidence": f"{ft_res.confidence:.2f}",
				"Time (s)": f"{time_ft:.2f}",
				"Expected": exp,
				"Correct (Y/N)": correct_ft
			})
			prog.progress(50 + int(j / max(1, len(qs)) * 50))
		prog.empty()
		df = pd.DataFrame(rows, columns=["Question", "Method", "Answer", "Confidence", "Time (s)", "Expected", "Correct (Y/N)"])
		st.dataframe(df, use_container_width=True)
		csv = df.to_csv(index=False).encode("utf-8")
		st.download_button("Download CSV", csv, file_name="qa_comparison.csv", mime="text/csv")

st.caption("Open-source models only. Guardrail triggers 'Data not in scope' for irrelevant queries.")
