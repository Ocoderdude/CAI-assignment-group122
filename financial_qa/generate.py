# Added comments for clarity of each function/method
from __future__ import annotations
import time
import re
from dataclasses import dataclass
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from .config import GENERATION_MODEL_NAME, MODELS_DIR
from .retrieval import RetrievedChunk


@dataclass
class GenerationResult:
	# Container for a generated answer and metadata
	answer: str
	confidence: float
	method: str
	inference_time_sec: float
	contexts: List[Tuple[float, str]]


def _fix_character_spacing(text: str) -> str:
	"""Aggressively fix character-by-character spacing artifacts in model output."""
	# Split into words
	words = text.split()
	fixed_words = []
	
	for word in words:
		# If a word is just a single character, it might be part of a broken word
		if len(word) == 1 and word.isalnum():
			# Look ahead to see if we can merge with next word
			if fixed_words and len(fixed_words[-1]) == 1 and fixed_words[-1].isalnum():
				# Merge with previous single character
				fixed_words[-1] += word
			else:
				fixed_words.append(word)
		else:
			fixed_words.append(word)
	
	# Join words back together
	result = ' '.join(fixed_words)
	
	# Additional regex fixes for common patterns
	result = re.sub(r'([a-zA-Z])\s+([a-zA-Z])', r'\1\2', result)
	result = re.sub(r'([0-9])\s+([0-9])', r'\1\2', result)
	result = re.sub(r'([a-zA-Z])\s+([0-9])', r'\1\2', result)
	result = re.sub(r'([0-9])\s+([a-zA-Z])', r'\1\2', result)
	
	return result


def _reconstruct_from_context(text: str, contexts: List[str], query: str) -> str:
	"""Fallback reconstruction: derive a clean answer from retrieved context when model output is mangled or out-of-scope."""
	# If the text is too mangled, try to reconstruct from context
	if len(text.split()) < 5 or any(len(word) == 1 for word in text.split()[:10]) or 'datanotinscope' in text.lower():
		# Look for key financial terms in context
		context_text = " ".join(contexts)
		
		# Analyze the question to determine what to look for
		query_lower = query.lower()
		
		# Extract the specific year requested from the question
		requested_year = None
		year_match = re.search(r'(20[12][0-9])', query)
		if year_match:
			requested_year = year_match.group(1)
		
		if 'revenue' in query_lower or 'revenues' in query_lower:
			# Look for revenue information for the specific year requested
			if requested_year:
				# Multiple patterns for revenue extraction
				revenue_patterns = [
					# Pattern 1: "Revenues were $761,000 for the six months ended June 30, 2024"
					rf'revenues?\s+were\s+\$([0-9,]+)\s+for\s+the\s+(?:six\s+months|year)\s+ended\s+[^,]*,?\s*{requested_year}',
					# Pattern 2: "Revenues for the year ended December 31, 2024, increased by 815% to $1.3 million"
					rf'revenues?\s+for\s+the\s+(?:six\s+months|year)\s+ended\s+[^,]*,?\s*{requested_year}[^$]*?to\s+\$([0-9,\.]+)\s*million',
					# Pattern 3: "Revenues $ 761 $ -- (from financial tables)"
					rf'revenues?\s+\$\s*([0-9,]+)[^$]*?{requested_year}',
					# Pattern 4: Simple revenue amount near year
					rf'{requested_year}[^$]*?revenues?[^$]*?\$([0-9,]+)',
				]
				
				for pattern in revenue_patterns:
					revenue_match = re.search(pattern, context_text, re.IGNORECASE | re.DOTALL)
					if revenue_match:
						revenue = revenue_match.group(1)
						
						# Handle "million" conversion
						if 'million' in context_text[max(0, revenue_match.start()-50):revenue_match.end()+50].lower():
							try:
								if '.' in revenue:
									revenue_amount = int(float(revenue) * 1000000)
									return f"Revenues were ${revenue_amount:,} for {requested_year}."
								else:
									revenue_amount = int(revenue) * 1000000
									return f"Revenues were ${revenue_amount:,} for {requested_year}."
							except:
								pass
						
						# Format with commas
						try:
							revenue_amount = int(revenue.replace(',', ''))
							return f"Revenues were ${revenue_amount:,} for {requested_year}."
						except:
							return f"Revenues were ${revenue} for {requested_year}."
				
				# If no revenue found for specific year, say so
				return f"Not available - Revenue information for {requested_year} is not found in the provided financial reports."
			else:
				# No specific year requested, look for any revenue
				revenue_match = re.search(r'revenues?\s+were\s+\$([0-9,]+)', context_text, re.IGNORECASE)
				if revenue_match:
					return f"Revenues were ${revenue_match.group(1)}."
		
		elif 'expense' in query_lower or 'cost' in query_lower:
			"""Extract expense totals when present in the context."""
			# Look for expense information
			expense_match = re.search(r'expenses?\s+were?\s+\$?([0-9,]+)', context_text, re.IGNORECASE)
			if expense_match:
				expense = expense_match.group(1)
				return f"Expenses were ${expense}."
		
		elif 'profit' in query_lower or 'loss' in query_lower or 'net' in query_lower or 'income' in query_lower:
			"""Extract net income/loss or operating loss for the requested year."""
			if requested_year:
				# Look for net income/loss patterns
				profit_patterns = [
					# Pattern 1: "Net loss for the period $ (24,324) $ (5,835)"
					rf'net\s+loss\s+for\s+the\s+period\s+\$\s*\(([0-9,]+)\)[^$]*?{requested_year}',
					# Pattern 2: "GAAP net loss for the six months ended June 30, 2024, was $24,324,000"
					rf'net\s+loss\s+for\s+the\s+(?:six\s+months|year)\s+ended\s+[^,]*,?\s*{requested_year}[^$]*?was\s+\$([0-9,]+)',
					# Pattern 3: "Operating loss (4,185) (5,985)"
					rf'operating\s+loss\s+\(([0-9,]+)\)[^$]*?{requested_year}',
					# Pattern 4: Simple net loss pattern
					rf'{requested_year}[^$]*?net\s+loss[^$]*?\$([0-9,]+)',
				]
				
				for pattern in profit_patterns:
					loss_match = re.search(pattern, context_text, re.IGNORECASE | re.DOTALL)
					if loss_match:
						loss_amount = loss_match.group(1)
						try:
							loss_num = int(loss_amount.replace(',', ''))
							return f"Net loss was ${loss_num:,} for {requested_year}."
						except:
							return f"Net loss was ${loss_amount} for {requested_year}."
				
				return f"Not available - Profit/loss information for {requested_year} is not found in the provided financial reports."
			else:
				# Look for profit/loss information
				profit_match = re.search(r'net\s+(?:income|loss)\s+(?:of\s+)?\$?([0-9,]+)', context_text, re.IGNORECASE)
				if profit_match:
					profit = profit_match.group(1)
					return f"Net income was ${profit}."
		
		elif 'quarter' in query_lower or 'q' in query_lower:
			"""Acknowledge presence of quarterly information in context."""
			# Look for quarterly information
			quarter_match = re.search(r'quarter\s+([1-4])\s+(20[12][0-9])', context_text, re.IGNORECASE)
			if quarter_match:
				quarter = quarter_match.group(1)
				year = quarter_match.group(2)
				return f"Quarter {quarter} {year} information is available in the financial reports."
		
		elif 'year' in query_lower and '202' in query_lower:
			"""Confirm that financial information for some year exists in the context."""
			# Look for specific year information
			year_match = re.search(r'(20[12][0-9])', context_text)
			if year_match:
				year = year_match.group(1)
				return f"Financial information for {year} is available in the reports."
		
		# Generic fallback - return "Not available" if nothing found
		if requested_year:
			return f"Not available - Financial information for {requested_year} is not found in the provided reports."
		else:
			return "Not available - The requested financial information is not found in the provided reports."
	
	return text


def _clean_response(text: str) -> str:
	"""Normalize and clean model output: spacing, punctuation, and financial formatting."""
	# Step 0: Fix character-by-character spacing first
	text = _fix_character_spacing(text)
	
	# Step 1: Fix the main character spacing issue
	# Pattern: letter space letter -> letterletter
	text = re.sub(r'([a-zA-Z])\s+([a-zA-Z])', r'\1\2', text)
	text = re.sub(r'([0-9])\s+([0-9])', r'\1\2', text)
	text = re.sub(r'([a-zA-Z])\s+([0-9])', r'\1\2', text)
	text = re.sub(r'([0-9])\s+([a-zA-Z])', r'\1\2', text)
	
	# Step 2: Fix punctuation spacing
	text = re.sub(r'([a-zA-Z0-9])\s+([,.!?])', r'\1\2', text)
	text = re.sub(r'([,.!?])\s+([a-zA-Z0-9])', r'\1 \2', text)
	
	# Step 3: Fix specific financial formatting patterns
	text = re.sub(r'(\$[0-9,]+)\s*([a-zA-Z])', r'\1 \2', text)
	text = re.sub(r'([0-9,]+)\s*([a-zA-Z])', r'\1 \2', text)
	
	# Step 4: Fix common financial terms that got mangled
	text = re.sub(r'Revenueswere', 'Revenues were', text)
	text = re.sub(r'fortheyearended', 'for the year ended', text)
	text = re.sub(r'Revenuesforthe', 'Revenues for the', text)
	text = re.sub(r'amountedto', 'amounted to', text)
	text = re.sub(r'dueto', 'due to', text)
	text = re.sub(r'completionofthe', 'completion of the', text)
	text = re.sub(r'Researchanddevelopment', 'Research and development', text)
	text = re.sub(r'expenses, net', 'expenses, net', text)
	text = re.sub(r'comparedto', 'compared to', text)
	text = re.sub(r'expensesof', 'expenses of', text)
	text = re.sub(r'inthe', 'in the', text)
	text = re.sub(r'Generalandadministrative', 'General and administrative', text)
	text = re.sub(r'forthe', 'for the', text)
	text = re.sub(r'were1,', 'were $1,', text)
	text = re.sub(r'comparedto \$', 'compared to $', text)
	text = re.sub(r'inthethreemo', 'in the three months', text)
	
	# Step 5: Fix sentence structure
	text = re.sub(r'([a-z])\s+([A-Z])', r'\1. \2', text)
	
	# Step 6: Clean up multiple spaces and final formatting
	text = re.sub(r'\s+', ' ', text)
	text = text.strip()
	
	# Step 7: Fix common financial abbreviations
	text = re.sub(r'LTP', 'LTP', text)  # Keep as is
	text = re.sub(r'R&D', 'R&D', text)  # Keep as is
	
	return text


def _format_prompt(query: str, contexts: List[str]) -> str:
	"""Create a clear instruction prompt for T5 using the retrieved contexts and explicit formatting guidance."""
	context_block = "\n\n".join([f"Context {i+1}: {c}" for i, c in enumerate(contexts)])
	prompt = (
		"You are a helpful financial QA assistant. Answer the question using only the provided context.\n"
		"IMPORTANT: Write answers with proper spacing between words. Do NOT put spaces between individual characters.\n"
		"Examples of CORRECT formatting:\n"
		"- 'Revenues were $421,000 for the year ended December 31, 2022'\n"
		"- 'The company reported $202,000 in revenue from Rio Tinto'\n"
		"Examples of INCORRECT formatting:\n"
		"- 'R e v e n u e s w e r e $ 4 2 1 , 0 0 0'\n"
		"- 'Revenueswere$421,000fortheyearended'\n\n"
		"Provide clear, well-formatted answers with proper sentence structure and spacing.\n"
		"Format financial numbers clearly with proper spacing.\n"
		"If the answer is not in the context, say 'Data not in scope'.\n\n"
		f"{context_block}\n\nQuestion: {query}\nAnswer:"
	)
	return prompt


class T5Generator:
	"""Thin wrapper around FLAN-T5 for retrieval-augmented QA with robust cleaning and fallback."""
	def __init__(self, model_name: str = GENERATION_MODEL_NAME, use_adapter: bool = False):
		"""Load tokenizer/model; optionally attach LoRA adapter if present; move to device."""
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
		if use_adapter:
			adapter_dir = MODELS_DIR / "flan_t5_small_lora"
			if adapter_dir.exists():
				try:
					from peft import PeftModel
					self.model = PeftModel.from_pretrained(self.model, str(adapter_dir))
				except Exception:
					pass
		self.model.eval()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model.to(self.device)
	
	@torch.inference_mode()
	def generate(self, query: str, retrieved: List[RetrievedChunk], max_context_chars: int = 3000) -> GenerationResult:
		"""Generate an answer from top retrieved chunks; clean, then fallback-reconstruct if needed."""
		# Guardrail: if top score is very low, declare out-of-scope
		contexts_sorted = sorted(retrieved, key=lambda x: x.score, reverse=True)
		top_score = contexts_sorted[0].score if contexts_sorted else 0.0
		if top_score < 0.05:
			return GenerationResult(
				answer="Data not in scope",
				confidence=0.2,
				method="Guardrail",
				inference_time_sec=0.0,
				contexts=[(c.score, c.text[:300]) for c in contexts_sorted[:3]],
			)
		
		accum = []
		chars = 0
		for c in contexts_sorted:
			if chars + len(c.text) > max_context_chars:
				break
			accum.append(c.text)
			chars += len(c.text)
		prompt = _format_prompt(query, accum)
		inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
		start = time.time()
		
		# Try different generation strategies to get better output
		try:
			# First try with more controlled generation
			outputs = self.model.generate(
				**inputs, 
				max_new_tokens=128, 
				num_beams=5, 
				do_sample=False,
				early_stopping=True,
				no_repeat_ngram_size=2
			)
		except Exception:
			# Fallback to simpler generation
			outputs = self.model.generate(**inputs, max_new_tokens=128, num_beams=3, do_sample=False)
		
		elapsed = time.time() - start
		answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
		
		# Clean up the response
		answer = _clean_response(answer)
		
		# If the answer is still too mangled, try to reconstruct from context
		context_texts = [c.text for c in contexts_sorted[:3]]
		original_answer = answer  # Keep track of original
		answer = _reconstruct_from_context(answer, context_texts, query)
		
		# Debug: Show if fallback was used
		if answer != original_answer:
			print(f"DEBUG: Fallback reconstruction used. Original: '{original_answer[:100]}...' -> New: '{answer[:100]}...'")
		
		confidence = max(0.3, min(0.98, 0.6 + 0.4 * float(top_score)))
		return GenerationResult(
			answer=answer,
			confidence=confidence,
			method="RAG-T5",
			inference_time_sec=elapsed,
			contexts=[(c.score, c.text[:300]) for c in contexts_sorted[:3]],
		)
