from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

from .config import QA_DATASET_PATH, MODELS_DIR, GENERATION_MODEL_NAME


def load_qa_pairs(path: Path = QA_DATASET_PATH) -> List[Dict]:
	pairs: List[Dict] = []
	if not path.exists():
		raise FileNotFoundError(f"QA pairs not found at {path}")
	with path.open() as f:
		for line in f:
			row = json.loads(line)
			if "question" in row and "answer" in row:
				pairs.append(row)
	return pairs


def build_dataset(pairs: List[Dict]):
	tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)
	def to_text(ex):
		prompt = f"Question: {ex['question']}\nAnswer:"
		return {"input_text": prompt, "target_text": ex["answer"]}
	mapped = list(map(to_text, pairs))
	ds = Dataset.from_list(mapped)
	def tokenize(batch):
		model_inputs = tokenizer(batch["input_text"], truncation=True, max_length=512)
		labels = tokenizer(batch["target_text"], truncation=True, max_length=128)
		model_inputs["labels"] = labels["input_ids"]
		return model_inputs
	return ds, tokenizer, tokenize


def finetune_lora(output_dir: Path | None = None, num_epochs: int = 3, learning_rate: float = 5e-4, batch_size: int = 4):
	pairs = load_qa_pairs()
	ds, tokenizer, tokenize = build_dataset(pairs)
	ds = ds.shuffle(seed=42)
	split = ds.train_test_split(test_size=0.1)
	train_ds = split["train"].map(tokenize, batched=True, remove_columns=["input_text", "target_text"])
	eval_ds = split["test"].map(tokenize, batched=True, remove_columns=["input_text", "target_text"])

	model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL_NAME)
	peft_config = LoraConfig(
		task_type="SEQ_2_SEQ_LM",
		r=16,
		lora_alpha=32,
		lora_dropout=0.05,
		target_modules=["q", "v"],
	)
	model = get_peft_model(model, peft_config)

	data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
	if output_dir is None:
		output_dir = MODELS_DIR / "flan_t5_small_lora"
		output_dir.mkdir(parents=True, exist_ok=True)

	args = TrainingArguments(
		output_dir=str(output_dir),
		per_device_train_batch_size=batch_size,
		per_device_eval_batch_size=batch_size,
		learning_rate=learning_rate,
		num_train_epochs=num_epochs,
		evaluation_strategy="epoch",
		save_strategy="epoch",
		logging_steps=50,
		fp16=torch.cuda.is_available(),
	)

	trainer = Trainer(
		model=model,
		args=args,
		train_dataset=train_ds,
		eval_dataset=eval_ds,
		data_collator=data_collator,
		tokenizer=tokenizer,
	)
	trainer.train()
	trainer.save_model(str(output_dir))
	print(f"Saved LoRA adapter to {output_dir}")
