import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import json, os

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

MODEL_NAME = "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled"

print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model in 4-bit: {MODEL_NAME}")
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("Loading dataset...")
dataset = load_dataset("json", data_files={"train": "train.jsonl", "validation": "valid.jsonl"})

def tokenize(example):
    # Build chat text manually if apply_chat_template fails
    try:
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    except Exception:
        # Fallback: manual formatting
        text = ""
        for msg in example["messages"]:
            role = msg["role"]
            content = msg["content"]
            text += f"<|{role}|>\n{content}\n"
        text += "<|assistant|>\n"
    
    tokens = tokenizer(text, truncation=True, max_length=4096, padding="max_length")
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

print("Tokenizing dataset...")
tokenized = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    warmup_steps=50,
    logging_steps=10,
    save_steps=200,
    eval_steps=200,
    eval_strategy="steps",
    save_total_limit=3,
    bf16=True,
    gradient_checkpointing=True,
    report_to="none",
    seed=42,
)

from transformers import Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
)

print("Starting training...")
trainer.train()

model.save_pretrained("./output/final-adapters")
tokenizer.save_pretrained("./output/final-adapters")
print("TRAINING_COMPLETE")
print("ADAPTERS_SAVED=true")
