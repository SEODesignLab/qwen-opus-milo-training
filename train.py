import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import gc, os

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram:.1f} GB")

MODEL_NAME = "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled"

print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model in 4-bit: {MODEL_NAME}")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 is more numerically stable than float16
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    # NOTE: do NOT set torch_dtype here — BitsAndBytesConfig controls compute dtype
    low_cpu_mem_usage=True,
)

# Clear cache before prep
gc.collect()
torch.cuda.empty_cache()

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=16,  # r=16 gives better quality; r=8 was too conservative for 24 GB RTX 4090
    lora_alpha=32,
    lora_dropout=0.05,
    # Target all attention projections for better fidelity
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

gc.collect()
torch.cuda.empty_cache()
print(f"VRAM after model load: {torch.cuda.memory_allocated()/1e9:.1f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

print("Loading dataset...")
dataset = load_dataset("json", data_files={"train": "train.jsonl", "validation": "valid.jsonl"})

# Avg example is ~700 tokens; max is ~950 tokens. seq_len=1024 covers all examples with no waste.
MAX_SEQ_LEN = 1024

def tokenize(example):
    try:
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    except Exception:
        text = ""
        for msg in example["messages"]:
            text += f"<|{msg['role']}|>\n{msg['content']}\n"

    tokens = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

print("Tokenizing dataset...")
tokenized = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # Effective batch size = 8
    learning_rate=2e-4,  # Higher LR is standard for LoRA (2e-4 not 2e-5)
    warmup_steps=30,
    logging_steps=5,
    save_steps=100,
    eval_steps=100,
    eval_strategy="steps",
    save_total_limit=2,
    bf16=True,       # bfloat16 matches compute dtype above; more stable than fp16
    fp16=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},  # non-reentrant is safer with PEFT
    report_to="none",
    seed=42,
    optim="paged_adamw_8bit",  # 8-bit optimizer saves VRAM
    max_grad_norm=0.3,
    dataloader_pin_memory=False,
    # Prevent VRAM fragmentation
    ddp_find_unused_parameters=False,
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
