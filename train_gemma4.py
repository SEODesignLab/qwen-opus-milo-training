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

# NOTE: "Gemma 4" from Google is branded as gemma-3-27b. This file trains Gemma 3 27B.
# We use fresh 4-bit quantization via BitsAndBytesConfig rather than the unsloth pre-quantized
# variant, which was causing load issues (torch_dtype conflict + 19.2 GB reported VRAM).
MODEL_NAME = "google/gemma-3-27b-it"

print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model in 4-bit NF4: {MODEL_NAME}")
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
    # NOTE: do NOT set torch_dtype here — BitsAndBytesConfig controls compute dtype.
    # Setting torch_dtype=float16 on a 4-bit model causes the non-quantized layers
    # (norms, lm_head) to load in fp16 AND can force the quantized weights to be
    # re-cast, which was the root cause of the 19.2 GB VRAM load (vs expected ~14 GB).
    low_cpu_mem_usage=True,
)

gc.collect()
torch.cuda.empty_cache()
print(f"VRAM after 4-bit model load: {torch.cuda.memory_allocated()/1e9:.1f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

gc.collect()
torch.cuda.empty_cache()
print(f"VRAM after LoRA setup: {torch.cuda.memory_allocated()/1e9:.1f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

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
            text += f"<start_of_turn>{msg['role']}\n{msg['content']}<end_of_turn>\n"

    tokens = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

print("Tokenizing dataset...")
tokenized = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

training_args = TrainingArguments(
    output_dir="./output-gemma3",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # Effective batch size = 8
    learning_rate=2e-4,  # Standard LoRA learning rate
    warmup_steps=30,
    logging_steps=5,
    save_steps=100,
    eval_steps=100,
    eval_strategy="steps",
    save_total_limit=2,
    bf16=True,
    fp16=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
    seed=42,
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    dataloader_pin_memory=False,
    ddp_find_unused_parameters=False,
)

from transformers import Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
)

print("Starting Gemma 3 27B training...")
trainer.train()

model.save_pretrained("./output-gemma3/final-adapters")
tokenizer.save_pretrained("./output-gemma3/final-adapters")
print("GEMMA3_TRAINING_COMPLETE")
print("ADAPTERS_SAVED=true")
