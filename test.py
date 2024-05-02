import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Install required packages
# !pip install -q -U trl transformers accelerate git+https://github.com/huggingface/peft.git
# !pip install -q -U datasets bitsandbytes

# Setup the model with 4-bit quantization
model_name = 'meta-llama/Llama-2-13b-chat-hf'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.config.use_cache = False

# Prompt Engineering
generator = pipeline(
    model=model, tokenizer=tokenizer,
    task='text-generation',
    temperature=0.001,
    max_new_tokens=500,
    repetition_penalty=1.1
)

prompt = "Could you explain to me how 4-bit quantization works?"
result = generator(prompt)
print(result[0]["generated_text"])

# Load dataset for Supervised Fine-Tuning
dataset = load_dataset('PericlesSavio/novel17_test', split="train")

# Parameter-Efficient Fine-Tuning (PEFT) setup
peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=16,
    bias="none",
    task_type="CAUSAL_LM"
)

lora_model = get_peft_model(model, peft_config)
print_trainable_parameters(lora_model)

# Training setup
output_dir = "./results"
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    max_grad_norm=0.3,
    max_steps=500,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    group_by_length=True,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
)

# Start training
trainer.train()

# Saving the adapter and model
model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
model_to_save.save_pretrained("outputs")

# Example text generation
text = "### Human: Écrire un texte dans un style baroque sur la glace et le feu ### Assistant: Si j'en luis éton."
device = "cuda:0"

inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=500)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
