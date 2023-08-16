from datasets import load_dataset
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainingArguments,
)
import torch
from peft import LoraConfig
import os

dataset = load_dataset("text", data_dir="data")
dataset["train"] = dataset["train"].filter(lambda e: len(e["text"].strip()) > 0)


model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-125m", load_in_8bit=True, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))


training_arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=10,
    optim="paged_adamw_32bit",
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,
    max_grad_norm=0.3,
    max_steps=1000,
    warmup_ratio=0.03,
    # group_by_length=True,
    lr_scheduler_type="constant",
)


trainer = SFTTrainer(
    model,
    train_dataset=dataset["train"],
    dataset_text_field="text",
    # torch_dtype=torch.bfloat16,
    peft_config=peft_config,
    callbacks=[PeftSavingCallback()],
    # max_seq_length=512,
    # dataset_batch_size=1,
    args=training_arguments,
)

trainer.train()
