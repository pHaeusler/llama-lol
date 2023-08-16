from datasets import load_dataset
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig
import os

dataset = load_dataset("text", data_dir="data")
dataset["train"] = dataset["train"].filter(lambda e: len(e["text"].strip()) > 0)


base_model_name = "meta-llama/Llama-2-7b-hf"

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    use_cache=False,
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True,
    load_in_8bit=True,
)

# More info: https://github.com/huggingface/transformers/pull/24906
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
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
    lr_scheduler_type="constant",
)


trainer = SFTTrainer(
    model,
    train_dataset=dataset["train"],
    dataset_text_field="text",
    peft_config=peft_config,
    callbacks=[PeftSavingCallback()],
    args=training_arguments,
)

trainer.train()
