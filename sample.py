from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

model = PeftModel.from_pretrained(
    base_model,
    "./results/checkpoint-200",
)
merged_model = model.merge_and_unload()

prompt = "You know, the thing about"

merged_model.to("cuda")
merged_model.eval()

input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

output = merged_model.generate(input_ids, max_length=500, do_sample=True)

output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print()
print(output_text)
