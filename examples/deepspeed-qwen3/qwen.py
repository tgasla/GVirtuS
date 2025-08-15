import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-4B"

# Step 1: Load tokenizer (this is fine)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 2: Load model manually onto CPU (no Accelerate hooks)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True  # This avoids auto offloading
)

# Step 3: Initialize DeepSpeed Inference (this will move model to GPU properly)
model = deepspeed.init_inference(
    model,
    mp_size=1,  # or >1 for multi-GPU
    dtype=torch.float16,
    replace_method="auto",
    replace_with_kernel_inject=True
)

# Step 4: Inference
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True
)

inputs = tokenizer([text], return_tensors="pt").to(model.module.device)  # model.module is the real model

outputs = model.generate(
    **inputs,
    max_new_tokens=256
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
