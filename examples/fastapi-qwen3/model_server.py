# model_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

model_name = "Qwen/Qwen3-8B"
enable_thinking = False
max_new_tokens = 16

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

app = FastAPI()

class InferenceRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(request: InferenceRequest):
    messages = [{"role": "user", "content": request.prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )

    start_time = time.time()
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )
    end_time = time.time()
    print(f"INFERENCE TIME: {end_time - start_time:.2f} seconds")

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    output = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    return {"response": output}
