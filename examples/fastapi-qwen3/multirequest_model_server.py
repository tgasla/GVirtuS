# model_server.py
from fastapi import FastAPI, Request
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
async def generate_text(request: InferenceRequest, raw_request: Request):
    request_arrival_time = time.time()

    # Optional: get client IP
    client_host = raw_request.client.host

    serving_start_time = time.time()

    messages = [{"role": "user", "content": request.prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )

    inference_start_time = time.time()
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )
    inference_end_time = time.time()

    serving_end_time = time.time()

    # Timing summary
    # print(f"Request from {client_host}")
    # print(f"Arrival time:         {request_arrival_time:.6f}")
    # print(f"Serving start time:   {serving_start_time:.6f}")
    # print(f"Inference start time: {inference_start_time:.6f}")
    # print(f"Inference end time:   {inference_end_time:.6f}")
    # print(f"Serving end time:     {serving_end_time:.6f}")
    # print(f"Inference duration:   {inference_end_time - inference_start_time:.2f} s")
    # print(f"Total serving time:   {serving_end_time - serving_start_time:.2f} s")
    # print(f"Total latency:        {serving_end_time - request_arrival_time:.2f} s")
    # print("-" * 60)

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    output = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    return {"response": output,
            "timing": {
                "arrival": request_arrival_time,
                "serving_start": serving_start_time,
                "inference_start": inference_start_time,
                "inference_end": inference_end_time,
                "serving_end": serving_end_time,
                "inference_duration": inference_end_time - inference_start_time,
                "total_serving_time": serving_end_time - serving_start_time,
                "total_latency": serving_end_time - request_arrival_time
            }
    }