import threading
import time
import queue
import uuid
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
NUM_WORKERS = 2          # GPU-native inference workers
GVIRTUS_WORKERS = 1      # CPU-only + gVirtuS workers
JOB_INTERVAL = 2         # Job arrival rate (seconds)
TOTAL_JOBS = 10
MODEL_NAME = "Qwen/Qwen3-8B"

# Load model once per process (simulate separate nodes)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def load_model():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )

# Queue to hold jobs
job_queue = queue.Queue()

def inference_worker(name, is_gvirtus=False):
    model = load_model()
    while True:
        try:
            job = job_queue.get(timeout=5)
        except queue.Empty:
            break

        start_infer = time.time()
        model_inputs = tokenizer([job['text']], return_tensors="pt").to(model.device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=256)
        end_infer = time.time()

        job['start_time'] = start_infer
        job['end_time'] = end_infer
        job['inference_time'] = end_infer - start_infer
        job['wait_time'] = start_infer - job['enqueue_time']
        job['location'] = "gVirtuS" if is_gvirtus else "native"

        print(f"[{name}] Job {job['id']} | {job['location']} | Wait: {job['wait_time']:.2f}s | Infer: {job['inference_time']:.2f}s")

def job_generator():
    for i in range(TOTAL_JOBS):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Give me a short intro to LLMs."}],
            tokenize=False,
            add_generation_prompt=True
        )
        job = {
            'id': str(uuid.uuid4())[:8],
            'text': text,
            'enqueue_time': time.time()
        }
        job_queue.put(job)
        time.sleep(JOB_INTERVAL)

# Start job generator
gen_thread = threading.Thread(target=job_generator)
gen_thread.start()

# Launch native GPU workers
for i in range(NUM_WORKERS):
    threading.Thread(target=inference_worker, args=(f"GPUWorker-{i}", False)).start()

# Launch gVirtuS workers
for i in range(GVIRTUS_WORKERS):
    threading.Thread(target=inference_worker, args=(f"gVirtuSWorker-{i}", True)).start()
