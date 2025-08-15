from vllm import LLM, SamplingParams

model_name = "Qwen/Qwen3-8B"
llm = LLM(
    model=model_name,
    dtype="float16",
    max_model_len=256,
    max_num_seqs=256,
    gpu_memory_utilization=1.0,
    max_num_batched_tokens=256
)

prompt = "Give me a short introduction to large language model."
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=256,
)

outputs = llm.generate([prompt], sampling_params)
print("content:", outputs[0].outputs[0].text)
