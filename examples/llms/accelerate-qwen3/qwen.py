from transformers import AutoModelForCausalLM, AutoTokenizer
import time

enable_thinking = False
model_name = "Qwen/Qwen3-8B"
# Qwen Alternatives:
# Qwen/Qwen3-14B
# Qwen/Qwen3-8B
# Qwen/Qwen3-4B
# Qwen/Qwen3-1.7B
# Qwen/Qwen3-0.6B

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=enable_thinking # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)


# start measuring time
start_time = time.time()

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=256
)

# end measuring time
end_time = time.time()
print(f"INFERENCE TIME: {end_time - start_time:.2f} seconds")

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

index = 0
# parsing thinking content
if enable_thinking:
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    print("thinking content:", thinking_content)

content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
print("content:", content)