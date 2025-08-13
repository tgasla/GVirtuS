from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import time

# Check if GPU is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  # Force CPU for this example
print(f"Using device: {device}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModelForMaskedLM.from_pretrained("xlm-roberta-base").to(device)
model.eval()

# Example batch of 4 masked input sentences
texts = [
    "Paris is the capital of <mask>.",
    "The <mask> is shining brightly today.",
    "He is a <mask> engineer.",
    "They went to the <mask> to buy food.",
    "Paris is the capital of <mask>.",
    "The <mask> is shining brightly today.",
    "He is a <mask> engineer.",
    "They went to the <mask> to buy food.",
    "Paris is the capital of <mask>.",
    "The <mask> is shining brightly today.",
    "He is a <mask> engineer.",
    "They went to the <mask> to buy food.",
    "Paris is the capital of <mask>.",
    "The <mask> is shining brightly today.",
    "He is a <mask> engineer.",
    "They went to the <mask> to buy food.",
    "Paris is the capital of <mask>.",
    "The <mask> is shining brightly today.",
    "He is a <mask> engineer.",
    "They went to the <mask> to buy food."
]

# Tokenize with padding and return PyTorch tensors
encoded_input = tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(device)

# Measure inference (generation) time
start_time = time.time()
with torch.no_grad():
    output = model(**encoded_input)
end_time = time.time()

# Time taken for inference
inference_duration = end_time - start_time
print(f"Inference time: {inference_duration:.4f} seconds")

# For each sentence, extract prediction for its <mask> token
# print("Predictions:")
for i, text in enumerate(texts):
    mask_token_index = (encoded_input['input_ids'][i] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    mask_token_logits = output.logits[i, mask_token_index, :]
    top_k = 5
    top_tokens = torch.topk(mask_token_logits, top_k, dim=1).indices[0].tolist()

    # print(f"\nInput: {text}")
    # print("Top predictions:")
    # for token_id in top_tokens:
        # print(f"- {tokenizer.decode([token_id])}")
