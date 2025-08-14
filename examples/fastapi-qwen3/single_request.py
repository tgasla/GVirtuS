# inference_client.py
import requests

prompt = "Give me a short introduction to large language model."
response = requests.post("http://localhost:8000/generate", json={"prompt": prompt})

print("Response:", response.json()["response"])
