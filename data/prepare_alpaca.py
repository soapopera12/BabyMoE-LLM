# data/prepare_alpaca.py
import json
import os
import requests
import tiktoken
import numpy as np

# Download tiny alpaca (just 100 examples for testing mechanics)
data_url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
file_path = "data/alpaca_data.json"

if not os.path.exists(file_path):
    print("Downloading Alpaca...")
    with open(file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(file_path, 'r') as f:
    raw_data = json.load(f)

# Limit to 500 items for speed/demo
raw_data = raw_data[:500] 

enc = tiktoken.get_encoding("gpt2")
data_ids = []

# Format: "User: <instruction> Assistant: <output>"
for item in raw_data:
    prompt = f"User: {item['instruction']}\nAssistant: "
    completion = f"{item['output']}\n"
    
    # We want the model to learn the completion, but we feed the whole thing
    full_text = prompt + completion
    ids = enc.encode(full_text)
    ids.append(50256) # <|endoftext|> token
    data_ids.extend(ids)

data_ids = np.array(data_ids, dtype=np.uint16)
data_ids.tofile("data/finetune_train.bin")
print(f"Saved {len(data_ids)} tokens for fine-tuning.")