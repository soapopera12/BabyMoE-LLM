import os
import requests
import tiktoken
import numpy as np

# 1. Download data
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

# 2. Tokenize using GPT-2 encoding
print(f"Length of dataset in characters: {len(data)}")
enc = tiktoken.get_encoding("gpt2")
train_data = enc.encode(data)
print(f"Total tokens: {len(train_data)}")

# 3. Split Train/Val (90/10)
n = int(0.9 * len(train_data))
train_ids = train_data[:n]
val_ids = train_data[n:]

print(f"Train tokens: {len(train_ids)}")
print(f"Val tokens: {len(val_ids)}")

# 4. Save to binary files (uint16 is enough for gpt2 vocab size ~50k)
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

print("Data preparation complete.")