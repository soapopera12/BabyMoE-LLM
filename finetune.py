import os
import time
import numpy as np
import torch
from src.model import GPT

# Config
learning_rate = 1e-5 # Low LR for fine-tuning
max_iters = 200      # Short run
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Pre-trained v2
print("Loading pre-trained model (v2)...")
checkpoint = torch.load('checkpoints/ckpt_v2.pt', map_location=device, weights_only=False)
config = checkpoint['config']
model = GPT(config)
model.load_state_dict(checkpoint['model'])
model.to(device)

# Load Fine-tune Data
if not os.path.exists('data/finetune_train.bin'):
    print("Error: data/finetune_train.bin not found! Run data/prepare_alpaca.py first.")
    exit()

data = np.memmap('data/finetune_train.bin', dtype=np.uint16, mode='r')

def get_batch():
    ix = torch.randint(len(data) - config.block_size, (8,))
    x = torch.stack([torch.from_numpy((data[i:i+config.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+config.block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Starting Fine-tuning...")
model.train()
for iter_num in range(max_iters):
    X, Y = get_batch()
    
    # CRITICAL UPDATE: Unpack 3 values (logits, loss, kv_cache)
    # We ignore kv_cache (_) during training
    logits, loss, _ = model(X, Y)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if iter_num % 10 == 0:
        print(f"iter {iter_num}: loss {loss.item():.4f}")

# Save Fine-tuned model v2
print("Saving fine-tuned model...")
torch.save({
    'model': model.state_dict(), 
    'config': config
}, 'checkpoints/finetuned_v2.pt')