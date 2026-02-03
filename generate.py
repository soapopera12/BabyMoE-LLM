import torch
import tiktoken
from src.model import GPT, ModelConfig

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
enc = tiktoken.get_encoding("gpt2")

# UPDATED: Point to v2 checkpoint
checkpoint_path = 'checkpoints/ckpt_v2.pt'

print(f"Loading model from {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
config = checkpoint['config']
model = GPT(config)

# Handle state dict prefix from torch.compile (starts with _orig_mod)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Generate
start_text = "To be, or not to be"
start_ids = enc.encode(start_text)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

print("Generating...")
# Generate 100 new tokens
# Note: The new model handles KV caching internally in generate()
y = model.generate(x, max_new_tokens=100, temperature=0.8, top_k=20)

print("--- Result ---")
print(enc.decode(y[0].tolist()))
print("--------------")