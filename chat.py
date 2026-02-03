import torch
import tiktoken
from src.model import GPT

# --- Configuration ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# UPDATED: Point to the v2 fine-tuned model
weights_path = 'checkpoints/finetuned_v2.pt'

if not torch.cuda.is_available():
    print("WARNING: Running on CPU. This will be slow.")

print(f"Loading Fine-Tuned Model from {weights_path}...")

# 1. Load Model
try:
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
except FileNotFoundError:
    print(f"Error: {weights_path} not found. Did you run finetune.py?")
    exit()

# 2. Setup Tokenizer
enc = tiktoken.get_encoding("gpt2")

print("\n" + "="*40)
print("🤖 BabyMoE v2 Chat (Type 'quit' to exit)")
print("="*40 + "\n")

while True:
    # 3. Get User Input
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit']:
        break
        
    # 4. Format Prompt (Prompt Engineering)
    prompt = f"User: {user_input}\nAssistant: "
    
    # 5. Encode
    start_ids = enc.encode(prompt)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    # 6. Generate
    # The new generate() function uses KV caching automatically for speed
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=100, temperature=0.7, top_k=20)

    # 7. Decode and Print
    output_text = enc.decode(y[0].tolist())
    
    # Extract only the answer
    answer = output_text[len(prompt):] 
    
    print(f"BabyMoE: {answer}")
    print("-" * 20)