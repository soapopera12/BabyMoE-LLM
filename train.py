import os
import time
import numpy as np
import torch
import wandb
from src.model import GPT, ModelConfig

# --- OPTIMIZATIONS ---
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True

# --- Hyperparameters ---
batch_size = 8          
gradient_accumulation_steps = 4 
max_iters = 5000        # High max_iters, relying on Early Stopping
learning_rate = 3e-4
eval_interval = 100     # Evaluate every 100 steps
patience = 5            # STOP if validation doesn't improve for 5 checks (500 steps)
device = 'cuda'

# --- Initialize WandB ---
# Make sure to login to wandb in terminal first: `wandb login`
wandb.init(project="baby-moe-llm", name="run-v2-rope-gqa-earlystop")

# --- Load Data ---
data_dir = os.path.join('data')
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - ModelConfig.block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+ModelConfig.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+ModelConfig.block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(50)
        for k in range(50):
            X, Y = get_batch(split)
            logits, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- Setup Model ---
config = ModelConfig()
# Override for Modern Config
config.n_head = 4
config.n_kv_head = 2 
config.dropout = 0.1 

print(f"Initializing Modern BabyMoE on {device}...")
model = GPT(config)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
scaler = torch.amp.GradScaler('cuda') 

# --- Training Loop ---
print("Starting training...")
iter_num = 0
best_val_loss = float('inf')
patience_counter = 0
t0 = time.time()

while iter_num < max_iters:
    
    # --- EVALUATION & LOGGING ---
    if iter_num % eval_interval == 0:
        losses = estimate_loss(model)
        dt = time.time() - t0
        t0 = time.time()
        
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Log to WandB
        wandb.log({
            "train_loss": losses['train'],
            "val_loss": losses['val'],
            "iter": iter_num,
            "best_val_loss": best_val_loss
        })
        
        # --- EARLY STOPPING LOGIC ---
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            patience_counter = 0 # Reset counter
            print(f"--> New Best Model! Saving (Val Loss: {best_val_loss:.4f})")
            
            # Save the best model
            checkpoint = {'model': model.state_dict(), 'config': config, 'iter': iter_num}
            torch.save(checkpoint, f'checkpoints/ckpt_v2.pt')
        else:
            patience_counter += 1
            print(f"--> No improvement. Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print(f"EARLY STOPPING TRIGGERED at step {iter_num}. Validation loss is rising.")
            break

    # --- FORWARD & BACKWARD ---
    for micro_step in range(gradient_accumulation_steps):
        X, Y = get_batch('train')
        with torch.amp.autocast('cuda', dtype=torch.float16): 
            logits, loss, _ = model(X, Y)
            loss = loss / gradient_accumulation_steps 
        scaler.scale(loss).backward()
    
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    
    iter_num += 1

print(f"Training Finished! Best Validation Loss: {best_val_loss}")
wandb.finish()