import os
import time
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from src.model import GPT, ModelConfig

# --- 1. DDP SETUP ---
# Detect if we are running under torchrun/DDP
ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)
    is_master = rank == 0
else:
    # Vanilla, non-DDP run (falls back smoothly)
    rank = 0
    local_rank = 0
    world_size = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    is_master = True

# --- 2. OPTIMIZATIONS ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- 3. HYPERPARAMETERS ---
batch_size = 32
block_size = 512
total_batch_size = 131072  # Desired total batch size in tokens (e.g., ~131k tokens)
max_iters = 5000
learning_rate = 3e-4
eval_interval = 100
patience = 5

# Toggle to run from scratch or resume from a checkpoint
init_from = 'scratch' # Options: 'scratch' or 'resume'

# --- 4. GRADIENT ACCUMULATION SETUP ---
# Set gradient accumulation size by DDP world size like Andrej Karpathy
tokens_per_micro_step = batch_size * block_size * world_size

# Ensure our desired total_batch_size is perfectly divisible by our micro-step tokens
assert total_batch_size % tokens_per_micro_step == 0, f"Make sure total_batch_size ({total_batch_size}) is divisible by B * T * world_size ({tokens_per_micro_step})"

gradient_accumulation_steps = total_batch_size // tokens_per_micro_step

if is_master:
    print(f"Total desired batch size: {total_batch_size} tokens")
    print(f"=> calculated gradient accumulation steps: {gradient_accumulation_steps}")
    wandb.init(project="baby-moe-llm", name="run-ddp")

# --- 5. LOAD DATA ---
data_dir = os.path.join('data')
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    
    # IMPORTANT: Shard data across GPUs securely 
    # (By choosing purely random offsets independently per DDP process)
    ix = torch.randint(
        len(data) - block_size,
        (batch_size,),
        device='cpu'
    )
    
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = torch.zeros(50, device=device)
    
    for k in range(50):
        X, Y = get_batch('val')
        _, loss, _ = model(X, Y)
        losses[k] = loss

    if ddp:
        # 🔥 sync losses across GPUs
        dist.all_reduce(losses, op=dist.ReduceOp.AVG)

    model.train()
    # Because losses is synced above, this mean() evaluates identically on every GPU
    return losses.mean().item()

# --- 6. SETUP MODEL & CHECKPOINTING ---
iter_num = 0
best_val_loss = float('inf')

if init_from == 'scratch':
    if is_master: print("Initializing a new model from scratch...")
    config = ModelConfig()
    config.block_size = block_size
    config.n_head = 4
    config.n_kv_head = 2
    config.dropout = 0.1
    config.vocab_size = 50304  # Setting vocab size explicitly
    
    model = GPT(config)
    
elif init_from == 'resume':
    if is_master: print("Resuming training from checkpoints/ckpt_ddp.pt...")
    ckpt_path = 'checkpoints/ckpt_ddp.pt'
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint['config']
    model = GPT(config)
    
    state_dict = checkpoint['model']
    # Fix state_dict keys if they were saved with `torch.compile` prefixes
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter']
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))

model.to(device)

# --- 7. COMPILE MODEL ---
if is_master:
    print("Compiling the model (this can take a minute)...")
model = torch.compile(model)

# --- 8. WRAP DDP ---
if ddp:
    model = DDP(model, device_ids=[local_rank])

# --- 9. OPTIMIZER & SCALER ---
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
if init_from == 'resume' and 'optimizer' in checkpoint:
    optimizer.load_state_dict(checkpoint['optimizer'])

scaler = torch.amp.GradScaler('cuda')

# --- 10. TRAINING LOOP ---
patience_counter = 0
t0 = time.time()

while iter_num < max_iters:

    # --- EVAL ---
    if iter_num % eval_interval == 0:
        val_loss = estimate_loss(model)

        if is_master:
            print(f"step {iter_num}: val loss {val_loss:.4f}")
            wandb.log({
                "val_loss": val_loss,
                "iter": iter_num,
                "best_val_loss": best_val_loss
            })

        # VERY IMPORTANT: Both ranks must execute this to keep 'patience_counter' synced
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Only the master handles the physical saving to avoid file corruption
            if is_master:
                print(f"--> Saving Best Model ({best_val_loss:.4f})")
                raw_model = model.module if ddp else model
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'iter': iter_num,
                    'best_val_loss': best_val_loss
                }
                os.makedirs('checkpoints', exist_ok=True)
                torch.save(checkpoint, 'checkpoints/ckpt_ddp.pt')
        else:
            patience_counter += 1
            if is_master:
                print(f"Patience: {patience_counter}/{patience}")

        # Both ranks will evaluate this to True simultaneously and break together
        if patience_counter >= patience:
            if is_master: 
                print("EARLY STOPPING")
            break

    # --- TRAIN ---
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for micro_step in range(gradient_accumulation_steps):
        # DDP Optmization: We only want to sync gradients at the last micro-step
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            
        X, Y = get_batch('train')

        with torch.amp.autocast('cuda', dtype=torch.float16):
            _, loss, _ = model(X, Y)
            loss = loss / gradient_accumulation_steps
            
            # --- DDP MoE FIX ---
            if ddp:
                dummy_loss = 0.0
                for p in model.parameters():
                    if p.requires_grad:
                        dummy_loss += p.sum() * 0.0
                loss = loss + dummy_loss

        scaler.scale(loss).backward()

    scaler.step(optimizer)
    scaler.update()

    if is_master and iter_num % 10 == 0:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        print(f"iter {iter_num}: loss {loss.item() * gradient_accumulation_steps:.4f}, time {dt*1000:.2f}ms")

    iter_num += 1

# --- 11. CLEANUP ---
if is_master:
    wandb.finish()

if ddp:
    dist.destroy_process_group()