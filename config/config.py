from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Model Architecture
    block_size: int = 256
    vocab_size: int = 50304
    n_layer: int = 4
    n_head: int = 4           # Number of Query Heads
    n_kv_head: int = 2        # New: Number of K/V Heads (GQA). 4/2 = 2 Queries per Key.
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = False
    
    # MoE
    num_experts: int = 4
    num_experts_per_tok: int = 2
    
    # System
    device: str = 'cuda'