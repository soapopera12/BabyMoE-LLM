import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

# --- 0. Configuration ---
@dataclass
class ModelConfig:
    # Model Architecture
    block_size: int = 256
    vocab_size: int = 50304
    n_layer: int = 4
    n_head: int = 4           # Number of Query Heads
    n_kv_head: int = 2        # New: Number of K/V Heads (GQA)
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = False
    
    # MoE
    num_experts: int = 4
    num_experts_per_tok: int = 2
    
    # System
    device: str = 'cuda'

# --- 1. RMSNorm (Llama Style Normalization) ---
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# --- 2. RoPE (Rotary Positional Embeddings) ---
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    # Reshape for broadcast
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:xq.shape[1]].view(1, xq.shape[1], 1, -1) # (1, T, 1, head_dim/2)
    
    # Rotate
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# --- 3. Attention with GQA and KV Cache ---
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        # GQA: We repeat K/V heads to match Q heads
        self.n_rep = self.n_head // self.n_kv_head
        
        self.wq = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)
        
        self.dropout = config.dropout

    def forward(self, x, freqs_cis=None, past_kv=None):
        B, T, C = x.size()
        
        xq = self.wq(x).view(B, T, self.n_head, self.head_dim)
        xk = self.wk(x).view(B, T, self.n_kv_head, self.head_dim)
        xv = self.wv(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply RoPE
        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # KV Cache Logic
        if past_kv is not None:
            k_cache, v_cache = past_kv
            xk = torch.cat([k_cache, xk], dim=1)
            xv = torch.cat([v_cache, xv], dim=1)
        
        # Save current state for next step
        current_kv = (xk, xv)

        # GQA: Repeat keys and values to match query heads
        # (B, T, n_kv_head, D) -> (B, T, n_head, D)
        if self.n_rep > 1:
            xk = xk.repeat_interleave(self.n_rep, dim=2)
            xv = xv.repeat_interleave(self.n_rep, dim=2)

        # Transpose for Flash Attention: (B, num_heads, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Flash Attention
        y = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y), current_kv

# --- 4. SwiGLU Expert (Better MLP) ---
class SwiGLUExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        # SwiGLU typically uses a wider hidden dim (e.g. 8/3 * embd)
        hidden_dim = int(2 * config.n_embd / 3) * 4 # Approximation of Llama sizing
        
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # SwiGLU logic: (SiLU(w1(x)) * w3(x)) -> w2
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.gate = nn.Linear(config.n_embd, self.num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLUExpert(config) for _ in range(self.num_experts)])

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        gate_logits = self.gate(x_flat)
        weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_tok)
        weights = F.softmax(weights, dim=-1, dtype=torch.float).to(x.dtype)
        results = torch.zeros_like(x_flat)
        
        for i in range(self.num_experts_per_tok):
            expert_id = selected_experts[:, i]
            weight = weights[:, i].unsqueeze(1)
            for j, expert in enumerate(self.experts):
                mask = (expert_id == j)
                if mask.any():
                    results[mask] += weight[mask] * expert(x_flat[mask])
        return results.view(B, T, C)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = RMSNorm(config.n_embd)
        self.moe = MoELayer(config)

    def forward(self, x, freqs_cis=None, past_kv=None):
        attn_out, current_kv = self.attn(self.ln1(x), freqs_cis, past_kv)
        x = x + attn_out
        x = x + self.moe(self.ln2(x))
        return x, current_kv

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            # 'wpe': Removed! We use RoPE now.
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            'ln_f': RMSNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        
        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(
            config.n_embd // config.n_head, 
            config.block_size * 2 # Double size just to be safe
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past_kvs=None):
        device = idx.device
        b, t = idx.size()
        
        # Embeddings
        x = self.transformer.wte(idx)
        x = self.transformer.drop(x)
        
        # Get RoPE for current sequence length
        # If we have past_kvs (inference), we need the offset positions
        start_pos = 0 if past_kvs is None else past_kvs[0][0].shape[1]
        freqs_cis = self.freqs_cis[start_pos : start_pos + t].to(device)

        new_kvs = []
        
        for i, block in enumerate(self.transformer.h):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, current_kv = block(x, freqs_cis, past_kv)
            new_kvs.append(current_kv)
            
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, new_kvs

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # KV Cache enabled Generation
        past_kvs = None
        
        for _ in range(max_new_tokens):
            # If we have cache, only feed the LAST token
            if past_kvs is not None:
                idx_cond = idx[:, -1:] 
            else:
                idx_cond = idx 
                
            logits, _, past_kvs = self(idx_cond, past_kvs=past_kvs)
            
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx