# BabyMoE-LLM

<p align="center">
  <img src="./assets/babymoe_rd.png" alt="Logo" width="1100" height="300">
</p>

**BabyMoE** is a **small, from-scratch Mixture of Experts (MoE) language model** implemented in pure PyTorch — created **exclusively for learning and experimentation**.

### Goals of this repository

- Understand and implement modern LLM building blocks from first principles
- Run meaningful training and inference on consumer-grade hardware  
  → single RTX 3050 (4–6 GB VRAM) + 16 GB system RAM on Windows
- Progressively add and compare recent architectural & training techniques
- Keep the codebase minimal, readable, well-commented and easy to step through


This is **not** intended to be SOTA, extremely efficient out-of-the-box, or production-ready.  
It is a **learning vehicle** — each major step is tagged so you can easily compare implementations and training behavior.

### Hardware target

- GPU: RTX 3050 (4 GB or 6 GB variant)
- VRAM usage kept ≤ ~3.8–5.2 GB during training
- Techniques used to fit: mixed precision (AMP), gradient accumulation, GQA, small context, RMSNorm, etc.

### Learning roadmap & Git tags

Each meaningful learning milestone gets its own **git tag** so you can:

- checkout a specific version
- read only the relevant code changes
- compare perplexity / generation quality / memory usage

You are welcome to follow along tag by tag, or jump to any point that interests you.

### Philosophy

> “The code should be simple enough that a motivated intermediate PyTorch user can understand almost every line — even if the model is using fairly advanced techniques.”

Happy learning!  
Feel free to open issues / PRs with questions, better explanations, or small cleanups.

---------------------------------

V1.0 features:

- Sparse MoE layer with top-k routing
- RMSNorm, RoPE, SwiGLU, Grouped-Query Attention (GQA)
- KV caching for fast generation
- Gradient accumulation + mixed precision (AMP)
- Weights & Biases (wandb) training logging
- Pre-training on TinyShakespeare → Instruction fine-tuning (Alpaca style)

## Directory Structure

```
BabyMoE/
├── config/
│   └── config.py               # Model & training hyperparameters
├── data/
│   ├── prepare_data.py         # Download & tokenize dataset
│   ├── input.txt               # Raw text (TinyShakespeare)
│   ├── train.bin               # Tokenized train split
│   └── val.bin                 # Tokenized validation split
├── src/
│   ├── model.py                # GPT-like MoE transformer architecture
│   ├── trainer.py              # Core training loop logic
│   ├── generate.py             # Basic sampling / generation script
│   └── dataset.py              # PyTorch Dataset + DataLoader
├── checkpoints/                # Saved model weights (.pt files)
├── train.py                    # Pre-training (next-token prediction)
├── finetune.py                 # Supervised fine-tuning (instruction following)
├── chat.py                     # Simple interactive chat interface
├── requirements.txt            # Dependencies
└── README.md
```

## Modern Techniques Implemented

| Technique          | Purpose / Benefit                                 | Inspired by     |
|--------------------|----------------------------------------------------|-----------------|
| **RMSNorm**        | Faster & more stable than LayerNorm               | Llama, Gemma    |
| **RoPE**           | Better length generalization & relative positions | Llama, PaLM     |
| **SwiGLU**         | Stronger activation than GELU                     | Llama, PaLM     |
| **Grouped-Query Attention (GQA)** | Much lower KV cache memory → longer context possible | Llama-2-70B     |
| **KV Caching**     | ~10× faster autoregressive generation             | Standard in all modern LLMs |
| **Top-k MoE routing** | Sparse expert activation → efficiency             | Mixtral, Switch Transformers |
| **Mixed Precision (AMP)** | Lower memory + faster training on RTX 30xx       | PyTorch native  |
| **Gradient Accumulation** | Simulate large batch sizes on 4 GB VRAM         | Common practice |

## Quick Start

### 1. Environment Setup

```bash
# Create and activate conda environment
conda create --name babymoe python=3.10 -y
conda activate babymoe

# Install PyTorch with CUDA 12.1 (recommended for RTX 3050 / 30-series)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install tiktoken requests tqdm numpy wandb
```

Later you may also want:

```bash
pip install matplotlib pandas    # optional: for analysis
```

### 2. Prepare Data (TinyShakespeare)

```bash
python data/prepare_data.py
```

Creates `data/train.bin` and `data/val.bin`

### 3. Training Pipeline

| Command              | What it does                                 | Typical runtime (RTX 3050 6GB) |
|----------------------|----------------------------------------------|--------------------------------|
| `python train.py`    | Pre-train on next-token prediction           | ~20-25 minutes                 |
| `python finetune.py` | Supervised fine-tuning (instruction format)  | ~10-15 minutes                 |
| `python generate.py` | Basic sampling from checkpoint               | —                              |
| `python chat.py`     | Interactive chat with the fine-tuned model   | —                              |

Example usage:

```bash
# 1. Pre-train
python train.py

# 2. After pre-training finishes (or from a checkpoint)
python finetune.py

# 3. Talk to your model!
python chat.py
```

## Training Tips for Low VRAM (4 GB)

- Use `--batch_size 4 --gradient_accumulation_steps 8` → effective batch size 32
- Enable AMP (`--dtype amp`)
- Keep context length ≤ 512–1024 tokens
- Use GQA (already enabled in config)
- Save checkpoints frequently (`--save_every 1000` steps)

## Planned Improvements / Roadmap

- Switch to OpenWebText / FineWeb / The Pile subset
- Implement learning rate scheduler (cosine with warmup)
- Add evaluation (perplexity, zero-shot tasks)
- Experiment with different #experts, top-k values
- Try Switch Transformers style routing
- LoRA / QLoRA fine-tuning support

## Acknowledgments

Inspired by:

- nanoGPT (Andrej Karpathy)
- llama2.c / lit-llama / lit-gpt
- Mixtral of Experts paper & open implementations
- Hugging Face Transformers (architecture reference)

Happy hacking! 🚀
