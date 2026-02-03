# BabyMoE-LLM
BabyMoE-LLM

1. Directory structure

BabyMoE/
│
├── config/
│   ├── config.py           # Hyperparameters (vocab size, dim, n_layers, etc.)
│
├── data/
│   ├── prepare_data.py     # Script to download and tokenize data
│   ├── input.txt           # Raw text file (TinyShakespeare)
│   ├── train.bin           # Tokenized training data
│   └── val.bin             # Tokenized validation data
│
├── src/
│   ├── model.py            # The GPT + MoE Architecture
│   ├── trainer.py          # The training loop logic
│   ├── generate.py         # Script to run inference/chat
│   └── dataset.py          # PyTorch Dataset/Dataloader class
│
├── checkpoints/            # Folder to save model weights
│
├── train.py                # Main entry point for pre-training
├── finetune.py             # Main entry point for SFT (Question Answering)
├── requirements.txt        # Libraries needed
└── README.md

2. The Implementation Plan

We will tackle this in 4 distinct phases.
Phase 1: Data Preparation & Tokenization

We need a dataset. We will start with TinyShakespeare (classic, small, fast). Later, we can swap this for OpenWebText.

    Goal: Convert raw text into integers (Tokens).

    Tool: We will use tiktoken (OpenAI's tokenizer) so we don't have to train our own tokenizer yet, giving us real-world token handling.

Phase 2: The Model Architecture (The "MoE" Part)

This is the core. We will build a Transformer Decoder.

    Standard Parts: Embeddings, Positional Encodings, LayerNorm, Self-Attention.

    The Twist (MoE): Instead of a standard Feed-Forward Network (FFN), we will implement a Sparse Mixture of Experts.

        A "Router" decides which expert handles which token.

        Top-k gating (e.g., pick top 2 experts out of 8).

Phase 3: Pre-training (Next Word Prediction)

We teach the model to speak English (or Shakespearean).

    Task: Given [A, B, C], predict D.

    Techniques to learn:

        Gradient Accumulation: To simulate a large batch size on your 4GB GPU.

        Mixed Precision (AMP): Using torch.amp to use half-precision (float16) to save memory and speed up math on the 3050.

        Checkpointing: Saving ckpt.pt so you can resume if the laptop crashes.

Phase 4: Instruction Fine-Tuning (SFT)

We teach the model to answer questions, not just complete sentences.

    Data: A tiny slice of the Alpaca dataset (Instruction/Response pairs).

    Technique: We load the pre-trained weights and train on the QA format: User: <question> Assistant: <answer>.


# env setup
Open Anaconda Prompt (search for it in your Windows Start menu).
Run these commands line-by-line

# 1. Create a new environment named 'babymoe' with Python 3.10
conda create --name babymoe python=3.10 -y

# 2. Activate the environment
conda activate babymoe

# 3. Install PyTorch with CUDA 12.1 support (Crucial for RTX 3050)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install the remaining requirements
pip install tiktoken requests tqdm numpy

# To activate this environment, use
#
#     $ conda activate babymoe
#
# To deactivate an active environment, use
#
#     $ conda deactivate


--------------------------------------------


Here is the BabyMoE v2 Roadmap. We will implement:

    RMSNorm: Replaces LayerNorm. It is faster and more stable (used in Llama/Gemma).

    RoPE (Rotary Positional Embeddings): Replaces standard learned embeddings. It helps the model understand positions much better.

    SwiGLU: A better activation function than GELU.

    GQA (Grouped Query Attention): The critical upgrade. It drastically reduces memory usage during inference, allowing larger context windows on your 4GB GPU.

    KV Caching: The technique to make generation 10x faster by not re-computing history.

    WandB Logging: Professional visual tracking of your training.


# Next word prediction Supervised training
python train.py

# testing
python generate.py

# supervised finetuning
python finetune.py

# final testing
python chat.py