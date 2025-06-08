
# 🦙 LLaMA Text Generation : A Journey into LLM Architecture

This project implements a wrapper around the LLaMA (Large Language Model Meta AI) Transformer model for performing text generation tasks like translation and explanation. It demonstrates how to load, configure, and use a pre-trained LLaMA model with custom prompts. Special thanks for Umar Jamil (https://www.youtube.com/@umarjamilai) for the project idea.

##  Features

- Loads and configures LLaMA model checkpoints
- Uses SentencePiece tokenizer for subword tokenization
- Performs efficient text generation using top-p (nucleus) sampling
- Supports batch generation and prompt continuation
- Device-aware execution (CPU, CUDA, MPS)

##  Technologies & Concepts used

| Technology | Description |
|------------|-------------|
| **PyTorch** | Used for tensor operations, model loading, and inference. |
| **Transformer Models** | Understanding of attention mechanisms and Transformer architectures used in LLaMA. |
| **SentencePiece** | Subword tokenization method for handling rare or unknown words. |
| **Top-p Sampling** | A probabilistic decoding method for generating more coherent outputs. |
| **MPS / CUDA** | Learned how to run models on different devices like GPU (CUDA), Apple Silicon (MPS), or fallback to CPU. |


## Core Learnings

### Transformer Architecture Deep Dive

Learned the fundementals of the transformer architecture. This included:

-   **Encoder-Decoder vs. Decoder-Only:**  Llama2's decoder-only architecture for generative tasks became more intuitive. Now I understand why this simplifies the model for text generation and avoids the need for an explicit encoder.
-   **Layer Stacking:**  Appreciating how stacking multiple identical layers (each containing multi-head attention and feed-forward networks) allows the model to learn increasingly complex representations.
-   **Residual Connections:**  The critical role of residual connections (x+textSublayer(x)) in enabling the training of very deep networks by mitigating vanishing gradients and facilitating information flow.

### Positional Encoding Variations

Llama2 utilizes  **Rotary Positional Embeddings (RoPE)**. What I learned was:

-   **Relative Position Information:**  Understanding how RoPE encodes relative positional information, which is crucial for handling variable sequence lengths and improving generalization. The rotation applied to query and key vectors before dot product interaction was a key insight.
-   **Computational Efficiency:**  How RoPE can be more computationally efficient than other complex positional encoding schemes during inference, especially with varying sequence lengths.

### Attention Mechanisms and Optimization

A significant portion of learning revolved around the attention mechanism:

-   **Multi-Head Attention:**  deeper understanding of how the classic attention mechanism works. Multiple "attention heads" allow the model to jointly attend to information from different representation subspaces at different positions.
-   **Scaled Dot-Product Attention:**  The importance of scaling by  sqrtd_k  to prevent the dot products from becoming too large and pushing the softmax function into regions with tiny gradients.
-   **Grouped-Query Attention (GQA):**  Llama2's use of GQA (and its predecessor, Multi-Query Attention - MQA) was a key learning. Understanding how sharing keys and values across multiple attention heads (or groups of heads) can significantly reduce memory bandwidth requirements and accelerate inference, especially for larger models.

### Inference Optimization

-   **KV Cache (Key-Value Cache):**  Explored the role of caching keys and values from previous tokens to avoid recomputing them at each step of auto-regressive generation. This dramatically speeds up inference.
-   **Batching during Inference:**  How batching multiple inference requests together can improve GPU utilization, even with the complexities of variable sequence lengths.
