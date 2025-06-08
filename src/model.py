from dataclasses import dataclass
from typing import Optional 
import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

@dataclass 
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 
    n_kv_heads: Optional[int] = None 
    vocab_size: int = -1 
    multiple_of: int = 256 
    ffn_dim_multiplier: Optional[float] = None 
    norm_eps: float = 1e-5 
    max_batch_size: int = 32 
    max_seq_len: int = 2048
    device: Optional[str] = None 

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps 
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * (x * norm).type_as(x)
    
def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, 
                                     theta: float = 1e4) -> torch.Tensor:
    """Precomputes rotary positional embeddings as complex numbers"""
    assert head_dim % 2 == 0, "head_dim must be divisible by two"
    theta_numerator_scaled = torch.arange(0, head_dim, 2, dtype=torch.float, device=device) / head_dim
    log_theta = torch.tensor(theta, dtype=torch.float, device=device).log()
    freqs = torch.exp(-log_theta * theta_numerator_scaled)
    m = torch.arange(seq_len, device=device)
    freqs = torch.outer(m, freqs)
    # radius is 1, so cos(freqs * position) + i * sin(freqs * position)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_embeddings(x: torch.Tensor, freqs: torch.Tensor, device: str) -> torch.Tensor:
    """Applies rotary positional embedding using complex mutliplication."""
    # separate into pairs. if head_dim is 64, the d_tensor becomes -1 (32 pairs), 2
    x = x.reshape(*x.shape[:-1], -1, 2).float()
    # interprets the last two dimensions as real and imaginary
    x_complex = torch.view_as_complex(x)
    # freqs has shape (seq_len, head_dim / 2) -> (1, seq_len, 1, head_dim / 2)
    # x has dimensions (batch_size, seq_len, num_heads, head_dim / 2)
    x_rotated = x_complex * freqs.unsqueeze(0).unsqueeze(2)
    # result has dimensions (batch_size, seq_len, num_heads, head_dim / 2) -> (batch_size, seq_len, num_heads, head_dim / 2, 2)
    # -> (batch_size, seq_len, num_heads, head_dim)
    return torch.view_as_real(x_rotated).reshape(*x.shape[:-2], -1).type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats KV (Key and Value) heads to match the number of Q (Query) heads.
    This is a common operation in Grouped Query Attention (GQA) or Multi-Query Attention (MQA)
    where the number of Key/Value heads is less than the number of Query heads.

    Args:
        x (torch.Tensor): The input tensor representing Key or Value heads.
                          Expected shape: (batch_size, sequence_length, num_kv_heads, head_dim)
        n_rep (int): The number of times to repeat each KV head. This value is typically
                     num_q_heads / num_kv_heads.

    Returns:
        torch.Tensor: The tensor with KV heads repeated.
                      Expected shape: (batch_size, sequence_length, num_q_heads, head_dim).
    """
    # num_q_heads == num_kv_heads
    if n_rep == 1:
        return x 
    # Get the dimensions of the input tensor x.
    # b: batch_size
    # s: sequence_length
    # h: num_kv_heads (number of Key/Value heads)
    # d: head_dim (dimension of each head)
    b, s, h, d = x.shape 

    # (b, s, h, d) -> (b, s, h, 1, d) -> (b, s, h, n_rep, d)
    expanded_x = x.unsqueeze(x).expand(b, s, h, n_rep, d)
    # h * n_rep = num_q_heads
    return expanded_x.reshape(b, s, h * n_rep, d)

class SelfAttention(nn.Module):
    """Multi-head Self Attention with rotary embeddings and caching"""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads 
        self.n_q_heads = args.n_heads 
        self.n_rep = self.n_q_heads // self.n_kv_heads 
        self.head_dim = args.dim // args.n_heads 
        
        self.wq = nn.Linear(args.dim, self.n_q_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        shape = (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        self.cache_k = torch.zeros(*shape, device=args.device)
        self.cache_v = torch.zeros(*shape, device=args.device)

    def forward(self, x: torch.Tensor, start_pos: int, freqs: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the self-attention mechanism.

        Args:
            x (torch.Tensor): The input tensor for queries, keys, and values.
                              Shape: (batch_size, sequence_length, model_dim).
            start_pos (int): The starting position in the sequence, used for KV caching.
                             For the first token, it's 0. For subsequent tokens, it's the
                             current token's position in the full sequence.
            freqs (torch.Tensor): Precomputed rotary positional embedding frequencies.
                                  Shape: (max_seq_len, head_dim / 2).

        Returns:
            torch.Tensor: The output of the self-attention layer.
                          Shape: (batch_size, sequence_length, model_dim).
        """
        # b: batch_size
        # s: sequence_length
        b, s, _ = x.shape 

        # project the q, k, v vectors
        xq = self.wq(x).view(b, s, self.n_q_heads, self.head_dim)
        xk = self.wk(x).view(b, s, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(b, s, self.n_kv_heads, self.head_dim)

        # apply RoPE to q, k
        xq = apply_rotary_embeddings(xq, freqs, x.device)
        xk = apply_rotary_embeddings(xk, freqs, x.device)

        # K, V caching in action
        self.cache_k[:b, start_pos: start_pos + s] = xk
        self.cache_v[:b, start_pos: start_pos + s] = xv 

        # for each attention step use all keys and values computed so far 
        keys = repeat_kv(self.cache_k[:b, :start_pos + s], self.n_rep)
        values = repeat_kv(self.cache_v[:b, :start_pos + s], self.n_rep)

        # (b, seq_len, num_heads, head_dim) -> (b, num_heads, seq_len, head_dim)
        swap = lambda x: x.transpose(1, 2)
        xq, keys, values = swap(xq), swap(keys), swap(values)
        # calculate attention scores: Q * K^T / sqrt(head_dim)
        attention_scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)

        # apply softmax to the scores to get attention probabilities 
        probs = F.softmax(attention_scores.float(), dim=-1).type_as(xq)

        # out shape: (b, n_heads_q, s, head_dim)
        out = torch.matmul(probs, values)

        # reshape the output to (batch_size, seq_eln, n_heads_q * head_dim)
        return self.wo(out.transpose(1, 2).reshape(b, s, -1))

class FeedForward(nn.Module):
    """Feed Forward Netowrk used in Transformer block"""

    def __init__(self, args: ModelArgs):
        super().__init__()
        # is args.ffn_dim_multiplier is None or 0 defaults to 2.66
        hidden_dim = int((args.ffn_dim_multiplier or 8 / 3) * args.dim)
        # rounds up the hidden dimension to be a multiple of args.multiple_of
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        # define the linear layers
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gate output
        gate_output = F.silu(self.w1(x))
        # value path
        value_output = self.w3(x)
        # elementwise multiplication of gate_output * value_path
        # allows to selectively pass information
        gated_result = gate_output * value_output
        return self.w2(gated_result)

class EncoderBlock(nn.Module):
    """Single Transformer Encoder Block"""
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs: torch.Tensor) -> torch.Tensor:
        # apply residual connection
        x = x + self.attention(self.attention_norm(x), start_pos, freqs)
        return x + self.feed_forward(self.ffn_norm(x))
    
class Transformer(nn.Module):
    """Transformer model with rotary embeddings"""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        assert args.vocab_size > 0, "Vocan size must be set"
        
        self.args = args 
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([EncoderBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        # max_seq_len * 2 is often used to precompute frequencies for a longer possible context
        self.freqs = precompute_theta_pos_frequencies(
            args.dim // args.n_heads, args.max_seq_len * 2, device=args.device
        )
    
    def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        b, s = tokens.shape 
        # only one token at a time is being processed
        # this confirms that the model is operating in a strict autoregrssive decoding mode.
        assert s == 1, "only one token at a time can be processed"

        # hidden state shape is (b, s, hidden_dim)
        h = self.tok_embeddings(tokens)

        # precompute the relevant rotary frequencies
        freqs = self.freqs[start_pos : start_pos + s]

        # push the hidden state through the layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs)

        # apply rms norm and project to vocab_size
        return self.output(self.norm(h)).float()
        

