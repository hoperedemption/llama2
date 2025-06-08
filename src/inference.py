from __future__ import annotations

from typing import Optional, List, Tuple
import torch 
import time 
from pathlib import Path 
import json 
from sentencepiece import SentencePieceProcessor 
from tqdm import tqdm

from model import ModelArgs, Transformer


class LLaMA:
    """
    Wrapper class for the LLaMa model, with tokenizer integration, 
    model instantiation and checkpoint loading
    """

    def __init__(
            self,
            model: Transformer,
            tokenizer: SentencePieceProcessor,
            model_args: ModelArgs
        ) -> None:
        self.model = model 
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod 
    def _load_checkpoint(checkpoints_dir: str) -> dict:
        checkpoints = sorted(Path(checkpoints_dir).glob('*.pth'))
        if not checkpoints:
            raise FileNotFoundError("No checkpoint files found in directory")
        chk_path = checkpoints[0]
        print(f'Loading checkpoint from {chk_path}')
        return torch.load(chk_path, map_location='cpu')
    
    @staticmethod
    def _load_model_args(
        checkpoints_dir: str, 
        max_seq_len: int, 
        max_batch_size: int, 
        device: str
    ) -> ModelArgs: 
        # construct the full path to the 'params.json' file within the checkpoints directory
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.load(f)
        return ModelArgs(
            max_seq_len=max_seq_len, 
            max_batch_size=max_batch_size, 
            device=device, 
            **params
        )
    
    @staticmethod 
    def _set_default_tensor_type(device: str) -> None:
        if device == "cuda":
            torch.set_default_dtype(torch.float16)
        else:
            torch.set_default_dtype(torch.bfloat16)

    @staticmethod 
    def build(
        checkpoints_dir: str,
        tokenizer_path: str,
        load_model: bool, 
        max_seq_len: int,
        max_batch_size: int,
        device: str
    ) -> LLaMA:
        """
        Factory method to construct and intialize an LLaMA instance

        Args:
            checkpoints_dir (str): Directory containing model checkpoints and params.json.
            tokenizer_path (str): Path to the SentencePiece tokenizer model.
            load_model (bool): Flag indicating whether to load model weights from checkpoint.
            max_seq_len (int): Maximum sequence length.
            max_batch_size (int): Maximum batch size.
            device (str): Device to load model onto ('cuda', 'mps', 'cpu').

        Returns: 
            LLaMA: Initialized LLaMA instance.
        """
        prev_time = time.time()
        checkpoint = None 
        if load_model:
            checkpoint = LLaMA._load_checkpoint(checkpoints_dir)
            print(f'Loading checkpoint took {time.time() - prev_time:.2f}s')
            prev_time = time.time()

        model_args = LLaMA._load_model_args(
            checkpoints_dir, max_seq_len, max_batch_size, device
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        LLaMA._set_default_tensor_type(device)
        
        model = Transformer(model_args).to(device)

        if load_model and checkpoint:
            checkpoint.pop("rope.freqs", None)
            model.load_state_dict(checkpoint, strict=True)
            print(f'Loaded state dict in {time.time() - prev_time:.2f}s')

        return LLaMA(model, tokenizer, model_args)
    
    def _sample_top_p(self, probs: torch.Tensor, p: float) -> torch.Tensor:
        """
        Sample next token using nucleus (top-p) sampling.

        Args:
            probs (Tensor): probability distribution over vocabulary
            p (float): cumulative probability threshold
        
        Returns:
            Tensor: Sampled token indices (batch:size, 1)
        """
        # sort the probabilities in descending order along the vocabulary dimension
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        # calculates the cumulative sum of sorted probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # tokens are masked to 0 if theird individual probability causes the cumulative sum to exceed p
        mask = cumulative_probs - sorted_probs > p 
        # fill the masked probabilities with 0.0 value
        sorted_probs.masked_fill_(mask, 0.0)
        # normalize the filtered probabilities by the sum of all probs
        sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))
        # sample a token with the multinomial distribution
        sampled = torch.multinomial(sorted_probs, num_samples=1)
        # map the sampled indices back to their original vocab indices
        return torch.gather(sorted_indices, dim=-1, index=sampled)

    def text_completion(
            self,
            prompts: List[str],
            device: str, 
            temperature: float=0.6,
            top_p: float=0.9,
            max_gen_len: Optional[int]=None
    ) -> Tuple[List[List[int]], List[str]]:
        """
        Generate text completions for a list of prompts using the model.

        Args:
            prompts (List[str]): Input prompt strings.
            device (str): Device to run the model on.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling probability.
            max_gen_len (Optional[int]): Maximum number of tokens to generate beyond prompt.

        Returns:
            Tuple[List[List[int]], List[str]]: Tuple of generated token lists and decoded texts.
        """
        max_gen_len = max_gen_len or (self.args.max_seq_len - 1)
        # add both Beggining of Sentence (BOS) and End of Sentence (EOS) tokens 
        prompt_tokens = [
            self.tokenizer.encode(p, out_type=int, add_bos=True, add_eos=True)
            for p in prompts 
        ]

        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, "Batch size exceeds model capacity"

        max_prompt_len = max(len(p) for p in prompt_tokens)
        assert max_prompt_len <= self.args.max_seq_len, (
            f"Prompt too long ({max_prompt_len} tokens). "
            f"Model max_seq_len is {self.args.max_seq_len}. "
            "Consider increasing max_seq_len in ModelArgs."
        )

        # avoid generating sequences longer than the model's capacity
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        # get the token ID for pad and end of sentence tokens
        pad_id = self.tokenizer.pad_id()
        eos_id = self.tokenizer.eos_id()

        # initialize the input token tensor for the entire batch
        tokens = torch.full(
            (batch_size, total_len), 
            pad_id, 
            dtype=torch.long, 
            device=device
        )

        # prompts are placed in the tokens tensor
        for i, p in enumerate(prompt_tokens):
            tokens[i, :len(p)] = torch.tensor(p, dtype=torch.long, device=device)

        # initialize a boolean vector to track which tokens in the batch reached eos
        eos_reached = torch.zeros(batch_size, dtype=torch.bool, device=device)
        # create a mask to identify which part of the tokens correspond to the initial prompt
        prompt_mask = tokens != pad_id 

        # main autoregressive generation loop
        # cur_pos starts from one because first token is the bos token
        # it iterates up to total_len generating one token at each step
        for cur_pos in tqdm(range(1, total_len), desc="Generating tokens"):
            # disable grad calculation for efficiency since we're doing inference
            with torch.no_grad():
                # tokens[:, cur_pos-1:cur_pos] extracts a slice of shape (batch_size, seq_len=1)
                # cur_pos is the token's position at each inference step
                # this allows to go through inference and update the KV cache correctly
                # logits shape is (batch_size, seq_len=1, vocab_size)
                logits = self.model(tokens[:, cur_pos-1:cur_pos], cur_pos)
                # select the logits for the last generated token
                logits = logits[:, -1] # shape (batch_size, vocab_size)

                # apply sampling strategy based on temperature
                if temperature > 0:
                    # if temperature > 0 apply softmax to logits and then perform nucleus (top-p) sampling.
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = self._sample_top_p(probs, top_p)
                else:
                    probs = torch.softmax(logits, dim=-1)  
                    # if temperature is 0 then perform greedy search 
                    next_token = torch.argmax(probs, dim=-1)
            
            # note that we still do inference on the token prompts since we need 
            # to feed the model the whole prompt sequence 
            next_token = torch.where(
                prompt_mask[:, cur_pos], # is this position part of the prompt?
                tokens[:, cur_pos], # if yes keep the prompt
                next_token.reshape(-1) # otherwise select next token
            )

            # place the next_token at the cur_pos
            tokens[:, cur_pos] = next_token

            # the current position is not part of the initial prompt and the next_token generated at this position is the EOS token
            eos_reached |= (~prompt_mask[:, cur_pos] & (next_token == eos_id))

            # if all sequences in the batch reqched the EOS token then stop iteration
            if eos_reached.all():
                break 
        

        breakpoint()
        # Post-processing: extract generted tokens and decode them into text
        out_tokens = []
        out_texts = []

        # iterate through each generated sequence in the batch 
        for row in tokens.tolist():
            # if an eos token is present in the generated row, truncate the sqeuence at that position
            if eos_id in row:
                row = row[:row.index(eos_id)]
            # append the processed token list to out_tokens
            out_tokens.append(row)
            # append the decoded tokens into the text list
            out_texts.append(self.tokenizer.decode(row))

        return out_tokens, out_texts


def main() -> None:
    torch.manual_seed(42)

    allow_cuda = False 
    device = (
        "cuda"
        if torch.cuda.is_available() and allow_cuda 
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device is {device}")

    prompts = [
        "Translate: Hello into French.", 
        "Translate the following English sentence into French: 'The quick brown fox jumps over the lazy dog.'", 
        "Explain the concept of quantum entanglement to a high school student. Use analogies to make it understandable."
    ]

    llama = LLaMA.build(
        checkpoints_dir='llama-2-7b', 
        tokenizer_path='tokenizer.model', 
        load_model=True, 
        max_seq_len=1024, 
        max_batch_size=3, 
        device=device
    )

    # Inference pipeline
    out_tokens, out_text = llama.text_completion(prompts, device, max_gen_len=64)
    for i in range(len(out_text)):
        print(f'{out_text[i]}')
        print('-' * 50)

    print(f'All OK')


if __name__ == "__main__":
    main()

