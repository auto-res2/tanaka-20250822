
import random
import numpy as np
import torch
from dataclasses import dataclass
from typing import List

def set_seed(seed: int = 1337):
    """Set deterministic behavior for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

@dataclass
class DecodeConfig:
    """Configuration for decoding parameters."""
    temperature: float
    top_p: float
    rep_pen: float
    ban_flag: int  # 0/1

class SyntheticPromptGenerator:
    """Generate synthetic prompts with domain-specific patterns."""
    
    def __init__(self, vocab_size: int, domain: str, seq_len: int = 32):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        
        rng = np.random.default_rng(123 if domain == 'chat' else 321)
        base = rng.random((vocab_size, vocab_size))
        
        for i in range(vocab_size):
            if domain == 'chat':
                base[i] += (i % 7) * 0.05  # Conversational patterns
            else:  # code
                base[i] += (i % 11) * 0.03  # Code-like patterns
                
        self.bigram = base / base.sum(axis=1, keepdims=True)
        
        self.start_probs = rng.random(vocab_size)
        self.start_probs /= self.start_probs.sum()

    def generate(self, n_prompts: int) -> List[List[int]]:
        """Generate n_prompts synthetic sequences."""
        prompts = []
        for _ in range(n_prompts):
            seq = [int(np.random.choice(self.vocab_size, p=self.start_probs))]
            
            for _ in range(self.seq_len - 1):
                prev = seq[-1]
                next_tok = int(np.random.choice(self.vocab_size, p=self.bigram[prev]))
                seq.append(next_tok)
                
            prompts.append(seq)
        return prompts

def apply_repetition_penalty(logits: torch.Tensor, prev_tokens: torch.Tensor, penalty: float):
    """Apply repetition penalty to logits."""
    if penalty == 1.0:
        return logits
    
    for t in prev_tokens.tolist():
        logits[..., t] /= penalty
    return logits

def top_p_filtering(logits: torch.Tensor, top_p: float = 1.0):
    """Apply top-p (nucleus) filtering to logits."""
    if top_p >= 1.0:
        return logits
        
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    
    mask = cumprobs > top_p
    mask[..., 0] = False  # Keep at least one token
    sorted_logits[mask] = float('-inf')
    
    logits[0, sorted_indices[0]] = sorted_logits[0]
    return logits

def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0):
    """Sample next token with temperature and top-p filtering."""
    logits = logits / max(temperature, 1e-6)
    logits = top_p_filtering(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    tok = torch.multinomial(probs, num_samples=1)
    return tok.squeeze(-1)

def create_decode_config_sampler():
    """Create a function that samples random decode configurations."""
    def sampler():
        temp = np.random.uniform(0.6, 1.2)
        top_p = np.random.uniform(0.8, 1.0)
        rep_pen = np.random.uniform(1.0, 1.3)
        ban_flag = np.random.choice([0, 1], p=[0.7, 0.3])
        return DecodeConfig(temp, top_p, rep_pen, ban_flag)
    return sampler
