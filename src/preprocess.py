import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Vocab:
    """Synthetic vocabulary for experimental workloads."""
    
    def __init__(self, n_base: int = 200, n_fact: int = 8):
        self.base_tokens = [f"TOK{i}" for i in range(n_base)]
        self.fact_tokens = [f"FACT_{i}" for i in range(n_fact)]
        self.ans_tokens = [f"ANS_{i}" for i in range(n_fact)]
        self.q_tokens = [f"Q_{i}" for i in range(n_fact)]  # ask for fact i
        self.summ_q = "SUMM_Q"
        self.summ_ans = "ANS_SUMM"
        self.code_q = "CODE_Q"
        self.code_ans = "ANS_CODE"
        self.code_fence = "```"
        self.bracket_l = "("
        self.bracket_r = ")"
        self.eos = "<eos>"
        
        self.tokens = (
            self.base_tokens
            + self.fact_tokens
            + self.ans_tokens
            + self.q_tokens
            + [self.summ_q, self.summ_ans, self.code_q, self.code_ans, 
               self.code_fence, self.bracket_l, self.bracket_r, self.eos]
        )
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.itos = {i: t for t, i in self.stoi.items()}
        self.n_base = n_base
        self.n_fact = n_fact

    def id(self, tok: str) -> int:
        return self.stoi[tok]

    def token(self, idx: int) -> str:
        return self.itos[idx]

    def size(self) -> int:
        return len(self.tokens)


def generate_rag_qa_sequence(vocab: Vocab, n_facts: int = 4, noise_len: int = 100) -> List[int]:
    """Generate RAG QA workload: facts + noise + question."""
    seq = []
    
    fact_positions = sorted(random.sample(range(noise_len), n_facts))
    noise_idx = 0
    
    for pos in fact_positions:
        while noise_idx < pos:
            seq.append(random.randint(0, vocab.n_base - 1))
            noise_idx += 1
        fact_id = random.randint(0, vocab.n_fact - 1)
        seq.append(vocab.id(vocab.fact_tokens[fact_id]))
        noise_idx += 1
    
    while noise_idx < noise_len:
        seq.append(random.randint(0, vocab.n_base - 1))
        noise_idx += 1
    
    query_fact = random.randint(0, vocab.n_fact - 1)
    seq.append(vocab.id(vocab.q_tokens[query_fact]))
    
    return seq


def generate_summarization_sequence(vocab: Vocab, doc_len: int = 150) -> List[int]:
    """Generate summarization workload with structural anchors."""
    seq = []
    
    for i in range(doc_len):
        if i % 20 == 0:  # Add structural markers
            seq.append(vocab.id(vocab.bracket_l))
        elif i % 20 == 19:
            seq.append(vocab.id(vocab.bracket_r))
        else:
            seq.append(random.randint(0, vocab.n_base - 1))
    
    seq.append(vocab.id(vocab.summ_q))
    
    return seq


def generate_coding_sequence(vocab: Vocab, code_len: int = 120) -> List[int]:
    """Generate coding workload with delimiters."""
    seq = []
    
    seq.append(vocab.id(vocab.code_fence))
    
    for i in range(code_len):
        if i % 15 == 0:
            seq.append(vocab.id(vocab.bracket_l))
        elif i % 15 == 14:
            seq.append(vocab.id(vocab.bracket_r))
        else:
            seq.append(random.randint(0, vocab.n_base - 1))
    
    seq.append(vocab.id(vocab.code_fence))
    seq.append(vocab.id(vocab.code_q))
    
    return seq


def generate_needle_sequence(vocab: Vocab, haystack_len: int = 200, needle_pos: int = 50) -> List[int]:
    """Generate needle-in-haystack workload."""
    seq = []
    
    for i in range(haystack_len):
        if i == needle_pos:
            seq.append(vocab.id(vocab.fact_tokens[0]))
        else:
            seq.append(random.randint(0, vocab.n_base - 1))
    
    seq.append(vocab.id(vocab.q_tokens[0]))
    
    return seq


def create_synthetic_workloads(vocab: Vocab, n_samples: int = 10) -> Dict[str, List[List[int]]]:
    """Create synthetic workloads for different tasks."""
    workloads = {
        'rag_qa': [],
        'summarization': [],
        'coding': [],
        'needle': []
    }
    
    for _ in range(n_samples):
        workloads['rag_qa'].append(generate_rag_qa_sequence(vocab))
        workloads['summarization'].append(generate_summarization_sequence(vocab))
        workloads['coding'].append(generate_coding_sequence(vocab))
        workloads['needle'].append(generate_needle_sequence(vocab))
    
    return workloads


def preprocess_data():
    """Main preprocessing function."""
    print("Preprocessing synthetic data for MiMoTA-KV experiments...")
    
    set_seed(42)
    
    vocab = Vocab(n_base=200, n_fact=8)
    print(f"Created vocabulary with {vocab.size()} tokens")
    
    workloads = create_synthetic_workloads(vocab, n_samples=5)
    
    print("Generated synthetic workloads:")
    for task, sequences in workloads.items():
        avg_len = sum(len(seq) for seq in sequences) / len(sequences)
        print(f"  {task}: {len(sequences)} sequences, avg length: {avg_len:.1f}")
    
    return vocab, workloads


if __name__ == "__main__":
    vocab, workloads = preprocess_data()
