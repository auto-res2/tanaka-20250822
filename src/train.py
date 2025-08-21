
import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

def seed_everything(seed: int = 42):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def quantize_per_row_int4(W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Row-wise symmetric affine int4 quantization."""
    assert W.dim() == 2
    V, H = W.shape
    device = W.device
    Wf = W.float()
    minv, maxv = Wf.min(dim=1).values, Wf.max(dim=1).values
    rangev = (maxv - minv).clamp(min=1e-6)
    scale = (rangev / 15.0).to(torch.float32)
    zp = (-minv / scale).round().clamp(0, 15).to(torch.int32)
    Q = ((Wf / scale.unsqueeze(1)) + zp.unsqueeze(1)).round().clamp(0, 15).to(torch.int32)
    
    if H % 2 != 0:
        Q = torch.cat([Q, Q[:, :1]], dim=1)
        H += 1
    Q_lo = (Q[:, 0::2] & 0x0F)
    Q_hi = ((Q[:, 1::2] & 0x0F) << 4)
    packed = (Q_lo | Q_hi).to(torch.uint8)
    return packed, scale, zp

class TinyBackbone(nn.Module):
    """Tiny toy language model backbone."""
    def __init__(self, vocab_size: int = 256, hidden_size: int = 128, seed: int = 0):
        super().__init__()
        g = torch.Generator(device='cpu')
        g.manual_seed(seed)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        
        self.W_packed: Optional[torch.Tensor] = None
        self.W_scale: Optional[torch.Tensor] = None
        self.W_zp: Optional[torch.Tensor] = None
        
        nn.init.normal_(self.embed.weight, std=0.02)
        for p in self.gru.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    def export_quantized_head(self):
        """Export quantized version of the embedding weights for efficient inference."""
        W = self.embed.weight.data.clone()
        W_packed, scale, zp = quantize_per_row_int4(W)
        self.W_packed = W_packed.to(W.device)
        self.W_scale = scale.to(W.device)
        self.W_zp = zp.to(W.device)

    def init_state(self, batch: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None:
            device = self.embed.weight.device
        return torch.zeros(batch, self.hidden_size, device=device)

    def step(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        e = self.embed(x)
        h_next = self.gru(e, h)
        return h_next

    def full_logits(self, h: torch.Tensor) -> torch.Tensor:
        return torch.matmul(h, self.embed.weight.T)

class FutureHead(nn.Module):
    """Future token prediction head for self-speculative decoding."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.act = nn.GELU()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h2 = self.fc2(self.act(self.fc1(self.ln(h))))
        return h2

class HydraSpecModel(nn.Module):
    """Complete HydraSpec model with backbone and future heads."""
    def __init__(self, vocab_size: int = 256, hidden_size: int = 128, num_heads: int = 3):
        super().__init__()
        self.backbone = TinyBackbone(vocab_size, hidden_size)
        self.future_heads = nn.ModuleList([FutureHead(hidden_size) for _ in range(num_heads)])
        self.skim_head = FutureHead(hidden_size)
        self.num_heads = num_heads

    def forward(self, sequences: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for training."""
        batch_size, seq_len = sequences.shape
        device = sequences.device
        
        h = self.backbone.init_state(batch_size, device)
        
        hidden_states = []
        
        for t in range(seq_len):
            h = self.backbone.step(h, sequences[:, t])
            hidden_states.append(h)
        
        hidden_states = torch.stack(hidden_states, dim=1)  # [B, T, H]
        
        backbone_logits = []
        for t in range(seq_len):
            logits = self.backbone.full_logits(hidden_states[:, t])
            backbone_logits.append(logits)
        backbone_logits = torch.stack(backbone_logits, dim=1)  # [B, T, V]
        
        future_logits = []
        for head in self.future_heads:
            head_logits = []
            for t in range(seq_len):
                h_proj = head(hidden_states[:, t])
                logits = torch.matmul(h_proj, self.backbone.embed.weight.T)
                head_logits.append(logits)
            head_logits = torch.stack(head_logits, dim=1)
            future_logits.append(head_logits)
        
        skim_logits = []
        for t in range(seq_len):
            h_proj = self.skim_head(hidden_states[:, t])
            logits = torch.matmul(h_proj, self.backbone.embed.weight.T)
            skim_logits.append(logits)
        skim_logits = torch.stack(skim_logits, dim=1)
        
        return {
            'backbone_logits': backbone_logits,
            'future_logits': future_logits,
            'skim_logits': skim_logits
        }

def train_model(model: HydraSpecModel, 
                train_sequences: np.ndarray, 
                train_targets: np.ndarray,
                num_epochs: int = 5,
                batch_size: int = 32,
                learning_rate: float = 1e-3,
                device: str = 'cpu') -> Dict[str, List[float]]:
    """Train the HydraSpec model."""
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    sequences = torch.from_numpy(train_sequences).long()
    targets = torch.from_numpy(train_targets).long()
    
    dataset = torch.utils.data.TensorDataset(sequences, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    losses = {'backbone': [], 'future': [], 'skim': [], 'total': []}
    
    model.train()
    for epoch in range(num_epochs):
        epoch_losses = {'backbone': 0.0, 'future': 0.0, 'skim': 0.0, 'total': 0.0}
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_seq, batch_targets in pbar:
            batch_seq = batch_seq.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(batch_seq)
            
            backbone_loss = F.cross_entropy(
                outputs['backbone_logits'].reshape(-1, model.backbone.vocab_size),
                batch_targets.reshape(-1)
            )
            
            future_loss = torch.tensor(0.0, device=batch_seq.device)
            for i, future_logits in enumerate(outputs['future_logits']):
                if i + 1 < batch_targets.shape[1]:
                    target_shift = batch_targets[:, i+1:]
                    pred_shift = future_logits[:, :target_shift.shape[1]]
                    future_loss += F.cross_entropy(
                        pred_shift.reshape(-1, model.backbone.vocab_size),
                        target_shift.reshape(-1)
                    )
            future_loss = future_loss / len(outputs['future_logits'])
            
            skim_loss = F.cross_entropy(
                outputs['skim_logits'].reshape(-1, model.backbone.vocab_size),
                batch_targets.reshape(-1)
            )
            
            total_loss = backbone_loss + 0.5 * future_loss + 0.3 * skim_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_losses['backbone'] += backbone_loss.item()
            epoch_losses['future'] += future_loss.item()
            epoch_losses['skim'] += skim_loss.item()
            epoch_losses['total'] += total_loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'backbone': f"{backbone_loss.item():.4f}",
                'future': f"{future_loss.item():.4f}",
                'skim': f"{skim_loss.item():.4f}"
            })
        
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            losses[key].append(epoch_losses[key])
        
        print(f"Epoch {epoch+1} - Total Loss: {epoch_losses['total']:.4f}")
    
    return losses

def plot_training_curves(losses: Dict[str, List[float]], save_path: str):
    """Plot and save training loss curves as PDF."""
    plt.figure(figsize=(12, 8))
    
    epochs = range(1, len(losses['total']) + 1)
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs, losses['total'], 'b-', linewidth=2)
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, losses['backbone'], 'r-', linewidth=2)
    plt.title('Backbone Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs, losses['future'], 'g-', linewidth=2)
    plt.title('Future Heads Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(epochs, losses['skim'], 'm-', linewidth=2)
    plt.title('Skim Head Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")

def train_main():
    """Main training function."""
    print("Starting HydraSpec model training...")
    
    seed_everything(42)
    
    from preprocess import load_data
    train_sequences, train_targets = load_data()
    
    print(f"Loaded {len(train_sequences)} training sequences")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = HydraSpecModel(vocab_size=256, hidden_size=128, num_heads=3)
    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    losses = train_model(
        model=model,
        train_sequences=train_sequences,
        train_targets=train_targets,
        num_epochs=5,
        batch_size=32,
        learning_rate=1e-3,
        device=device
    )
    
    model.backbone.export_quantized_head()
    
    os.makedirs("models", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': 256,
        'hidden_size': 128,
        'num_heads': 3
    }, "models/hydraspec_model.pt")
    
    os.makedirs(".research/iteration1/images", exist_ok=True)
    plot_training_curves(losses, ".research/iteration1/images/training_curves.pdf")
    
    print("Training completed successfully!")

if __name__ == "__main__":
    train_main()
