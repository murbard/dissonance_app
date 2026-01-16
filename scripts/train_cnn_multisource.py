"""
CNN training with multi-source synthetic generator and Monte Carlo validation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Import the synthetic generator
sys.path.insert(0, os.path.dirname(__file__))
from synth_generator import generate_random_sample, MultiSourceSample


class RoughnessCNN(nn.Module):
    """1D CNN for roughness estimation."""
    def __init__(self, base_channels=32):
        super().__init__()
        
        self.conv1 = nn.Conv1d(1, base_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(base_channels)
        
        self.block1 = self._make_block(base_channels, base_channels * 2, stride=4)
        self.block2 = self._make_block(base_channels * 2, base_channels * 4, stride=4)
        self.block3 = self._make_block(base_channels * 4, base_channels * 8, stride=4)
        self.block4 = self._make_block(base_channels * 8, base_channels * 8, stride=4)
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def _make_block(self, in_ch, out_ch, stride):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.head(x).squeeze(-1)


def generate_batch(batch_size: int, sr: int, duration: float, device: str):
    """Generate a batch of multi-source samples with targets."""
    signals = []
    targets = []
    
    for _ in range(batch_size):
        sample = generate_random_sample(
            n_harmonics=16, sr=sr, duration=duration,
            min_sources=1, max_sources=4,
            octave_spread=0.5, inharmonicity_cents=10.0
        )
        sig = sample.generate_signal(device)
        target = sample.compute_sethares_dissonance()
        signals.append(sig)
        targets.append(target)
    
    return torch.stack(signals), torch.tensor(targets, dtype=torch.float32, device=device)


def monte_carlo_validation(model, sr, duration, device, batch_size=64, target_cv=0.01, max_batches=100):
    """
    Compute validation loss until std/mean < target_cv (coefficient of variation).
    Returns mean loss and number of batches used.
    """
    from tqdm import tqdm
    
    model.eval()
    mse = nn.MSELoss()
    
    # Online computation of mean and variance
    sum_x = 0.0
    sum_x2 = 0.0
    n = 0
    
    pbar = tqdm(range(max_batches), desc="Validation", leave=False)
    
    with torch.no_grad():
        for i in pbar:
            signals, targets = generate_batch(batch_size, sr, duration, device)
            preds = model(signals)
            
            # Normalize
            p_norm = (preds - preds.min()) / (preds.max() - preds.min() + 1e-8)
            t_norm = (targets - targets.min()) / (targets.max() - targets.min() + 1e-8)
            
            loss = mse(p_norm, t_norm).item()
            
            # Update running sums
            sum_x += loss
            sum_x2 += loss * loss
            n += 1
            
            # Check convergence after at least 5 batches
            if n >= 5:
                mean_loss = sum_x / n
                # Var(X) = E(X^2) - E(X)^2
                var_loss = (sum_x2 / n) - (mean_loss ** 2)
                std_loss = np.sqrt(max(0, var_loss))
                # Standard error of the mean
                sem = std_loss / np.sqrt(n)
                cv = sem / (mean_loss + 1e-8)
                cv_pct = cv * 100
                
                pbar.set_postfix({'n': n, 'SEM%': f'{cv_pct:.2f}'})
                
                if cv < target_cv:
                    pbar.close()
                    break
    
    return sum_x / n, n


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    sr = 22050
    duration = 0.5
    model = RoughnessCNN(base_channels=32).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Learnable parameters: {n_params}")
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    
    # Load checkpoint if exists
    checkpoint_path = 'best_cnn_multisource.pt'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if not result.missing_keys and not result.unexpected_keys:
                best_val_loss = checkpoint.get('val_loss', float('inf'))
                print(f"Loaded checkpoint, best_val_loss: {best_val_loss:.5f}")
            else:
                best_val_loss = float('inf')
        else:
            best_val_loss = float('inf')
    else:
        best_val_loss = float('inf')
        print("No checkpoint found, starting fresh.")
    
    batch_size = 64
    n_steps = 100000
    val_every = 10000  # Validate every N steps
    log_every = 10     # Log EMA every N steps
    mse_criterion = nn.MSELoss()
    
    # EMA for loss tracking
    ema_loss = None
    ema_alpha = 0.1  # EMA decay factor
    
    print("Starting training with multi-source generator...")
    print(f"Validation every {val_every} steps using Monte Carlo (1% CV target, max 10k batches)")
    print(f"EMA loss reported every {log_every} steps")
    
    for step in range(n_steps):
        model.train()
        optimizer.zero_grad()
        
        # Generate batch
        signals, targets = generate_batch(batch_size, sr, duration, device)
        
        preds = model(signals)
        
        # Normalize
        p_norm = (preds - preds.min()) / (preds.max() - preds.min() + 1e-8)
        t_norm = (targets - targets.min()) / (targets.max() - targets.min() + 1e-8)
        
        loss = mse_criterion(p_norm, t_norm)
        
        if torch.isnan(loss):
            print(f"Step {step}: NaN loss")
            continue
        
        # Update EMA
        loss_val = loss.item()
        if ema_loss is None:
            ema_loss = loss_val
        else:
            ema_loss = ema_alpha * loss_val + (1 - ema_alpha) * ema_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Log EMA loss
        if step % log_every == 0 and step % val_every != 0:
            print(f"Step {step}: EMA Loss {ema_loss:.5f}")
        
        # Monte Carlo validation
        if step % val_every == 0:
            val_loss, n_batches = monte_carlo_validation(
                model, sr, duration, device,
                batch_size=64, target_cv=0.01, max_batches=10000
            )
            
            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]['lr']
            print(f"Step {step}: EMA {ema_loss:.5f}, ValLoss {val_loss:.5f} ({n_batches} batches), lr={lr:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"  >>> NEW BEST!")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                    'step': step
                }, checkpoint_path)
    
    print(f"\nDone. Best ValLoss: {best_val_loss:.5f}")


if __name__ == "__main__":
    train()
