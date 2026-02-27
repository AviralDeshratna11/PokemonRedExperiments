"""
behavioral_cloning.py
---------------------
Phase 1: Behavioral Cloning (BC) from recorded human gameplay.

Pipeline:
  1. BCDataset    — loads HDF5 files, sequences-of-frames for temporal context
  2. BCPolicy     — CNN + RAM encoder + LSTM + actor head
  3. BCTrainer    — trains with cross-entropy loss on expert actions
  4. train()      — full training loop with validation & checkpointing

The trained weights serve as the initialization for Phase 2 (PPO fine-tuning).

Usage:
    python bc/behavioral_cloning.py \
        --data_dir data/ \
        --output   checkpoints/bc_policy.pt \
        --epochs   30

Architecture overview:
    screen (3×144×160) ──→ CNN encoder ──┐
                                          ├──→ fusion ──→ LSTM ──→ actor (8 logits)
    ram    (128,)      ──→ MLP encoder ──┘
"""

import sys
import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────────────────────────────────────

class BCDataset(Dataset):
    """
    Loads sequences of (screen, ram, action) tuples from HDF5 files.

    Each item is a sliding window of `seq_len` consecutive steps.
    The target is the action at the LAST step in the window.
    Earlier steps provide temporal context for the LSTM.

    Args:
        h5_paths:  list of HDF5 file paths
        seq_len:   sequence length fed to LSTM (e.g. 16)
        stride:    step between consecutive windows (1 = maximum overlap)
        min_rtg:   minimum return-to-go to include a sample (filters bad play)
    """

    def __init__(
        self,
        h5_paths:  List[str],
        seq_len:   int   = 16,
        stride:    int   = 4,
        min_rtg:   float = -np.inf,
        augment:   bool  = True,
    ):
        self.seq_len = seq_len
        self.augment = augment
        self.samples: List[Tuple[np.ndarray, np.ndarray, int]] = []
        # self.samples entries: (screen_seq, ram_seq, action_target)

        for h5_path in h5_paths:
            with h5py.File(h5_path, "r") as f:
                episodes = sorted(k for k in f.keys() if k.startswith("episode_"))
                for ep_key in episodes:
                    grp      = f[ep_key]
                    n        = grp.attrs["n_steps"]
                    if n < seq_len:
                        continue

                    screens  = grp["screens"][:]    # (N, 3, H, W) uint8
                    rams     = grp["rams"][:]        # (N, 128)     float32
                    actions  = grp["actions"][:]     # (N,)         int32
                    rtgs     = grp["returns_to_go"][:] if "returns_to_go" in grp else np.zeros(n)

                    # Slide window
                    for start in range(0, n - seq_len + 1, stride):
                        end     = start + seq_len
                        rtg_val = float(rtgs[end - 1])
                        if rtg_val < min_rtg:
                            continue

                        self.samples.append((
                            screens[start:end].copy(),  # (seq_len, 3, H, W)
                            rams[start:end].copy(),      # (seq_len, 128)
                            int(actions[end - 1]),       # scalar action target
                            rtg_val,
                        ))

        print(f"Dataset: {len(self.samples):,} windows from {len(h5_paths)} file(s)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        screens, rams, action, rtg = self.samples[idx]

        # Convert screen to float [0, 1]
        screens_f = torch.from_numpy(screens).float() / 255.0  # (T, 3, H, W)

        # Optional: horizontal flip augmentation (Pokemon maps are mostly symmetric)
        if self.augment and random.random() < 0.3:
            screens_f = torch.flip(screens_f, dims=[-1])

        rams_t    = torch.from_numpy(rams).float()            # (T, 128)
        action_t  = torch.tensor(action, dtype=torch.long)
        rtg_t     = torch.tensor(rtg,    dtype=torch.float32)

        return screens_f, rams_t, action_t, rtg_t


def build_dataloaders(
    data_dir:   str,
    seq_len:    int = 16,
    batch_size: int = 64,
    val_split:  float = 0.1,
    num_workers:int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Split HDF5 files in data_dir into train/val dataloaders."""
    h5_files = sorted(Path(data_dir).glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {data_dir}")

    random.shuffle(h5_files)
    n_val   = max(1, int(len(h5_files) * val_split))
    val_f   = [str(f) for f in h5_files[:n_val]]
    train_f = [str(f) for f in h5_files[n_val:]]

    print(f"Train files: {len(train_f)} | Val files: {len(val_f)}")

    train_ds = BCDataset(train_f, seq_len=seq_len, stride=4,   augment=True)
    val_ds   = BCDataset(val_f,   seq_len=seq_len, stride=8,   augment=False)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

    return train_dl, val_dl


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class CNNEncoder(nn.Module):
    """
    Atari-style CNN for encoding 144×160 game frames.
    Output: (batch, 3136) flat feature vector.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,  32, kernel_size=8, stride=4),  # → (32, 35, 39)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # → (64, 16, 18)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # → (64, 14, 16)
            nn.ReLU(),
            nn.Flatten(),                                  # → 64*14*16 = 14336
        )
        # compute actual output size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 144, 160)
            self._out_size = self.net(dummy).shape[1]

    def forward(self, x):
        return self.net(x)

    @property
    def out_size(self):
        return self._out_size


class RAMEncoder(nn.Module):
    """MLP for encoding the 128-dim RAM feature vector."""
    def __init__(self, in_dim=128, hidden=256, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )
        self.out_size = out_dim

    def forward(self, x):
        return self.net(x)


class BCPolicy(nn.Module):
    """
    Behavioural Cloning policy.

    Input per time step:
      screen: (batch, T, 3, 144, 160) float32
      ram:    (batch, T, 128)         float32

    The CNN and RAM encoders process each frame independently.
    The fused features are passed through an LSTM over the T steps.
    The LSTM hidden state at step T is used to predict the action.

    This is identical to the PPO policy used in Phase 2,
    allowing direct weight transfer.
    """

    def __init__(
        self,
        n_actions:   int   = 8,
        lstm_hidden: int   = 512,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.cnn_enc  = CNNEncoder()
        self.ram_enc  = RAMEncoder(in_dim=128, out_dim=256)
        self.n_actions = n_actions

        fused_size = self.cnn_enc.out_size + self.ram_enc.out_size

        self.fusion = nn.Sequential(
            nn.Linear(fused_size, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size  = lstm_hidden,
            hidden_size = lstm_hidden,
            num_layers  = 1,
            batch_first = True,
        )

        self.actor  = nn.Linear(lstm_hidden, n_actions)
        self.critic = nn.Linear(lstm_hidden, 1)   # value head (for PPO later)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Actor head: small init so policy starts near uniform
        nn.init.orthogonal_(self.actor.weight, gain=0.01)

    def init_lstm_state(self, batch_size: int, device):
        h = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
        return (h, c)

    def forward(self, screens, rams, lstm_state=None):
        """
        Args:
            screens:    (B, T, 3, H, W) float32
            rams:       (B, T, 128)     float32
            lstm_state: optional (h, c) tuple

        Returns:
            logits:     (B, n_actions)   — logits at last time step
            value:      (B, 1)           — value estimate
            lstm_state: (h, c)           — updated LSTM state
        """
        B, T, C, H, W = screens.shape

        # Encode frames: flatten batch+time, encode, reshape
        screens_flat   = screens.view(B * T, C, H, W)
        cnn_out        = self.cnn_enc(screens_flat)          # (B*T, cnn_out)
        rams_flat      = rams.view(B * T, -1)
        ram_out        = self.ram_enc(rams_flat)             # (B*T, 256)

        fused          = torch.cat([cnn_out, ram_out], dim=-1)  # (B*T, fused)
        fused          = self.fusion(fused)                   # (B*T, lstm_hidden)
        fused          = fused.view(B, T, -1)                # (B, T, lstm_hidden)

        # LSTM over time dimension
        if lstm_state is None:
            lstm_state = self.init_lstm_state(B, screens.device)

        lstm_out, new_state = self.lstm(fused, lstm_state)   # (B, T, hidden)

        # Use only the last time step for prediction
        last = lstm_out[:, -1, :]                            # (B, hidden)

        logits = self.actor(last)                            # (B, n_actions)
        value  = self.critic(last)                           # (B, 1)

        return logits, value, new_state

    def get_action(self, screens, rams, lstm_state=None, deterministic=False):
        """Convenience: get action index and updated LSTM state."""
        with torch.no_grad():
            logits, _, new_state = self.forward(screens, rams, lstm_state)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = torch.distributions.Categorical(logits=logits).sample()
        return action, new_state


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class BCTrainer:
    """
    Trains BCPolicy via cross-entropy loss on expert actions.

    Features:
      - Weighted CE loss: samples with higher return-to-go contribute more
      - Gradient clipping
      - LR schedule (cosine annealing)
      - Early stopping on validation accuracy
      - Checkpoint saving
    """

    def __init__(
        self,
        policy:      BCPolicy,
        train_dl:    DataLoader,
        val_dl:      DataLoader,
        lr:          float = 3e-4,
        weight_decay:float = 1e-4,
        grad_clip:   float = 0.5,
        device:      str   = "cuda",
        output_dir:  str   = "checkpoints",
    ):
        self.policy      = policy.to(device)
        self.train_dl    = train_dl
        self.val_dl      = val_dl
        self.device      = device
        self.grad_clip   = grad_clip
        self.output_dir  = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            policy.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.best_val_acc = 0.0
        self.patience_counter = 0

    def train_epoch(self, epoch: int) -> dict:
        self.policy.train()
        total_loss = 0.0
        correct    = 0
        total      = 0

        pbar = tqdm(self.train_dl, desc=f"Epoch {epoch:3d} [train]", leave=False)
        for screens, rams, actions, rtgs in pbar:
            screens = screens.to(self.device)   # (B, T, 3, H, W)
            rams    = rams.to(self.device)       # (B, T, 128)
            actions = actions.to(self.device)    # (B,)
            rtgs    = rtgs.to(self.device)       # (B,)

            # Forward
            logits, _, _ = self.policy(screens, rams)

            # Weighted cross-entropy
            # Weight = softmax of RTG (encourages imitating high-quality play)
            rtg_weights = torch.softmax(rtgs / (rtgs.std() + 1e-8), dim=0) * len(rtgs)
            loss = F.cross_entropy(logits, actions, reduction="none")
            loss = (loss * rtg_weights).mean()

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            preds      = logits.argmax(dim=-1)
            correct    += (preds == actions).sum().item()
            total      += len(actions)

            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             acc=f"{correct/total:.3f}")

        return {
            "loss": total_loss / len(self.train_dl),
            "acc":  correct / total,
        }

    @torch.no_grad()
    def val_epoch(self) -> dict:
        self.policy.eval()
        total_loss = 0.0
        correct    = 0
        total      = 0

        for screens, rams, actions, rtgs in self.val_dl:
            screens = screens.to(self.device)
            rams    = rams.to(self.device)
            actions = actions.to(self.device)

            logits, _, _ = self.policy(screens, rams)
            loss         = F.cross_entropy(logits, actions)

            total_loss  += loss.item()
            correct     += (logits.argmax(-1) == actions).sum().item()
            total       += len(actions)

        return {
            "loss": total_loss / len(self.val_dl),
            "acc":  correct / total,
        }

    def save(self, path: str, extra: dict = None):
        state = {
            "model_state":     self.policy.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        if extra:
            state.update(extra)
        torch.save(state, path)
        print(f"  Saved: {path}")

    def train(self, n_epochs: int = 30, patience: int = 5) -> BCPolicy:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs, eta_min=1e-5
        )

        print(f"\n{'='*60}")
        print(f"  Behavioral Cloning Training — {n_epochs} epochs")
        print(f"  Device: {self.device}")
        print(f"{'='*60}\n")

        for epoch in range(1, n_epochs + 1):
            train_stats = self.train_epoch(epoch)
            val_stats   = self.val_epoch()
            scheduler.step()

            print(
                f"  Epoch {epoch:3d}/{n_epochs} | "
                f"Train loss: {train_stats['loss']:.4f}  acc: {train_stats['acc']:.3f} | "
                f"Val   loss: {val_stats['loss']:.4f}  acc: {val_stats['acc']:.3f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

            # Save best
            if val_stats["acc"] > self.best_val_acc:
                self.best_val_acc     = val_stats["acc"]
                self.patience_counter = 0
                self.save(
                    str(self.output_dir / "bc_best.pt"),
                    extra={"epoch": epoch, "val_acc": val_stats["acc"]},
                )
            else:
                self.patience_counter += 1

            # Save every 5 epochs
            if epoch % 5 == 0:
                self.save(str(self.output_dir / f"bc_epoch_{epoch:03d}.pt"))

            # Early stopping
            if self.patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(val acc didn't improve for {patience} epochs).")
                break

        # Final save
        self.save(str(self.output_dir / "bc_final.pt"),
                  extra={"best_val_acc": self.best_val_acc})
        print(f"\n  Best val accuracy: {self.best_val_acc:.3f}")
        print("  BC training complete.")
        return self.policy


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  default="data",         help="Dir with .h5 files")
    p.add_argument("--output",    default="checkpoints",   help="Output dir")
    p.add_argument("--epochs",    type=int,  default=30)
    p.add_argument("--batch",     type=int,  default=64)
    p.add_argument("--lr",        type=float,default=3e-4)
    p.add_argument("--seq_len",   type=int,  default=16,
                   help="Sequence length fed to LSTM")
    p.add_argument("--patience",  type=int,  default=7,
                   help="Early stopping patience in epochs")
    p.add_argument("--workers",   type=int,  default=4)
    p.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def train():
    args     = parse_args()
    device   = args.device

    # Build dataloaders
    train_dl, val_dl = build_dataloaders(
        args.data_dir,
        seq_len     = args.seq_len,
        batch_size  = args.batch,
        num_workers = args.workers,
    )

    # Build policy
    policy  = BCPolicy(n_actions=8, lstm_hidden=512, dropout=0.1)
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Policy parameters: {n_params:,}")

    # Train
    trainer = BCTrainer(
        policy     = policy,
        train_dl   = train_dl,
        val_dl     = val_dl,
        lr         = args.lr,
        device     = device,
        output_dir = args.output,
    )
    trained_policy = trainer.train(n_epochs=args.epochs, patience=args.patience)

    print(f"\nBC policy saved to {args.output}/bc_best.pt")
    print("Next: run Phase 2 PPO fine-tuning starting from these weights.")


if __name__ == "__main__":
    train()
