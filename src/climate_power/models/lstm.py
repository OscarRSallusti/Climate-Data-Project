"""Sequence-to-sequence LSTM.

The encoder reads the last week of prices and weather; its final state seeds a
decoder that walks the 336 target hours, fed with the weather expected at each
hour plus calendar features. The network outputs a correction to the 7-day
hour-of-day price profile rather than the price itself.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from ..features import DEC_PROFILE, Samples


class Seq2Seq(nn.Module):
    def __init__(self, n_enc: int, n_dec: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.LSTM(n_enc, hidden, batch_first=True)
        self.decoder = nn.LSTM(n_dec, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden + n_dec, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, enc: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        _, state = self.encoder(enc)
        out, _ = self.decoder(dec, state)
        return self.head(torch.cat([out, dec], dim=-1)).squeeze(-1) + dec[..., DEC_PROFILE]


def _tensors(s: Samples):
    return torch.from_numpy(s.enc), torch.from_numpy(s.dec), torch.from_numpy(s.y.astype("float32"))


class LSTMForecaster:
    """Small ensemble of identically configured networks with different seeds."""

    def __init__(self, seeds=(0, 1, 2), hidden: int = 64, epochs: int = 80, patience: int = 12):
        self.seeds, self.hidden, self.epochs, self.patience = seeds, hidden, epochs, patience
        self.nets: list[Seq2Seq] = []

    def _fit_one(self, seed: int, train: Samples, val: Samples) -> Seq2Seq:
        torch.manual_seed(seed)
        enc, dec, y = _tensors(train)
        v_enc, v_dec, v_y = _tensors(val)
        net = Seq2Seq(enc.shape[-1], dec.shape[-1], self.hidden)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
        loss_fn = nn.L1Loss()
        best, best_state, stale = float("inf"), None, 0
        gen = torch.Generator().manual_seed(seed)
        for epoch in range(self.epochs):
            net.train()
            for batch in torch.randperm(len(enc), generator=gen).split(32):
                opt.zero_grad()
                loss = loss_fn(net(enc[batch], dec[batch]), y[batch])
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            net.eval()
            with torch.no_grad():
                val_loss = loss_fn(net(v_enc, v_dec), v_y).item()
            sched.step(val_loss)
            if val_loss < best - 1e-4:
                best, best_state, stale = val_loss, copy.deepcopy(net.state_dict()), 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        print(f"  lstm seed {seed}: best val MAE {best:.3f} (normalised) after {epoch + 1} epochs", flush=True)
        net.load_state_dict(best_state)
        return net.eval()

    def fit(self, train: Samples, val: Samples) -> LSTMForecaster:
        torch.set_num_threads(max(1, torch.get_num_threads()))
        self.nets = [self._fit_one(seed, train, val) for seed in self.seeds]
        return self

    def predict(self, s: Samples) -> np.ndarray:
        enc, dec = torch.from_numpy(s.enc), torch.from_numpy(s.dec)
        with torch.no_grad():
            return torch.stack([net(enc, dec) for net in self.nets]).mean(0).numpy()
