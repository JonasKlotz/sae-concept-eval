import os

import torch
from torch import nn as nn


class LinearAttrPredictor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        cap_to_k: bool = True,
        default_thresh: float = 0.5,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.cap_to_k = cap_to_k
        self.nb_concepts = out_dim

        # Stored with the model, moved to GPU with .to(device), saved in state_dict.
        self.register_buffer(
            "thresholds", torch.full((out_dim,), float(default_thresh))
        )

    def set_thresholds(self, thresholds: torch.Tensor):
        """
        thresholds: shape (C,) in probability space, values in [0,1].
        """
        if thresholds.ndim != 1 or thresholds.numel() != self.nb_concepts:
            raise ValueError(
                f"Expected thresholds of shape ({self.nb_concepts},), got {tuple(thresholds.shape)}."
            )
        thresholds = thresholds.detach().to(
            self.thresholds.device, dtype=self.thresholds.dtype
        )
        self.thresholds.copy_(thresholds)

    def forward(self, x):
        return self.linear(x)  # logits

    @torch.no_grad()
    def encode(self, x, k: int = 32, return_binary: bool = True):
        """
        Returns:
            If cap_to_k=True:
                acts_topk: (B, C) probs with only top-k entries kept (others 0)
                probs:     (B, C) full probabilities
            If cap_to_k=False:
                preds:     (B, C) binary (or float 0/1) predictions based on per-label thresholds
                probs:     (B, C) full probabilities
        """
        logits = self.forward(x)
        probs = torch.sigmoid(logits)  # (B, C)

        if not self.cap_to_k:
            # per-label thresholding in probability space
            thr = self.thresholds.view(1, -1)  # (1, C)
            preds = probs >= thr
            if return_binary:
                preds = preds.to(probs.dtype)
            return preds, probs

        B, C = probs.shape
        k = min(k, C)
        topk_vals, topk_idx = torch.topk(probs, k=k, dim=1)

        acts_topk = torch.zeros_like(probs)
        acts_topk.scatter_(1, topk_idx, topk_vals)
        return acts_topk, probs


def train_linear_predictor(
    train_data_loader,
    val_data_loader=None,
    epochs: int = 10,
    lr: float = 1e-3,
    device=None,
    cap_to_k=True,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    features, labels, _ = next(iter(train_data_loader))
    in_dim = features.shape[1]
    out_dim = labels.shape[1]

    model = LinearAttrPredictor(in_dim, out_dim, cap_to_k=cap_to_k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_model_state = None

    for ep in range(epochs):
        # Training
        model.train()
        total, n = 0.0, 0
        number_of_labels_per_sample = 0
        for features, labels, _ in train_data_loader:
            features = features.to(device).float()
            labels = labels.to(device).float()

            logits = model(features)
            loss = loss_fn(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = features.size(0)
            total += loss.item() * bs
            n += bs
            number_of_labels_per_sample += labels.sum().item() / bs

        train_loss = total / n
        number_of_labels_per_sample /= len(train_data_loader)
        print(f"epoch {ep + 1:02d}  train_loss {train_loss:.4f}")

        # Validation every 5 epochs (and at the last epoch)
        if val_data_loader is not None and ((ep + 1) % 5 == 0 or ep == epochs - 1):
            model.eval()
            val_total, val_n = 0.0, 0
            with torch.no_grad():
                for features, labels, _ in val_data_loader:
                    features = features.to(device).float()
                    labels = labels.to(device).float()

                    logits = model(features)
                    loss = loss_fn(logits, labels)

                    bs = features.size(0)
                    val_total += loss.item() * bs
                    val_n += bs

            val_loss = val_total / val_n
            print(f"         val_loss {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                print(f"         New best model!")

    print(
        f"Avg number of positive labels per sample: {number_of_labels_per_sample:.4f}"
    )

    # Load best model if validation was used
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with val_loss {best_val_loss:.4f}")

    return model


def get_or_train_linear_predictor(
    data_loader, val_loader, save_path, epochs=20, lr=1e-3, device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    features, labels, _ = next(iter(data_loader))
    in_dim = features.shape[1]
    out_dim = labels.shape[1]

    model = LinearAttrPredictor(in_dim, out_dim).to(device)

    if os.path.exists(save_path):
        print(f"Loading linear predictor from {save_path}")
        state = torch.load(save_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model

    print("Training linear predictor from scratch")
    model = train_linear_predictor(
        data_loader, val_data_loader=val_loader, epochs=epochs, lr=lr, device=device
    )

    torch.save(model.state_dict(), save_path)
    print(f"Saved linear predictor to {save_path}")
    model.eval()
    return model
