from pathlib import Path
from typing import Any
import hydra
import numpy as np
import rootutils
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
from omegaconf import OmegaConf


root = Path(rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False))
from src.utils import resolvers  # noqa: F401 ensures resolver is registered

from src.overcomplete.metrics import l0_eps
from src.overcomplete.sae.trackers import DeadCodeTracker
from src.overcomplete.sae.train import _compute_reconstruction_error, extract_input
from src.overcomplete.metrics import l0, hoyer, r2_score, relative_avg_l2_loss
from src.visualization.vis_utils import index_to_label_dict

from src.utils.model_load_utils import load_sae, get_image_encoder
from src.utils.data_utils import load_embedding_datamodule


class LitSparseAutoencoder(pl.LightningModule):
    def __init__(self, cfg, criterion=None, pretrained: bool = False):
        super().__init__()
        self.cfg = cfg
        self.model = load_sae(cfg, pretrained=pretrained)
        self.learning_rate = cfg.sae.lr
        self.nb_concepts = self.model.nb_concepts

        # AuxK hyper-parameters
        self.alpha = cfg.sae.get("alpha", 1 / 32)
        self.top_k_aux = cfg.sae.get("top_k_aux", 512)

        self.clip_model = None
        self.text_features = None

        # default MSE criterion
        if criterion is None:
            self.criterion = lambda x, x_hat, *args: F.mse_loss(
                x_hat, x, reduction="mean"
            )
        else:
            self.criterion = criterion

        # accumulate statistics each epoch
        self.register_buffer("_epoch_loss", torch.tensor(0.0), persistent=False)
        self.register_buffer("_epoch_error", torch.tensor(0.0), persistent=False)
        self.register_buffer("_epoch_sparsity", torch.tensor(0.0), persistent=False)
        self.register_buffer("_batch_count", torch.tensor(0), persistent=False)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def on_train_epoch_start(self):
        # reset counters
        self._epoch_loss.zero_()
        self._epoch_error.zero_()
        self._epoch_sparsity.zero_()
        self._batch_count.zero_()

    def training_step(self, batch, batch_idx):
        x = extract_input(batch).to(self.device, non_blocking=True)

        output_dict = self.model(x)
        loss = output_dict["loss"]
        dead_features = output_dict["num_dead_features"]

        z = output_dict["feature_acts"]
        x_hat = output_dict["sae_out"]

        rec_err = _compute_reconstruction_error(x, x_hat)
        sparsity = l0_eps(z, 0).sum()
        self._epoch_loss += loss.detach()
        self._epoch_error += rec_err
        self._epoch_sparsity += sparsity
        self._batch_count += 1

        # log per‐step loss
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log(
            "train/dead_features",
            dead_features,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def on_after_backward(self):
        if hasattr(self.model, "make_decoder_weights_and_grad_unit_norm"):
            self.model.make_decoder_weights_and_grad_unit_norm()

    def on_train_epoch_end(self):
        # compute averages
        count = self._batch_count.item()
        avg_loss = (self._epoch_loss / count).item()
        avg_r2 = (self._epoch_error / count).item()
        avg_l0 = (self._epoch_sparsity / count).item()

        # log epoch metrics
        self.log("train/avg_loss", avg_loss)
        self.log("train/r2", avg_r2)
        self.log("train/z_sparsity", avg_l0)

    def validation_step(self, batch, batch_idx):
        x = extract_input(batch).to(self.device, non_blocking=True)

        output_dict = self.model(x)
        loss = output_dict["loss"]

        z = output_dict["feature_acts"]
        x_hat = output_dict["sae_out"]

        rec_err = _compute_reconstruction_error(x, x_hat)
        sparsity = l0_eps(z, 0).sum()

        l0_score = l0(z).item()
        hs = hoyer(z).mean().item()
        r2 = r2_score(x, x_hat).item()
        avg_l2_loss = relative_avg_l2_loss(x, x_hat)
        if batch_idx % 10 == 0 and self.cfg.model.name == "clip":
            acc_recon, acc_orig, preds, preds_orig = (
                self.calculate_zero_shot_reconstruction_metric(embeddings_batch=batch)
            )
            self.log("val/acc_recon", acc_recon, on_epoch=True, prog_bar=True)
            self.log("val/acc_orig", acc_orig, on_epoch=True)

        # Lightning will average across batches
        self.log("val/loss", loss.detach(), prog_bar=True, on_epoch=True)

        self.log("val/r2", rec_err, on_epoch=True)
        self.log("val/z_sparsity", sparsity, on_epoch=True)
        self.log("val/l0", l0_score, on_epoch=True)
        self.log("val/hoyer", hs, on_epoch=True)
        self.log("val/r2_score", r2, on_epoch=True)
        self.log("val/relative_avg_l2_loss", avg_l2_loss, on_epoch=True)

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        """
        Test step is the same as validation step in this case.
        """
        return self.validation_step(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            eps=6.25e-10,  # both from scaling paper eps and betas (standard for adam
            betas=(0.9, 0.999),
        )
        # if self.cfg.train_sae.use_scheduler:
        #     scheduler = torch.optim.lr_scheduler.YourSchedulerHere(
        #         optimizer,
        #         **self.cfg.train_sae.scheduler_kwargs
        #     )
        #     return {
        #         "optimizer": optimizer,
        #         "lr_scheduler": {
        #             "scheduler": scheduler,
        #             "interval": "step"  # or "epoch"
        #         }
        #     }
        return optimizer

    def calculate_zero_shot_reconstruction_metric(
        self,
        embeddings_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        """
        Calculates zero-shot reconstruction accuracy for EuroSAT using CLIP features.

        Args:
            embeddings_batch: Tuple containing image features, labels, and keys as torch.Tensors.

        Returns:
            acc_recon: Accuracy for reconstructed features.
            acc_orig: Accuracy for original features.
            preds: Predicted class indices for reconstructed features.
            preds_orig: Predicted class indices for original features.
        """
        if self.clip_model is None:
            self.clip_model, _, _ = get_image_encoder(
                self.cfg.model, device=self.device
            )

        self.clip_model.eval()
        self.model.eval()
        #  encode class labels
        # Calculate overall zero-shot accuracy for EuroSAT on reconstructed features
        # Get CLIP text features for EuroSAT classes

        if self.text_features is None:
            idx2label = index_to_label_dict(
                self.cfg.dataset.name, self.cfg.dataset.get("path", None)
            )
            classes = list(idx2label.values())

            text_tokens = self.clip_model.tokenizer(classes).to(self.device)
            self.text_features = self.clip_model.encode_text(text_tokens)
            self.text_features = self.text_features / self.text_features.norm(
                dim=-1, keepdim=True
            )
            # to cpu
            self.text_features = self.text_features.cpu()
        image_features, labels, keys = embeddings_batch

        # Reconstruct features using topk_sae
        with torch.no_grad():
            z, _ = self.model.encode(image_features.to(self.device))
            X_hat = self.model.decode(z)
            X_hat = X_hat / X_hat.norm(dim=-1, keepdim=True)

        # Compute similarity and predict for reconstructed features
        logits = X_hat.cpu() @ self.text_features.T
        preds = logits.argmax(dim=-1).cpu().numpy()

        # Compute similarity and predict for original CLIP features
        X_orig = image_features / image_features.norm(dim=-1, keepdim=True)
        logits_orig = X_orig.cpu() @ self.text_features.T
        preds_orig = logits_orig.argmax(dim=-1).cpu().numpy()
        labels = labels.argmax(dim=-1).cpu().numpy()

        acc_recon = (preds == labels).mean()
        acc_orig = (preds_orig == labels).mean()

        return acc_recon, acc_orig, preds, preds_orig


def build_wandb_logger(run_name: str, tags: list, debug=False) -> WandbLogger:
    wandb_dir = root / "logs" / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    mode = "offline" if not debug else "online"
    print(f"Using wandb mode: {mode}")
    return WandbLogger(
        project="sae_training",
        name=run_name,
        tags=tags,
        mode=mode,
        dir=wandb_dir,
    )


@hydra.main(
    config_path=str(root / "config"),
    config_name="base.yaml",
    version_base="1.2",
)
def train_with_lightning(cfg: DictConfig, override: bool = False):
    if cfg.sae.model_type in ["random", "frozen"]:
        print("SAE is random, skipping training.")
        return

    final_sae_path = Path(cfg.paths.final_sae_path)
    final_sae_path.parent.mkdir(parents=True, exist_ok=True)
    if final_sae_path.exists() and not override:
        print(
            f"Final SAE path {final_sae_path} already exists. Skipping training. Use override=True to force retrain."
        )
        return
    model = LitSparseAutoencoder(cfg)
    dm = load_embedding_datamodule(cfg)
    dm.setup()

    print(f"Datamodule has {len(dm.train_dataset)} training samples.")
    print(f"Datamodule has {len(dm.val_dataset)} validation samples.")
    print(f"Datamodule has {len(dm.test_dataset)} test samples.")

    # --- Path preparation ---
    date_and_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    print(f"Training SAE with run name: {cfg.train_sae.run_name}")
    output_root = Path(cfg.paths.trained_sae_dir) / f"{date_and_time}"
    run_name = f"{cfg.train_sae.run_name}"
    # append date and time to run name
    run_name += f"_{date_and_time}"
    tags = [
        cfg.dataset.name,
        cfg.model.model_identifier,
        cfg.seed,
        cfg.train_sae.run_name,
    ]
    debug = cfg.get("debug", False)
    # wandb_logger = build_wandb_logger(run_name, tags, True)
    # wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    base_logger = pl.loggers.CSVLogger(save_dir=output_root)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=output_root,
            filename=f"{run_name}-best",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.TQDMProgressBar(refresh_rate=10),
        pl.callbacks.EarlyStopping(
            monitor="val/loss",
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.train_sae.epochs,
        callbacks=callbacks,
        gradient_clip_val=True,
        log_every_n_steps=50,
        logger=base_logger,
        default_root_dir=root / "logs" / "lightning" / run_name,
    )
    print(f"Fit model")
    # create wandb logger

    trainer.fit(model, datamodule=dm)

    # todo save best model instead
    print("Training complete. Saving model...")
    output_root.mkdir(parents=True, exist_ok=True)
    model_save_path = output_root / f"{cfg.train_sae.run_name}.pt"
    torch.save(model.model.state_dict(), model_save_path)
    print(f"✅ Model saved to: {model_save_path}")

    # if the cfg.paths.final_sae_path file does not exist yet, also save the model there

    final_sae_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.model.state_dict(), final_sae_path)
    print(f"✅ Model saved to: {final_sae_path}")


if __name__ == "__main__":
    train_with_lightning()
