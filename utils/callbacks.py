#callbacks.py
import torch
from pytorch_lightning.callbacks import Callback

import os


class PretrainCheckpointCallback(Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint_filename = (
            "-".join(
                [
                    "pretrain",
                    pl_module.hparams.arch,
                    pl_module.hparams.dataset,
                    pl_module.hparams.comment,
                ]
            )
            + ".cp"
        )
        checkpoint_path = os.path.join(pl_module.hparams.checkpoint_dir, checkpoint_filename)
        torch.save(pl_module.model.state_dict(), checkpoint_path)


class DiscoverCheckpointCallback(Callback):
    """
    Callback to save CB-NCD discovery model checkpoint.
    Saves the full model (encoder + concept layer + clustering heads) for later
    interpretability analysis.
    """
    def __init__(self, checkpoint_dir="checkpoints"):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.best_loss = float('inf')

    def on_validation_epoch_end(self, trainer, pl_module):
        # Save checkpoint at the end of training or when loss improves
        current_loss = trainer.callback_metrics.get("loss", float('inf'))

        # Save best model
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self._save_checkpoint(pl_module, suffix="best")

    def on_train_end(self, trainer, pl_module):
        # Always save final model
        self._save_checkpoint(pl_module, suffix="final")

    def _save_checkpoint(self, pl_module, suffix="final"):
        """Save the discovery model checkpoint."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        mode = "CB-NCD" if pl_module.use_concepts else "NCD"

        checkpoint_filename = (
            "-".join(
                [
                    "discover",
                    mode,
                    pl_module.hparams.arch,
                    pl_module.hparams.dataset,
                    pl_module.hparams.comment,
                    suffix,
                ]
            )
            + ".pth"
        )
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)

        # Build checkpoint dict
        checkpoint = {
            'model_state_dict': pl_module.model.state_dict(),
            'hparams': dict(pl_module.hparams),
            'best_head': torch.argmin(pl_module.loss_per_head).item(),
            'loss_per_head': pl_module.loss_per_head.cpu(),
            'use_concepts': pl_module.use_concepts,
        }

        if pl_module.use_concepts:
            checkpoint['num_concepts'] = pl_module.num_concepts
            checkpoint['concepts'] = pl_module.concepts
            checkpoint['concept_weights'] = pl_module.concept_weights.cpu()

        torch.save(checkpoint, checkpoint_path)
        print(f"\nSaved {mode} discovery checkpoint: {checkpoint_path}")
