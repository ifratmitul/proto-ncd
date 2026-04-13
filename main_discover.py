#discover.py
import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchmetrics import Accuracy

from utils.data import get_datamodule
from utils.nets import MultiHeadResNet
from utils.eval import ClusterMetrics
from utils.sinkhorn_knopp import SinkhornKnopp
from utils.callbacks import DiscoverCheckpointCallback

import numpy as np
from argparse import ArgumentParser
from datetime import datetime


parser = ArgumentParser()
parser.add_argument("--dataset", default="CIFAR100", type=str, help="dataset")
parser.add_argument("--imagenet_split", default="A", type=str, help="imagenet split [A,B,C]")
parser.add_argument("--download", default=False, action="store_true", help="wether to download")
parser.add_argument("--data_dir", default="datasets", type=str, help="data directory")
parser.add_argument("--log_dir", default="logs", type=str, help="log directory")
parser.add_argument("--batch_size", default=256, type=int, help="batch size")
parser.add_argument("--num_workers", default=10, type=int, help="number of workers")
parser.add_argument("--arch", default="resnet18", type=str, help="backbone architecture")
parser.add_argument("--base_lr", default=0.4, type=float, help="learning rate")
parser.add_argument("--min_lr", default=0.001, type=float, help="min learning rate")
parser.add_argument("--momentum_opt", default=0.9, type=float, help="momentum for optimizer")
parser.add_argument("--weight_decay_opt", default=1.5e-4, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="warmup epochs")
parser.add_argument("--proj_dim", default=256, type=int, help="projected dim")
parser.add_argument("--hidden_dim", default=2048, type=int, help="hidden dim in proj/pred head")
parser.add_argument("--overcluster_factor", default=3, type=int, help="overclustering factor")
parser.add_argument("--num_heads", default=5, type=int, help="number of heads for clustering")
parser.add_argument("--num_hidden_layers", default=1, type=int, help="number of hidden layers")
parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")
parser.add_argument("--epsilon_sk", default=0.05, type=float, help="epsilon for the Sinkhorn")
parser.add_argument("--temperature", default=0.1, type=float, help="softmax temperature")
parser.add_argument("--comment", default=datetime.now().strftime("%b%d_%H-%M-%S"), type=str)
parser.add_argument("--project", default="CB-NCD", type=str, help="wandb project")
parser.add_argument("--entity", default="ifrat-mitul-university-of-south-dakota", type=str, help="wandb entity")
parser.add_argument("--offline", default=False, action="store_true", help="disable wandb")
parser.add_argument("--num_labeled_classes", default=80, type=int, help="number of labeled classes")
parser.add_argument("--num_unlabeled_classes", default=20, type=int, help="number of unlab classes")
parser.add_argument("--pretrained", type=str, help="pretrained checkpoint path")
parser.add_argument("--multicrop", default=False, action="store_true", help="activates multicrop")
parser.add_argument("--num_large_crops", default=2, type=int, help="number of large crops")
parser.add_argument("--num_small_crops", default=2, type=int, help="number of small crops")

#CB-NCD arguments
parser.add_argument("--concept_checkpoint", type=str, default=None,
                    help="Path to trained concept layer checkpoint (for CB-NCD)")
parser.add_argument("--concept_lr", default=0.04, type=float,
                    help="Learning rate for concept layer W_c (typically 10x lower than base_lr)")
parser.add_argument("--lambda_sparse", default=0.0, type=float,
                    help="Weight for concept sparsity loss (encourage few concepts per cluster)")
parser.add_argument("--lambda_intra", default=0.0, type=float,
                    help="Weight for intra-cluster concept consistency loss")
parser.add_argument("--no_pretrained_weights", default=False, action="store_true",
                    help="disable ImageNet pretrained weights (use for ImageNet dataset)")


class Discoverer(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters({k: v for (k, v) in kwargs.items() if not callable(v)})

        #Load concept checkpoint if provided
        self.use_concepts = self.hparams.concept_checkpoint is not None
        if self.use_concepts:
            print(f"\n{'='*60}")
            print(f"Loading concept checkpoint for CB-NCD...")
            print(f"{'='*60}\n")

            concept_ckpt = torch.load(self.hparams.concept_checkpoint, map_location='cpu')
            self.num_concepts = concept_ckpt['num_concepts']
            self.concepts = concept_ckpt['concepts']
            self.concept_weights = concept_ckpt['W_c']
            # Load normalization parameters
            self.proj_mean = concept_ckpt.get('proj_mean', None)
            self.proj_std = concept_ckpt.get('proj_std', None)

            print(f"✓ Loaded {self.num_concepts} concepts")
            print(f"✓ Sample concepts: {self.concepts[:5]}")
            print(f"✓ Concept weights shape: {self.concept_weights.shape}")
            if self.proj_mean is not None:
                print(f"✓ Normalization parameters loaded")
            else:
                print(f"⚠ No normalization parameters found (older checkpoint)")
        else:
            print(f"\n{'='*60}")
            print(f"Running standard NCD (no concepts)")
            print(f"{'='*60}\n")
            self.num_concepts = None

        # build model
        self.model = MultiHeadResNet(
            arch=self.hparams.arch,
            dataset=self.hparams.dataset,
            num_labeled=self.hparams.num_labeled_classes,
            num_unlabeled=self.hparams.num_unlabeled_classes,
            proj_dim=self.hparams.proj_dim,
            hidden_dim=self.hparams.hidden_dim,
            overcluster_factor=self.hparams.overcluster_factor,
            num_heads=self.hparams.num_heads,
            num_hidden_layers=self.hparams.num_hidden_layers,
            num_concepts=self.num_concepts,  #CB-NCD: set num_concepts
            pretrained=not self.hparams.no_pretrained_weights,
        )

        # Load pretrained encoder
        state_dict = torch.load(self.hparams.pretrained, map_location=self.device)
        # Filter out unlabeled heads
        state_dict = {k: v for k, v in state_dict.items() if ("unlab" not in k)}
        
        if self.use_concepts:
            state_dict = {k: v for k, v in state_dict.items() if ("head_lab" not in k)}
        self.model.load_state_dict(state_dict, strict=False)

        # Load trained concept layer weights and normalization
        if self.use_concepts:
            print(f"\n{'='*60}")
            print(f"Loading concept layer weights into model...")
            print(f"{'='*60}\n")

            self.model.concept_layer.weight.data = self.concept_weights.to(self.device)

            # Load normalization parameters
            if self.proj_mean is not None and self.proj_std is not None:
                self.model.proj_mean = self.proj_mean.to(self.device)
                self.model.proj_std = self.proj_std.to(self.device)
                print(f"✓ Normalization parameters loaded into model")
            else:
                print(f"⚠ Using identity normalization (no proj_mean/proj_std in checkpoint)")

            # This allows W_c to adapt to novel class discovery
            for param in self.model.concept_layer.parameters():
                param.requires_grad = True

            print(f"✓ Concept layer weights loaded")
            print(f"✓ Concept layer trainable: {self.model.concept_layer.weight.requires_grad}")
            print(f"✓ Concept layer LR: {self.hparams.concept_lr}\n")

        # Sinkorn-Knopp
        self.sk = SinkhornKnopp(
            num_iters=self.hparams.num_iters_sk, epsilon=self.hparams.epsilon_sk
        )

        # metrics
        self.metrics = torch.nn.ModuleList(
            [
                ClusterMetrics(self.hparams.num_heads),
                ClusterMetrics(self.hparams.num_heads),
                Accuracy(task="multiclass", num_classes=self.hparams.num_labeled_classes),
            ]
        )
        self.metrics_inc = torch.nn.ModuleList(
            [
                ClusterMetrics(self.hparams.num_heads),
                ClusterMetrics(self.hparams.num_heads),
                Accuracy(task="multiclass", num_classes=self.hparams.num_labeled_classes + self.hparams.num_unlabeled_classes),
            ]
        )

        # buffer for best head tracking
        self.register_buffer("loss_per_head", torch.zeros(self.hparams.num_heads))

    def configure_optimizers(self):
        # Separate parameter groups for concept layer vs rest of model
        if self.use_concepts:
            # Get concept layer parameters
            concept_params = list(self.model.concept_layer.parameters())
            concept_param_ids = set(id(p) for p in concept_params)

            # Get all other parameters (exclude concept layer)
            other_params = [p for p in self.model.parameters()
                          if id(p) not in concept_param_ids]

            # Create parameter groups with different learning rates
            param_groups = [
                {'params': other_params, 'lr': self.hparams.base_lr},
                {'params': concept_params, 'lr': self.hparams.concept_lr},
            ]

            print(f"\n{'='*60}")
            print(f"Optimizer Configuration:")
            print(f"  - Other params LR: {self.hparams.base_lr}")
            print(f"  - Concept layer LR: {self.hparams.concept_lr}")
            print(f"{'='*60}\n")
        else:
            #learning rate for all parameters
            param_groups = [{'params': self.model.parameters(), 'lr': self.hparams.base_lr}]

        optimizer = torch.optim.SGD(
            param_groups,
            momentum=self.hparams.momentum_opt,
            weight_decay=self.hparams.weight_decay_opt,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
            warmup_start_lr=self.hparams.min_lr,
            eta_min=self.hparams.min_lr,
        )
        return [optimizer], [scheduler]

    def cross_entropy_loss(self, preds, targets):
        preds = F.log_softmax(preds / self.hparams.temperature, dim=-1)
        return torch.mean(-torch.sum(targets * preds, dim=-1), dim=-1)

    def swapped_prediction(self, logits, targets):
        loss = 0
        for view in range(self.hparams.num_large_crops):
            for other_view in np.delete(range(self.hparams.num_crops), view):
                loss += self.cross_entropy_loss(logits[other_view], targets[view])
        return loss / (self.hparams.num_large_crops * (self.hparams.num_crops - 1))


    def _analyze_cluster_concepts(self):
        """Analyze which concepts define each discovered cluster."""
        print(f"\n{'='*60}")
        print(f"CLUSTER-CONCEPT ANALYSIS (Epoch {self.current_epoch})")
        print(f"{'='*60}\n")
        
        # Get a batch from validation set
        val_loader = self.trainer.datamodule.val_dataloader()[1]  # unlabeled val
        
        all_concepts = []
        all_preds = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                if batch_idx >= 10: 
                    break
                    
                images = images.to(self.device)
                
                # Get concepts
                feats = self.model.encoder(images)
                concepts_raw = self.model.concept_layer(feats)
                concepts = (concepts_raw - self.model.proj_mean) / self.model.proj_std
                
                # Get predictions (best head)
                outputs = self.model(images)
                best_head = torch.argmin(self.loss_per_head)
                preds = outputs["logits_unlab"][best_head].argmax(dim=-1)
                
                all_concepts.append(concepts.cpu())
                all_preds.append(preds.cpu())
                all_labels.append(labels)
        
        all_concepts = torch.cat(all_concepts, dim=0)
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Analyze each discovered cluster
        num_clusters = self.hparams.num_unlabeled_classes
        
        for cluster_id in range(num_clusters):
            cluster_mask = all_preds == cluster_id
            if cluster_mask.sum() < 5:
                continue
                
            cluster_concepts = all_concepts[cluster_mask]
            cluster_labels = all_labels[cluster_mask]
            
            # Mean concept activation for this cluster
            mean_concepts = cluster_concepts.mean(dim=0)
            top3_idx = torch.topk(mean_concepts, k=3).indices
            
            # What ground-truth classes are in this cluster?
            unique_labels, counts = torch.unique(cluster_labels, return_counts=True)
            dominant_label = unique_labels[counts.argmax()].item()
            purity = counts.max().item() / cluster_mask.sum().item()
            
            print(f"Cluster {cluster_id}: {cluster_mask.sum().item()} samples, purity={purity:.2f}")
            print(f"  Dominant class: {dominant_label}")
            print(f"  Top concepts:")
            for idx in top3_idx:
                print(f"    - {self.concepts[idx]}: {mean_concepts[idx].item():.3f}")
            print()
        
        print(f"{'='*60}\n")

    # Concept sparsity loss
    def concept_sparsity_loss(self, concept_activations, cluster_assignments, alpha=0.5):
        """
        Encourage sparse concept usage per cluster.

        L_sparse = (||c̄||₁ - α||c̄||₂) / num_concepts

        This promotes few dominant concepts per cluster rather than
        uniform activation across all concepts.

        Args:
            concept_activations: [batch, num_concepts] concept values
            cluster_assignments: [batch, num_clusters] soft cluster assignments
            alpha: Balance parameter (default 0.5)

        Returns:
            loss: Scalar sparsity loss (normalized)
        """
        num_concepts = concept_activations.shape[1]

        # Compute average concept activation per cluster
        # cluster_concepts: [num_clusters, num_concepts]
        cluster_concepts = torch.matmul(
            cluster_assignments.T, concept_activations
        ) / (cluster_assignments.sum(dim=0, keepdim=True).T + 1e-8)

        # Sparsity loss: L1 - alpha * L2 (per cluster, then average)
        # Normalize by num_concepts to keep gradients stable
        l1_norms = torch.norm(cluster_concepts, p=1, dim=1) / num_concepts  # [num_clusters]
        l2_norms = torch.norm(cluster_concepts, p=2, dim=1) / (num_concepts ** 0.5)  # [num_clusters]

        # Higher L1/L2 ratio = more sparse (few large values)
        loss = (l1_norms - alpha * l2_norms).mean()

        return loss

    # Intra-cluster concept consistency loss
    def concept_consistency_loss(self, concept_activations, cluster_assignments):
        """
        Minimize concept variance within each cluster (normalized).

        Args:
            concept_activations: [batch, num_concepts] concept values
            cluster_assignments: [batch, num_clusters] soft cluster assignments

        Returns:
            loss: Scalar consistency loss (normalized)
        """
        num_clusters = cluster_assignments.shape[1]
        num_concepts = concept_activations.shape[1]

        # Compute global variance for normalization
        global_var = concept_activations.var(dim=0).mean() + 1e-8

        loss = 0
        valid_clusters = 0

        for k in range(num_clusters):
            # Get samples belonging to cluster k
            weights = cluster_assignments[:, k:k+1]  # [batch, 1]

            if weights.sum() < 1e-8:
                continue

            # Weighted mean
            mean_concepts = (weights * concept_activations).sum(dim=0) / (weights.sum() + 1e-8)

            # Weighted variance
            diff = concept_activations - mean_concepts.unsqueeze(0)
            variance = (weights * diff.pow(2)).sum(dim=0) / (weights.sum() + 1e-8)

            # Normalize by global variance to keep loss scale ~1
            loss += (variance / global_var).mean()
            valid_clusters += 1

        if valid_clusters == 0:
            return torch.tensor(0.0).to(concept_activations.device)

        return loss / valid_clusters

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self):
        self.loss_per_head = torch.zeros_like(self.loss_per_head)

    def unpack_batch(self, batch):
        if self.hparams.dataset == "ImageNet":
            views_lab, labels_lab, views_unlab, labels_unlab = batch
            views = [torch.cat([vl, vu]) for vl, vu in zip(views_lab, views_unlab)]
            labels = torch.cat([labels_lab, labels_unlab])
        else:
            views, labels = batch
        mask_lab = labels < self.hparams.num_labeled_classes
        return views, labels, mask_lab

    def training_step(self, batch, _):
        views, labels, mask_lab = self.unpack_batch(batch)
        nlc = self.hparams.num_labeled_classes

        # normalize prototypes
        self.model.normalize_prototypes()

        # forward
        outputs = self.model(views)

            # DEBUG: Log concept statistics every 10 epochs
        if self.use_concepts and self.current_epoch % 10 == 0 and self.global_step % 100 == 0:
            with torch.no_grad():
                feats = self.model.encoder(views[0])
                concepts_raw = self.model.concept_layer(feats)
                concepts = (concepts_raw - self.model.proj_mean) / self.model.proj_std
                
                # Overall concept activation statistics
                print(f"\n{'='*60}")
                print(f"DEBUG: Epoch {self.current_epoch}, Step {self.global_step}")
                print(f"{'='*60}")
                print(f"Concept activations (normalized):")
                print(f"  Mean: {concepts.mean().item():.4f}")
                print(f"  Std:  {concepts.std().item():.4f}")
                print(f"  Min:  {concepts.min().item():.4f}")
                print(f"  Max:  {concepts.max().item():.4f}")
                
                # Per-concept activation
                concept_means = concepts.mean(dim=0)  # [num_concepts]
                top5_idx = torch.topk(concept_means, k=5).indices
                bot5_idx = torch.topk(concept_means, k=5, largest=False).indices
                
                print(f"\nTop 5 most active concepts:")
                for idx in top5_idx:
                    print(f"  {self.concepts[idx]}: {concept_means[idx].item():.4f}")
                
                print(f"\nBottom 5 least active concepts:")
                for idx in bot5_idx:
                    print(f"  {self.concepts[idx]}: {concept_means[idx].item():.4f}")
                
                # Concept discrimination: variance across samples (higher = more discriminative)
                concept_vars = concepts.var(dim=0)  # [num_concepts]
                top5_var_idx = torch.topk(concept_vars, k=5).indices
                
                print(f"\nTop 5 most discriminative concepts (highest variance):")
                for idx in top5_var_idx:
                    print(f"  {self.concepts[idx]}: var={concept_vars[idx].item():.4f}")
                
                # Labeled vs Unlabeled concept differences
                concepts_lab = concepts[mask_lab]
                concepts_unlab = concepts[~mask_lab]
                
                print(f"\nLabeled vs Unlabeled concept stats:")
                print(f"  Labeled   - Mean: {concepts_lab.mean().item():.4f}, Std: {concepts_lab.std().item():.4f}")
                print(f"  Unlabeled - Mean: {concepts_unlab.mean().item():.4f}, Std: {concepts_unlab.std().item():.4f}")
                
                # Concept sparsity check
                active_threshold = 0.5  # Consider concept "active" if > 0.5
                num_active_per_sample = (concepts > active_threshold).float().sum(dim=1)
                print(f"\nConcept sparsity (threshold={active_threshold}):")
                print(f"  Avg active concepts per sample: {num_active_per_sample.mean().item():.1f} / {concepts.shape[1]}")
                print(f"  Min: {num_active_per_sample.min().item():.0f}, Max: {num_active_per_sample.max().item():.0f}")
                
                print(f"{'='*60}\n")

        # gather outputs
        outputs["logits_lab"] = (
            outputs["logits_lab"].unsqueeze(1).expand(-1, self.hparams.num_heads, -1, -1)
        )
        logits = torch.cat([outputs["logits_lab"], outputs["logits_unlab"]], dim=-1)
        logits_over = torch.cat([outputs["logits_lab"], outputs["logits_unlab_over"]], dim=-1)

        # create targets
        targets_lab = (
            F.one_hot(labels[mask_lab], num_classes=self.hparams.num_labeled_classes)
            .float()
            .to(self.device)
        )
        targets = torch.zeros_like(logits)
        targets_over = torch.zeros_like(logits_over)

        # generate pseudo-labels with sinkhorn-knopp and fill unlab targets
        for v in range(self.hparams.num_large_crops):
            for h in range(self.hparams.num_heads):
                targets[v, h, mask_lab, :nlc] = targets_lab.type_as(targets)
                targets_over[v, h, mask_lab, :nlc] = targets_lab.type_as(targets)
                targets[v, h, ~mask_lab, nlc:] = self.sk(
                    outputs["logits_unlab"][v, h, ~mask_lab]
                ).type_as(targets)
                targets_over[v, h, ~mask_lab, nlc:] = self.sk(
                    outputs["logits_unlab_over"][v, h, ~mask_lab]
                ).type_as(targets)

        # compute swapped prediction loss
        loss_cluster = self.swapped_prediction(logits, targets)
        loss_overcluster = self.swapped_prediction(logits_over, targets_over)

        # update best head tracker
        self.loss_per_head += loss_cluster.clone().detach()

        # total loss
        loss_cluster = loss_cluster.mean()
        loss_overcluster = loss_overcluster.mean()
        loss = (loss_cluster + loss_overcluster) / 2

        # Experimenting with concept-based regularization losses
        if self.use_concepts and (self.hparams.lambda_sparse > 0 or self.hparams.lambda_intra > 0):
            
            feats_unlab = self.model.encoder(views[0][~mask_lab])
            concepts_raw = self.model.concept_layer(feats_unlab)

            # Apply output normalization
            concepts = (concepts_raw - self.model.proj_mean) / self.model.proj_std

            # Get cluster assignments for unlabeled samples (averaged over heads)
            # Detach cluster_probs to prevent gradients flowing back through clustering head
            cluster_probs = F.softmax(
                outputs["logits_unlab"][0, :, ~mask_lab, :].detach() / self.hparams.temperature,
                dim=-1
            ).mean(dim=0)  # Average over heads: [num_unlab, num_clusters]

            # Compute regularization losses if any
            if self.hparams.lambda_sparse > 0:
                loss_sparse = self.concept_sparsity_loss(concepts, cluster_probs)
                loss = loss + self.hparams.lambda_sparse * loss_sparse
            else:
                loss_sparse = torch.tensor(0.0).to(self.device)

            if self.hparams.lambda_intra > 0:
                loss_intra = self.concept_consistency_loss(concepts, cluster_probs)
                loss = loss + self.hparams.lambda_intra * loss_intra
            else:
                loss_intra = torch.tensor(0.0).to(self.device)

            # Log concept losses
            self.log("loss_sparse", loss_sparse.detach(), on_step=False, on_epoch=True, sync_dist=True)
            self.log("loss_intra", loss_intra.detach(), on_step=False, on_epoch=True, sync_dist=True)

        # log
        results = {
            "loss": loss.detach(),
            "loss_cluster": loss_cluster.mean(),
            "loss_overcluster": loss_overcluster.mean(),
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
        }
        # Add concept layer LR if using concepts
        if self.use_concepts:
            results["lr_concept"] = self.trainer.optimizers[0].param_groups[1]["lr"]

        self.log_dict(results, on_step=False, on_epoch=True, sync_dist=True)
        return loss
    
    def on_after_backward(self):
        """Check if gradients are flowing through concept layer."""
        if self.use_concepts and self.global_step % 500 == 0:
            if self.model.concept_layer.weight.grad is not None:
                grad_norm = self.model.concept_layer.weight.grad.norm().item()
                print(f"Step {self.global_step}: Concept layer grad norm = {grad_norm:.6f}")
                
                # Log to wandb
                self.log("debug/concept_grad_norm", grad_norm, sync_dist=True)
            else:
                print(f"Step {self.global_step}: Concept layer grad is None (frozen)")

    def validation_step(self, batch, batch_idx, dl_idx):
        images, labels = batch
        tag = self.trainer.datamodule.dataloader_mapping[dl_idx]

        # forward
        outputs = self(images)

        if "unlab" in tag:  # use clustering head
            preds = outputs["logits_unlab"]
            preds_inc = torch.cat(
                [
                    outputs["logits_lab"].unsqueeze(0).expand(self.hparams.num_heads, -1, -1),
                    outputs["logits_unlab"],
                ],
                dim=-1,
            )
        else:  # use supervised classifier
            preds = outputs["logits_lab"]
            best_head = torch.argmin(self.loss_per_head)
            preds_inc = torch.cat(
                [outputs["logits_lab"], outputs["logits_unlab"][best_head]], dim=-1
            )
        preds = preds.max(dim=-1)[1]
        preds_inc = preds_inc.max(dim=-1)[1]

        self.metrics[dl_idx].update(preds, labels)
        self.metrics_inc[dl_idx].update(preds_inc, labels)

    def save_finetuned_concept_layer(self, suffix="finetuned"):
        """
        Save the fine-tuned concept layer W_c as a SEPARATE checkpoint.
        This is independent from the Stage 2 checkpoint and can be used
        for inference or further analysis.
        """
        if not self.use_concepts:
            print("No concept layer to save (not using CB-NCD mode)")
            return

        import os
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create filename
        filename = f"concept_layer_finetuned-{self.hparams.arch}-{self.hparams.dataset}-{self.hparams.comment}-{suffix}.pth"
        save_path = os.path.join(checkpoint_dir, filename)

        # Build checkpoint with fine-tuned weights
        checkpoint = {
            # Fine-tuned weights from discovery training
            'W_c': self.model.concept_layer.weight.data.cpu().clone(),
            # Concept metadata
            'concepts': self.concepts,
            'num_concepts': self.num_concepts,
            # Normalization parameters
            'proj_mean': self.model.proj_mean.cpu().clone(),
            'proj_std': self.model.proj_std.cpu().clone(),
            # Training info
            'epoch': self.current_epoch,
            'dataset': self.hparams.dataset,
            'arch': self.hparams.arch,
            'concept_lr': self.hparams.concept_lr,
            'base_lr': self.hparams.base_lr,
            'is_finetuned': True        
            }

        torch.save(checkpoint, save_path)

        print(f"\n{'='*60}")
        print(f"SAVED FINE-TUNED CONCEPT LAYER")
        print(f"{'='*60}")
        print(f"  Path: {save_path}")
        print(f"  Epoch: {self.current_epoch}")
        print(f"  Num concepts: {self.num_concepts}")
        print(f"  W_c shape: {checkpoint['W_c'].shape}")
        print(f"  Original checkpoint: {self.hparams.concept_checkpoint}")
        print(f"{'='*60}\n")

        return save_path

    def on_validation_epoch_end(self):
        results = [m.compute() for m in self.metrics]
        results_inc = [m.compute() for m in self.metrics_inc]

        # DEBUG: Cluster-Concept Association Analysis 
        if self.use_concepts and self.current_epoch % 20 == 0:
            self._analyze_cluster_concepts()

        # Save fine-tuned concept layer at the end of training 
        # Only save at the final epoch (max_epochs - 1)
        if self.use_concepts and self.current_epoch == self.hparams.max_epochs - 1:
            self.save_finetuned_concept_layer(suffix="final")

        # log metrics
        for dl_idx, (result, result_inc) in enumerate(zip(results, results_inc)):
            prefix = self.trainer.datamodule.dataloader_mapping[dl_idx]
            prefix_inc = "incremental/" + prefix
            if "unlab" in prefix:
                for (metric, values), (_, values_inc) in zip(result.items(), result_inc.items()):
                    name = "/".join([prefix, metric])
                    name_inc = "/".join([prefix_inc, metric])
                    avg = torch.stack(values).mean()
                    avg_inc = torch.stack(values_inc).mean()
                    best = values[torch.argmin(self.loss_per_head)]
                    best_inc = values_inc[torch.argmin(self.loss_per_head)]
                    self.log(name + "/avg", avg, sync_dist=True)
                    self.log(name + "/best", best, sync_dist=True)
                    self.log(name_inc + "/avg", avg_inc, sync_dist=True)
                    self.log(name_inc + "/best", best_inc, sync_dist=True)
            else:
                self.log(prefix + "/acc", result, sync_dist=True)
                self.log(prefix_inc + "/acc", result_inc, sync_dist=True)


def main(args):
    dm = get_datamodule(args, "discover")

    mode = "CB-NCD" if args.concept_checkpoint else "NCD"
    run_name = "-".join([mode, "discover", args.arch, args.dataset, args.comment])
    
    wandb_logger = pl.loggers.WandbLogger(
        save_dir=args.log_dir,
        name=run_name,
        project=args.project,
        entity=args.entity,
        offline=args.offline,
    )

    model = Discoverer(**args.__dict__)

    # Add checkpoint callback to save discovery model
    checkpoint_callback = DiscoverCheckpointCallback(checkpoint_dir="checkpoints")

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    if not args.multicrop:
        args.num_small_crops = 0
    args.num_crops = args.num_large_crops + args.num_small_crops

    main(args)