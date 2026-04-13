#nets.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import sys
import os

# Add cifar-10-model to path for importing CIFAR-specific ResNet
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cifar-10-model'))
from resnet import resnet18 as cifar_resnet18


def get_encoder(arch, dataset, pretrained=True):
    """
    Get the appropriate encoder based on architecture and dataset.

    Args:
        arch: Architecture name (only 'resnet18' supported for now)
        dataset: Dataset name ('CIFAR10', 'CIFAR100', 'CUB', 'ImageNet')
        pretrained: Whether to load pretrained weights
                    - For CIFAR: ignored (always reinitialize)
                    - For CUB/ImageNet: use ImageNet pretrained if True

    Returns:
        encoder: The encoder network with fc removed
        feat_dim: Feature dimension of the encoder
    """
    if arch != "resnet18":
        raise ValueError(f"Only resnet18 is supported, got {arch}")

    if "CIFAR" in dataset:
        # Use CIFAR-optimized ResNet18 (3x3 conv1, stride=1)
        # Always reinitialize weights for CIFAR (no pretrained)
        encoder = cifar_resnet18(pretrained=False, num_classes=10)
        feat_dim = 512
        # Remove maxpool to preserve spatial resolution for 32x32 images
        encoder.maxpool = nn.Identity()
        # Remove the classification head
        encoder.fc = nn.Identity()
        # Reinitialize conv1 weights
        nn.init.kaiming_normal_(encoder.conv1.weight, mode="fan_out", nonlinearity="relu")
    else:
        # Use standard torchvision ResNet18 for CUB/ImageNet (224x224 images)
        weights = 'IMAGENET1K_V1' if pretrained else None
        encoder = models.resnet18(weights=weights)
        feat_dim = encoder.fc.weight.shape[1]  # 512
        encoder.fc = nn.Identity()

    return encoder, feat_dim


class Prototypes(nn.Module):
    def __init__(self, output_dim, num_prototypes):
        super().__init__()

        self.prototypes = nn.Linear(output_dim, num_prototypes, bias=False)

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.prototypes.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.prototypes.weight.copy_(w)

    def forward(self, x):
        return self.prototypes(x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers=1):
        super().__init__()

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_hidden_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            ]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MultiHead(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, num_prototypes, num_heads, num_hidden_layers=1
    ):
        super().__init__()
        self.num_heads = num_heads

        # projectors
        self.projectors = torch.nn.ModuleList(
            [MLP(input_dim, hidden_dim, output_dim, num_hidden_layers) for _ in range(num_heads)]
        )

        # prototypes
        self.prototypes = torch.nn.ModuleList(
            [Prototypes(output_dim, num_prototypes) for _ in range(num_heads)]
        )
        self.normalize_prototypes()

    @torch.no_grad()
    def normalize_prototypes(self):
        for p in self.prototypes:
            p.normalize_prototypes()

    def forward_head(self, head_idx, feats):
        z = self.projectors[head_idx](feats)
        z = F.normalize(z, dim=1)
        return self.prototypes[head_idx](z), z

    def forward(self, feats):
        out = [self.forward_head(h, feats) for h in range(self.num_heads)]
        return [torch.stack(o) for o in map(list, zip(*out))]


class MultiHeadResNet(nn.Module):
    def __init__(
        self,
        arch,
        dataset,
        num_labeled,
        num_unlabeled,
        hidden_dim=2048,
        proj_dim=256,
        overcluster_factor=3,
        num_heads=5,
        num_hidden_layers=1,
        num_concepts=None,  #None means no concept bottleneck (Stage 1)
        pretrained=True,  # Use ImageNet pretrained weights for CUB/ImageNet
    ):
        super().__init__()

        # Get appropriate encoder for the dataset
        # CIFAR -> CIFAR-optimized ResNet18 (no pretrained, no maxpool)
        # CUB/ImageNet -> standard torchvision ResNet18 with ImageNet pretrained
        self.encoder, self.feat_dim = get_encoder(arch, dataset, pretrained)

        # Concept bottleneck layer
        self.use_concepts = num_concepts is not None
        if self.use_concepts:
            self.concept_layer = nn.Linear(self.feat_dim, num_concepts, bias=False)
            head_input_dim = num_concepts  # Heads take concepts as input
            # Normalization parameters
            # These will be loaded from checkpoint, initialized to identity transform
            self.register_buffer('proj_mean', torch.zeros(num_concepts))
            self.register_buffer('proj_std', torch.ones(num_concepts))
        else:
            head_input_dim = self.feat_dim  # Heads take features as input (Stage 1)

        # Heads input dimension
        self.head_lab = Prototypes(head_input_dim, num_labeled)
        
        if num_heads is not None:
            self.head_unlab = MultiHead(
                input_dim=head_input_dim,  # uses concepts or features
                hidden_dim=hidden_dim,
                output_dim=proj_dim,
                num_prototypes=num_unlabeled,
                num_heads=num_heads,
                num_hidden_layers=num_hidden_layers,
            )
            self.head_unlab_over = MultiHead(
                input_dim=head_input_dim,  # uses concepts or features
                hidden_dim=hidden_dim,
                output_dim=proj_dim,
                num_prototypes=num_unlabeled * overcluster_factor,
                num_heads=num_heads,
                num_hidden_layers=num_hidden_layers,
            )

    @torch.no_grad()
    def normalize_prototypes(self):
        self.head_lab.normalize_prototypes()
        if getattr(self, "head_unlab", False):
            self.head_unlab.normalize_prototypes()
            self.head_unlab_over.normalize_prototypes()

    def forward_heads(self, feats):
        # Apply concept bottleneck if enabled
        if self.use_concepts:
            # Project features to concept space
            concepts_raw = self.concept_layer(feats)
            concepts = (concepts_raw - self.proj_mean) / self.proj_std
            head_input = concepts
        else:
            # Use features directly (Stage 1 pretraining)
            head_input = feats

        # Normalize before classification heads
        out = {"logits_lab": self.head_lab(F.normalize(head_input))}
        
        if hasattr(self, "head_unlab"):
            logits_unlab, proj_feats_unlab = self.head_unlab(head_input)
            logits_unlab_over, proj_feats_unlab_over = self.head_unlab_over(head_input)
            out.update(
                {
                    "logits_unlab": logits_unlab,
                    "proj_feats_unlab": proj_feats_unlab,
                    "logits_unlab_over": logits_unlab_over,
                    "proj_feats_unlab_over": proj_feats_unlab_over,
                }
            )
        return out

    def forward(self, views):
        if isinstance(views, list):
            feats = [self.encoder(view) for view in views]
            out = [self.forward_heads(f) for f in feats]
            out_dict = {"feats": torch.stack(feats)}
            for key in out[0].keys():
                out_dict[key] = torch.stack([o[key] for o in out])
            return out_dict
        else:
            feats = self.encoder(views)
            out = self.forward_heads(feats)
            out["feats"] = feats
            return out