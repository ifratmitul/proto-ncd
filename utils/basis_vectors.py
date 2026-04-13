#basis_vectors.py
"""
Learnable Basis Vector Pool for Proto-NCD.

This module implements the core components for the Proto-NCD approach:
- BasisVectorPool: Learnable attribute basis vectors
- Activation profile computation via cosine similarity
- Regularization losses (orthogonality, compactness, separation, consistency)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasisVectorPool(nn.Module):
    """
    Learnable pool of basis vectors for Proto-NCD.

    B = {b_1, ..., b_N} ∈ R^(N×dz) where N = num_unlabeled_classes × num_basis_per_class

    Each basis vector represents a learnable visual attribute direction in the
    encoder feature space. Activation profiles are computed via cosine similarity.

    Args:
        num_unlabeled_classes: Number of novel/unlabeled classes (Cu)
        num_basis_per_class: Number of basis vectors per class (M)
        feat_dim: Feature dimension from encoder (dz, typically 512)
    """

    def __init__(self, num_unlabeled_classes, num_basis_per_class, feat_dim):
        super().__init__()
        self.num_unlabeled_classes = num_unlabeled_classes
        self.num_basis_per_class = num_basis_per_class
        self.feat_dim = feat_dim
        self.num_basis = num_unlabeled_classes * num_basis_per_class  # N = Cu × M

        # Learnable basis vectors: (N, dz)
        # Initialize as random unit vectors
        basis = torch.randn(self.num_basis, feat_dim)
        basis = F.normalize(basis, dim=1, p=2)
        self.basis_vectors = nn.Parameter(basis)

    @torch.no_grad()
    def normalize_basis_vectors(self):
        """Re-normalize basis vectors to unit length after gradient step."""
        self.basis_vectors.data = F.normalize(self.basis_vectors.data, dim=1, p=2)

    def forward(self, feats):
        """
        Compute activation profiles via cosine similarity.

        a_j(x) = z(x)^T b_j / (||z(x)|| · ||b_j||)

        Args:
            feats: Encoder features (batch_size, feat_dim)

        Returns:
            activation_profile: (batch_size, num_basis) where each element is
                               cosine similarity between feature and basis vector
        """
        # Normalize features and basis vectors for cosine similarity
        feats_norm = F.normalize(feats, dim=1, p=2)  # (B, dz)
        basis_norm = F.normalize(self.basis_vectors, dim=1, p=2)  # (N, dz)

        # Compute cosine similarities: (B, N)
        activation_profile = torch.mm(feats_norm, basis_norm.t())

        return activation_profile

    def orthogonality_loss(self):
        """
        Within-Pool Orthogonality Loss (Eq. 13).

        L_orth = (1/N^2) * ||B̃B̃^T - I_N||_F^2

        Prevents basis vectors from collapsing to similar directions.
        Encourages diverse visual attribute capture.

        Returns:
            loss: Scalar tensor
        """
        basis_norm = F.normalize(self.basis_vectors, dim=1, p=2)  # (N, dz)
        gram_matrix = torch.mm(basis_norm, basis_norm.t())  # (N, N)
        identity = torch.eye(self.num_basis, device=self.basis_vectors.device)

        loss = torch.norm(gram_matrix - identity, p='fro') ** 2
        loss = loss / (self.num_basis ** 2)

        return loss

    def compactness_loss(self, feats):
        """
        Compactness Loss (Eq. 14).

        L_compact = (1/Nu) * sum_{x in Du} min_j (1 - a_j(x))

        Each novel image should be well represented by at least one basis vector.
        Minimizes distance between each novel image and its nearest basis vector.

        Args:
            feats: Encoder features for unlabeled samples (batch_size, feat_dim)

        Returns:
            loss: Scalar tensor
        """
        activation_profile = self.forward(feats)  # (B, N)
        # For each sample, find max similarity (closest basis vector)
        max_similarity, _ = activation_profile.max(dim=1)  # (B,)
        # Loss is (1 - max_similarity) averaged over batch
        loss = (1.0 - max_similarity).mean()

        return loss

    def separation_loss(self, labeled_head_weights):
        """
        Separation Loss (Eq. 15).

        L_sep = (1/N) * sum_j max_k (cos_sim(b_j, w_k^h))

        Novel basis vectors should capture visual attributes distinct from
        those already represented by known class directions.

        Args:
            labeled_head_weights: Weight matrix from labeled head
                                  Shape: (num_labeled, feat_dim) if operating in feature space
                                  or (num_labeled, num_basis) if operating on activation profiles

        Returns:
            loss: Scalar tensor
        """
        basis_norm = F.normalize(self.basis_vectors, dim=1, p=2)  # (N, dz)
        weights_norm = F.normalize(labeled_head_weights, dim=1, p=2)  # (Cl, dim)

        # Check if we're comparing in feature space or activation space
        if labeled_head_weights.shape[1] == self.feat_dim:
            # Both in feature space (dz dimensions)
            # Cosine similarity between each basis vector and each labeled class weight
            similarity = torch.mm(basis_norm, weights_norm.t())  # (N, Cl)
        elif labeled_head_weights.shape[1] == self.num_basis:
            # Labeled head operates on activation profiles (N dimensions)
            # Compare basis vectors indices with labeled weights
            # This is a conceptual comparison - use identity mapping
            # Each basis j has similarity with weight k based on j-th component of w_k
            similarity = weights_norm.t()  # (N, Cl) - j-th basis, k-th class
        else:
            # Dimension mismatch - return 0 as fallback
            return torch.tensor(0.0, device=self.basis_vectors.device)

        # For each basis vector, find max similarity across all labeled classes
        max_similarity, _ = similarity.max(dim=1)  # (N,)
        loss = max_similarity.mean()

        return loss

    def consistency_loss(self, feats_view1, feats_view2):
        """
        Augmentation Consistency Loss (Eq. 16).

        L_cons = (1/Nu) * sum_x (1/N) * ||a(v1) - a(v2)||_2^2

        Activation profiles should be consistent across augmented views.
        Encourages basis vectors to capture stable visual attributes
        invariant to common image augmentations.

        Args:
            feats_view1: Features from first augmented view (batch_size, feat_dim)
            feats_view2: Features from second augmented view (batch_size, feat_dim)

        Returns:
            loss: Scalar tensor
        """
        activation1 = self.forward(feats_view1)  # (B, N)
        activation2 = self.forward(feats_view2)  # (B, N)

        # MSE between activation profiles, normalized by N
        diff = activation1 - activation2
        loss = (diff ** 2).sum(dim=1).mean() / self.num_basis

        return loss

    def get_activation_dim(self):
        """Return the dimension of activation profiles (N = Cu × M)."""
        return self.num_basis

    def extra_repr(self):
        return (f'num_unlabeled_classes={self.num_unlabeled_classes}, '
                f'num_basis_per_class={self.num_basis_per_class}, '
                f'feat_dim={self.feat_dim}, '
                f'total_basis={self.num_basis}')
