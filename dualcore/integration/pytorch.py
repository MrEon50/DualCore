import torch
import torch.nn as nn
from typing import List, Dict, Optional
import numpy as np

class DualLayer(nn.Module):
    """
    A PyTorch layer that incorporates DualCore axes into a neural network.
    It can either use fixed axes to provide interpretable features or 
    learn its own axes.
    """
    def __init__(self, input_dim: int, num_axes: int = 8, 
                 trainable_axes: bool = True, 
                 initial_pole_vecs: Optional[torch.Tensor] = None):
        super().__init__()
        self.input_dim = input_dim
        self.num_axes = num_axes
        
        if initial_pole_vecs is not None:
            # Expecting shape (num_axes, 2, input_dim) where index 1 is poles (A and B)
            self.poles = nn.Parameter(initial_pole_vecs, requires_grad=trainable_axes)
        else:
            # Random initialization
            self.poles = nn.Parameter(torch.randn(num_axes, 2, input_dim), 
                                     requires_grad=trainable_axes)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects input x onto the dual axes.
        x: (batch_size, input_dim)
        Returns: (batch_size, num_axes)
        """
        # x is (B, D)
        # poles is (N, 2, D)
        pole_a = self.poles[:, 0, :] # (N, D)
        pole_b = self.poles[:, 1, :] # (N, D)
        
        # Calculate axis vectors
        axis_vecs = pole_b - pole_a # (N, D)
        axis_len_sq = torch.sum(axis_vecs**2, dim=1, keepdim=True).T # (1, N)
        
        # Vector from A to x
        # x: (B, D), pole_a: (N, D)
        # Compute (x_b,d - a_n,d)
        # We need broadcast: x is (B, 1, D), pole_a is (1, N, D)
        point_vecs = x.unsqueeze(1) - pole_a.unsqueeze(0) # (B, N, D)
        
        # Dot product (point_vecs . axis_vecs)
        # axis_vecs needs to be (1, N, D)
        dot_probs = torch.sum(point_vecs * axis_vecs.unsqueeze(0), dim=2) # (B, N)
        
        # Projection
        pos = dot_probs / (axis_len_sq + 1e-8) # (B, N)
        
        return torch.clamp(pos, 0.0, 1.0)

class DualCoreModel(nn.Module):
    """
    Example model combining a standard encoder with a DualLayer.
    """
    def __init__(self, encoder: nn.Module, embedding_dim: int, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.dual_layer = DualLayer(embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        emb = self.encoder(x)
        dual_profile = self.dual_layer(emb)
        logits = self.classifier(dual_profile)
        return logits, dual_profile
