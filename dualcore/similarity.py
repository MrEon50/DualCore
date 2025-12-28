import numpy as np
from typing import Dict, List, Optional
from .core import AxisPosition

def dual_profile_similarity(profile_a: Dict[str, AxisPosition], 
                            profile_b: Dict[str, AxisPosition], 
                            weights: Optional[Dict[str, float]] = None) -> float:
    """
    Calculates similarity between two dual profiles.
    
    Args:
        profile_a: First profile.
        profile_b: Second profile.
        weights: Optional dictionary mapping axis names to weight (0.0 to 1.0).
        
    Returns:
        Similarity score from 0.0 to 1.0.
    """
    shared_axes = set(profile_a.keys()) & set(profile_b.keys())
    if not shared_axes:
        return 0.0
        
    total_diff = 0.0
    total_weight = 0.0
    
    for name in shared_axes:
        weight = weights.get(name, 1.0) if weights else 1.0
        diff = abs(profile_a[name].position - profile_b[name].position)
        total_diff += diff * weight
        total_weight += weight
        
    avg_diff = total_diff / total_weight
    return 1.0 - avg_diff

def find_most_similar(target_profile: Dict[str, AxisPosition], 
                      candidates: List[Dict[str, AxisPosition]], 
                      top_k: int = 5) -> List[int]:
    """Returns indices of most similar profiles."""
    scores = [dual_profile_similarity(target_profile, c) for c in candidates]
    return np.argsort(scores)[::-1][:top_k].tolist()

def detect_polar_opposites(profile_a: Dict[str, AxisPosition], 
                           profile_b: Dict[str, AxisPosition]) -> List[str]:
    """Identifies axes where the two profiles are at opposite ends."""
    opposites = []
    for name in profile_a:
        if name in profile_b:
            if abs(profile_a[name].position - profile_b[name].position) > 0.7:
                opposites.append(name)
    return opposites
