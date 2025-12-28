import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from .utils import get_embedding, EmbeddingManager, get_default_embedding_manager

@dataclass
class AxisPosition:
    position: float  # 0 to 1  (where on the axis)
    confidence: float  # 0 to 1 (how relevant is this axis to this concept)
    label: str
    axis_name: str

class DualAxis:
    """
    Represents a single dual axis between two semantic poles.
    """
    def __init__(self, name: str, pole_a: Union[str, List[str]], pole_b: Union[str, List[str]], 
                 description: str = "", curvature: float = 1.0):
        self.name = name
        self.pole_a_anchors = [pole_a] if isinstance(pole_a, str) else pole_a
        self.pole_b_anchors = [pole_b] if isinstance(pole_b, str) else pole_b
        self.pole_a_name = self.pole_a_anchors[0] # For backward compatibility in labels
        self.pole_b_name = self.pole_b_anchors[0]
        self.description = description
        self.curvature = curvature
        
        self._pole_a_vec = None
        self._pole_b_vec = None

    def _ensure_vectors(self):
        if self._pole_a_vec is None:
            # Average embeddings of all anchors to create a stable pole vector
            vecs_a = [get_embedding(a) for a in self.pole_a_anchors]
            self._pole_a_vec = get_default_embedding_manager().normalize(np.mean(vecs_a, axis=0))
            
            vecs_b = [get_embedding(b) for b in self.pole_b_anchors]
            self._pole_b_vec = get_default_embedding_manager().normalize(np.mean(vecs_b, axis=0))

    def project(self, embedding: np.ndarray) -> Tuple[float, float]:
        """
        Projects an embedding onto this dual axis using relative cosine distance.
        Returns:
            position: float in [0, 1] — where on the axis
            confidence: float in [0, 1] — how relevant this axis is to the concept
        """
        self._ensure_vectors()
        
        # Calculate cosine similarities
        sim_a = np.dot(embedding, self._pole_a_vec)
        sim_b = np.dot(embedding, self._pole_b_vec)
        
        # Calculate cosine distances (1 - cosine similarity)
        dist_a = 1.0 - sim_a
        dist_b = 1.0 - sim_b
        
        # Avoid division by zero
        total_dist = dist_a + dist_b
        if total_dist < 1e-8:
            return 0.5, 0.0  # No confidence if equidistant from both
            
        # Position is relative proximity to A vs B
        pos = dist_a / total_dist
        
        # CONFIDENCE: How well does the concept align with the axis?
        # High confidence = close to one pole (high similarity to one, low to other)
        # Low confidence = far from both poles OR equidistant
        
        # Method 1: Spread-based (difference between similarities)
        max_sim = max(sim_a, sim_b)
        min_sim = min(sim_a, sim_b)
        spread = max_sim - min_sim  # 0 to ~2
        
        # Method 2: Max proximity (how close to at least one pole)
        # Penalize if both similarities are low (concept is orthogonal to the axis)
        max_proximity = max_sim  # -1 to 1
        
        # Method 3: Position extremity (how far from center 0.5)
        extremity = abs(pos - 0.5) * 2  # 0 to 1
        
        # Method 4: Orthogonality penalty
        # If both sim_a and sim_b are low (< 0.3), this axis doesn't apply to the concept
        avg_similarity = (sim_a + sim_b) / 2
        orthogonality_penalty = max(0, 0.3 - avg_similarity) * 3  # Penalty up to 0.9
        
        # Combine methods with boosting
        raw_confidence = (
            spread * 0.35 +                    # Spread contributes 35%
            (max_proximity + 1) / 2 * 0.25 +   # Proximity contributes 25%
            extremity * 0.25 +                 # Extremity contributes 25%
            avg_similarity * 0.15              # Average relevance contributes 15%
        )
        
        # Apply orthogonality penalty (reduces confidence for unrelated concepts)
        raw_confidence = max(0, raw_confidence - orthogonality_penalty)
        
        # Apply mild boost to spread the values
        confidence = np.clip(raw_confidence * 1.3, 0, 1)
        
        # Apply curvature to position if needed
        if self.curvature != 1.0:
            centered = pos - 0.5
            pos = 0.5 + np.sign(centered) * (abs(centered * 2) ** (1/self.curvature)) / 2
            
        return np.clip(float(pos), 0, 1), float(confidence)

    def get_position_label(self, position: float) -> str:
        """Categorizes the numerical position into a human label."""
        if position < 0.2:
            return f"Extreme {self.pole_a_name}"
        elif position < 0.4:
            return f"Strongly {self.pole_a_name}"
        elif position < 0.6:
            return "Balanced"
        elif position < 0.8:
            return f"Strongly {self.pole_b_name}"
        else:
            return f"Extreme {self.pole_b_name}"

class DualCoreSystem:
    """
    Manages a set of DualAxes and provides higher-level analysis.
    """
    DEFAULT_AXES_CONFIG = [
        ("Simple-Complex", "simple", "complex", "Structural complexity"),
        ("Concrete-Abstract", "concrete", "abstract", "Level of abstraction"),
        ("Local-Global", "local", "global", "Scope and impact"),
        ("Specific-General", "specific", "general", "Detail level"),
        ("Fast-Slow", "fast", "slow", "Temporal dynamics"),
        ("Analytic-Intuitive", "analytic", "intuitive", "Reasoning style"),
        ("Conservative-Creative", "conservative", "creative", "Innovation level"),
        ("Independent-Contextual", "independent", "contextual", "Relational dependence")
    ]

    def __init__(self, axes: Optional[List[DualAxis]] = None, config_path: Optional[str] = None):
        if axes:
            self.axes = axes
        else:
            # Try to load from JSON
            import json
            import os
            
            if config_path is None:
                # Default path relative to this file
                base_dir = os.path.dirname(__file__)
                config_path = os.path.join(base_dir, "data", "default_axes.json")
            
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.axes = [DualAxis(
                        name=c['name'], 
                        pole_a=c['pole_a'], 
                        pole_b=c['pole_b'], 
                        description=c.get('description', "")
                    ) for c in config]
            else:
                self.axes = [DualAxis(*config) for config in self.DEFAULT_AXES_CONFIG]
            
        self.manager = get_default_embedding_manager()

    def analyze(self, text: str, context: Optional[str] = None) -> Dict[str, AxisPosition]:
        """
        Analyzes a piece of text on all dimensions.
        Optional 'context' string can shift the perception (e.g. 'quantum physics').
        
        Returns dict of AxisPosition with position AND confidence for each axis.
        """
        embedding = self.manager.get_embedding(text)
        
        if context:
            # Shift embedding slightly towards the context to 'view through a lens'
            context_vec = self.manager.get_embedding(context)
            # 0.2 is the 'influence' of context on perception
            embedding = self.manager.normalize(embedding * 0.8 + context_vec * 0.2)
            
        profile = {}
        for axis in self.axes:
            pos, conf = axis.project(embedding)
            profile[axis.name] = AxisPosition(
                position=float(pos),
                confidence=float(conf),
                label=axis.get_position_label(pos),
                axis_name=axis.name
            )
        return profile

    def compare(self, text_a: str, text_b: str) -> Dict[str, Any]:
        """Compares two texts in the dual axis space."""
        profile_a = self.analyze(text_a)
        profile_b = self.analyze(text_b)
        
        similarities = {}
        diffs = []
        for name in profile_a:
            diff = abs(profile_a[name].position - profile_b[name].position)
            similarities[name] = 1.0 - diff
            diffs.append(diff)
            
        global_sim = 1.0 - np.mean(diffs)
        
        return {
            "global_similarity": float(global_sim),
            "axis_similarities": similarities,
            "profiles": {"a": profile_a, "b": profile_b}
        }

    def mix_profiles(self, profile_a: Dict[str, AxisPosition], 
                     profile_b: Dict[str, AxisPosition], 
                     alpha: float = 0.5) -> Dict[str, AxisPosition]:
        """Interpolates between two profiles."""
        mixed = {}
        for name in profile_a:
            pos = profile_a[name].position * (1-alpha) + profile_b[name].position * alpha
            conf = profile_a[name].confidence * (1-alpha) + profile_b[name].confidence * alpha
            # Find the corresponding axis to get the label
            axis = next(ax for ax in self.axes if ax.name == name)
            mixed[name] = AxisPosition(
                position=float(pos),
                confidence=float(conf),
                label=axis.get_position_label(pos),
                axis_name=name
            )
        return mixed

    def get_profile_coherence(self, profile: Dict[str, AxisPosition]) -> Dict[str, Any]:
        """
        Calculates how coherent/well-defined a concept's profile is.
        
        Returns:
            overall_coherence: float 0-1 (average confidence across axes)
            strong_axes: list of axes where the concept is clearly positioned
            weak_axes: list of axes where the concept is ambiguous
            interpretation: human-readable summary
        """
        confidences = [p.confidence for p in profile.values()]
        overall = float(np.mean(confidences))
        
        strong = [name for name, p in profile.items() if p.confidence > 0.3]
        weak = [name for name, p in profile.items() if p.confidence < 0.15]
        
        if overall > 0.35:
            interpretation = "Well-defined concept with clear cognitive signature."
        elif overall > 0.2:
            interpretation = "Moderately defined. Clear on some axes, ambiguous on others."
        else:
            interpretation = "Ambiguous or novel concept. Does not fit standard axes well."
            
        return {
            "overall_coherence": overall,
            "strong_axes": strong,
            "weak_axes": weak,
            "interpretation": interpretation
        }
