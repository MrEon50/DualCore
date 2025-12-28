"""
Composite Axes Module for DualCore.

Allows creation of second-order dimensions by combining base axes.
Example: "Elegance" = (Simple + Beautiful) / 2
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .core import AxisPosition, DualCoreSystem


@dataclass
class CompositeAxisDefinition:
    """Defines a composite axis from base axes."""
    name: str
    components: List[Tuple[str, float]]  # [(axis_name, weight), ...]
    description: str = ""
    
    def compute_position(self, profile: Dict[str, AxisPosition]) -> float:
        """Computes the composite position from a profile."""
        total_weight = sum(w for _, w in self.components)
        weighted_sum = 0.0
        
        for axis_name, weight in self.components:
            if axis_name in profile:
                weighted_sum += profile[axis_name].position * weight
                
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def compute_confidence(self, profile: Dict[str, AxisPosition]) -> float:
        """Average confidence of component axes."""
        confidences = [profile[ax].confidence for ax, _ in self.components if ax in profile]
        return float(np.mean(confidences)) if confidences else 0.0


# Predefined useful composites
DEFAULT_COMPOSITES = [
    CompositeAxisDefinition(
        name="Elegance",
        components=[("Simple-Complex", -1.0), ("Beautiful-Ugly", -1.0)],
        description="Simple + Beautiful (inverted so higher = more elegant)"
    ),
    CompositeAxisDefinition(
        name="Wisdom",
        components=[("Concrete-Abstract", 0.5), ("True-False", -1.0), ("Certain-Uncertain", -1.0)],
        description="Abstract understanding grounded in truth and certainty"
    ),
    CompositeAxisDefinition(
        name="Innovation",
        components=[("Static-Dynamic", 1.0), ("Analytic-Intuitive", 0.5)],
        description="Dynamic change with intuitive leaps"
    ),
    CompositeAxisDefinition(
        name="Danger",
        components=[("Good-Bad", 1.0), ("Certain-Uncertain", 1.0)],
        description="Bad + Uncertain = High risk"
    ),
    CompositeAxisDefinition(
        name="Clarity",
        components=[("Simple-Complex", -1.0), ("Concrete-Abstract", -1.0), ("Certain-Uncertain", -1.0)],
        description="Simple, concrete, and certain"
    )
]


class CompositeAxisSystem:
    """
    Extends DualCoreSystem with composite axes.
    """
    def __init__(self, base_system: DualCoreSystem, 
                 composites: Optional[List[CompositeAxisDefinition]] = None):
        self.base = base_system
        self.composites = composites or DEFAULT_COMPOSITES
        
    def analyze_with_composites(self, text: str, context: Optional[str] = None) -> Dict:
        """
        Returns both base profile and composite scores.
        """
        base_profile = self.base.analyze(text, context)
        
        composite_scores = {}
        for comp in self.composites:
            pos = comp.compute_position(base_profile)
            conf = comp.compute_confidence(base_profile)
            composite_scores[comp.name] = {
                "position": pos,
                "confidence": conf,
                "description": comp.description
            }
            
        return {
            "base_profile": base_profile,
            "composites": composite_scores
        }
    
    def add_composite(self, composite: CompositeAxisDefinition):
        """Adds a custom composite axis."""
        self.composites.append(composite)
        
    def discover_composites(self, profiles: List[Dict[str, AxisPosition]], 
                           min_correlation: float = 0.7) -> List[CompositeAxisDefinition]:
        """
        Analyzes a set of profiles to discover axes that correlate together.
        Returns candidate composite definitions.
        """
        if len(profiles) < 10:
            return []
            
        # Extract position vectors
        axis_names = list(profiles[0].keys())
        data = np.array([[p[ax].position for ax in axis_names] for p in profiles])
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(data.T)
        
        # Find highly correlated pairs
        candidates = []
        for i in range(len(axis_names)):
            for j in range(i+1, len(axis_names)):
                if abs(corr_matrix[i, j]) >= min_correlation:
                    sign = 1.0 if corr_matrix[i, j] > 0 else -1.0
                    candidates.append(CompositeAxisDefinition(
                        name=f"{axis_names[i].split('-')[0]}+{axis_names[j].split('-')[0]}",
                        components=[(axis_names[i], 1.0), (axis_names[j], sign)],
                        description=f"Auto-discovered: {axis_names[i]} {'positively' if sign > 0 else 'negatively'} correlates with {axis_names[j]}"
                    ))
                    
        return candidates
