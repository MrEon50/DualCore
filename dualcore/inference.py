"""
Pole Inference System for DualCore.

Enables the system to INFER the existence of opposite poles,
even when only seeing one side. This is a step toward true reasoning.

Philosophy: If you see "left", you should KNOW "right" exists,
even if you've never seen it. This is the essence of understanding duality.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .core import DualCoreSystem, AxisPosition
from .utils import get_default_embedding_manager


@dataclass
class InferredPole:
    """A pole that was inferred (not directly observed)."""
    concept: str  # The observed concept
    inferred_opposite: str  # The hypothesized opposite
    axis_name: str  # Which axis this relates to
    confidence: float  # How confident we are in the inference
    reasoning: str  # Why we inferred this


class PoleInferenceEngine:
    """
    Infers the existence of opposite poles from observed concepts.
    
    This is the bridge between "pattern matching" and "understanding":
    - Pattern matching: See "hot", know it exists
    - Understanding: See "hot", INFER that "cold" must exist as its opposite
    """
    
    def __init__(self, system: DualCoreSystem):
        self.system = system
        self.manager = get_default_embedding_manager()
        
        # Knowledge base of known dualities (for bootstrapping)
        self.known_dualities = {
            "left": "right", "right": "left",
            "up": "down", "down": "up",
            "hot": "cold", "cold": "hot",
            "light": "dark", "dark": "light",
            "good": "evil", "evil": "good",
            "true": "false", "false": "true",
            "life": "death", "death": "life",
            "love": "hate", "hate": "love",
            "order": "chaos", "chaos": "order",
            "creation": "destruction", "destruction": "creation",
            "beginning": "end", "end": "beginning",
            "cause": "effect", "effect": "cause",
            "question": "answer", "answer": "question",
            "problem": "solution", "solution": "problem",
        }
        
    def infer_opposite(self, concept: str) -> InferredPole:
        """
        Given a concept, infer what its logical opposite would be.
        
        This uses three methods:
        1. Knowledge base lookup (known dualities)
        2. Axis projection (find the opposite pole on the axis where concept scores extreme)
        3. Embedding space reflection (compute the "anti-concept")
        """
        concept_lower = concept.lower().strip()
        
        # Method 1: Known duality lookup
        if concept_lower in self.known_dualities:
            return InferredPole(
                concept=concept,
                inferred_opposite=self.known_dualities[concept_lower],
                axis_name="Semantic",
                confidence=1.0,
                reasoning=f"Known duality: '{concept}' is canonically opposite to '{self.known_dualities[concept_lower]}'"
            )
        
        # Method 2: Axis-based inference
        profile = self.system.analyze(concept)
        
        # Find the axis where this concept scores most extreme
        most_extreme_axis = None
        most_extreme_deviation = 0
        extreme_direction = 0  # -1 for pole_a, +1 for pole_b
        
        for axis_name, pos in profile.items():
            deviation = abs(pos.position - 0.5)
            if deviation > most_extreme_deviation and pos.confidence > 0.2:
                most_extreme_deviation = deviation
                most_extreme_axis = axis_name
                extreme_direction = 1 if pos.position > 0.5 else -1
        
        if most_extreme_axis and most_extreme_deviation > 0.1:
            # The opposite is the other pole of this axis
            axis = next(ax for ax in self.system.axes if ax.name == most_extreme_axis)
            opposite_pole = axis.pole_a_name if extreme_direction > 0 else axis.pole_b_name
            
            return InferredPole(
                concept=concept,
                inferred_opposite=opposite_pole,
                axis_name=most_extreme_axis,
                confidence=min(most_extreme_deviation * 2, 1.0),
                reasoning=f"'{concept}' scores extreme on {most_extreme_axis} axis, so its opposite is '{opposite_pole}'"
            )
        
        # Method 3: Embedding reflection (compute anti-concept)
        # This is more speculative but can work for abstract concepts
        concept_embedding = self.manager.get_embedding(concept)
        
        # Find the concept in our vocabulary that is most "anti-correlated"
        # For now, we approximate by using the axis system
        opposite_profile = {ax: 1.0 - profile[ax].position for ax in profile}
        
        return InferredPole(
            concept=concept,
            inferred_opposite=f"anti-{concept}",
            axis_name="Inferred",
            confidence=0.5,
            reasoning=f"No clear axis opposition found. Hypothetical opposite would have inverted profile."
        )
    
    def detect_missing_poles(self, observed_concepts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyzes a set of concepts and detects which "poles" are missing.
        
        For example, if you see many "positive" emotion words but no "negative" ones,
        the system should flag: "The 'negative emotion' pole is not represented."
        """
        # Analyze all concepts
        profiles = [self.system.analyze(c) for c in observed_concepts]
        
        # For each axis, check if only one pole is represented
        missing_poles = []
        
        for axis in self.system.axes:
            positions = [p[axis.name].position for p in profiles]
            avg_pos = np.mean(positions)
            std_pos = np.std(positions)
            
            # Check for imbalance
            if avg_pos < 0.35 and std_pos < 0.15:
                # Skewed toward pole A, missing pole B
                missing_poles.append({
                    "axis": axis.name,
                    "observed_pole": axis.pole_a_name,
                    "missing_pole": axis.pole_b_name,
                    "skew": float(avg_pos),
                    "recommendation": f"Consider adding concepts related to '{axis.pole_b_name}' for balance"
                })
            elif avg_pos > 0.65 and std_pos < 0.15:
                # Skewed toward pole B, missing pole A
                missing_poles.append({
                    "axis": axis.name,
                    "observed_pole": axis.pole_b_name,
                    "missing_pole": axis.pole_a_name,
                    "skew": float(avg_pos),
                    "recommendation": f"Consider adding concepts related to '{axis.pole_a_name}' for balance"
                })
                
        return missing_poles
    
    def generate_balanced_opposites(self, concepts: List[str]) -> List[Tuple[str, str]]:
        """
        For each concept, generates its inferred opposite.
        Returns pairs of (original, inferred_opposite).
        """
        pairs = []
        for concept in concepts:
            inference = self.infer_opposite(concept)
            pairs.append((concept, inference.inferred_opposite))
        return pairs
    
    def understand_duality(self, concept: str) -> Dict[str, Any]:
        """
        Full duality analysis: given a concept, explain its place in the dual structure.
        
        This is the "understanding" function - not just classification,
        but explaining WHY this concept exists in opposition to something else.
        """
        profile = self.system.analyze(concept)
        inference = self.infer_opposite(concept)
        coherence = self.system.get_profile_coherence(profile)
        
        # Find which axes this concept is most aligned with
        aligned_axes = []
        for axis_name, pos in profile.items():
            if pos.confidence > 0.3:
                pole = axis_name.split("-")[0] if pos.position < 0.5 else axis_name.split("-")[1]
                aligned_axes.append({
                    "axis": axis_name,
                    "aligned_pole": pole,
                    "position": pos.position,
                    "confidence": pos.confidence
                })
        
        aligned_axes.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "concept": concept,
            "profile": profile,
            "coherence": coherence,
            "primary_axis": aligned_axes[0] if aligned_axes else None,
            "inferred_opposite": inference,
            "understanding": self._generate_understanding(concept, aligned_axes, inference)
        }
    
    def _generate_understanding(self, concept: str, axes: List[Dict], inference: InferredPole) -> str:
        """
        Generates a human-readable explanation of the concept's duality.
        """
        if not axes:
            return f"'{concept}' exists in a unique cognitive space, not strongly aligned with standard dualities."
        
        primary = axes[0]
        
        explanation = f"'{concept}' fundamentally embodies the '{primary['aligned_pole']}' pole of the {primary['axis']} duality. "
        explanation += f"By the law of opposites, this implies the existence of its counterpart: '{inference.inferred_opposite}'. "
        
        if len(axes) > 1:
            secondary = axes[1]
            explanation += f"Secondarily, it also expresses '{secondary['aligned_pole']}' (from {secondary['axis']})."
        
        return explanation


class NeuralDualCoreInterface:
    """
    Interface for connecting DualCore with neural networks.
    
    DualCore acts as a "cognitive layer" that provides:
    1. Interpretable structure to neural network outputs
    2. Logical constraints (e.g., if output says "hot", check it doesn't also say "cold")
    3. Missing knowledge inference (if NN only learned "positive", DualCore infers "negative" exists)
    
    This enables co-evolution: NN learns patterns, DualCore provides reasoning framework.
    """
    
    def __init__(self, system: DualCoreSystem):
        self.system = system
        self.inference_engine = PoleInferenceEngine(system)
        
    def constrain_output(self, text: str) -> Dict[str, Any]:
        """
        Analyzes neural network output and flags logical inconsistencies.
        """
        from .reasoning import ParadoxDetector
        
        detector = ParadoxDetector(self.system)
        paradox_report = detector.detect_paradox(text)
        
        return {
            "text": text,
            "is_consistent": not paradox_report.is_paradox,
            "issues": paradox_report.conflicting_elements if paradox_report.is_paradox else [],
            "recommendation": "Regenerate without contradictions" if paradox_report.is_paradox else "Output is logically consistent"
        }
    
    def enrich_with_duality(self, concepts: List[str]) -> Dict[str, Any]:
        """
        Takes concepts from a neural network and enriches them with duality understanding.
        
        This is how DualCore "co-evolves" with NN:
        - NN provides raw concepts
        - DualCore provides the logical structure and infers missing pieces
        """
        enriched = []
        missing = self.inference_engine.detect_missing_poles(concepts)
        
        for concept in concepts:
            understanding = self.inference_engine.understand_duality(concept)
            enriched.append(understanding)
            
        return {
            "concepts": enriched,
            "missing_poles": missing,
            "balance_recommendation": self._generate_balance_recommendation(missing)
        }
    
    def _generate_balance_recommendation(self, missing: List[Dict]) -> str:
        if not missing:
            return "The concept space is well-balanced across all axes."
        
        axes = [m["axis"] for m in missing]
        return f"Consider exploring the opposite poles of: {', '.join(axes)}"
    
    def create_training_signal(self, concept: str) -> Dict[str, float]:
        """
        Creates a training signal for neural networks based on DualCore profile.
        
        This allows neural networks to learn the cognitive structure of DualCore,
        enabling them to internalize the dual axis framework.
        """
        profile = self.system.analyze(concept)
        
        # Convert profile to a normalized vector
        signal = {}
        for axis_name, pos in profile.items():
            signal[axis_name] = pos.position
            signal[f"{axis_name}_conf"] = pos.confidence
            
        return signal
    
    def validate_neural_concept(self, concept: str, neural_embedding: np.ndarray) -> Dict[str, Any]:
        """
        Validates that a neural network's understanding of a concept
        aligns with DualCore's logical structure.
        
        This is the "sanity check" that prevents neural networks from
        learning inconsistent representations.
        """
        # Get DualCore's profile
        dc_profile = self.system.analyze(concept)
        
        # Project neural embedding onto DualCore axes
        neural_profile = {}
        for axis in self.system.axes:
            pos, conf = axis.project(neural_embedding)
            neural_profile[axis.name] = {"position": pos, "confidence": conf}
        
        # Compare
        alignment_scores = {}
        for axis_name in dc_profile:
            dc_pos = dc_profile[axis_name].position
            nn_pos = neural_profile[axis_name]["position"]
            alignment = 1.0 - abs(dc_pos - nn_pos)
            alignment_scores[axis_name] = alignment
            
        overall_alignment = np.mean(list(alignment_scores.values()))
        
        return {
            "concept": concept,
            "alignment": overall_alignment,
            "axis_alignment": alignment_scores,
            "is_aligned": overall_alignment > 0.7,
            "recommendation": "Neural representation is consistent" if overall_alignment > 0.7 
                            else "Neural network may have learned inconsistent representation"
        }
