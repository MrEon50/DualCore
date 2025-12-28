"""
Reasoning Module for DualCore.

Provides logical analysis including paradox detection and analogical reasoning.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from .core import AxisPosition, DualCoreSystem


@dataclass
class ParadoxReport:
    """Report of detected logical contradictions."""
    is_paradox: bool
    severity: float  # 0-1, how severe the contradiction
    conflicting_elements: List[Dict[str, Any]]
    explanation: str


# Define axis pairs that are logically mutually exclusive
# If a concept scores extreme on BOTH sides of these pairs, it's a paradox
MUTUALLY_EXCLUSIVE_PAIRS = [
    # Physical impossibilities
    ("Fast-Slow", "Fast-Slow"),  # Can't be both fast and slow literally
    ("Static-Dynamic", "Static-Dynamic"),  # Can't be frozen and moving
    ("Concrete-Abstract", "Concrete-Abstract"),  # Same object can't be both
    
    # Logical impossibilities  
    ("True-False", "True-False"),  # Can't be both true and false
    ("Certain-Uncertain", "True-False"),  # Certain falsehood is different from uncertain truth
    
    # Value impossibilities
    ("Good-Bad", "Good-Bad"),  # Same action can't be purely good AND purely bad
    ("Beautiful-Ugly", "Beautiful-Ugly"),  # Same object, same perspective
]

# Semantic pairs where extremes conflict (e.g., "hot ice")
SEMANTIC_CONFLICTS = [
    # (keyword_a, axis_a, expected_position_a, keyword_b, axis_b, expected_position_b)
    # If concept contains keyword_a but scores opposite on axis_a, AND same for b -> conflict
    
    # Temperature-state conflicts
    ("hot", "Fast-Slow", "low", "frozen", "Static-Dynamic", "low"),
    ("cold", "Fast-Slow", "high", "boiling", "Static-Dynamic", "high"),
    
    # Logical conflicts
    ("true", "True-False", "low", "false", "True-False", "high"),
    ("certain", "Certain-Uncertain", "low", "doubtful", "Certain-Uncertain", "high"),
    
    # Moral conflicts
    ("good", "Good-Bad", "low", "evil", "Good-Bad", "high"),
    ("virtue", "Good-Bad", "low", "sin", "Good-Bad", "high"),
]


class ParadoxDetector:
    """
    Detects logical contradictions and impossibilities in concepts.
    """
    
    def __init__(self, system: DualCoreSystem):
        self.system = system
        
    def detect_paradox(self, text: str, context: Optional[str] = None) -> ParadoxReport:
        """
        Analyzes text for logical contradictions.
        
        A paradox occurs when:
        1. The concept contains mutually exclusive terms (e.g., "hot ice")
        2. The profile shows extreme positions on conflicting axes
        3. The semantic content contradicts the profile (e.g., "cold" scoring as "hot")
        
        Returns:
            ParadoxReport with severity and explanation
        """
        profile = self.system.analyze(text, context)
        text_lower = text.lower()
        
        conflicts = []
        
        # Check 1: Profile extremes on same axis (internal contradiction)
        for axis_name, pos in profile.items():
            # If position is very close to center with HIGH confidence,
            # it might indicate competing forces (not paradox)
            # But if confidence is LOW and position is extreme, something's wrong
            
            # Actually for paradox: we need BOTH poles to be strongly indicated
            # This is detected by looking for conflicting keywords in text
            pass
        
        # Check 2: Semantic keyword conflicts
        for kw_a, axis_a, expected_a, kw_b, axis_b, expected_b in SEMANTIC_CONFLICTS:
            if kw_a in text_lower and kw_b in text_lower:
                # Both conflicting keywords present
                pos_a = profile.get(axis_a)
                pos_b = profile.get(axis_b)
                
                if pos_a and pos_b:
                    conflicts.append({
                        "type": "semantic_conflict",
                        "terms": [kw_a, kw_b],
                        "axes": [axis_a, axis_b],
                        "severity": 0.9,  # High severity for direct keyword conflict
                        "explanation": f"'{kw_a}' and '{kw_b}' are logically incompatible"
                    })
        
        # Check 3: Look for oxymoron patterns
        oxymoron_patterns = [
            # Temperature/State
            ("hot", "cold"), ("hot", "frozen"), ("hot", "ice"), ("warm", "frozen"),
            ("frozen", "fire"), ("freezing", "fire"), ("icy", "fire"), ("cold", "fire"),
            ("boiling", "frozen"), ("burning", "ice"),
            
            # Physical states
            ("dry", "water"), ("dry", "wet"), ("dry", "rain"), ("dry", "ocean"),
            ("solid", "liquid"), ("solid", "gas"),
            
            # Sound
            ("silent", "scream"), ("silent", "loud"), ("quiet", "noisy"), ("mute", "shout"),
            
            # Light
            ("bright", "dark"), ("light", "darkness"), ("dark", "bright"),
            ("visible", "invisible"), ("seen", "invisible"),
            
            # Life
            ("alive", "dead"), ("living", "dead"), ("life", "death"),
            
            # Truth
            ("true", "false"), ("true", "lie"), ("truth", "lie"), ("honest", "lie"),
            
            # Morality
            ("good", "evil"), ("pure", "corrupt"), ("saint", "sinner"),
            ("virtue", "sin"), ("holy", "evil"),
            
            # Motion
            ("still", "moving"), ("static", "dynamic"), ("frozen", "flowing"),
        ]
        
        for term_a, term_b in oxymoron_patterns:
            if term_a in text_lower and term_b in text_lower:
                conflicts.append({
                    "type": "oxymoron",
                    "terms": [term_a, term_b],
                    "severity": 1.0,  # Maximum severity for direct opposites
                    "explanation": f"'{term_a}' and '{term_b}' are mutually exclusive in literal sense"
                })
        
        # Check 4: Negation paradoxes (e.g., "not not true" is fine, but structural issues)
        if " not " in text_lower:
            # Count negations - odd number is fine, check for contradictory structures
            pass
        
        # Aggregate results
        if not conflicts:
            return ParadoxReport(
                is_paradox=False,
                severity=0.0,
                conflicting_elements=[],
                explanation="No logical contradictions detected."
            )
        
        max_severity = max(c["severity"] for c in conflicts)
        
        # Determine if it's a TRUE paradox or just metaphorical
        # True paradox = literal impossibility
        # Metaphor = unusual but interpretable combination
        
        is_true_paradox = max_severity >= 0.8
        
        if is_true_paradox:
            explanation = f"LOGICAL PARADOX DETECTED: {conflicts[0]['explanation']}. "
            explanation += "This concept contains mutually exclusive elements that cannot coexist literally."
        else:
            explanation = f"Potential tension detected: {conflicts[0]['explanation']}. "
            explanation += "This may be metaphorical or require context to resolve."
        
        return ParadoxReport(
            is_paradox=is_true_paradox,
            severity=max_severity,
            conflicting_elements=conflicts,
            explanation=explanation
        )
    
    def check_physical_possibility(self, text: str) -> Dict[str, Any]:
        """
        Checks if the concept describes something physically possible.
        """
        report = self.detect_paradox(text)
        
        if report.is_paradox:
            return {
                "physically_possible": False,
                "reason": report.explanation,
                "conflicts": report.conflicting_elements
            }
        
        return {
            "physically_possible": True,
            "confidence": 1.0 - report.severity,
            "notes": "No obvious physical impossibilities detected."
        }


class AnalogyEngine:
    """
    Performs analogical reasoning in dual axis space.
    
    Example: "King is to Queen as Man is to ?" -> Woman
    """
    
    def __init__(self, system: DualCoreSystem):
        self.system = system
        
    def compute_relation_vector(self, concept_a: str, concept_b: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Computes the "direction" from A to B in profile space.
        Returns both the relation vector AND the weights (based on magnitude).
        """
        profile_a = self.system.analyze(concept_a)
        profile_b = self.system.analyze(concept_b)
        
        relation = {}
        weights = {}
        
        for axis_name in profile_a:
            delta = profile_b[axis_name].position - profile_a[axis_name].position
            relation[axis_name] = delta
            # Weight is proportional to how much this axis changed
            weights[axis_name] = abs(delta)
            
        # Normalize weights so they sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            # Equal weights if no change
            weights = {k: 1.0 / len(weights) for k in weights}
            
        return relation, weights
    
    def apply_relation(self, concept_c: str, relation: Dict[str, float]) -> Dict[str, float]:
        """
        Applies a relation vector to concept C to predict D's profile.
        """
        profile_c = self.system.analyze(concept_c)
        
        predicted_d = {}
        for axis_name in profile_c:
            new_pos = profile_c[axis_name].position + relation.get(axis_name, 0)
            predicted_d[axis_name] = np.clip(new_pos, 0, 1)
            
        return predicted_d
    
    def complete_analogy(self, a: str, b: str, c: str, 
                        candidates: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Completes "A is to B as C is to ?"
        
        Uses weighted similarity where axes that changed most in A->B 
        are weighted higher when comparing candidates.
        
        Also detects "opposition" relations (A is opposite of B) and applies
        semantic opposition bonus to candidates that are antonyms of C.
        """
        relation, weights = self.compute_relation_vector(a, b)
        predicted_d = self.apply_relation(c, relation)
        
        # Detect if A->B is an opposition relation
        is_opposition = self._detect_opposition(a, b)
        
        result = {
            "relation_vector": relation,
            "axis_weights": weights,
            "predicted_profile": predicted_d,
            "interpretation": self._interpret_relation(relation),
            "is_opposition": is_opposition
        }
        
        if candidates:
            # Rank candidates by WEIGHTED similarity to predicted profile
            scores = []
            for cand in candidates:
                cand_profile = self.system.analyze(cand)
                similarity = self._weighted_profile_similarity(predicted_d, cand_profile, weights)
                
                # If opposition relation detected, give bonus to known antonyms
                if is_opposition:
                    opposition_bonus = self._compute_opposition_bonus(c, cand)
                    similarity += opposition_bonus * 0.1  # 10% bonus for antonyms
                
                scores.append((cand, similarity))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            result["ranked_candidates"] = scores
            result["best_match"] = scores[0][0] if scores else None
            
        return result
    
    def _detect_opposition(self, a: str, b: str) -> bool:
        """
        Detects if A and B are opposites (antonyms).
        Uses profile analysis: if they differ significantly on many axes,
        they're likely opposites.
        """
        profile_a = self.system.analyze(a)
        profile_b = self.system.analyze(b)
        
        # Count axes where they differ by more than 0.15
        significant_diffs = 0
        for axis_name in profile_a:
            diff = abs(profile_a[axis_name].position - profile_b[axis_name].position)
            if diff > 0.1:
                significant_diffs += 1
                
        # If they differ on many axes, it's likely an opposition
        return significant_diffs >= 3
    
    def _compute_opposition_bonus(self, c: str, candidate: str) -> float:
        """
        Computes a bonus for candidates that are semantically opposite to C.
        """
        # Known antonym pairs
        antonym_pairs = [
            ("light", "dark"), ("dark", "light"),
            ("day", "night"), ("night", "day"),
            ("hot", "cold"), ("cold", "hot"),
            ("fast", "slow"), ("slow", "fast"),
            ("good", "evil"), ("evil", "good"),
            ("truth", "lie"), ("lie", "truth"),
            ("big", "small"), ("small", "big"),
            ("tall", "short"), ("short", "tall"),
            ("up", "down"), ("down", "up"),
            ("left", "right"), ("right", "left"),
            ("love", "hate"), ("hate", "love"),
            ("life", "death"), ("death", "life"),
        ]
        
        c_lower = c.lower()
        cand_lower = candidate.lower()
        
        for word_a, word_b in antonym_pairs:
            if word_a in c_lower and word_b in cand_lower:
                return 1.0
            if word_b in c_lower and word_a in cand_lower:
                return 1.0
                
        return 0.0
    
    def _weighted_profile_similarity(self, predicted: Dict[str, float], 
                                     actual: Dict[str, AxisPosition],
                                     weights: Dict[str, float]) -> float:
        """
        Compute WEIGHTED similarity between predicted and actual profile.
        Axes that changed more in the analogy relation are weighted higher.
        """
        weighted_diff = 0.0
        total_weight = 0.0
        
        for axis_name in predicted:
            if axis_name in actual:
                diff = abs(predicted[axis_name] - actual[axis_name].position)
                weight = weights.get(axis_name, 1.0)
                weighted_diff += diff * weight
                total_weight += weight
                
        if total_weight == 0:
            return 0.0
            
        avg_diff = weighted_diff / total_weight
        return 1.0 - avg_diff
    
    def _interpret_relation(self, relation: Dict[str, float]) -> str:
        """Creates human-readable interpretation of the relation vector."""
        significant = [(ax, delta) for ax, delta in relation.items() if abs(delta) > 0.05]
        
        if not significant:
            return "Minimal transformation (concepts are similar)"
        
        significant.sort(key=lambda x: abs(x[1]), reverse=True)
        
        parts = []
        for ax, delta in significant[:3]:
            direction = "increases" if delta > 0 else "decreases"
            pole = ax.split("-")[1] if delta > 0 else ax.split("-")[0]
            parts.append(f"{direction} toward '{pole}'")
            
        return "Transformation: " + ", ".join(parts)
