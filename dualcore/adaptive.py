"""
Adaptive Axis Discovery for DualCore.

Enables the system to discover new axes when encountering data
that doesn't fit well on existing dimensions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .core import DualAxis, DualCoreSystem, AxisPosition
from .utils import get_default_embedding_manager


@dataclass
class DiscoveredAxis:
    """A newly discovered axis from data analysis."""
    name: str
    pole_a_concepts: List[str]  # Concepts defining pole A
    pole_b_concepts: List[str]  # Concepts defining pole B
    separation_strength: float  # How well-separated the poles are (0-1)
    coverage: float  # What fraction of outliers this axis explains
    suggested_labels: Tuple[str, str]  # Suggested names for poles


class AdaptiveAxisDiscovery:
    """
    Discovers new axes by analyzing concepts that don't fit existing dimensions.
    """
    
    def __init__(self, base_system: DualCoreSystem):
        self.base_system = base_system
        self.manager = get_default_embedding_manager()
        self.outlier_buffer: List[Tuple[str, Dict]] = []  # (concept, profile)
        self.discovered_axes: List[DiscoveredAxis] = []
        
    def analyze_and_buffer(self, text: str) -> Dict[str, Any]:
        """
        Analyzes a concept, and if it has low coherence, buffers it for axis discovery.
        """
        profile = self.base_system.analyze(text)
        coherence = self.base_system.get_profile_coherence(profile)
        
        result = {
            "profile": profile,
            "coherence": coherence,
            "is_outlier": False
        }
        
        # If coherence is low, this concept might need a new axis
        if coherence["overall_coherence"] < 0.25:
            self.outlier_buffer.append((text, profile))
            result["is_outlier"] = True
            result["note"] = "Low coherence - buffered for axis discovery"
            
        return result
    
    def discover_axes(self, min_outliers: int = 10) -> List[DiscoveredAxis]:
        """
        Attempts to discover new axes from buffered outliers.
        
        Returns list of candidate axes found in the data.
        """
        if len(self.outlier_buffer) < min_outliers:
            return []
        
        # Get embeddings for all outliers
        texts = [t for t, _ in self.outlier_buffer]
        embeddings = np.array([self.manager.get_embedding(t) for t in texts])
        
        # Find the direction of maximum variance (potential new axis)
        pca = PCA(n_components=min(5, len(embeddings)))
        pca.fit(embeddings)
        
        discovered = []
        
        for i, component in enumerate(pca.components_[:3]):  # Top 3 principal components
            # Project all embeddings onto this component
            projections = embeddings @ component
            
            # Find extreme concepts on this axis
            sorted_indices = np.argsort(projections)
            
            pole_a_indices = sorted_indices[:3]  # 3 most negative
            pole_b_indices = sorted_indices[-3:]  # 3 most positive
            
            pole_a_concepts = [texts[i] for i in pole_a_indices]
            pole_b_concepts = [texts[i] for i in pole_b_indices]
            
            # Calculate separation strength
            pole_a_mean = np.mean(projections[pole_a_indices])
            pole_b_mean = np.mean(projections[pole_b_indices])
            separation = abs(pole_b_mean - pole_a_mean) / (np.std(projections) + 1e-8)
            
            # Suggest labels based on the concepts
            suggested_a = self._suggest_label(pole_a_concepts)
            suggested_b = self._suggest_label(pole_b_concepts)
            
            axis = DiscoveredAxis(
                name=f"Emergent-{i+1}",
                pole_a_concepts=pole_a_concepts,
                pole_b_concepts=pole_b_concepts,
                separation_strength=min(separation / 3, 1.0),  # Normalize to 0-1
                coverage=pca.explained_variance_ratio_[i],
                suggested_labels=(suggested_a, suggested_b)
            )
            
            discovered.append(axis)
            
        self.discovered_axes.extend(discovered)
        return discovered
    
    def create_axis_from_discovery(self, discovery: DiscoveredAxis) -> DualAxis:
        """
        Creates a usable DualAxis from a discovered axis.
        """
        return DualAxis(
            name=f"{discovery.suggested_labels[0]}-{discovery.suggested_labels[1]}",
            pole_a=discovery.pole_a_concepts,
            pole_b=discovery.pole_b_concepts,
            description=f"Emergent axis discovered from data (separation: {discovery.separation_strength:.2f})"
        )
    
    def extend_system(self, discovery: DiscoveredAxis) -> None:
        """
        Adds a discovered axis to the base system.
        """
        new_axis = self.create_axis_from_discovery(discovery)
        self.base_system.axes.append(new_axis)
        
    def _suggest_label(self, concepts: List[str]) -> str:
        """
        Suggests a label for a pole based on its defining concepts.
        Uses the shortest concept word as a heuristic.
        """
        # Extract key words from concepts
        words = []
        for concept in concepts:
            # Get significant words (longer than 3 chars)
            concept_words = [w for w in concept.lower().split() if len(w) > 3]
            words.extend(concept_words)
        
        if words:
            # Return most common short word
            from collections import Counter
            common = Counter(words).most_common(1)
            return common[0][0].capitalize() if common else "Unknown"
        return "Unknown"
    
    def detect_domain_axes(self, domain_concepts: List[str]) -> List[DiscoveredAxis]:
        """
        Analyzes a set of domain-specific concepts to discover axes
        relevant to that domain.
        
        Example: For music domain, might discover "Melodic-Rhythmic" axis.
        """
        if len(domain_concepts) < 5:
            return []
            
        embeddings = np.array([self.manager.get_embedding(c) for c in domain_concepts])
        
        # Use K-means to find natural clusters (potential poles)
        n_clusters = min(4, len(domain_concepts) // 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        discovered = []
        
        # Find pairs of clusters that are most distant (potential axis poles)
        centers = kmeans.cluster_centers_
        
        for i in range(n_clusters):
            for j in range(i+1, n_clusters):
                # Distance between cluster centers
                distance = np.linalg.norm(centers[i] - centers[j])
                
                if distance > 0.5:  # Significant separation
                    pole_a = [domain_concepts[k] for k in range(len(labels)) if labels[k] == i][:3]
                    pole_b = [domain_concepts[k] for k in range(len(labels)) if labels[k] == j][:3]
                    
                    axis = DiscoveredAxis(
                        name=f"Domain-{i}-{j}",
                        pole_a_concepts=pole_a,
                        pole_b_concepts=pole_b,
                        separation_strength=min(distance, 1.0),
                        coverage=0.5,  # Approximate
                        suggested_labels=(
                            self._suggest_label(pole_a),
                            self._suggest_label(pole_b)
                        )
                    )
                    discovered.append(axis)
        
        return discovered


class SelfEvolvingDualCore:
    """
    A DualCore system that can evolve its axes over time.
    """
    
    def __init__(self, initial_system: Optional[DualCoreSystem] = None):
        self.base = initial_system or DualCoreSystem()
        self.discoverer = AdaptiveAxisDiscovery(self.base)
        self.evolution_history: List[Dict] = []
        
    def analyze(self, text: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyzes text and tracks outliers for potential axis evolution.
        """
        result = self.discoverer.analyze_and_buffer(text)
        result["base_profile"] = self.base.analyze(text, context)
        return result
    
    def evolve(self, min_outliers: int = 10, auto_extend: bool = False) -> List[DiscoveredAxis]:
        """
        Triggers axis discovery from accumulated outliers.
        
        If auto_extend=True, automatically adds discovered axes to the system.
        """
        discoveries = self.discoverer.discover_axes(min_outliers)
        
        for disc in discoveries:
            self.evolution_history.append({
                "type": "axis_discovered",
                "axis": disc.name,
                "separation": disc.separation_strength,
                "poles": disc.suggested_labels
            })
            
            if auto_extend and disc.separation_strength > 0.5:
                self.discoverer.extend_system(disc)
                self.evolution_history.append({
                    "type": "axis_added",
                    "axis": disc.name
                })
                
        return discoveries
    
    def adapt_to_domain(self, domain_name: str, concepts: List[str]) -> List[DiscoveredAxis]:
        """
        Adapts the system to a new domain by discovering domain-specific axes.
        """
        discoveries = self.discoverer.detect_domain_axes(concepts)
        
        for disc in discoveries:
            disc.name = f"{domain_name}-{disc.name}"
            
        return discoveries
    
    def get_axis_count(self) -> Dict[str, int]:
        """Returns count of base vs discovered axes."""
        return {
            "base_axes": len(self.base.axes),
            "discovered_pending": len(self.discoverer.discovered_axes),
            "outliers_buffered": len(self.discoverer.outlier_buffer)
        }
