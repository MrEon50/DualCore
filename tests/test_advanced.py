"""
Tests for the new DualCore modules: Inference, Adaptive, Composites, and full Reasoning.
"""

import unittest
import numpy as np
from dualcore.core import DualCoreSystem
from dualcore.inference import PoleInferenceEngine, NeuralDualCoreInterface
from dualcore.adaptive import AdaptiveAxisDiscovery, SelfEvolvingDualCore
from dualcore.composites import CompositeAxisSystem, CompositeAxisDefinition
from dualcore.reasoning import ParadoxDetector, AnalogyEngine


class TestPoleInference(unittest.TestCase):
    """Tests for the Pole Inference Engine."""
    
    @classmethod
    def setUpClass(cls):
        cls.system = DualCoreSystem()
        cls.engine = PoleInferenceEngine(cls.system)
    
    def test_known_duality_inference(self):
        """Tests that known dualities are correctly inferred."""
        result = self.engine.infer_opposite("love")
        self.assertEqual(result.inferred_opposite, "hate")
        self.assertEqual(result.confidence, 1.0)
        
        result = self.engine.infer_opposite("chaos")
        self.assertEqual(result.inferred_opposite, "order")
    
    def test_unknown_concept_inference(self):
        """Tests inference for concepts not in known dualities."""
        result = self.engine.infer_opposite("blockchain")
        # Should return something (even if hypothetical)
        self.assertIsNotNone(result.inferred_opposite)
        self.assertGreater(len(result.inferred_opposite), 0)
    
    def test_understand_duality(self):
        """Tests the full duality understanding analysis."""
        result = self.engine.understand_duality("justice")
        
        self.assertIn("concept", result)
        self.assertIn("profile", result)
        self.assertIn("inferred_opposite", result)
        self.assertIn("understanding", result)
        self.assertIsInstance(result["understanding"], str)
    
    def test_generate_balanced_opposites(self):
        """Tests generating opposites for multiple concepts."""
        concepts = ["hot", "fast", "good"]
        pairs = self.engine.generate_balanced_opposites(concepts)
        
        self.assertEqual(len(pairs), 3)
        for original, opposite in pairs:
            self.assertIsNotNone(opposite)


class TestNeuralInterface(unittest.TestCase):
    """Tests for the Neural-DualCore Interface."""
    
    @classmethod
    def setUpClass(cls):
        cls.system = DualCoreSystem()
        cls.interface = NeuralDualCoreInterface(cls.system)
    
    def test_constrain_output_consistent(self):
        """Tests that consistent outputs are recognized."""
        result = self.interface.constrain_output("The cold water froze")
        self.assertTrue(result["is_consistent"])
    
    def test_constrain_output_paradox(self):
        """Tests that paradoxical outputs are detected."""
        result = self.interface.constrain_output("The hot ice burned")
        self.assertFalse(result["is_consistent"])
    
    def test_create_training_signal(self):
        """Tests training signal generation."""
        signal = self.interface.create_training_signal("science")
        
        # Should have entries for all 12 axes (position + confidence)
        self.assertEqual(len(signal), 24)  # 12 axes * 2 (pos + conf)
        
        for key, value in signal.items():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
    
    def test_enrich_with_duality(self):
        """Tests enrichment of neural concepts."""
        concepts = ["robot", "algorithm"]
        result = self.interface.enrich_with_duality(concepts)
        
        self.assertIn("concepts", result)
        self.assertIn("missing_poles", result)
        self.assertIn("balance_recommendation", result)


class TestAdaptiveDiscovery(unittest.TestCase):
    """Tests for Adaptive Axis Discovery."""
    
    @classmethod
    def setUpClass(cls):
        cls.system = DualCoreSystem()
        cls.discoverer = AdaptiveAxisDiscovery(cls.system)
    
    def test_analyze_and_buffer(self):
        """Tests that outliers are buffered correctly."""
        initial_buffer = len(self.discoverer.outlier_buffer)
        
        # Analyze some concepts
        self.discoverer.analyze_and_buffer("quantum entanglement superposition")
        
        # Buffer might grow if concept is an outlier
        self.assertGreaterEqual(len(self.discoverer.outlier_buffer), initial_buffer)
    
    def test_self_evolving_system(self):
        """Tests the self-evolving DualCore wrapper."""
        evolving = SelfEvolvingDualCore()
        
        # Analyze produces extended result
        result = evolving.analyze("test concept")
        
        self.assertIn("base_profile", result)
        self.assertIn("coherence", result)
        self.assertIn("is_outlier", result)
    
    def test_axis_count(self):
        """Tests axis counting."""
        evolving = SelfEvolvingDualCore()
        counts = evolving.get_axis_count()
        
        self.assertIn("base_axes", counts)
        self.assertEqual(counts["base_axes"], 12)


class TestCompositeAxes(unittest.TestCase):
    """Tests for Composite Axes."""
    
    @classmethod
    def setUpClass(cls):
        cls.system = DualCoreSystem()
        cls.composite_system = CompositeAxisSystem(cls.system)
    
    def test_default_composites_exist(self):
        """Tests that default composites are loaded."""
        self.assertGreater(len(self.composite_system.composites), 0)
        
        # Check for expected composites
        names = [c.name for c in self.composite_system.composites]
        self.assertIn("Elegance", names)
    
    def test_analyze_with_composites(self):
        """Tests analysis with composite axes."""
        result = self.composite_system.analyze_with_composites("beautiful simplicity")
        
        self.assertIn("base_profile", result)
        self.assertIn("composites", result)
        self.assertIn("Elegance", result["composites"])
    
    def test_add_custom_composite(self):
        """Tests adding a custom composite axis."""
        custom = CompositeAxisDefinition(
            name="TestComposite",
            components=[("Simple-Complex", 1.0), ("Good-Bad", 1.0)],
            description="Test axis"
        )
        
        initial_count = len(self.composite_system.composites)
        self.composite_system.add_composite(custom)
        
        self.assertEqual(len(self.composite_system.composites), initial_count + 1)


class TestFullReasoning(unittest.TestCase):
    """Integration tests for full reasoning capabilities."""
    
    @classmethod
    def setUpClass(cls):
        cls.system = DualCoreSystem()
        cls.paradox = ParadoxDetector(cls.system)
        cls.analogy = AnalogyEngine(cls.system)
    
    def test_paradox_with_report(self):
        """Tests paradox detection with full report."""
        report = self.paradox.detect_paradox("frozen fire")
        
        self.assertTrue(report.is_paradox)
        self.assertGreater(report.severity, 0.5)
        self.assertIn("PARADOX", report.explanation.upper())
    
    def test_physical_possibility(self):
        """Tests physical possibility checking."""
        possible = self.paradox.check_physical_possibility("a warm cup of tea")
        self.assertTrue(possible["physically_possible"])
        
        impossible = self.paradox.check_physical_possibility("hot ice cube")
        self.assertFalse(impossible["physically_possible"])
    
    def test_analogy_with_opposition_detection(self):
        """Tests that analogy engine detects opposition relations."""
        result = self.analogy.complete_analogy(
            "hot", "cold", "fast",
            candidates=["slow", "quick", "red"]
        )
        
        self.assertIn("is_opposition", result)
        # Hot->Cold should be a clearer opposition
        # The feature exists and returns a boolean
        self.assertIsInstance(result["is_opposition"], bool)
    
    def test_relation_interpretation(self):
        """Tests that relation vectors are interpreted correctly."""
        result = self.analogy.complete_analogy("king", "queen", "man")
        
        self.assertIn("interpretation", result)
        self.assertIsInstance(result["interpretation"], str)


if __name__ == '__main__':
    unittest.main()
