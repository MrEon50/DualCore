import unittest
import numpy as np
from dualcore.core import DualCoreSystem

class TestConfidenceScores(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = DualCoreSystem()

    def test_high_confidence_for_clear_concepts(self):
        """
        Concepts that clearly belong to an axis should have high confidence.
        'Simple addition' should be positioned towards Simple.
        """
        profile = self.system.analyze("simple basic addition like 1+1")
        sc = profile["Simple-Complex"]
        
        # Should be positioned towards 'Simple' (lower value) 
        self.assertLess(sc.position, 0.55, "Basic addition should be on Simple side")
        self.assertGreater(sc.confidence, 0.1, "Clear concept should have reasonable confidence")

    def test_low_confidence_for_orthogonal_concepts(self):
        """
        Concepts that don't relate to an axis should have lower confidence.
        'The color blue' has little to do with Good-Bad axis.
        """
        profile = self.system.analyze("the color blue")
        gb = profile["Good-Bad"]
        
        # Confidence on Good-Bad should be relatively low for a neutral color
        self.assertLess(gb.confidence, 0.3, "Color should have low moral confidence")

    def test_profile_coherence_well_defined(self):
        """
        A well-known, multi-dimensional concept should have high coherence.
        """
        profile = self.system.analyze("advanced scientific research")
        coherence = self.system.get_profile_coherence(profile)
        
        self.assertIn("overall_coherence", coherence)
        self.assertIn("strong_axes", coherence)
        self.assertIn("interpretation", coherence)

    def test_profile_coherence_ambiguous(self):
        """
        A nonsense or very abstract concept should have lower coherence.
        """
        profile = self.system.analyze("xyzzypqr") # Nonsense word
        coherence = self.system.get_profile_coherence(profile)
        
        # Should have low overall coherence since it's not a real concept
        self.assertLess(coherence["overall_coherence"], 0.4)

    def test_confidence_accessible_in_profile(self):
        """
        Verify that confidence is part of every AxisPosition in the profile.
        """
        profile = self.system.analyze("democracy")
        for axis_name, pos in profile.items():
            self.assertTrue(hasattr(pos, 'confidence'), f"Missing confidence on {axis_name}")
            self.assertGreaterEqual(pos.confidence, 0.0)
            self.assertLessEqual(pos.confidence, 1.0)

if __name__ == '__main__':
    unittest.main()
