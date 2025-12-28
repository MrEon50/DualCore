import unittest
import numpy as np
from dualcore.core import DualCoreSystem, DualAxis, AxisPosition
from dualcore.utils import get_embedding
from dualcore.similarity import dual_profile_similarity

class TestDualCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = DualCoreSystem()

    def test_embedding_generation(self):
        emb = get_embedding("test")
        self.assertIsInstance(emb, np.ndarray)
        self.assertAlmostEqual(np.linalg.norm(emb), 1.0, places=5)

    def test_axis_projection(self):
        axis = DualAxis("Test", "A", "B")
        emb = get_embedding("A")
        pos, conf = axis.project(emb)  # Now returns (position, confidence)
        self.assertGreaterEqual(pos, 0.0)
        self.assertLessEqual(pos, 1.0)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)
        # "A" should be closer to 0 than 1
        pos_a, _ = axis.project(get_embedding("A"))
        pos_b, _ = axis.project(get_embedding("B"))
        self.assertLess(pos_a, pos_b)

    def test_system_analyze(self):
        profile = self.system.analyze("artificial intelligence")
        self.assertEqual(len(profile), 12)  # Updated to 12 axes
        for name, pos in profile.items():
            self.assertIsInstance(pos, AxisPosition)
            self.assertGreaterEqual(pos.position, 0.0)
            self.assertLessEqual(pos.position, 1.0)

    def test_semantic_logic(self):
        # "2+2=4" should be simpler than "General Relativity"
        simple_prof = self.system.analyze("2+2=4")
        complex_prof = self.system.analyze("The Einstein field equations in general relativity")
        
        self.assertLess(
            simple_prof["Simple-Complex"].position, 
            complex_prof["Simple-Complex"].position,
            "2+2=4 should be simpler than General Relativity"
        )
        
        # "A dog" should be more concrete than "Philosophy"
        dog_prof = self.system.analyze("a golden retriever dog")
        phil_prof = self.system.analyze("the branch of philosophy known as ontology")
        
        self.assertLess(
            dog_prof["Concrete-Abstract"].position,
            phil_prof["Concrete-Abstract"].position,
            "Dog should be more concrete than Ontology"
        )

    def test_similarity(self):
        prof_a = self.system.analyze("happy")
        prof_b = self.system.analyze("joyful")
        prof_c = self.system.analyze("quantum physics")
        
        sim_ab = dual_profile_similarity(prof_a, prof_b)
        sim_ac = dual_profile_similarity(prof_a, prof_c)
        
        self.assertGreater(sim_ab, sim_ac, "Happy/Joyful should be more similar than Happy/Quantum")

    def test_compare(self):
        # 'ice' and 'fire' should be quite different on several dimensions
        res = self.system.compare("ice", "fire")
        self.assertIn("global_similarity", res)
        self.assertIn("axis_similarities", res)
        # Even if they share some 'Concrete' or 'Physical' traits, 
        # they should differ enough to not be 100% similar.
        self.assertLess(res["global_similarity"], 0.98) 

if __name__ == '__main__':
    unittest.main()
