import unittest
import numpy as np
from dualcore.core import DualCoreSystem
from dualcore.similarity import dual_profile_similarity

class TestCognitiveReasoning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = DualCoreSystem()

    def test_analogical_consistency(self):
        """
        Tests if the system maintains consistent profiles for analogous concepts.
        Concept: 'Small/Young' vs 'Large/Old'
        """
        # Set 1: Animals
        puppy = self.system.analyze("a small puppy")
        dog = self.system.analyze("an adult dog")
        
        # Set 2: Plants
        sapling = self.system.analyze("a young tree sapling")
        tree = self.system.analyze("a fully grown oak tree")
        
        # Difference profiles
        diff_animal = {name: puppy[name].position - dog[name].position for name in puppy}
        diff_plant = {name: sapling[name].position - tree[name].position for name in sapling}
        
        # Check if the 'direction' of difference is similar on key axes
        # Both puppy and sapling should be more 'Simple' or 'Local' than their adult versions
        axes_to_check = ["Simple-Complex", "Local-Global"]
        
        for axis in axes_to_check:
            # They don't have to be identical, but the SIGN of difference should be consistent
            sign_animal = np.sign(diff_animal[axis])
            sign_plant = np.sign(diff_plant[axis])
            
            if abs(diff_animal[axis]) > 0.02 and abs(diff_plant[axis]) > 0.02:
                self.assertEqual(sign_animal, sign_plant, f"Analogy failed on {axis}")

    def test_paradox_detection(self):
        """
        Tests how the system handles concepts that blend opposites.
        """
        # 'A logical intuition' balances two poles
        logical_intuition = self.system.analyze("a logical intuition")
        pos = logical_intuition["Analytic-Intuitive"].position
        
        # Should be closer to center (0.5) than pure 'mathematical proof'
        math_proof = self.system.analyze("a formal mathematical proof")
        pure_hunch = self.system.analyze("a random gut feeling")
        
        dist_to_center_paradox = abs(pos - 0.5)
        dist_to_center_proof = abs(math_proof["Analytic-Intuitive"].position - 0.5)
        dist_to_center_hunch = abs(pure_hunch["Analytic-Intuitive"].position - 0.5)
        
        self.assertLess(dist_to_center_paradox, max(dist_to_center_proof, dist_to_center_hunch))

    def test_moral_valuation(self):
        """
        Tests the ethical axis.
        """
        help_str = self.system.analyze("helping an elderly person cross the street")
        hurt_str = self.system.analyze("intentionally causing harm to an innocent being")
        
        # Good-Bad axis: lower is Good (Pole A), higher is Bad (Pole B)
        # Note: Depending on default_axes.json, pole_a is 'good', pole_b is 'bad'
        # So helping should be closer to 0, harming closer to 1.
        self.assertLess(help_str["Good-Bad"].position, hurt_str["Good-Bad"].position)

if __name__ == '__main__':
    unittest.main()
