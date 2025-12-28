"""
Demonstration of DualCore's Pole Inference and Neural Integration capabilities.

This shows how DualCore can:
1. INFER opposites it has never seen (true understanding)
2. Detect missing poles in datasets
3. Interface with neural networks for co-evolution
"""

from dualcore.core import DualCoreSystem
from dualcore.inference import PoleInferenceEngine, NeuralDualCoreInterface


def main():
    print("=" * 70)
    print("DualCore: Pole Inference & Reasoning Demonstration")
    print("=" * 70)
    
    # Initialize systems
    dc = DualCoreSystem()
    inference = PoleInferenceEngine(dc)
    neural_interface = NeuralDualCoreInterface(dc)
    
    # ===========================================
    # 1. INFERRING OPPOSITES
    # ===========================================
    print("\n" + "-" * 50)
    print("1. POLE INFERENCE: Seeing one side, knowing the other exists")
    print("-" * 50)
    
    test_concepts = ["love", "chaos", "simplicity", "artificial intelligence", "quantum"]
    
    for concept in test_concepts:
        result = inference.infer_opposite(concept)
        print(f"\n  Concept: '{concept}'")
        print(f"  Inferred Opposite: '{result.inferred_opposite}'")
        print(f"  Axis: {result.axis_name}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Reasoning: {result.reasoning}")
    
    # ===========================================
    # 2. DETECTING MISSING POLES
    # ===========================================
    print("\n" + "-" * 50)
    print("2. MISSING POLE DETECTION: Finding imbalance in knowledge")
    print("-" * 50)
    
    # Simulate a biased dataset (only positive concepts)
    biased_concepts = [
        "happiness", "joy", "love", "kindness", "hope",
        "success", "beauty", "harmony", "peace"
    ]
    
    print(f"\n  Analyzing biased dataset: {biased_concepts}")
    missing = inference.detect_missing_poles(biased_concepts)
    
    if missing:
        print("\n  IMBALANCE DETECTED:")
        for m in missing:
            print(f"    - Axis: {m['axis']}")
            print(f"      Observed: '{m['observed_pole']}' | Missing: '{m['missing_pole']}'")
            print(f"      Recommendation: {m['recommendation']}")
    else:
        print("\n  Dataset is balanced across all axes.")
    
    # ===========================================
    # 3. UNDERSTANDING DUALITY
    # ===========================================
    print("\n" + "-" * 50)
    print("3. DUALITY UNDERSTANDING: Deep analysis of a concept")
    print("-" * 50)
    
    deep_concept = "democracy"
    understanding = inference.understand_duality(deep_concept)
    
    print(f"\n  Concept: '{deep_concept}'")
    print(f"\n  Understanding:")
    print(f"    {understanding['understanding']}")
    
    if understanding['primary_axis']:
        print(f"\n  Primary Axis: {understanding['primary_axis']['axis']}")
        print(f"  Aligned Pole: {understanding['primary_axis']['aligned_pole']}")
    
    print(f"\n  Inferred Opposite: {understanding['inferred_opposite'].inferred_opposite}")
    
    # ===========================================
    # 4. NEURAL NETWORK INTERFACE
    # ===========================================
    print("\n" + "-" * 50)
    print("4. NEURAL NETWORK INTEGRATION: Co-evolution capability")
    print("-" * 50)
    
    # Simulate neural network concepts
    nn_concepts = ["robot", "algorithm", "data", "computation"]
    
    print(f"\n  Enriching neural network concepts: {nn_concepts}")
    enriched = neural_interface.enrich_with_duality(nn_concepts)
    
    print(f"\n  Balance Analysis: {enriched['balance_recommendation']}")
    
    # Constraint checking
    print("\n  Logical Consistency Check:")
    
    test_outputs = [
        "The water is cold and frozen",  # Consistent
        "The hot ice melted quickly",     # Paradox!
        "She spoke in a silent whisper", # Edge case
    ]
    
    for output in test_outputs:
        constraint = neural_interface.constrain_output(output)
        status = "OK" if constraint['is_consistent'] else "PARADOX"
        print(f"    [{status}] '{output}'")
    
    # Training signal generation
    print("\n  Training Signal for 'intelligence':")
    signal = neural_interface.create_training_signal("intelligence")
    
    # Show top 3 most confident axes
    conf_items = [(k, v) for k, v in signal.items() if '_conf' in k]
    conf_items.sort(key=lambda x: x[1], reverse=True)
    
    for axis_conf, conf_val in conf_items[:3]:
        axis_name = axis_conf.replace('_conf', '')
        pos = signal[axis_name]
        print(f"    {axis_name}: position={pos:.2f}, confidence={conf_val:.2f}")
    
    print("\n" + "=" * 70)
    print("CONCLUSION: DualCore can now INFER, not just DETECT.")
    print("It understands that if WHITE exists, BLACK must exist too.")
    print("This is the foundation for neural-symbolic co-evolution.")
    print("=" * 70)


if __name__ == "__main__":
    main()
