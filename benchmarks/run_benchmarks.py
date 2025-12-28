"""
Benchmark Suite for DualCore.

Tests the system against known semantic relationships to quantify accuracy.
"""

import time
from typing import List, Tuple, Dict
from dualcore.core import DualCoreSystem
from dualcore.reasoning import ParadoxDetector, AnalogyEngine


def run_benchmarks():
    print("=" * 60)
    print("DualCore Benchmark Suite")
    print("=" * 60)
    
    dc = DualCoreSystem()
    
    results = {
        "semantic_ordering": run_semantic_ordering_benchmark(dc),
        "paradox_detection": run_paradox_benchmark(dc),
        "analogy_reasoning": run_analogy_benchmark(dc),
        "confidence_calibration": run_confidence_benchmark(dc),
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_score = 0
    total_tests = 0
    
    for name, res in results.items():
        score = res.get("accuracy", res.get("score", 0))
        tests = res.get("total", 1)
        total_score += score * tests
        total_tests += tests
        print(f"{name}: {score:.1%} ({res.get('passed', '?')}/{tests})")
    
    overall = total_score / total_tests if total_tests else 0
    print(f"\nOVERALL ACCURACY: {overall:.1%}")
    
    return results


def run_semantic_ordering_benchmark(dc: DualCoreSystem) -> Dict:
    """
    Tests if DualCore correctly orders concepts on each axis.
    """
    print("\n--- Semantic Ordering Benchmark ---")
    
    # (axis_name, concept_A, concept_B) where A should score LOWER than B
    test_cases = [
        # Simple-Complex (lower = simpler)
        ("Simple-Complex", "1+1=2", "differential equations in fluid dynamics"),
        ("Simple-Complex", "a ball", "a supercomputer"),
        ("Simple-Complex", "hello", "Shakespeare's complete works"),
        
        # Concrete-Abstract (lower = more concrete)
        ("Concrete-Abstract", "a wooden chair", "the concept of justice"),
        ("Concrete-Abstract", "a red apple", "mathematical infinity"),
        ("Concrete-Abstract", "a hammer", "democracy"),
        
        # Good-Bad (lower = more good)
        ("Good-Bad", "helping the poor", "murder"),
        ("Good-Bad", "kindness", "cruelty"),
        ("Good-Bad", "charity", "theft"),
        
        # True-False (lower = more true)
        ("True-False", "scientific fact", "conspiracy theory"),
        ("True-False", "verified data", "rumor"),
        
        # Fast-Slow (lower = faster)
        ("Fast-Slow", "lightning bolt", "glacier movement"),
        ("Fast-Slow", "instant messaging", "postal mail"),
        
        # Certain-Uncertain (lower = more certain)
        ("Certain-Uncertain", "mathematical proof", "wild guess"),
        ("Certain-Uncertain", "law of gravity", "fortune telling"),
    ]
    
    passed = 0
    failed_cases = []
    
    for axis, concept_a, concept_b in test_cases:
        profile_a = dc.analyze(concept_a)
        profile_b = dc.analyze(concept_b)
        
        pos_a = profile_a[axis].position
        pos_b = profile_b[axis].position
        
        if pos_a < pos_b:
            passed += 1
            status = "PASS"
        else:
            failed_cases.append((axis, concept_a, concept_b, pos_a, pos_b))
            status = "FAIL"
        
        print(f"  [{status}] {axis}: '{concept_a[:20]}...' < '{concept_b[:20]}...' ({pos_a:.2f} vs {pos_b:.2f})")
    
    accuracy = passed / len(test_cases)
    print(f"\nSemantic Ordering: {passed}/{len(test_cases)} = {accuracy:.1%}")
    
    return {"accuracy": accuracy, "passed": passed, "total": len(test_cases), "failed": failed_cases}


def run_paradox_benchmark(dc: DualCoreSystem) -> Dict:
    """
    Tests paradox detection accuracy.
    """
    print("\n--- Paradox Detection Benchmark ---")
    
    detector = ParadoxDetector(dc)
    
    # (text, should_be_paradox)
    test_cases = [
        # True paradoxes (logical impossibilities)
        ("hot ice", True),
        ("frozen fire", True),
        ("living dead person", True),
        ("true lie", True),
        ("bright darkness", True),
        ("dry water", True),
        ("silent scream", True),
        ("visible invisible object", True),
        
        # Not paradoxes (normal concepts)
        ("cold ice", False),
        ("hot coffee", False),
        ("bright sun", False),
        ("quiet library", False),
        ("wet rain", False),
        
        # Metaphors (debatable - system should flag but lower severity)
        ("frozen heart", False),  # Metaphor, not literal
        ("burning passion", False),  # Metaphor
    ]
    
    passed = 0
    
    for text, expected_paradox in test_cases:
        report = detector.detect_paradox(text)
        detected = report.is_paradox
        
        if detected == expected_paradox:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"
        
        print(f"  [{status}] '{text}' -> paradox={detected} (expected {expected_paradox})")
    
    accuracy = passed / len(test_cases)
    print(f"\nParadox Detection: {passed}/{len(test_cases)} = {accuracy:.1%}")
    
    return {"accuracy": accuracy, "passed": passed, "total": len(test_cases)}


def run_analogy_benchmark(dc: DualCoreSystem) -> Dict:
    """
    Tests analogical reasoning.
    """
    print("\n--- Analogy Reasoning Benchmark ---")
    
    engine = AnalogyEngine(dc)
    
    # (A, B, C, correct_D, candidate_pool)
    test_cases = [
        ("king", "queen", "man", "woman", ["woman", "child", "dog", "tree"]),
        ("hot", "cold", "fast", "slow", ["slow", "quick", "red", "heavy"]),
        ("big", "small", "tall", "short", ["short", "wide", "heavy", "loud"]),
        ("day", "night", "light", "dark", ["dark", "bright", "warm", "cold"]),
        ("good", "evil", "truth", "lie", ["lie", "fact", "story", "news"]),
    ]
    
    passed = 0
    
    for a, b, c, correct_d, candidates in test_cases:
        result = engine.complete_analogy(a, b, c, candidates)
        best = result.get("best_match")
        
        if best == correct_d:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"
        
        print(f"  [{status}] {a}:{b} :: {c}:? -> {best} (expected {correct_d})")
        print(f"         Ranking: {result.get('ranked_candidates', [])[:3]}")
    
    accuracy = passed / len(test_cases)
    print(f"\nAnalogy Reasoning: {passed}/{len(test_cases)} = {accuracy:.1%}")
    
    return {"accuracy": accuracy, "passed": passed, "total": len(test_cases)}


def run_confidence_benchmark(dc: DualCoreSystem) -> Dict:
    """
    Tests if confidence scores are calibrated correctly.
    """
    print("\n--- Confidence Calibration Benchmark ---")
    
    # Concepts that should have HIGH confidence on specific axes
    high_confidence_cases = [
        ("mathematics", "Simple-Complex"),
        ("rock", "Concrete-Abstract"),
        ("charity", "Good-Bad"),
    ]
    
    # Concepts that should have LOW confidence on specific axes (orthogonal)
    low_confidence_cases = [
        ("blue color", "Good-Bad"),  # Color is not moral
        ("number seven", "Beautiful-Ugly"),  # Number is not aesthetic
    ]
    
    passed = 0
    total = len(high_confidence_cases) + len(low_confidence_cases)
    
    for text, axis in high_confidence_cases:
        profile = dc.analyze(text)
        conf = profile[axis].confidence
        
        if conf > 0.15:  # Reasonable threshold for "high"
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"
        
        print(f"  [{status}] '{text}' on {axis}: conf={conf:.2f} (expected HIGH)")
    
    for text, axis in low_confidence_cases:
        profile = dc.analyze(text)
        conf = profile[axis].confidence
        
        if conf < 0.3:  # Reasonable threshold for "low"
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"
        
        print(f"  [{status}] '{text}' on {axis}: conf={conf:.2f} (expected LOW)")
    
    accuracy = passed / total
    print(f"\nConfidence Calibration: {passed}/{total} = {accuracy:.1%}")
    
    return {"accuracy": accuracy, "passed": passed, "total": total}


if __name__ == "__main__":
    start = time.time()
    results = run_benchmarks()
    elapsed = time.time() - start
    print(f"\nBenchmark completed in {elapsed:.2f} seconds.")
