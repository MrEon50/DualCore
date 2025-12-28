# DualCore 🧠 — Cognitive Architecture for AI Reasoning

**DualCore** is a revolutionary cognitive architecture that provides AI with a human-like coordinate system for understanding, reasoning, and evaluating concepts. Instead of operating in undefined high-dimensional embedding spaces, DualCore maps everything onto **12 fundamental Dual Axes** — giving AI a "north star" for thought.

> **"If you see WHITE, you should KNOW that BLACK exists — even without seeing it."**
> — This is the essence of understanding duality, and now AI can do it too.
>
> [Statistics] -----> [DualCore] -----> [Cognitive Core] -----> [AGI]
    GPT-4                HERE            ????? (Syntheos)       ?????

"I know patterns" "I know structure" "I'm learning structure" "I am"

---

## 🚀 Key Features

### ⚓ Anchor Sets — The "White vs Black" Detection
Each pole (e.g., *Good* or *Evil*) is anchored by a cluster of semantically rich concepts. This ensures the system always has fixed reference points for extremes, enabling precise detection of opposites.

### 🧠 Pole Inference — True Understanding
DualCore doesn't just classify — it **infers**. If it sees "love", it knows "hate" must exist. This is the bridge between pattern matching and genuine understanding.

### 🔍 Paradox Detection — Logical Impossibility Filter
The system detects logical contradictions like "hot ice" or "true lie" and flags them as paradoxes. This enables AI safety checks and prevents generation of impossible statements.

### 🧬 Adaptive Axis Discovery — Self-Evolution
When encountering concepts that don't fit existing axes, DualCore can discover new dimensions automatically. The system evolves with your data.

### 🔗 Neural Network Integration — Co-Evolution
DualCore provides a cognitive layer for neural networks, enabling them to learn structured reasoning. The `NeuralDualCoreInterface` allows bidirectional communication.

### 📊 Confidence Scores — Know What You Don't Know
Every projection includes a confidence score indicating how relevant an axis is to the concept. Orthogonal concepts get low confidence, aligned concepts get high.

---

## 🛠 Installation

```bash
pip install -r requirements.txt
pip install -e .
```

---

## 📖 Quick Start

### Basic Analysis
```python
from dualcore.core import DualCoreSystem

dc = DualCoreSystem()

# Analyze a concept
profile = dc.analyze("artificial intelligence", context="computer science")

# Access interpretable positions with confidence
for axis_name, pos in profile.items():
    print(f"{axis_name}: {pos.position:.2f} (conf: {pos.confidence:.2f}) — {pos.label}")
```

### Paradox Detection
```python
from dualcore.reasoning import ParadoxDetector

detector = ParadoxDetector(dc)

# Check for logical impossibilities
report = detector.detect_paradox("the hot ice melted")
print(f"Is Paradox: {report.is_paradox}")  # True
print(f"Explanation: {report.explanation}")
```

### Pole Inference — Understanding Duality
```python
from dualcore.inference import PoleInferenceEngine

engine = PoleInferenceEngine(dc)

# Infer what the opposite would be
result = engine.infer_opposite("chaos")
print(f"Opposite of 'chaos': {result.inferred_opposite}")  # "order"

# Full duality understanding
understanding = engine.understand_duality("democracy")
print(understanding["understanding"])
```

### Composite Axes
```python
from dualcore.composites import CompositeAxisSystem

composite = CompositeAxisSystem(dc)

# Analyze with second-order dimensions
result = composite.analyze_with_composites("elegant mathematical proof")
print(f"Elegance Score: {result['composites']['Elegance']['position']:.2f}")
```

### Neural Network Integration
```python
from dualcore.inference import NeuralDualCoreInterface

interface = NeuralDualCoreInterface(dc)

# Generate training signal for neural networks
signal = interface.create_training_signal("intelligence")

# Validate neural network output for logical consistency
constraint = interface.constrain_output("The frozen fire burned brightly")
print(f"Consistent: {constraint['is_consistent']}")  # False
```

### Self-Evolving System
```python
from dualcore.adaptive import SelfEvolvingDualCore

evolving = SelfEvolvingDualCore()

# Analyze and buffer outliers
for concept in your_domain_concepts:
    evolving.analyze(concept)

# Discover new axes from accumulated outliers
new_axes = evolving.evolve(auto_extend=True)
```

---

## ⚖️ The 12 Fundamental Axes

### 🏗️ Structure
| Axis | Description |
|------|-------------|
| **Simple ↔ Complex** | Structural and relational depth |
| **Concrete ↔ Abstract** | Physical tangibility vs. conceptual theory |
| **Local ↔ Global** | Scope of influence and relevance |
| **Specific ↔ General** | Precision vs. universal applicability |

### ⚙️ Process
| Axis | Description |
|------|-------------|
| **Fast ↔ Slow** | Temporal dynamics and speed |
| **Analytic ↔ Intuitive** | Systematic vs. instinctive reasoning |
| **Static ↔ Dynamic** | Stability vs. continuous evolution |
| **Controlled ↔ Automatic** | Deliberate vs. spontaneous reaction |

### 💎 Value (Axiology/Epistemology)
| Axis | Description |
|------|-------------|
| **Certain ↔ Uncertain** | Epistemic confidence and predictability |
| **True ↔ False** | Veracity and logical correctness |
| **Good ↔ Bad** | Moral valence and ethical value |
| **Beautiful ↔ Ugly** | Aesthetic harmony and quality |

---

## 🎨 Composite Axes (Second-Order Dimensions)

Pre-defined composite dimensions for richer analysis:

| Composite | Formula | Description |
|-----------|---------|-------------|
| **Elegance** | Simple + Beautiful | Simple and aesthetically pleasing |
| **Wisdom** | Abstract + True + Certain | Deep, verified understanding |
| **Innovation** | Dynamic + Intuitive | Creative, evolving breakthroughs |
| **Danger** | Bad + Uncertain | High-risk, unpredictable threats |
| **Clarity** | Simple + Concrete + Certain | Clear, grounded, reliable |

---

## 🧪 Testing & Validation

DualCore includes a comprehensive test suite and benchmark:

```bash
# Run all 32 unit tests
python -m unittest discover -s tests

# Run accuracy benchmark
python benchmarks/run_benchmarks.py
```

### Current Benchmark Results (v0.3.0)
| Category | Accuracy |
|----------|----------|
| Semantic Ordering | 93.3% |
| Paradox Detection | 100.0% |
| Analogy Reasoning | 60.0% |
| Confidence Calibration | 80.0% |
| **Overall** | **90.0%** |

---

## 📁 Project Structure

```
DualCore/
├── dualcore/
│   ├── core.py           # Main system (12 axes, confidence)
│   ├── reasoning.py      # Paradox detection + Analogies
│   ├── inference.py      # Pole inference + Neural API
│   ├── adaptive.py       # Self-evolving axes
│   ├── composites.py     # Second-order dimensions
│   ├── visualization.py  # Radar charts
│   └── integration/      # PyTorch integration
├── tests/                # 32 comprehensive tests
├── benchmarks/           # Accuracy measurement
└── demos/                # Usage examples
```

---

## 🔮 Philosophy

DualCore is built on a fundamental insight: **Human thought operates in dualities**. We understand "hot" because we know "cold" exists. We grasp "good" by its contrast with "evil". This architecture gives AI the same cognitive structure.

Unlike black-box embeddings, DualCore provides:
- **Interpretability**: Know *why* concepts are similar
- **Inference**: Deduce what must exist from what is observed
- **Constraints**: Prevent logically impossible outputs
- **Evolution**: Adapt to new domains automatically

---

## 📄 License

MIT License — Free for research and commercial use.

---

## 🤝 Contributing

Contributions welcome! See `ROADMAP.md` for planned features and areas needing work.

---

*DualCore: Teaching AI to think in opposites, so it can understand the whole.*

