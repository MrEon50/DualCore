from dualcore.core import DualCoreSystem
from dualcore.visualization import plot_radar
import os

def main():
    print("Initializing DualCore System...")
    dc = DualCoreSystem()
    
    concepts = [
        "A piece of bread",
        "A complex distributed cloud architecture",
        "Deep meditation and inner peace",
        "A fast sports car"
    ]
    
    profiles = {}
    for text in concepts:
        print(f"Analyzing: '{text}'")
        profile = dc.analyze(text)
        profiles[text] = profile
        
        # Print summary
        print(f"   Profile Summary:")
        for name, pos in profile.items():
            if pos.position > 0.7 or pos.position < 0.3:
                print(f"     - {name}: {pos.label} ({pos.position:.2f})")
        print("-" * 30)

    # Comparison example
    print("\nComparing 'Democracy' and 'Anarchy'...")
    comp = dc.compare("Democracy", "Anarchy")
    print(f"Global Cognitive Similarity: {comp['global_similarity']:.2%}")
    
    # Visualization
    print("\nGenerating Radar Chart for 'Cloud Architecture'...")
    # Since we can't show plots here, we'll just simulate the call
    # In a real environment, this would pop up a window or save a file
    # plot_radar(profiles["A complex distributed cloud architecture"], title="Cloud Architecture Profile")
    print("Done! (Plot would appear in a GUI environment)")

if __name__ == "__main__":
    main()
