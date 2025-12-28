import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Optional
from .core import AxisPosition

def plot_radar(profile: Dict[str, AxisPosition], 
               title: str = "DualCore Cognitive Profile", 
               save_path: Optional[str] = None):
    """
    Plots a radar chart for a single profile.
    """
    categories = list(profile.keys())
    values = [p.position for p in profile.values()]
    
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Complete the loop
    values += values[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=7)
    plt.ylim(0, 1)
    
    # Plot data
    ax.plot(angles, values, linewidth=2, linestyle='solid', label="Profile")
    ax.fill(angles, values, 'b', alpha=0.1)
    
    plt.title(title, size=20, color='blue', y=1.1)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_comparison(profile_a: Dict[str, AxisPosition], 
                   profile_b: Dict[str, AxisPosition],
                   label_a: str = "A", label_b: str = "B",
                   title: str = "Cognitive Comparison"):
    """
    Plots a dual radar chart for comparison.
    """
    categories = list(profile_a.keys())
    values_a = [profile_a[c].position for c in categories]
    values_b = [profile_b[c].position for c in categories]
    
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    values_a += values_a[:1]
    values_b += values_b[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    plt.ylim(0, 1)
    
    # Plot A
    ax.plot(angles, values_a, linewidth=2, linestyle='solid', label=label_a, color='blue')
    ax.fill(angles, values_a, 'blue', alpha=0.1)
    
    # Plot B
    ax.plot(angles, values_b, linewidth=2, linestyle='solid', label=label_b, color='red')
    ax.fill(angles, values_b, 'red', alpha=0.1)
    
    plt.title(title, size=20, color='black', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.show()

def plot_axis_distribution(axis_positions: List[float], axis_name: str):
    """Plots distribution of many concepts on a single axis."""
    plt.figure(figsize=(10, 4))
    sns.histplot(axis_positions, bins=20, kde=True)
    plt.title(f"Distribution on {axis_name}")
    plt.xlabel("Position (0=Left, 1=Right)")
    plt.xlim(0, 1)
    plt.show()
