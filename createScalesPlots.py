import json
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import os

# Load scales from JSON file
with open('scales.json', 'r') as f:
    scales_data = json.load(f)

with open('normalizedScales.json', 'r') as f:
    norm_scales_data = json.load(f)

# Amino acid order
aa_order = ['A', 'I', 'L', 'M', 'P', 'V', 'F', 'W', 'Y', 'C',
            'G', 'N', 'Q', 'S', 'T', 'D', 'E', 'H', 'K', 'R']

colors = {
    'Alpha Helix': 'green',
    'Beta Sheet': 'green', 
    'Coil': 'green',
    'Free Energy': 'red',
    'Hydrophobicity': 'yellow',
    'Mutability': 'grey',
    'Stability': 'brown',
    'Volume': 'blue',
}

scale_names = list(scales_data.keys())

# Optional: output directory
out_dir = "plots"
os.makedirs(out_dir, exist_ok=True)

for scale_name in scale_names:
    # Normalized values (your original)
    values_norm = np.array([scales_data[scale_name].get(aa, 0) for aa in aa_order])

    # Original scales (multiple per feature) - plot ALL of them
    values_orig_list = []
    if scale_name in norm_scales_data:
        for orig_scale_name in norm_scales_data[scale_name]:
            values_orig = np.array([norm_scales_data[scale_name][orig_scale_name].get(aa, 0) for aa in aa_order])
            values_orig_list.append(values_orig)
    else:
        values_orig_list = [np.zeros(len(aa_order))]  # Fallback

    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    fig.subplots_adjust(right=1.05)

    # Plot 2: Original (half opacity, dashed orange)
    colorOr = 'magenta'
    if scale_name in scale_name:
        colorOr = colors[scale_name]

    for i, values_orig in enumerate(values_orig_list):
        ax.plot(range(len(aa_order)), values_orig, '-', markersize=0, color=colorOr, alpha=0.5, linewidth=1.2, label='AAindex scales' if i == 0 else "")

    # Line plot
    ax.plot(range(len(aa_order)), values_norm, '-', linewidth=2.2, markersize=5, color='black', markerfacecolor='white', markeredgewidth=2, label='Created Scale')

    # Styling
    ax.axhline(y=0, color='gray', linewidth=0.8, alpha=0.7)
    ax.set_title(scale_name.replace('_', ' ').title(), fontsize=13, pad=10, weight='bold')
    ax.set_xlabel('Amino acid', fontsize=10)
    ax.set_ylabel('Scale value', fontsize=10)
    ax.set_xticks(range(len(aa_order)))
    ax.set_xticklabels(aa_order, rotation=45, ha='right')
    ax.grid(False)
    ax.margins(x=0.05, y=0.1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_elements = [
        patches.Patch(facecolor='black', label='Created Scale'),
        patches.Patch(facecolor=colorOr, label='AAindex scales', alpha=0.5)
    ]
    ax.legend(handles=legend_elements, loc='upper left', 
          bbox_to_anchor=(1.02, 0.98), frameon=False, 
          fontsize=8, handlelength=1.2, handleheight=1.2)

    plt.tight_layout()

    # Safe filename: e.g. hydroscales -> hydroscales.png
    fname = "scale_" + scale_name.replace(" ", "_").replace("/", "_") + ".png"
    fpath = os.path.join(out_dir, fname)
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved {scale_name} to {fpath}")

print(f"Created {len(scale_names)} separate line plot files in '{out_dir}'")