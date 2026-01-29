import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Load scales from JSON file
with open('scales.json', 'r') as f:
    scales_data = json.load(f)

# Amino acid order
aa_order = ['A', 'I', 'L', 'M', 'P', 'V', 'F', 'W', 'Y', 'C',
            'G', 'N', 'Q', 'S', 'T', 'D', 'E', 'H', 'K', 'R']

scale_names = list(scales_data.keys())

# Optional: output directory
out_dir = "plots"
os.makedirs(out_dir, exist_ok=True)

for scale_name in scale_names:
    values = np.array([scales_data[scale_name].get(aa, 0) for aa in aa_order])

    fig, ax = plt.subplots(figsize=(6, 4))

    # Line plot
    ax.plot(range(len(aa_order)), values, 'o-', linewidth=2.5, markersize=6,
            color='steelblue', markerfacecolor='white', markeredgewidth=2)

    # Styling
    ax.axhline(y=0, color='r', linestyle='-', linewidth=1.5)
    ax.set_title(scale_name.replace('_', ' ').title(), fontsize=12, pad=15)
    ax.set_xlabel('Amino acid', fontsize=10)
    ax.set_ylabel('Scale value', fontsize=10)
    ax.set_xticks(range(len(aa_order)))
    ax.set_xticklabels(aa_order, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.margins(x=0.05, y=0.1)

    plt.tight_layout()

    # Safe filename: e.g. hydroscales -> hydroscales.png
    fname = "scale_" + scale_name.replace(" ", "_").replace("/", "_") + ".png"
    fpath = os.path.join(out_dir, fname)
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved {scale_name} to {fpath}")

print(f"Created {len(scale_names)} separate line plot files in '{out_dir}'")