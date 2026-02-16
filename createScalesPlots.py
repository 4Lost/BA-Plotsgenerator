import json
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np
import os

# ---------- CONFIG ----------
AA_ORDER = ['A', 'I', 'L', 'M', 'P', 'V', 'F', 'W', 'Y', 'C',
            'G', 'N', 'Q', 'S', 'T', 'D', 'E', 'H', 'K', 'R']
COLORS = {
    'Alpha Helix': 'green',
    'Beta Sheet': 'green', 
    'Coil': 'green',
    'Free Energy': 'red',
    'Hydrophobicity': 'magenta',
    'Mutability': 'grey',
    'Stability': 'brown',
    'Volume': 'blue',
}
OUTPUT_DIRECTORY = "plots"

# ---------- HELPERS ----------
def loadScales():
    with open('scales.json', 'r') as f:
        createdScales = json.load(f)

    with open('normalizedScales.json', 'r') as f:
        normalizedScales = json.load(f)

    return createdScales, normalizedScales


def getValues(createdScales, normalizedScales, scaleName):
    valuesNormalized = np.array([createdScales[scaleName].get(aa, 0) for aa in AA_ORDER])
    originalValues = []

    if scaleName in normalizedScales:
        for originalScaleName in normalizedScales[scaleName]:
            values_orig = np.array([normalizedScales[scaleName][originalScaleName].get(aa, 0) for aa in AA_ORDER])
            originalValues.append(values_orig)
    else:
        originalValues = [np.zeros(len(AA_ORDER))]
    
    return valuesNormalized, originalValues


def generatePlot(scaleName, valuesNormalized, originalValues):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    fig.subplots_adjust(right=1.05)

    colorOr = 'magenta'
    if scaleName in scaleName:
        colorOr = COLORS[scaleName]

    for i, values_orig in enumerate(originalValues):
        ax.plot(range(len(AA_ORDER)), values_orig, '-', markersize=0, color=colorOr, alpha=0.5, linewidth=1.2, label='AAindex scales' if i == 0 else "")

    ax.plot(range(len(AA_ORDER)), valuesNormalized, '-', linewidth=2.2, markersize=5, color='black', markerfacecolor='white', markeredgewidth=2, label='Created Scale')

    ax.axhline(y=0, color='gray', linewidth=0.8, alpha=0.7)
    ax.set_title(scaleName.replace('_', ' ').title(), fontsize=13, pad=10, weight='bold')
    ax.set_xlabel('Amino acid', fontsize=10)
    ax.set_ylabel('Scale value', fontsize=10)
    ax.set_xticks(range(len(AA_ORDER)))
    ax.set_xticklabels(AA_ORDER, rotation=45, ha='right')
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

    fname = "scale_" + scaleName.replace(" ", "_").replace("/", "_") + ".png"
    fpath = os.path.join(OUTPUT_DIRECTORY, fname)
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved {scaleName} to {fpath}")


# ---------- MAIN ----------
def main():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    createdScales, normalizedScales = loadScales()
    scaleNames = list(createdScales.keys())

    for scaleName in scaleNames:
        valuesNormalized, originalValues = getValues(createdScales, normalizedScales, scaleName)
        generatePlot(scaleName, valuesNormalized, originalValues)

    print(f"Created {len(scaleNames)} separate line plot files in '{OUTPUT_DIRECTORY}'")


if __name__ == "__main__":
    main()