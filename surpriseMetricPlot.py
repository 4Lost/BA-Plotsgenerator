import json
import os
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np

# ---------- HELPERS ----------
def loadDatasets():
    jsonFiles = ["UniProt50-surpriseMetric.json", "SCL-surpriseMetric.json", "SCL-UniProt50-surpriseMetric.json"]
    datasetNames = ['UniProt50', 'SCL', 'SCL over UniProt50'] #TODO search better names
    datasets = []

    for i, file_path in enumerate(jsonFiles):
        with open(file_path, 'r') as f:
            data = json.load(f)

        total = sum(data.values())

        displayValues = []
        displayLabels = []

        for category, value in data.items():
            if category != "ordinary":
                pct = (value / total) * 100
                displayValues.append(pct)
                displayLabels.append(category)
        
        datasets.append((displayLabels, displayValues, datasetNames[i]))

    return datasets


def getMaxY(datasets):
    return max(max(values) for _, values, _ in datasets)


def generatePlot():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.clear()
    datasets = loadDatasets()

    nCategories = len(datasets[0][1])
    barWidth = 0.22
    xPos = np.arange(nCategories)

    maxY = getMaxY(datasets)

    colors = ['blue', 'orange', 'green']
    datasetNames = []

    for i, (_, values, name) in enumerate(datasets):
        offset = (i - 1) * barWidth
        heights = [v / maxY * 0.85 for v in values]

        ax.bar(xPos + offset, heights, barWidth, 
                        label=name, color=colors[i],
                        edgecolor='black', linewidth=0.8)
        datasetNames.append(name)

    labels = datasets[0][0]
    
    ax.set_xticks(xPos)
    ax.set_xticklabels(labels, rotation=0, fontsize=12)

    yTicks = np.linspace(0, maxY, 6)
    ax.set_yticks(np.linspace(0, 0.85, 6))
    ax.set_yticklabels([f'{y:.1f}%' for y in yTicks], fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    ax.set_xlabel("Surprise Level", fontsize=12, fontweight='bold')
    ax.set_ylabel("Percentage of Dataset", fontsize=12, fontweight='bold')
    ax.set_title('Surprise Metric Distribution (excluding ordinary class)', 
                        fontsize=14, fontweight='bold')
    legend_elements = [
        patches.Patch(facecolor=colors[0], label=datasetNames[0]),
        patches.Patch(facecolor=colors[1], label=datasetNames[1]), #  alpha=0.5
        patches.Patch(facecolor=colors[2], label=datasetNames[2])
    ]
    ax.legend(handles=legend_elements, loc='upper left', 
        bbox_to_anchor=(1.02, 0.98), frameon=False, 
        fontsize=8, handlelength=1.2, handleheight=1.2)

    plt.tight_layout()
        
    fig.savefig(
        "plots/surprise_metric.png",
        dpi=300,
        bbox_inches="tight",
        format="jpg"
    )


# ---------- MAIN ----------
def main():
    os.makedirs("plots", exist_ok=True)

    generatePlot()

    print(f"Plot created")


if __name__ == "__main__":
    main()










    
