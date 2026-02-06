import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# ---------- CONFIG ----------
N_WORKERS = 12
FEATURES = [
    "hydrophobicity",
    "freeEnergie",
    "stability",
    "volume",
    "alpha-helix",
    "beta-sheet",
    "coil",
    "mutability",
]
OUTPUT_DIR = "plots"


# ---------- HELPERS ----------
def loadValues(feature):
    fname = f"distribution_scales_{feature}.json"
    with open(fname, "r") as f:
        data = json.load(f)

    return data["perSequence"], data["stats"]


def printStats(feature, stats, count):
    print(f"\n=== Feature: {feature} ===")
    print(f"Number of sequences : {count}")
    print(f"Min                 : {stats['min']:.4f}")
    print(f"Max                 : {stats['max']:.4f}")
    print(f"Mean                : {stats['mean']:.4f}")
    print(f"Standard deviation  : {stats['std_dev']:.4f}")
    print(f"Median              : {stats['median']:.4f}")


def generateKDE(values, numPoints=200):
    arr = np.array(values, dtype=float)

    if arr.size < 2 or np.all(arr == arr[0]):
        x = np.linspace(arr[0] - 1, arr[0] + 1, numPoints)
        y = np.zeros_like(x)
        return x, y
    
    kde = gaussian_kde(arr)
    xMin, xMax = arr.min(), arr.max()
    x = np.linspace(xMin, xMax, numPoints)
    y = kde(x)

    return x, y


def plotKDE(feature, values, stats):
    x, y = generateKDE(values)
    m = stats["mean"]
    s = stats["std_dev"]

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, color="steelblue", label="KDE")
    plt.axvline(m, color="red", linestyle="--", label="mean")
    plt.axvline(m - s, color="gray", linestyle=":", label="mean ± std")
    plt.axvline(m + s, color="gray", linestyle=":")

    plt.title(f"{feature} KDE")
    plt.xlabel(feature)
    plt.ylabel("Density (KDE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"kde_{feature}.png"), dpi=200)
    plt.close()


def normalize(distDict):
    total = sum(distDict.values())
    if total == 0:
        return {k: 0.0 for k in distDict}
    return {str(k): (v / total) * 100.0 for k, v in distDict.items()}


def loadAA():
    if os.path.exists("distribution_AA.json"):
        with open("distribution_AA.json", "r") as f:
            return json.load(f)

    from collections import defaultdict

    dist = defaultdict(int)
    posDist = defaultdict(lambda: defaultdict(int))
    lenDist = defaultdict(int)

    for fname in glob.glob("worker_*_distribution.jsonl"):
        with open(fname, "r") as f:
            for lineNumber, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    for aa, cnt in data.items():
                        dist[aa] += cnt
                except json.JSONDecodeError as e:
                    print(f"[ERROR] {fname} (distribution) line {lineNumber}: {line}")
                    raise e

    for fname in glob.glob("worker_*_positionalDistribution.jsonl"):
        with open(fname, "r") as f:
            for lineNumber, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    for pos, sub in data.items():
                        for aa, cnt in sub.items():
                            posDist[int(pos)][aa] += cnt
                except json.JSONDecodeError as e:
                    print(f"[ERROR] {fname} (positional) line {lineNumber}: {line}")
                    raise e

    for fname in glob.glob("worker_*_lengthDistribution.jsonl"):
        with open(fname, "r") as f:
            for lineNumber, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    for length, cnt in data.items():
                        lenDist[int(length)] += cnt
                except json.JSONDecodeError as e:
                    print(f"[ERROR] {fname} (length) line {lineNumber}: {line}")
                    raise e

    posDistNorm = {}
    for pos, sub in posDist.items():
        total = sum(sub.values())
        if total == 0:
            posDistNorm[str(pos)] = {aa: 0.0 for aa in sub}
        else:
            posDistNorm[str(pos)] = {aa: (cnt / total) * 100.0 for aa, cnt in sub.items()}

    # length stats
    lengths = np.array(list(lenDist.keys()), dtype=float)
    counts = np.array(list(lenDist.values()), dtype=float)
    meanLen = np.average(lengths, weights=counts)
    varianceLen = np.average((lengths - meanLen) ** 2, weights=counts)
    lengthStats = {
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "mean": float(meanLen),
        "std_dev": float(np.sqrt(varianceLen)),
    }

    result = {
        "lengthKde": {str(k): int(v) for k, v in lenDist.items()},
        "lengthStats": lengthStats,
        "distribution": normalize(dist),
        "positionalDistribution": posDistNorm,
    }

    # Save for later reuse
    with open("distribution_AA.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def printAAStats(aa_json):
    print("\n=== Amino acid global distribution (percent) ===")
    dist = aa_json["distribution"]
    for aa in sorted(dist.keys()):
        print(f"{aa}: {dist[aa]:6.2f} %")

    ls = aa_json["lengthStats"]
    print("\n=== Sequence length stats ===")
    print(f"Min length         : {ls['min']}")
    print(f"Max length         : {ls['max']}")
    print(f"Mean length        : {ls['mean']:.2f}")
    print(f"Standard deviation : {ls['std_dev']:.2f}")



def plotAADistribution(aaJson):
    # Global AA distribution barplot (KEEP AS BAR)
    dist = aaJson["distribution"]
    aas = sorted(dist.keys())
    vals = [dist[aa] for aa in aas]

    plt.figure(figsize=(8, 4))
    plt.bar(aas, vals, color="darkorange", edgecolor="black")
    plt.title("Amino acid composition (percent)")
    plt.xlabel("Amino acid")
    plt.ylabel("Frequency [%]")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "kde_AA_distribution.png"), dpi=200)
    plt.close()

    # Sequence length KDE (NEW - replaces bar chart)
    lengthKde = aaJson["lengthKde"]
    lengths = np.array([int(k) for k in lengthKde.keys()], dtype=float)
    counts = np.array(list(lengthKde.values()), dtype=float)
    repeated_lengths = np.repeat(lengths, counts.astype(int))
    
    stats = aaJson["lengthStats"]
    
    x, y = generateKDE(repeated_lengths)
    
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, color="seagreen", label="KDE")
    m = stats["mean"]
    s = stats["std_dev"]
    plt.axvline(m, color="red", linestyle="--", label="mean")
    plt.axvline(m - s, color="gray", linestyle=":", label="mean ± std")
    plt.axvline(m + s, color="gray", linestyle=":")
    
    plt.title("Sequence length KDE")
    plt.xlabel("Sequence length")
    plt.ylabel("Density (KDE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "kde_length.png"), dpi=200)
    plt.close()



# ---------- MAIN ----------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== Per-feature stats and plots ===")
    for feature in FEATURES:
        try:
            per_seq, stats = loadValues(feature)
            if not per_seq:
                print(f"\n=== Feature: {feature} ===")
                print("No data found, skipping.")
                continue

            values = list(per_seq.values())
            printStats(feature, stats, len(values))
            plotKDE(feature, values, stats)
        except FileNotFoundError:
            print(f"\n=== Feature: {feature} ===")
            print("distribution_scales_{feature}.json not found, skipping.")

    print("\n=== Amino acid and length stats ===")
    aa_json = loadAA()
    printAAStats(aa_json)
    plotAADistribution(aa_json)

    print(f"\nAll plots saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
