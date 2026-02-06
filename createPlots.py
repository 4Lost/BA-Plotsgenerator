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
os.makedirs(OUTPUT_DIR, exist_ok=True)



# ---------- HELPERS ----------
def load_feature_values(feature):
    fname = f"distribution_scales_{feature}.json"
    with open(fname, "r") as f:
        data = json.load(f)
    # return both per-sequence values and precomputed stats
    return data["perSequence"], data["stats"]



def print_feature_stats(feature, stats, count):
    print(f"\n=== Feature: {feature} ===")
    print(f"Number of sequences : {count}")
    print(f"Min                 : {stats['min']:.4f}")
    print(f"Max                 : {stats['max']:.4f}")
    print(f"Mean                : {stats['mean']:.4f}")
    print(f"Standard deviation  : {stats['std_dev']:.4f}")
    print(f"Median              : {stats['median']:.4f}")



def kde_1d(values, num_points=200):
    arr = np.array(values, dtype=float)
    if arr.size < 2 or np.all(arr == arr[0]):
        # Degenerate case: no spread
        x = np.linspace(arr[0] - 1, arr[0] + 1, num_points)
        y = np.zeros_like(x)
        return x, y
    kde = gaussian_kde(arr)
    x_min, x_max = arr.min(), arr.max()
    x = np.linspace(x_min, x_max, num_points)
    y = kde(x)
    return x, y



def plot_feature_kde(feature, values, stats):
    x, y = kde_1d(values)

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, color="steelblue", label="KDE")
    # mean and std dev lines
    m = stats["mean"]
    s = stats["std_dev"]
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



def load_AA_distribution():
    if os.path.exists("distribution_AA.json"):
        with open("distribution_AA.json", "r") as f:
            return json.load(f)

    # Fallback: build them from worker files
    from collections import defaultdict

    dist = defaultdict(int)
    posDist = defaultdict(lambda: defaultdict(int))
    lenDist = defaultdict(int)

    # distribution
    for fname in glob.glob("worker_*_distribution.jsonl"):
        with open(fname, "r") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    for aa, cnt in data.items():
                        dist[aa] += cnt
                except json.JSONDecodeError as e:
                    print(f"[ERROR] {fname} (distribution) line {line_number}: {line}")
                    raise e

    # positional distribution
    for fname in glob.glob("worker_*_positionalDistribution.jsonl"):
        with open(fname, "r") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    for pos, sub in data.items():
                        for aa, cnt in sub.items():
                            posDist[int(pos)][aa] += cnt
                except json.JSONDecodeError as e:
                    print(f"[ERROR] {fname} (positional) line {line_number}: {line}")
                    raise e

    # length distribution
    for fname in glob.glob("worker_*_lengthDistribution.jsonl"):
        with open(fname, "r") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    for length, cnt in data.items():
                        lenDist[int(length)] += cnt
                except json.JSONDecodeError as e:
                    print(f"[ERROR] {fname} (length) line {line_number}: {line}")
                    raise e

    def normalize(dist_dict):
        total = sum(dist_dict.values())
        if total == 0:
            return {k: 0.0 for k in dist_dict}
        return {str(k): (v / total) * 100.0 for k, v in dist_dict.items()}

    # normalize positional distribution per position
    posDist_norm = {}
    for pos, sub in posDist.items():
        total = sum(sub.values())
        if total == 0:
            posDist_norm[str(pos)] = {aa: 0.0 for aa in sub}
        else:
            posDist_norm[str(pos)] = {aa: (cnt / total) * 100.0 for aa, cnt in sub.items()}

    # length stats
    lengths = np.array(list(lenDist.keys()), dtype=float)
    counts = np.array(list(lenDist.values()), dtype=float)
    mean_len = np.average(lengths, weights=counts)
    variance_len = np.average((lengths - mean_len) ** 2, weights=counts)
    length_stats = {
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "mean": float(mean_len),
        "std_dev": float(np.sqrt(variance_len)),
    }

    result = {
        "lengthKde": {str(k): int(v) for k, v in lenDist.items()},
        "lengthStats": length_stats,
        "distribution": normalize(dist),
        "positionalDistribution": posDist_norm,
    }

    # Save for later reuse
    with open("distribution_AA.json", "w") as f:
        json.dump(result, f, indent=2)

    return result



def print_AA_stats(aa_json):
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



def plot_AA_distribution(aa_json):
    # Global AA distribution barplot (KEEP AS BAR)
    dist = aa_json["distribution"]
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
    lengthKde = aa_json["lengthKde"]
    lengths = np.array([int(k) for k in lengthKde.keys()], dtype=float)
    counts = np.array(list(lengthKde.values()), dtype=float)
    repeated_lengths = np.repeat(lengths, counts.astype(int))
    
    stats = aa_json["lengthStats"]
    
    x, y = kde_1d(repeated_lengths)
    
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
    print("=== Per-feature stats and plots ===")
    for feature in FEATURES:
        try:
            per_seq, stats = load_feature_values(feature)
            if not per_seq:
                print(f"\n=== Feature: {feature} ===")
                print("No data found, skipping.")
                continue

            values = list(per_seq.values())
            print_feature_stats(feature, stats, len(values))
            plot_feature_kde(feature, values, stats)
        except FileNotFoundError:
            print(f"\n=== Feature: {feature} ===")
            print("distribution_scales_{feature}.json not found, skipping.")

    print("\n=== Amino acid and length stats ===")
    aa_json = load_AA_distribution()
    print_AA_stats(aa_json)
    plot_AA_distribution(aa_json)

    print(f"\nAll plots saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
