import json
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np

class CustomBarPlot:
    def __init__(self, json_files, x_label="Category", y_label="Percentage", max_label_len=12):
        self.json_files = json_files
        self.x_label = x_label
        self.y_label = y_label
        self.max_label_len = max_label_len
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        
    def _load_datasets(self):
        """Load the 3 specific JSON files, exclude 'ordinary' from display but include in total"""
        datasets = []
        dataset_names = ['UniProt50', 'SCL', 'SCL over UniProt50'] #TODO search better names
        
        for i, file_path in enumerate(self.json_files):
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Calculate total including 'ordinary'
            total = sum(data.values())
            
            # Get values excluding 'ordinary' for display, but use full total for %
            display_values = []
            display_labels = []
            
            for category, value in data.items():
                if category != "ordinary":
                    pct = (value / total) * 100
                    display_values.append(pct)
                    display_labels.append(category)
            
            datasets.append((display_labels, display_values, dataset_names[i]))
        return datasets
    
    def _get_max_y(self, datasets):
        return max(max(values) for _, values, _ in datasets)
    
    def _truncate_label(self, label):
        if len(label) > self.max_label_len:
            return label[:self.max_label_len-2] + '..'
        return label
    
    def render(self):
        """Main render method - shows percentages (excluding ordinary from bars)"""
        self.ax.clear()
        datasets = self._load_datasets()
        
        # Grouped bar positioning - now 4 categories (excluding ordinary)
        n_categories = len(datasets[0][1])  # 4 surprise levels
        bar_width = 0.22
        x_pos = np.arange(n_categories)
        
        max_y = self._get_max_y(datasets)
        
        # Draw bars for each dataset
        colors = ['blue', 'orange', 'green']
        rects_list = []
        
        for i, (_, values, name) in enumerate(datasets):
            offset = (i - 1) * bar_width
            heights = [v / max_y * 0.85 for v in values]
            
            rects = self.ax.bar(x_pos + offset, heights, bar_width, 
                               label=name, color=colors[i],
                               edgecolor='black', linewidth=0.8)
            rects_list.append(rects)
        
        # X-axis labels (excluding ordinary)
        labels = datasets[0][0]  # extremely surprising, highly surprising, etc.
        
        self.ax.set_xticks(x_pos)
        self.ax.set_xticklabels(labels, rotation=0, fontsize=12)
        
        # Y-axis shows percentages
        y_ticks = np.linspace(0, max_y, 6)
        self.ax.set_yticks(np.linspace(0, 0.85, 6))
        self.ax.set_yticklabels([f'{y:.1f}%' for y in y_ticks], fontsize=12)
        
        # Clean styling
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_linewidth(1.2)
        self.ax.spines['bottom'].set_linewidth(1.2)
        
        self.ax.set_xlabel(self.x_label, fontsize=12, fontweight='bold')
        self.ax.set_ylabel(self.y_label, fontsize=12, fontweight='bold')
        self.ax.set_title('Surprise Metric Distribution (excluding ordinary class)', 
                         fontsize=14, fontweight='bold')
        legend_elements = [
            patches.Patch(facecolor=colors[0], label=dataset_names[0]),
            patches.Patch(facecolor=colors[1], label=dataset_names[1]), #  alpha=0.5
            patches.Patch(facecolor=colors[2], label=dataset_names[2])
        ]
        self.ax.legend(handles=legend_elements, loc='upper left', 
          bbox_to_anchor=(1.02, 0.98), frameon=False, 
          fontsize=8, handlelength=1.2, handleheight=1.2)
    
        plt.tight_layout()
        self.fig.savefig(
            "plots/surprise_metric.jpg",
            dpi=300,
            bbox_inches="tight",
            format="jpg"
        )

# Usage with YOUR exact files
if __name__ == "__main__":
    plot = CustomBarPlot(
        ["UniProt50-surpriseMetric.json", "SCL-surpriseMetric.json", "SCL-UniProt50-surpriseMetric.json"],
        x_label="Surprise Level",
        y_label="Percentage of Dataset"
    )
    plot.render()
