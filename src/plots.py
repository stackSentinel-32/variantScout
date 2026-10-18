import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .msa import AA


def heatmap(df, title, path):
    grid = df.pivot_table(index="mt", columns="pos", values="DMS_score").reindex(list(AA))
    plt.figure(figsize=(16, 5))
    sns.heatmap(grid, cmap="viridis", cbar_kws={"label": "DMS_score"})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
