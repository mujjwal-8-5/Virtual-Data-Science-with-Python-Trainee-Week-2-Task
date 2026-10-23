from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "wine.csv"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "summary_tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
df = pd.read_csv(DATA)

print("Shape:", df.shape)
print("\\nData types:\\n", df.dtypes)
print("\\nMissing values:\\n", df.isna().sum())
print("\\nDuplicate rows:", df.duplicated().sum())
print("\\nDescriptive statistics:\\n", df.select_dtypes("number").describe().T)
print("\\nClass counts:\\n", df["target_name"].value_counts())

df.isna().sum().rename("missing_count").to_csv(TAB / "missing_values.csv")
df.describe(include="all").T.to_csv(TAB / "descriptive_statistics.csv")
df.groupby("target_name").mean(numeric_only=True).T.to_csv(TAB / "classwise_means.csv")
corr = df.select_dtypes("number").drop(columns=["target"], errors="ignore").corr()
corr.to_csv(TAB / "correlation_matrix.csv")

fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=df, x="target_name", order=sorted(df["target_name"].unique()),
              hue="target_name", palette="deep", legend=False, ax=ax)
ax.set(title="Wine Samples by Cultivar Class", xlabel="Wine class", ylabel="Number of samples")
for container in ax.containers: ax.bar_label(container)
fig.tight_layout(); fig.savefig(FIG / "01_class_distribution.png", dpi=180, bbox_inches="tight"); plt.close(fig)

cols = ["alcohol", "malic_acid", "color_intensity", "proline"]
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.flat, cols):
    sns.histplot(data=df, x=col, kde=True, color="steelblue", ax=ax)
    ax.set(title=f"Distribution of {col.replace('_', ' ').title()}",
           xlabel=col.replace("_", " ").title(), ylabel="Sample count")
fig.tight_layout(); fig.savefig(FIG / "02_feature_distributions.png", dpi=180, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="target_name", y="alcohol", order=sorted(df["target_name"].unique()),
            hue="target_name", palette="deep", legend=False, ax=ax)
ax.set(title="Alcohol Content by Wine Class", xlabel="Wine class", ylabel="Alcohol measurement")
fig.tight_layout(); fig.savefig(FIG / "03_alcohol_by_class.png", dpi=180, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, cbar_kws={"label": "Pearson correlation"})
ax.set_title("Correlation Matrix of Numeric Features (target excluded)", weight="bold")
fig.tight_layout(); fig.savefig(FIG / "04_correlation_heatmap.png", dpi=180, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=df, x="flavanoids", y="color_intensity", hue="target_name",
                style="target_name", s=65, ax=ax)
ax.set(title="Flavanoids vs Colour Intensity by Class",
       xlabel="Flavanoids", ylabel="Colour intensity")
ax.legend(title="Wine class"); fig.tight_layout()
fig.savefig(FIG / "05_selected_feature_scatter.png", dpi=180, bbox_inches="tight"); plt.close(fig)

cols = ["malic_acid", "color_intensity", "proline", "proanthocyanins"]
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, col in zip(axes.flat, cols):
    sns.boxplot(x=df[col], color="steelblue", ax=ax)
    ax.set(title=col.replace("_", " ").title(), xlabel=col.replace("_", " ").title())
fig.tight_layout(); fig.savefig(FIG / "06_numeric_outlier_scan.png", dpi=180, bbox_inches="tight"); plt.close(fig)

# Save IQR outlier counts without deleting observations.
counts = {}
for col in df.select_dtypes("number").columns:
    if col == "target": continue
    q1, q3 = df[col].quantile([.25, .75]); iqr = q3-q1
    counts[col] = int(((df[col] < q1-1.5*iqr) | (df[col] > q3+1.5*iqr)).sum())
pd.DataFrame({"feature": list(counts), "iqr_outlier_count": list(counts.values())}).sort_values(
    "iqr_outlier_count", ascending=False).to_csv(TAB / "iqr_outlier_counts.csv", index=False)
print(f"Done. Figures: {FIG}; summary tables: {TAB}")
