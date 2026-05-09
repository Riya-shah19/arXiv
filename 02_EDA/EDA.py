import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
import os

os.makedirs("plots", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")

df = pd.read_csv("arxiv_monthly_counts.csv")

print("=" * 50)
print("  DATASET OVERVIEW")
print("=" * 50)
print(f"  Rows          : {len(df):,}")
print(f"  Columns       : {df.columns.tolist()}")
print(f"  Fields        : {sorted(df['field'].unique())}")
print(f"  Subcategories : {df['sub_field'].nunique()}")
print(f"  Year range    : {df['year'].min()} – {df['year'].max()}")
print(f"  Total papers  : {df['paper_count'].sum():,}")
print(f"\n  First 5 rows:")
print(df.head())

# Basic statistics
print("\n" + "=" * 50)
print("  PAPERS BY FIELD")
print("=" * 50)
field_totals = df.groupby("field")["paper_count"].sum().sort_values(ascending=False)
for field, total in field_totals.items():
    print(f"  {field:<10} → {total:>12,} papers")

print("\n  Top 10 subcategories:")
top10 = (df.groupby("sub_field")["paper_count"]
           .sum()
           .nlargest(10)
           .reset_index())
print(top10.to_string(index=False))

# Plot 1 - Total papers per year by field 

yearly = df.groupby(["field","year"])["paper_count"].sum().reset_index()
colors = {"cs":"#2196F3", "stat":"#FF9800", "physics":"#4CAF50"}

plt.figure(figsize=(14, 6))
for field in ["cs","stat","physics"]:
    data = yearly[yearly["field"] == field]
    plt.plot(data["year"], data["paper_count"],
             label=field.upper(),
             color=colors[field],
             linewidth=2.5,
             marker="o",
             markersize=4)
    
plt.title("Total arXiv Papers per Year by Field", fontsize=16, fontweight="bold")
plt.xlabel("Year", fontsize=13)
plt.ylabel("Total Papers", fontsize=13)
plt.legend(title="Field", fontsize=11)
plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
plt.tight_layout()
plt.savefig("plots/plot1_yearly_trend.png", dpi=150)
plt.show()

# Plot 2 - Average papers per month (seasonal pattern)

month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

monthly = df.groupby(["field","month"])["paper_count"].mean().reset_index()

plt.figure(figsize=(14, 6))
for field in ["cs","stat","physics"]:
    data = monthly[monthly["field"] == field].sort_values("month")
    plt.plot(data["month"], data["paper_count"],
             label=field.upper(),
             color=colors[field],
             linewidth=2.5,
             marker="o",
             markersize=6)
    
plt.title("Average Monthly Submission Pattern by Field", fontsize=16, fontweight="bold")
plt.xlabel("Month", fontsize=13)
plt.ylabel("Average Papers", fontsize=13)
plt.xticks(range(1, 13), month_labels)
plt.legend(title="Field", fontsize=11)
plt.tight_layout()
plt.savefig("plots/plot2_seasonal_pattern.png", dpi=150)
plt.show()

# Plot 3 - Hearmap: CS subcategories * Year

cs_pivot = (df[(df["field"]=="cs") & (df["year"]>=1991)]
            .groupby(["sub_field","year"])["paper_count"]
            .sum()
            .unstack("year")
            .fillna(0))

plt.figure(figsize=(18, 12))
sns.heatmap(cs_pivot, cmap="YlOrRd",
            linewidths=0.3,
            linecolor="white",
            cbar_kws={"label":"Paper Count"})
plt.title("CS Subcategories × Year Heatmap (1991–2026)",
          fontsize=16, fontweight="bold")
plt.xlabel("Year", fontsize=13)
plt.ylabel("Subcategory", fontsize=13)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("plots/plot3_cs_heatmap.png", dpi=150)
plt.show()

# Plot 4 - Heatmap: Statistics subcategories * Year 

stat_pivot = (df[(df["field"]=="stat") & (df["year"]>=1991)]
              .groupby(["sub_field","year"])["paper_count"]
              .sum()
              .unstack("year")
              .fillna(0))

plt.figure(figsize=(18, 6))
sns.heatmap(stat_pivot, cmap="YlOrRd",
            linewidths=0.3,
            linecolor="white",
            annot=True, fmt=".0f",
            cbar_kws={"label":"Paper Count"})
plt.title("Statistics Subcategories x Year Heatmap (1991-2026)",
          fontsize=16, fontweight="bold")
plt.xlabel("Year", fontsize=13)
plt.ylabel("Subcategory", fontsize=13)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("plots/plot4_stat_heatmap.png", dpi=150)
plt.show()

# Plot 5 - Heatmap: Physics subcategories * Year

phys_pivot = (df[(df["field"]=="physics") & (df["year"]>=1991)]
              .groupby(["sub_field","year"])["paper_count"]
              .sum()
              .unstack("year")
              .fillna(0))

plt.figure(figsize=(18, 10))
sns.heatmap(phys_pivot, cmap="YlOrRd",
            linewidths=0.3,
            linecolor="white",
            cbar_kws={"label":"Paper Count"})
plt.title("Physics Subcategories × Year Heatmap (1991–2026)",
          fontsize=16, fontweight="bold")
plt.xlabel("Year", fontsize=13)
plt.ylabel("Subcategory", fontsize=13)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("plots/plot5_physics_heatmap.png", dpi=150)
plt.show()

# Plot 6 - Top 10 fastest growing subcategories

early  = df[(df["year"] >= 2015) & (df["year"] <= 2019)]
recent = df[(df["year"] >= 2020) & (df["year"] <= 2024)]

early_avg  = early.groupby(["field","sub_field"])["paper_count"].mean()
recent_avg = recent.groupby(["field","sub_field"])["paper_count"].mean()

growth = pd.concat([early_avg, recent_avg], axis=1, keys=["early","recent"])
growth = growth.dropna()
growth["growth_pct"] = ((growth["recent"] - growth["early"]) / growth["early"] * 100).round(1)
growth = growth.reset_index()
growth = growth.sort_values("growth_pct", ascending=False).head(10)
growth = growth.sort_values("growth_pct", ascending=True)

bar_colors = [colors.get(f, "#9C27B0") for f in growth["field"]]

fig, ax = plt.subplots(figsize=(14, 8))
bars = ax.barh(growth["sub_field"], growth["growth_pct"],
               color=bar_colors, edgecolor="white", height=0.6)

for bar, (_, row) in zip(bars, growth.iterrows()):
    ax.text(bar.get_width() + 2,
            bar.get_y() + bar.get_height() / 2,
            f"{row['growth_pct']:.0f}%  ({int(row['early']):,} → {int(row['recent']):,} avg/month)",
            va="center", fontsize=9.5)

ax.axvline(x=100, color="gray", linestyle="--", linewidth=1, alpha=0.6)
ax.text(101, -0.6, "100%\n(doubled)", color="gray", fontsize=8)

legend_handles = [
    mpatches.Patch(color=c, label=f.upper())
    for f, c in colors.items()
]
ax.legend(handles=legend_handles, title="Field", fontsize=10, loc="lower right")
ax.set_title("Top 10 Fastest Growing Subcategories\n(Average papers/month: 2015–2019 vs 2020–2024)",
             fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("Growth (%)", fontsize=13)
ax.set_ylabel("Subcategory", fontsize=13)
ax.set_xlim(0, growth["growth_pct"].max() * 1.35)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}%"))
sns.despine()
plt.tight_layout()
plt.savefig("plots/plot6_growth_rate.png", dpi=150)
plt.show()

# Plot 7 - CS vs Physics comparison 

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Computer Science vs Physics — arXiv Paper Comparison",
             fontsize=16, fontweight="bold", y=1.02)
 
#  Left panel
 
yearly = df.groupby(["field","year"])["paper_count"].sum().reset_index()
 
for field, color, label in [("cs","#2196F3","Computer Science"),
                              ("physics","#4CAF50","Physics")]:
    data = yearly[yearly["field"] == field]
    ax1.plot(data["year"], data["paper_count"],
             color=color, linewidth=2.5,
             marker="o", markersize=4,
             label=label)
 
cs_data  = yearly[yearly["field"]=="cs"].set_index("year")["paper_count"]
phy_data = yearly[yearly["field"]=="physics"].set_index("year")["paper_count"]
common_years = cs_data.index.intersection(phy_data.index)
 
ax1.fill_between(common_years,
                 cs_data[common_years],
                 phy_data[common_years],
                 alpha=0.08, color="purple",
                 label="Gap between fields")
 
ax1.set_title("Yearly Paper Count Trend", fontsize=13, fontweight="bold")
ax1.set_xlabel("Year", fontsize=11)
ax1.set_ylabel("Total Papers", fontsize=11)
ax1.legend(fontsize=10)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
ax1.set_xlim(1991, 2026)
sns.despine(ax=ax1)
 
#  Right panel:
 
top5_cs = (df[df["field"]=="cs"]
           .groupby("sub_field")["paper_count"]
           .sum()
           .nlargest(5)
           .reset_index())
 
top5_ph = (df[df["field"]=="physics"]
           .groupby("sub_field")["paper_count"]
           .sum()
           .nlargest(5)
           .reset_index())
 

top5_cs["field"]     = "cs"
top5_ph["field"]     = "physics"
combined             = pd.concat([top5_cs, top5_ph])
combined["rank"]     = combined.groupby("field")["paper_count"].rank(
                           ascending=False).astype(int)
combined["label"]    = combined["sub_field"] + "\n(#" + combined["rank"].astype(str) + ")"
 
x      = range(5)
width  = 0.35
 
cs_vals  = top5_cs.sort_values("paper_count", ascending=False)["paper_count"].values
phy_vals = top5_ph.sort_values("paper_count", ascending=False)["paper_count"].values
cs_subs  = top5_cs.sort_values("paper_count", ascending=False)["sub_field"].values
phy_subs = top5_ph.sort_values("paper_count", ascending=False)["sub_field"].values
 
bars1 = ax2.bar([i - width/2 for i in x], cs_vals,
                width, label="Computer Science",
                color="#2196F3", edgecolor="white")
bars2 = ax2.bar([i + width/2 for i in x], phy_vals,
                width, label="Physics",
                color="#4CAF50", edgecolor="white")
 

for bar, val in zip(bars1, cs_vals):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 500,
             f"{int(val):,}", ha="center",
             fontsize=8, color="#2196F3", fontweight="bold")
 
for bar, val in zip(bars2, phy_vals):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 500,
             f"{int(val):,}", ha="center",
             fontsize=8, color="#4CAF50", fontweight="bold")
 

tick_labels = [f"CS: {c}\nPHY: {p}"
               for c, p in zip(cs_subs, phy_subs)]
ax2.set_xticks(list(x))
ax2.set_xticklabels(tick_labels, fontsize=8.5)
ax2.set_title("Top 5 Subcategories — CS vs Physics", fontsize=13, fontweight="bold")
ax2.set_ylabel("Total Papers", fontsize=11)
ax2.legend(fontsize=10)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
sns.despine(ax=ax2)
 
plt.tight_layout()
plt.savefig("plots/plot7_cs_vs_physics.png", dpi=150, bbox_inches="tight")
plt.show()

# PLOT 8 — All 5 Fields: Yearly Trend

yearly = df.groupby(["field","year"])["paper_count"].sum().reset_index()

colors = {
    "cs"     : "#2196F3",
    "stat"   : "#FF9800",
    "physics": "#4CAF50",
    "math"   : "#9C27B0",
    "eess"   : "#F44336",
}

fig, ax = plt.subplots(figsize=(16, 7))

for field in ["cs","math","physics","stat","eess"]:
    data = yearly[yearly["field"] == field]
    if data.empty:
        continue
    ax.plot(data["year"], data["paper_count"],
            label=field.upper(),
            color=colors[field],
            linewidth=2.5,
            marker="o",
            markersize=4)

ax.set_title("Total arXiv Papers per Year — All 5 Fields (1991–2026)",
             fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("Year", fontsize=13)
ax.set_ylabel("Total Papers", fontsize=13)
ax.legend(title="Field", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
ax.set_xlim(1991, 2026)
sns.despine()
plt.tight_layout()
plt.savefig("plots/plot8_all_fields_yearly.png", dpi=150, bbox_inches="tight")
plt.show()

# PLOT 9 — Mathematics Subcategories Heatmap (2010–2026)

math_pivot = (df[(df["field"] == "math") & (df["year"] >= 2010)]
              .groupby(["sub_field","year"])["paper_count"]
              .sum()
              .unstack("year")
              .fillna(0))

fig, ax = plt.subplots(figsize=(20, 14))
sns.heatmap(math_pivot,
            cmap="Purples",
            linewidths=0.3,
            linecolor="white",
            annot=False,
            cbar_kws={"label": "Paper Count"},
            ax=ax)

ax.set_title("Mathematics Subcategories - Year Heatmap (2010-2026)",
             fontsize=16, fontweight="bold", pad=15)
ax.set_xlabel("Year", fontsize=13)
ax.set_ylabel("Subcategory", fontsize=13)
ax.tick_params(axis="x", rotation=45)
ax.tick_params(axis="y", rotation=0)
plt.tight_layout()
plt.savefig("plots/plot9_math_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# PLOT 10 — EESS Subcategories Yearly Trend

eess_yearly = (df[df["field"] == "eess"]
               .groupby(["sub_field","year"])["paper_count"]
               .sum()
               .reset_index())

eess_colors = {
    "eess.AS": "#E91E63",
    "eess.IV": "#FF5722",
    "eess.SP": "#FF9800",
    "eess.SY": "#FFC107",
}

eess_labels = {
    "eess.AS": "Audio & Speech Processing",
    "eess.IV": "Image & Video Processing",
    "eess.SP": "Signal Processing",
    "eess.SY": "Systems & Control",
}

fig, ax = plt.subplots(figsize=(14, 6))

for subcat in ["eess.AS","eess.IV","eess.SP","eess.SY"]:
    data = eess_yearly[eess_yearly["sub_field"] == subcat]
    if data.empty:
        continue
    ax.plot(data["year"], data["paper_count"],
            label=eess_labels[subcat],
            color=eess_colors[subcat],
            linewidth=2.5,
            marker="o",
            markersize=5)

ax.set_title("EESS Subcategories — Yearly Paper Count",
             fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("Year", fontsize=13)
ax.set_ylabel("Total Papers", fontsize=13)
ax.legend(title="Subcategory", fontsize=10, loc="upper left")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
sns.despine()
plt.tight_layout()
plt.savefig("plots/plot10_eess_yearly.png", dpi=150, bbox_inches="tight")
plt.show()

# PLOT 11 — Top 10 Mathematics Subcategories by Total Papers

math_names = {
    "math.AC":"Commutative Algebra",   "math.AG":"Algebraic Geometry",
    "math.AP":"Analysis of PDEs",      "math.AT":"Algebraic Topology",
    "math.CA":"Classical Analysis",    "math.CO":"Combinatorics",
    "math.CT":"Category Theory",       "math.CV":"Complex Variables",
    "math.DG":"Differential Geometry", "math.DS":"Dynamical Systems",
    "math.FA":"Functional Analysis",   "math.GM":"General Mathematics",
    "math.GN":"General Topology",      "math.GR":"Group Theory",
    "math.GT":"Geometric Topology",    "math.HO":"History & Overview",
    "math.IT":"Information Theory",    "math.KT":"K-Theory",
    "math.LO":"Logic",                 "math.MG":"Metric Geometry",
    "math.MP":"Mathematical Physics",  "math.NA":"Numerical Analysis",
    "math.NT":"Number Theory",         "math.OA":"Operator Algebras",
    "math.OC":"Optimisation & Control","math.PR":"Probability",
    "math.QA":"Quantum Algebra",       "math.RA":"Rings & Algebras",
    "math.RT":"Representation Theory", "math.SG":"Symplectic Geometry",
    "math.SP":"Spectral Theory",       "math.ST":"Statistics Theory",
}

top10_math = (df[df["field"] == "math"]
              .groupby("sub_field")["paper_count"]
              .sum()
              .nlargest(10)
              .reset_index()
              .sort_values("paper_count", ascending=True))

top10_math["label"] = (top10_math["sub_field"] + " — " +
                       top10_math["sub_field"].map(math_names))

fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.barh(top10_math["label"], top10_math["paper_count"],
               color="#9C27B0", edgecolor="white", height=0.6)

for bar, val in zip(bars, top10_math["paper_count"]):
    ax.text(bar.get_width() + 100,
            bar.get_y() + bar.get_height()/2,
            f"{int(val):,}", va="center", fontsize=9)

ax.set_title("Top 10 Mathematics Subcategories by Total Papers",
             fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("Total Papers", fontsize=13)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
sns.despine()
plt.tight_layout()
plt.savefig("plots/plot11_math_top10.png", dpi=150, bbox_inches="tight")
plt.show()

# PLOT 12 — CS vs Math vs EESS Side by Side Comparison

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("CS vs Mathematics vs EESS — Yearly Growth Comparison",
             fontsize=16, fontweight="bold")

fields_to_compare = [
    ("cs",   "Computer Science",           "#2196F3"),
    ("math", "Mathematics",                "#9C27B0"),
    ("eess", "Electrical Engineering (EESS)","#F44336"),
]

for ax, (field, title, color) in zip(axes, fields_to_compare):
    data = yearly[yearly["field"] == field]
    ax.fill_between(data["year"], data["paper_count"],
                    alpha=0.2, color=color)
    ax.plot(data["year"], data["paper_count"],
            color=color, linewidth=2.5,
            marker="o", markersize=4)
    ax.set_title(title, fontsize=13, fontweight="bold", color=color)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Total Papers", fontsize=11)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    ax.set_xlim(1991, 2026)
    sns.despine(ax=ax)

plt.tight_layout()
plt.savefig("plots/plot12_cs_math_eess_comparison.png",
            dpi=150, bbox_inches="tight")
plt.show()

# PLOT 13 — EESS Heatmap: Subcategories × Year

eess_pivot = (df[df["field"] == "eess"]
              .groupby(["sub_field","year"])["paper_count"]
              .sum()
              .unstack("year")
              .fillna(0))

fig, ax = plt.subplots(figsize=(20, 5))
sns.heatmap(eess_pivot,
            cmap="Reds",
            linewidths=0.4,
            linecolor="white",
            annot=True,
            fmt=".0f",
            cbar_kws={"label": "Paper Count"},
            ax=ax)

ax.set_title("EESS Subcategories × Year Heatmap",
             fontsize=16, fontweight="bold", pad=15)
ax.set_xlabel("Year", fontsize=13)
ax.set_ylabel("Subcategory", fontsize=13)
ax.tick_params(axis="x", rotation=45)
ax.tick_params(axis="y", rotation=0)
plt.tight_layout()
plt.savefig("plots/plot13_eess_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Plot 13 saved")

# PLOT 14 — Pie Chart: Share of Total Papers by Field

field_totals = (df.groupby("field")["paper_count"]
                  .sum()
                  .sort_values(ascending=False))

field_labels = {
    "cs"     : "Computer Science",
    "math"   : "Mathematics",
    "physics": "Physics",
    "stat"   : "Statistics",
    "eess"   : "Elec. Engineering",
}

labels = [field_labels.get(f, f.upper()) for f in field_totals.index]
clrs   = [colors.get(f, "#607D8B") for f in field_totals.index]

fig, ax = plt.subplots(figsize=(10, 8))
wedges, texts, autotexts = ax.pie(
    field_totals.values,
    labels=labels,
    colors=clrs,
    autopct="%1.1f%%",
    startangle=140,
    pctdistance=0.82,
    wedgeprops={"edgecolor":"white","linewidth":2}
)

for text in texts:
    text.set_fontsize(11)
for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_fontweight("bold")

ax.set_title("Share of Total arXiv Papers by Field",
             fontsize=15, fontweight="bold", pad=20)

total = field_totals.sum()
ax.text(0, 0, f"Total\n{total:,}",
        ha="center", va="center",
        fontsize=12, fontweight="bold", color="gray")

plt.tight_layout()
plt.savefig("plots/plot14_field_share_pie.png", dpi=150, bbox_inches="tight")
plt.show()

# PLOT 15 — CS Heatmap WITHOUT cs.AI and cs.LG (better visibility)

cs_filtered = df[
    (df["field"] == "cs") &
    (df["year"] >= 2010) &
    (~df["sub_field"].isin(["cs.AI", "cs.LG"]))   # remove dominant fields
]

cs_filtered_pivot = (cs_filtered
                     .groupby(["sub_field","year"])["paper_count"]
                     .sum()
                     .unstack("year")
                     .fillna(0))

fig, ax = plt.subplots(figsize=(20, 14))
sns.heatmap(cs_filtered_pivot,
            cmap="YlOrRd",
            linewidths=0.3,
            linecolor="white",
            annot=False,
            cbar_kws={"label":"Paper Count"},
            ax=ax)

ax.set_title(
    "CS Subcategories - Year Heatmap (1991-2026)\n"
    "[ cs.AI and cs.LG removed for better visibility ]",
    fontsize=15, fontweight="bold", pad=15
)
ax.set_xlabel("Year", fontsize=13)
ax.set_ylabel("Subcategory", fontsize=13)
ax.tick_params(axis="x", rotation=45)
ax.tick_params(axis="y", rotation=0)
plt.tight_layout()
plt.savefig("plots/plot16_cs_heatmap_filtered.png", dpi=150, bbox_inches="tight")
plt.show()


# PLOT 17 — Statistics Heatmap WITHOUT stat.ML (better visibility)

stat_filtered = df[
    (df["field"] == "stat") &
    (df["year"] >= 2010) &
    (~df["sub_field"].isin(["stat.ML"]))   # remove dominant field
]

stat_filtered_pivot = (stat_filtered
                       .groupby(["sub_field","year"])["paper_count"]
                       .sum()
                       .unstack("year")
                       .fillna(0))

fig, ax = plt.subplots(figsize=(18, 6))
sns.heatmap(stat_filtered_pivot,
            cmap="YlOrRd",
            linewidths=0.3,
            linecolor="white",
            annot=True,
            fmt=".0f",
            cbar_kws={"label":"Paper Count"},
            ax=ax)

ax.set_title(
    "Statistics Subcategories - Year Heatmap (1991-2026)\n"
    "[ stat.ML removed for better visibility ]",
    fontsize=15, fontweight="bold", pad=15
)
ax.set_xlabel("Year", fontsize=13)
ax.set_ylabel("Subcategory", fontsize=13)
ax.tick_params(axis="x", rotation=45)
ax.tick_params(axis="y", rotation=0)
plt.tight_layout()
plt.savefig("plots/plot17_stat_heatmap_filtered.png", dpi=150, bbox_inches="tight")
plt.show()
