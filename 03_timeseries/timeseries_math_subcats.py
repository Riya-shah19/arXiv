"""
timeseries_math_subcats.py
==========================
Time Series Analysis + Changepoint Detection
Field: Mathematics (math.*)
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import ruptures as rpt
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# Set working directory
os.chdir(r"C:\Users\riyas\OneDrive\ARXIV project")

# Create output folder
os.makedirs("plots/timeseries/math", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# KNOWN REAL WORLD EVENTS
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_EVENTS = {
    1991: "arXiv launched",
    2000: "arXiv math section created",
    2008: "Financial crisis - mathematical modelling in demand",
    2010: "Compressed sensing and sparse methods boom",
    2012: "Deep learning - maths of optimisation needed",
    2014: "ML needs probability and statistical theory",
    2016: "Deep learning theory research begins",
    2018: "Neural tangent kernel - maths of deep learning",
    2020: "COVID-19 - mathematical modelling surge",
    2022: "ChatGPT - theory of LLMs needs mathematics",
    2023: "Geometric deep learning and topology in ML",
}

# Full names for key Mathematics subcategories
MATH_NAMES = {
    "math.AC": "Commutative Algebra",
    "math.AG": "Algebraic Geometry",
    "math.AP": "Analysis of PDEs",
    "math.AT": "Algebraic Topology",
    "math.CA": "Classical Analysis",
    "math.CO": "Combinatorics",
    "math.CT": "Category Theory",
    "math.CV": "Complex Variables",
    "math.DG": "Differential Geometry",
    "math.DS": "Dynamical Systems",
    "math.FA": "Functional Analysis",
    "math.GM": "General Mathematics",
    "math.GN": "General Topology",
    "math.GR": "Group Theory",
    "math.GT": "Geometric Topology",
    "math.HO": "History and Overview",
    "math.IT": "Information Theory",
    "math.KT": "K-Theory and Homology",
    "math.LO": "Logic",
    "math.MG": "Metric Geometry",
    "math.MP": "Mathematical Physics",
    "math.NA": "Numerical Analysis",
    "math.NT": "Number Theory",
    "math.OA": "Operator Algebras",
    "math.OC": "Optimisation and Control",
    "math.PR": "Probability",
    "math.QA": "Quantum Algebra",
    "math.RA": "Rings and Algebras",
    "math.RT": "Representation Theory",
    "math.SG": "Symplectic Geometry",
    "math.SP": "Spectral Theory",
    "math.ST": "Statistics Theory",
}

# Most important subcategories for dissertation
# These are the ones most connected to AI and data science growth
IMPORTANT_SUBCATS = {
    "math.OC": "Optimisation and Control",
    "math.PR": "Probability",
    "math.ST": "Statistics Theory",
    "math.NA": "Numerical Analysis",
    "math.CO": "Combinatorics",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# Identical approach to Statistics and CS for consistency
# ─────────────────────────────────────────────────────────────────────────────

def prepare_monthly_series(df, field=None, subfield=None):
    """
    Prepares a clean monthly time series.
    Filters data, creates date column, fills missing months.
    """
    if subfield:
        filtered = df[(df["sub_field"] == subfield) &
                      (df["year"] <= 2025)].copy()
    else:
        filtered = df[(df["field"] == field) &
                      (df["year"] <= 2025)].copy()

    monthly = (filtered
               .groupby(["year","month"])["paper_count"]
               .sum()
               .reset_index())

    monthly["date"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" +
        monthly["month"].astype(str).str.zfill(2) + "-01"
    )

    monthly = monthly.sort_values("date").reset_index(drop=True)
    ts = monthly.set_index("date")["paper_count"]
    ts = ts.asfreq("MS", fill_value=0)

    return ts, monthly


def find_closest_event(year, events, tolerance=2):
    """
    Matches a changepoint year to the closest known event.
    Returns event label if within tolerance, else None.
    """
    closest = min(events.keys(), key=lambda y: abs(y - year))
    if abs(closest - year) <= tolerance:
        return events[closest]
    return None


def detect_changepoints(ts, pen=10):
    """
    Detects changepoints using PELT on the trend component.
    Returns changepoint dates and decomposition object.
    """
    decomp   = seasonal_decompose(ts, model="additive", period=12)
    trend    = decomp.trend.dropna()
    model    = rpt.Pelt(model="rbf").fit(trend.values)
    breaks   = model.predict(pen=pen)
    cp_dates = [trend.index[i-1] for i in breaks[:-1]]
    return cp_dates, decomp


def plot_decomposition(ts, title, color, save_path):
    """
    4-panel decomposition plot.
    Observed, Trend, Seasonality, Residual.
    """
    decomp = seasonal_decompose(ts, model="additive", period=12)

    fig, axes = plt.subplots(4, 1, figsize=(18, 12))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    components = [
        (ts,              color,     "Observed\n(raw data)"),
        (decomp.trend,    "#2196F3", "Trend\n(long-term direction)"),
        (decomp.seasonal, "#FF9800", "Seasonality\n(yearly pattern)"),
        (decomp.resid,    "#F44336", "Residual\n(noise)"),
    ]

    for ax, (data, col, label) in zip(axes, components):
        ax.plot(data.index, data.values, color=col, linewidth=1.5)
        ax.set_ylabel(label, fontsize=9, labelpad=10)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
        sns.despine(ax=ax)

    axes[-1].set_xlabel("Date", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Decomposition saved -> {save_path}")


def plot_changepoints(ts_monthly, changepoint_dates,
                      title, color, save_path):
    """
    Time series plot with changepoint lines and event labels.
    Top panel    -> series with date labels
    Bottom strip -> event labels
    """
    fig, (ax, ax_labels) = plt.subplots(
        2, 1, figsize=(20, 9),
        gridspec_kw={"height_ratios": [5, 1]}
    )

    # ── Main plot ──────────────────────────────────────────
    ax.plot(ts_monthly["date"], ts_monthly["paper_count"],
            color=color, linewidth=1.2, alpha=0.5,
            label="Monthly papers")

    rolling = (ts_monthly.set_index("date")["paper_count"]
               .rolling(window=12, center=True).mean())
    ax.plot(rolling.index, rolling.values,
            color=color, linewidth=2.5,
            label="12-month rolling average")

    y_min   = ts_monthly["paper_count"].min()
    y_max   = ts_monthly["paper_count"].max()
    y_range = y_max - y_min

    date_heights = [0.92, 0.78, 0.92, 0.78, 0.92, 0.78]

    for i, cp in enumerate(changepoint_dates):
        ax.axvline(x=cp, color="#F44336", linewidth=2,
                   linestyle="--", alpha=0.8,
                   label="Changepoint" if i == 0 else "")

        ax.text(cp,
                y_min + y_range * date_heights[i % len(date_heights)],
                cp.strftime("%b\n%Y"),
                ha="center", fontsize=9,
                color="#F44336", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white",
                          edgecolor="#F44336",
                          alpha=0.95))

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Papers per Month", fontsize=12)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    ax.set_xlim(ts_monthly["date"].min(),
                ts_monthly["date"].max())
    ax.set_xticks([])
    sns.despine(ax=ax)

    # ── Event label strip ──────────────────────────────────
    ax_labels.set_xlim(ax.get_xlim())
    ax_labels.set_ylim(0, 1)
    ax_labels.axis("off")

    for i, cp in enumerate(changepoint_dates):
        event = find_closest_event(cp.year, KNOWN_EVENTS)
        label = event if event else "Unknown shift"
        y_pos = 0.75 if i % 2 == 0 else 0.15

        ax_labels.text(
            mdates.date2num(cp), y_pos,
            label,
            ha="center", va="center",
            fontsize=8.5, color="#1A237E",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="#F3E5F5",
                      edgecolor="#9C27B0",
                      alpha=0.95)
        )

    ax_labels.set_xlabel("Date", fontsize=12)
    plt.tight_layout(h_pad=0)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Changepoint plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  MATHEMATICS - TIME SERIES ANALYSIS")
print("=" * 60)

df = pd.read_csv("C:\\Users\\riyas\\OneDrive\\ARXIV project\\01_data_collection\\arxiv_monthly_counts.csv")

math_rows = df[df["field"] == "math"]
print(f"\n  Math rows found    : {len(math_rows):,}")
print(f"  Math subcategories : {math_rows['sub_field'].nunique()}")
print(f"  Year range         : "
      f"{math_rows['year'].min()} - {math_rows['year'].max()}")
print(f"  Total papers       : {math_rows['paper_count'].sum():,}")
print(f"\n  Papers by subcategory:")
for sub, total in (math_rows.groupby("sub_field")["paper_count"]
                             .sum()
                             .sort_values(ascending=False)
                             .items()):
    name = MATH_NAMES.get(sub, sub)
    print(f"    {sub:<12} {name:<30} -> {total:>8,}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — OVERALL MATHEMATICS RAW TIME SERIES
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "-"*60)
print("  PART 1 - Overall Mathematics Raw Time Series")
print("-"*60)

math_ts, math_monthly = prepare_monthly_series(df, field="math")

fig, ax = plt.subplots(figsize=(18, 6))

ax.plot(math_monthly["date"], math_monthly["paper_count"],
        color="#CE93D8", linewidth=1.2, alpha=0.6,
        label="Monthly papers")

rolling = math_ts.rolling(window=12, center=True).mean()
ax.plot(rolling.index, rolling.values,
        color="#9C27B0", linewidth=2.5,
        linestyle="--", label="12-month rolling average")

ax.set_title(
    "Mathematics - Monthly Paper Submissions (1991-2025)",
    fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("Date", fontsize=13)
ax.set_ylabel("Papers per Month", fontsize=13)
ax.legend(fontsize=11)
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
sns.despine()
plt.tight_layout()
plt.savefig("plots/timeseries/math/math_overall_raw.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(" Raw time series saved")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 - DECOMPOSITION OF OVERALL MATHEMATICS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "-"*60)
print("  PART 2 - Decomposition")
print("-"*60)

plot_decomposition(
    math_ts,
    title="Mathematics - Time Series Decomposition\n"
          "Observed | Trend | Seasonality | Residual",
    color="#9C27B0",
    save_path="plots/timeseries/math/math_overall_decomposition.png"
)


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — CHANGEPOINT DETECTION ON OVERALL MATHEMATICS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "-"*60)
print("  PART 3 - Changepoint Detection (Overall Mathematics)")
print("-"*60)

cp_dates, _ = detect_changepoints(math_ts, pen=10)

print(f"  Changepoints detected: {len(cp_dates)}")
for cp in cp_dates:
    event = find_closest_event(cp.year, KNOWN_EVENTS)
    print(f"  -> {cp.strftime('%B %Y')}  "
          f"|  {event or 'No known event nearby'}")

plot_changepoints(
    math_monthly, cp_dates,
    title="Mathematics - Changepoint Detection (1991-2025)\n"
          "Red dashed lines show detected structural breaks",
    color="#9C27B0",
    save_path="plots/timeseries/math/math_overall_changepoints.png"
)

print("\n  CHANGEPOINT SUMMARY - OVERALL MATHEMATICS")
print("  " + "-"*55)
for i, cp in enumerate(cp_dates, 1):
    before = math_monthly[
        math_monthly["date"] < cp]["paper_count"].mean()
    after  = math_monthly[
        math_monthly["date"] >= cp]["paper_count"].mean()
    change = ((after - before) / before * 100)
    event  = find_closest_event(cp.year, KNOWN_EVENTS)
    print(f"\n  Changepoint {i}: {cp.strftime('%B %Y')}")
    print(f"    Avg before   : {before:>8,.0f} papers/month")
    print(f"    Avg after    : {after:>8,.0f} papers/month")
    print(f"    Change       : {change:>+.1f}%")
    print(f"    Likely cause : {event or 'Unknown'}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 4 — IMPORTANT SUBCATEGORIES
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "-"*60)
print("  PART 4 - Important Subcategories")
print("-"*60)
print(f"  Analysing {len(IMPORTANT_SUBCATS)} key subcategories:")
for sub, name in IMPORTANT_SUBCATS.items():
    print(f"    -> {sub} - {name}")

all_results = []

for subcat, name in IMPORTANT_SUBCATS.items():
    print(f"\n  {'-'*50}")
    print(f"  Processing: {subcat} - {name}")
    print(f"  {'-'*50}")

    try:
        ts, monthly = prepare_monthly_series(df, subfield=subcat)

        if len(monthly) < 24 or monthly["paper_count"].sum() < 50:
            print(f"     Not enough data - skipping")
            continue

        # Decomposition
        plot_decomposition(
            ts,
            title=f"{subcat} - {name}\n"
                  f"Time Series Decomposition",
            color="#9C27B0",
            save_path=(f"plots/timeseries/math/"
                       f"{subcat.replace('.','_')}_decomposition.png")
        )

        # Changepoint detection
        cp_dates_sub, _ = detect_changepoints(ts, pen=8)

        print(f"    Changepoints found: {len(cp_dates_sub)}")
        for cp in cp_dates_sub:
            event = find_closest_event(cp.year, KNOWN_EVENTS)
            print(f"   -> {cp.strftime('%B %Y')} | "
                  f"{event or 'No known event nearby'}")

        # Changepoint plot
        plot_changepoints(
            monthly, cp_dates_sub,
            title=f"{subcat} - {name}\n"
                  f"Changepoint Detection (1991-2025)",
            color="#9C27B0",
            save_path=(f"plots/timeseries/math/"
                       f"{subcat.replace('.','_')}_changepoints.png")
        )

        # Store results
        for cp in cp_dates_sub:
            before = monthly[
                monthly["date"] < cp]["paper_count"].mean()
            after  = monthly[
                monthly["date"] >= cp]["paper_count"].mean()
            change = ((after - before) / before * 100)
            event  = find_closest_event(cp.year, KNOWN_EVENTS)
            all_results.append({
                "sub_field"   : subcat,
                "name"        : name,
                "changepoint" : cp.strftime("%B %Y"),
                "avg_before"  : round(before, 1),
                "avg_after"   : round(after, 1),
                "change_pct"  : round(change, 1),
                "likely_cause": event or "Unknown",
            })

        print(f"    Done")

    except Exception as e:
        print(f"    Error: {e} - skipping")
        continue


# ─────────────────────────────────────────────────────────────────────────────
# PART 5 — SAVE SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "-"*60)
print("  PART 5 - Summary Table")
print("-"*60)

results_df = pd.DataFrame(all_results)
results_df.to_csv(
    "plots/timeseries/math/math_changepoints_summary.csv",
    index=False)

print(f"\n  Total changepoints found : {len(results_df)}")
if not results_df.empty:
    print(f"\n  Full summary:")
    print(results_df[["sub_field","name",
                       "changepoint","change_pct",
                       "likely_cause"]]
          .to_string(index=False))

print("\n" + "="*60)
print("  MATHEMATICS TIME SERIES COMPLETE!")
print("="*60)
print("\n  Files saved in: plots/timeseries/math/")
print("   math_overall_raw.png")
print("   math_overall_decomposition.png")
print("   math_overall_changepoints.png")
for sub in IMPORTANT_SUBCATS:
    s = sub.replace(".","_")
    print(f"   {s}_decomposition.png")
    print(f"   {s}_changepoints.png")
print("   math_changepoints_summary.csv")
