"""
timeseries_stat.py
==================
Time Series Analysis + Changepoint Detection
Field: Statistics (stat.*)
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
os.makedirs("plots/timeseries/stat", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# KNOWN REAL WORLD EVENTS
# Events that likely caused changepoints in Statistics research
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_EVENTS = {
    1991: "arXiv launched",
    2006: "R language became widely adopted",
    2008: "Financial crisis - statistics in demand",
    2010: "Big Data era begins",
    2012: "Deep learning revolution / Data science boom",
    2014: "Data science becomes mainstream career",
    2016: "Reproducibility crisis in statistics",
    2017: "Transformer paper - ML meets statistics",
    2019: "Federated learning / Privacy statistics",
    2020: "COVID-19 - statistics in global spotlight",
    2022: "ChatGPT - LLM and Bayesian methods surge",
    2023: "Causal AI and uncertainty quantification boom",
}

# Full names for Statistics subcategories
STAT_NAMES = {
    "stat.AP": "Applications",
    "stat.CO": "Computation",
    "stat.ME": "Methodology",
    "stat.ML": "Machine Learning",
    "stat.OT": "Other Statistics",
    "stat.TH": "Theory",
}

# Most important subcategories to analyse
IMPORTANT_SUBCATS = {
    "stat.ML": "Machine Learning",
    "stat.ME": "Methodology",
    "stat.TH": "Theory",
    "stat.AP": "Applications",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# (same approach as CS - consistent methodology across all fields)
# ─────────────────────────────────────────────────────────────────────────────

def prepare_monthly_series(df, field=None, subfield=None):
    """
    Prepares a clean monthly time series.
    Filters by field or subfield, removes 2026,
    creates proper date column and fills missing months.
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
    Finds the closest known event to a detected changepoint year.
    Returns event label if within tolerance years, else None.
    """
    closest = min(events.keys(), key=lambda y: abs(y - year))
    if abs(closest - year) <= tolerance:
        return events[closest]
    return None


def detect_changepoints(ts, pen=10):
    """
    Detects changepoints using PELT algorithm on the trend component.
    Returns list of changepoint dates and the decomposition object.
    """
    decomp = seasonal_decompose(ts, model="additive", period=12)
    trend  = decomp.trend.dropna()
    model  = rpt.Pelt(model="rbf").fit(trend.values)
    breaks = model.predict(pen=pen)
    cp_dates = [trend.index[i-1] for i in breaks[:-1]]
    return cp_dates, decomp


def plot_decomposition(ts, title, color, save_path):
    """
    Plots the 4-panel decomposition chart.
    Shows observed, trend, seasonality and residual.
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
    print(f"  Decomposition saved → {save_path}")


def plot_changepoints(ts_monthly, changepoint_dates,
                      title, color, save_path):
    """
    Plots time series with changepoint lines and event labels.
    Top panel    -> time series with date labels on changepoints
    Bottom strip -> event labels aligned to each changepoint
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

    # Alternate date label heights to avoid overlap
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
                      facecolor="#E8EAF6",
                      edgecolor="#FF9800",
                      alpha=0.95)
        )

    ax_labels.set_xlabel("Date", fontsize=12)
    plt.tight_layout(h_pad=0)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Changepoint plot saved -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  STATISTICS - TIME SERIES ANALYSIS")
print("=" * 60)

df = pd.read_csv("C:\\Users\\riyas\\OneDrive\\ARXIV project\\01_data_collection\\arxiv_monthly_counts.csv")

stat_rows = df[df["field"] == "stat"]
print(f"\n  Stat rows found    : {len(stat_rows):,}")
print(f"  Stat subcategories : {stat_rows['sub_field'].nunique()}")
print(f"  Year range         : "
      f"{stat_rows['year'].min()} - {stat_rows['year'].max()}")
print(f"  Total papers       : {stat_rows['paper_count'].sum():,}")
print(f"\n  Papers by subcategory:")
for sub, total in (stat_rows.groupby("sub_field")["paper_count"]
                             .sum()
                             .sort_values(ascending=False)
                             .items()):
    name = STAT_NAMES.get(sub, sub)
    print(f"    {sub:<12} {name:<25} -> {total:>8,}")


# ─────────────────────────────────────────────────────────────────────────────
#  OVERALL STATISTICS RAW TIME SERIES
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "-"*60)
print("  PART 1 - Overall Statistics Raw Time Series")
print("-"*60)

stat_ts, stat_monthly = prepare_monthly_series(df, field="stat")

fig, ax = plt.subplots(figsize=(18, 6))

ax.plot(stat_monthly["date"], stat_monthly["paper_count"],
        color="#FFCC80", linewidth=1.2, alpha=0.6,
        label="Monthly papers")

rolling = stat_ts.rolling(window=12, center=True).mean()
ax.plot(rolling.index, rolling.values,
        color="#FF9800", linewidth=2.5,
        linestyle="--", label="12-month rolling average")

ax.set_title(
    "Statistics - Monthly Paper Submissions (1991-2025)",
    fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("Date", fontsize=13)
ax.set_ylabel("Papers per Month", fontsize=13)
ax.legend(fontsize=11)
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
sns.despine()
plt.tight_layout()
plt.savefig("plots/timeseries/stat/stat_overall_raw.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  Raw time series saved")


# ─────────────────────────────────────────────────────────────────────────────
# DECOMPOSITION OF OVERALL STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "-"*60)
print("  PART 2 - Decomposition")
print("-"*60)

plot_decomposition(
    stat_ts,
    title="Statistics - Time Series Decomposition\n"
          "Observed | Trend | Seasonality | Residual",
    color="#FF9800",
    save_path="plots/timeseries/stat/stat_overall_decomposition.png"
)


# ─────────────────────────────────────────────────────────────────────────────
# CHANGEPOINT DETECTION ON OVERALL STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "-"*60)
print("  PART 3 - Changepoint Detection (Overall Statistics)")
print("-"*60)

cp_dates, _ = detect_changepoints(stat_ts, pen=10)

print(f"  Changepoints detected: {len(cp_dates)}")
for cp in cp_dates:
    event = find_closest_event(cp.year, KNOWN_EVENTS)
    print(f"    -> {cp.strftime('%B %Y')}  "
          f"|  {event or 'No known event nearby'}")

plot_changepoints(
    stat_monthly, cp_dates,
    title="Statistics - Changepoint Detection (1991-2025)\n"
          "Red dashed lines show detected structural breaks",
    color="#FF9800",
    save_path="plots/timeseries/stat/stat_overall_changepoints.png"
)

# Summary table
print("\n  CHANGEPOINT SUMMARY - OVERALL STATISTICS")
print("  " + "-"*55)
for i, cp in enumerate(cp_dates, 1):
    before = stat_monthly[
        stat_monthly["date"] < cp]["paper_count"].mean()
    after  = stat_monthly[
        stat_monthly["date"] >= cp]["paper_count"].mean()
    change = ((after - before) / before * 100)
    event  = find_closest_event(cp.year, KNOWN_EVENTS)
    print(f"\n  Changepoint {i}: {cp.strftime('%B %Y')}")
    print(f"    Avg before   : {before:>8,.0f} papers/month")
    print(f"    Avg after    : {after:>8,.0f} papers/month")
    print(f"    Change       : {change:>+.1f}%")
    print(f"    Likely cause : {event or 'Unknown'}")


# ─────────────────────────────────────────────────────────────────────────────
# IMPORTANT SUBCATEGORIES
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
            print(f"    Not enough data — skipping")
            continue

        # Decomposition
        plot_decomposition(
            ts,
            title=f"{subcat} - {name}\nTime Series Decomposition",
            color="#FF9800",
            save_path=(f"plots/timeseries/stat/"
                       f"{subcat.replace('.','_')}_decomposition.png")
        )

        # Changepoint detection
        cp_dates_sub, _ = detect_changepoints(ts, pen=8)

        print(f"    Changepoints found: {len(cp_dates_sub)}")
        for cp in cp_dates_sub:
            event = find_closest_event(cp.year, KNOWN_EVENTS)
            print(f"  -> {cp.strftime('%B %Y')} | "
                  f"{event or 'No known event nearby'}")

        # Changepoint plot
        plot_changepoints(
            monthly, cp_dates_sub,
            title=f"{subcat} - {name}\n"
                  f"Changepoint Detection (1991-2025)",
            color="#FF9800",
            save_path=(f"plots/timeseries/stat/"
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

        print(f" Done")

    except Exception as e:
        print(f"    Error: {e} - skipping")
        continue


# ─────────────────────────────────────────────────────────────────────────────
#  SAVE SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "-"*60)
print("  PART 5 - Summary Table")
print("-"*60)

results_df = pd.DataFrame(all_results)
results_df.to_csv(
    "plots/timeseries/stat/stat_changepoints_summary.csv",
    index=False)

print(f"\n  Total changepoints found : {len(results_df)}")
print(f"\n  Full summary:")
if not results_df.empty:
    print(results_df[["sub_field","name",
                       "changepoint","change_pct",
                       "likely_cause"]]
          .to_string(index=False))

print("\n" + "="*60)
print("  STATISTICS TIME SERIES COMPLETE!")
print("="*60)
print("\n  Files saved in: plots/timeseries/stat/")
print("   stat_overall_raw.png")
print("   stat_overall_decomposition.png")
print("   stat_overall_changepoints.png")
for sub in IMPORTANT_SUBCATS:
    s = sub.replace(".","_")
    print(f"   {s}_decomposition.png")
    print(f"   {s}_changepoints.png")
print("   stat_changepoints_summary.csv")