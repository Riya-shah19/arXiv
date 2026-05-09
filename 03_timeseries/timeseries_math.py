"""
timeseries_math.py
==================
Time Series Analysis + Changepoint Detection
Field: Mathematics (math.*)
Data: Monthly paper counts 1991-2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates        
import matplotlib.patches as mpatches
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import ruptures as rpt
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
import os
os.makedirs("plots/timeseries", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load and prepare the data
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_csv(r"C:\Users\riyas\OneDrive\ARXIV project\01_data_collection\arxiv_monthly_counts.csv")

# Filter Mathematics only, remove 2026
math_df = df[
    (df["field"] == "math") &
    (df["year"] <= 2025)
].copy()

# Quick check — see what's in the data
print("  Math rows found:", len(math_df))
print("  Sample:")
print(math_df.head())
print("  Year values:", sorted(math_df["year"].unique()))
print("  Month values:", sorted(math_df["month"].unique()))

# Aggregate ALL math subcategories into one total per month
math_monthly = (math_df
                .groupby(["year","month"])["paper_count"]
                .sum()
                .reset_index())

# Create a proper date column (first day of each month)
math_monthly["date"] = pd.to_datetime(
    math_monthly["year"].astype(str) + "-" +
    math_monthly["month"].astype(str).str.zfill(2) + "-01"
)

# Drop any rows where date failed to parse
before = len(math_monthly)
math_monthly = math_monthly.dropna(subset=["date"])
after  = len(math_monthly)

if before != after:
    print(f"  ⚠ Dropped {before - after} rows with invalid dates")

# Sort by date
math_monthly = math_monthly.sort_values("date").reset_index(drop=True)


print("=" * 55)
print("  MATHEMATICS TIME SERIES")
print("=" * 55)
print(f"  Date range  : {math_monthly['date'].min().strftime('%b %Y')} "
      f"– {math_monthly['date'].max().strftime('%b %Y')}")
print(f"  Total months: {len(math_monthly)}")
print(f"  Total papers: {math_monthly['paper_count'].sum():,}")
print(f"\n  First 5 rows:")
print(math_monthly[["date","paper_count"]].head())


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Plot raw monthly time series
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(18, 6))

ax.plot(math_monthly["date"], math_monthly["paper_count"],
        color="#9C27B0", linewidth=1.5, alpha=0.8, label="Monthly papers")

# Add a rolling 12-month average to show the trend clearly
rolling_avg = math_monthly["paper_count"].rolling(window=12, center=True).mean()
ax.plot(math_monthly["date"], rolling_avg,
        color="#4A148C", linewidth=2.5,
        linestyle="--", label="12-month rolling average")

ax.set_title("Mathematics — Monthly Paper Submissions (1991–2025)",
             fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("Date", fontsize=13)
ax.set_ylabel("Papers per Month", fontsize=13)
ax.legend(fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
sns.despine()
plt.tight_layout()
plt.savefig("plots/timeseries/math_ts_raw.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Raw time series plot saved")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Decompose the time series
#
# What decomposition does:
#   It splits your data into 3 separate components:
#   1. Trend     → the overall long-term direction (going up/down)
#   2. Seasonal  → repeating patterns every 12 months
#   3. Residual  → what's left after removing trend and seasonality (noise)
# ─────────────────────────────────────────────────────────────────────────────

# Set date as index for decomposition
ts = math_monthly.set_index("date")["paper_count"]

# Fill any missing months with 0
ts = ts.asfreq("MS", fill_value=0)

# Decompose — period=12 means we expect yearly seasonal patterns
decomposition = seasonal_decompose(ts, model="additive", period=12)

fig, axes = plt.subplots(4, 1, figsize=(18, 14))
fig.suptitle("Mathematics — Time Series Decomposition",
             fontsize=16, fontweight="bold", y=1.01)

components = [
    (ts,                        "#9C27B0", "Observed\n(raw data)"),
    (decomposition.trend,       "#2196F3", "Trend\n(long-term direction)"),
    (decomposition.seasonal,    "#FF9800", "Seasonality\n(repeating yearly pattern)"),
    (decomposition.resid,       "#F44336", "Residual\n(noise / unexplained)"),
]

for ax, (data, color, title) in zip(axes, components):
    ax.plot(data.index, data.values, color=color, linewidth=1.5)
    ax.set_ylabel(title, fontsize=10, labelpad=10)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    sns.despine(ax=ax)

axes[-1].set_xlabel("Date", fontsize=12)
plt.tight_layout()
plt.savefig("plots/timeseries/math_ts_decomposition.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✓ Decomposition plot saved")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Changepoint Detection
#
# What changepoints are:
#   Moments in time where the pattern of submissions SUDDENLY changed.
#   The ruptures library automatically finds these points.
#
# We use the PELT algorithm:
#   PELT = Pruned Exact Linear Time
#   It finds the most significant shifts in the data efficiently.
# ─────────────────────────────────────────────────────────────────────────────

# Use the trend component for changepoint detection
# (cleaner signal — removes seasonal noise)
trend_values = decomposition.trend.dropna().values
trend_dates  = decomposition.trend.dropna().index

# Run changepoint detection
# pen=10 controls sensitivity:
#   lower pen → more changepoints detected
#   higher pen → fewer, more significant changepoints
model  = rpt.Pelt(model="rbf").fit(trend_values)
breaks = model.predict(pen=10)

# Convert break indices to actual dates
changepoint_dates = [trend_dates[i-1] for i in breaks[:-1]]

print(f"\n  Changepoints detected: {len(changepoint_dates)}")
for cp in changepoint_dates:
    print(f"    → {cp.strftime('%B %Y')}")

# Known real world events that may explain math changepoints
KNOWN_EVENTS = {
    1991: "arXiv\nlaunched",
    2000: "arXiv math\nsection created",
    2008: "Global financial\ncrisis (math finance)",
    2012: "Deep learning\nrevolution begins",
    2015: "ML + math\nconvergence",
    2020: "COVID-19\n(online research surge)",
    2022: "ChatGPT\nreleased",
}
# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Plot changepoints on the time series
# ─────────────────────────────────────────────────────────────────────────────

fig, (ax, ax_labels) = plt.subplots(
    2, 1,
    figsize=(20, 9),
    gridspec_kw={"height_ratios": [5, 1]}  # main plot is 5x taller than label strip
)

# ── Main plot ────────────────────────────────────────────
ax.plot(math_monthly["date"], math_monthly["paper_count"],
        color="#CE93D8", linewidth=1.2, alpha=0.6, label="Monthly papers")

rolling = math_monthly.set_index("date")["paper_count"].rolling(
    window=12, center=True).mean()
ax.plot(rolling.index, rolling.values,
        color="#9C27B0", linewidth=2.5, label="12-month average")

y_min = math_monthly["paper_count"].min()
y_max = math_monthly["paper_count"].max()
y_range = y_max - y_min

# Alternate date label heights so they don't overlap each other
date_heights = [0.92, 0.78, 0.92, 0.78, 0.92, 0.78]

for i, cp in enumerate(changepoint_dates):
    # Vertical line on main plot
    ax.axvline(x=cp, color="#F44336", linewidth=2,
               linestyle="--", alpha=0.8,
               label="Changepoint" if i == 0 else "")

    # Date label on main plot
    ax.text(cp,
            y_min + y_range * date_heights[i % len(date_heights)],
            cp.strftime("%b\n%Y"),
            ha="center", fontsize=9,
            color="#F44336", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white",
                      edgecolor="#F44336",
                      alpha=0.95))

    # Vertical line on label strip below
    ax_labels.axvline(x=cp, color="#F44336",
                      linewidth=2, linestyle="--", alpha=0.8)

ax.set_title("Mathematics — Changepoint Detection (1991–2025)\n"
             "Red dashed lines show detected structural breaks",
             fontsize=15, fontweight="bold", pad=15)
ax.set_ylabel("Papers per Month", fontsize=13)
ax.legend(fontsize=11)
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
ax.set_xlim(math_monthly["date"].min(), math_monthly["date"].max())
ax.set_xticks([])   # hide x ticks on main plot
sns.despine(ax=ax)

# ── Label strip below main plot ───────────────────────────
ax_labels.set_xlim(ax.get_xlim())
ax_labels.set_ylim(0, 1)
ax_labels.axis("off")

for i, cp in enumerate(changepoint_dates):
    # Find closest known event
    matched_event = None
    for event_year, event_label in KNOWN_EVENTS.items():
        if abs(cp.year - event_year) <= 2:
            matched_event = event_label.replace("\n", " ")
            break

    label_text = matched_event if matched_event else "Unknown shift"

    # Alternate between top and bottom of the strip
    y_pos = 0.75 if i % 2 == 0 else 0.15

    ax_labels.text(
        mdates.date2num(cp),   # ← convert date to number for positioning
        y_pos,
        label_text,
        ha="center", va="center",
        fontsize=8.5,
        color="#1A237E",
        style="italic",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#E8EAF6",
            edgecolor="#9C27B0",
            alpha=0.95
        )
    )

ax_labels.set_xlabel("Date", fontsize=13)

plt.tight_layout(h_pad=0)
plt.savefig("plots/timeseries/math_ts_changepoints.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("✓ Changepoint plot saved")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Summary table of changepoints
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("  CHANGEPOINT SUMMARY — MATHEMATICS")
print("=" * 55)

for i, cp in enumerate(changepoint_dates, 1):
    # Find the average paper count before and after changepoint
    before = math_monthly[math_monthly["date"] < cp]["paper_count"].mean()
    after  = math_monthly[math_monthly["date"] >= cp]["paper_count"].mean()
    change = ((after - before) / before * 100)

    # Find closest known event
    closest_event = min(KNOWN_EVENTS.keys(),
                        key=lambda y: abs(y - cp.year))
    event_distance = abs(closest_event - cp.year)
    event_note = (KNOWN_EVENTS[closest_event]
                  if event_distance <= 2
                  else "No known event nearby")
    event_note = event_note.replace("\n", " ")

    print(f"\n  Changepoint {i}: {cp.strftime('%B %Y')}")
    print(f"    Avg before : {before:>8,.0f} papers/month")
    print(f"    Avg after  : {after:>8,.0f} papers/month")
    print(f"    Change     : {change:>+.1f}%")
    print(f"    Likely cause: {event_note}")

print("\n" + "=" * 55)
print("  FILES SAVED")
print("=" * 55)
print("  📊 plots/timeseries/math_ts_raw.png")
print("  📊 plots/timeseries/math_ts_decomposition.png")
print("  📊 plots/timeseries/math_ts_changepoints.png")
print("\n  Done! ✅")