"""
timeseries_cs.py
================
Time Series Analysis + Changepoint Detection
Field: Computer Science (cs.*)

What this script does:
    1. Plots overall CS monthly submissions (1991-2025)
    2. Decomposes the series into trend, seasonality and noise
    3. Detects changepoints on overall CS trend
    4. Plots time series for every CS subcategory individually
    5. Detects changepoints for each subcategory

Why we do this:
    CS is the fastest growing field on arXiv. By analysing
    each subcategory separately we can see exactly WHEN and
    WHERE the growth happened and link it to real world events
    like the deep learning revolution, transformer models,
    and the rise of large language models.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")   # ← must be BEFORE importing pyplot
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import ruptures as rpt
import warnings
import os

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# Create output folders
os.makedirs("plots/timeseries/cs", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# KNOWN REAL WORLD EVENTS
# These are events that likely caused changepoints in CS research
# The algorithm will automatically detect changepoints and we will
# match them to these known events
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_EVENTS = {
    1991: "arXiv launched",
    1998: "Google founded / Web boom",
    2001: "Dot-com crash",
    2008: "Financial crisis / Cloud computing",
    2012: "AlexNet - Deep learning revolution",
    2014: "GANs invented / Deep RL",
    2017: "Transformer paper published",
    2018: "BERT / GPT-1 released",
    2020: "COVID-19 research surge",
    2022: "ChatGPT / LLM explosion",
    2023: "GPT-4 / Generative AI boom",
}

# Full names for all CS subcategories
CS_NAMES = {
    "cs.AI":"Artificial Intelligence",
    "cs.AR":"Hardware Architecture",
    "cs.CC":"Computational Complexity",
    "cs.CE":"Computational Engineering",
    "cs.CG":"Computational Geometry",
    "cs.CL":"Computation & Language (NLP)",
    "cs.CR":"Cryptography & Security",
    "cs.CV":"Computer Vision",
    "cs.CY":"Computers & Society",
    "cs.DB":"Databases",
    "cs.DC":"Distributed Computing",
    "cs.DL":"Digital Libraries",
    "cs.DM":"Discrete Mathematics",
    "cs.DS":"Data Structures & Algorithms",
    "cs.ET":"Emerging Technologies",
    "cs.FL":"Formal Languages",
    "cs.GL":"General Literature",
    "cs.GR":"Graphics",
    "cs.GT":"Game Theory",
    "cs.HC":"Human-Computer Interaction",
    "cs.IR":"Information Retrieval",
    "cs.IT":"Information Theory",
    "cs.LG":"Machine Learning",
    "cs.LO":"Logic in CS",
    "cs.MA":"Multiagent Systems",
    "cs.MM":"Multimedia",
    "cs.MS":"Mathematical Software",
    "cs.NA":"Numerical Analysis",
    "cs.NE":"Neural & Evolutionary Computing",
    "cs.NI":"Networking & Internet",
    "cs.OH":"Other Computer Science",
    "cs.OS":"Operating Systems",
    "cs.PF":"Performance",
    "cs.PL":"Programming Languages",
    "cs.RO":"Robotics",
    "cs.SC":"Symbolic Computation",
    "cs.SD":"Sound",
    "cs.SE":"Software Engineering",
    "cs.SI":"Social & Information Networks",
    "cs.SY":"Systems & Control",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def prepare_monthly_series(df, field=None, subfield=None):
    """
    Prepares a clean monthly time series from the dataframe.

    What this does:
        - Filters by field or subfield
        - Removes 2026 (incomplete year)
        - Groups by year+month and sums paper counts
        - Creates a proper date column
        - Fills any missing months with 0

    Parameters
    ----------
    df       : full dataframe
    field    : e.g. "cs" for all CS combined
    subfield : e.g. "cs.LG" for just Machine Learning

    Returns
    -------
    A clean monthly time series with date as index
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

    # Set date as index and fill missing months
    ts = monthly.set_index("date")["paper_count"]
    ts = ts.asfreq("MS", fill_value=0)

    return ts, monthly


def find_closest_event(year, events, tolerance=2):
    """
    Finds the closest known event to a given year.

    Parameters
    ----------
    year      : the changepoint year
    events    : dictionary of {year: event_label}
    tolerance : how many years away counts as a match

    Returns
    -------
    Event label string or None
    """
    closest = min(events.keys(), key=lambda y: abs(y - year))
    if abs(closest - year) <= tolerance:
        return events[closest]
    return None


def detect_changepoints(ts, pen=10):
    """
    Detects changepoints in a time series using the PELT algorithm.

    What PELT does:
        Looks for points where the mean of the data shifts significantly.
        pen controls sensitivity:
            low pen  = finds more changepoints (more sensitive)
            high pen = finds fewer changepoints (only big shifts)

    Parameters
    ----------
    ts  : time series as pandas Series
    pen : penalty value (default 10)

    Returns
    -------
    List of changepoint dates
    """
    # Use trend from decomposition for cleaner signal
    decomp  = seasonal_decompose(ts, model="additive", period=12)
    trend   = decomp.trend.dropna()

    model   = rpt.Pelt(model="rbf").fit(trend.values)
    breaks  = model.predict(pen=pen)

    # Convert indices to dates
    cp_dates = [trend.index[i-1] for i in breaks[:-1]]
    return cp_dates, decomp


def plot_changepoints(ts_monthly, changepoint_dates,
                      title, color, save_path):
    """
    Plots the time series with changepoints and event labels.

    Layout:
        Top panel    → time series with rolling average and
                       changepoint lines with date labels
        Bottom strip → event labels aligned to each changepoint

    Parameters
    ----------
    ts_monthly        : monthly dataframe with date and paper_count
    changepoint_dates : list of changepoint dates
    title             : plot title
    color             : line colour
    save_path         : where to save the plot
    """
    fig, (ax, ax_labels) = plt.subplots(
        2, 1, figsize=(20, 9),
        gridspec_kw={"height_ratios": [5, 1]}
    )

    # ── Main plot ──────────────────────────────────────────
    ax.plot(ts_monthly["date"], ts_monthly["paper_count"],
            color=color, linewidth=1.2, alpha=0.5,
            label="Monthly papers")

    # Rolling 12-month average
    rolling = (ts_monthly.set_index("date")["paper_count"]
               .rolling(window=12, center=True).mean())
    darker  = color
    ax.plot(rolling.index, rolling.values,
            color=darker, linewidth=2.5,
            label="12-month rolling average")

    y_min   = ts_monthly["paper_count"].min()
    y_max   = ts_monthly["paper_count"].max()
    y_range = y_max - y_min

    # Alternate date label heights
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
                      edgecolor="#9C27B0",
                      alpha=0.95)
        )

    ax_labels.set_xlabel("Date", fontsize=12)
    plt.tight_layout(h_pad=0)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  COMPUTER SCIENCE — TIME SERIES ANALYSIS")
print("=" * 60)

df = pd.read_csv(r"C:\Users\riyas\OneDrive\ARXIV project\01_data_collection\arxiv_monthly_counts.csv")

# Quick check
cs_rows = df[df["field"] == "cs"]
print(f"\n  CS rows found    : {len(cs_rows):,}")
print(f"  CS subcategories : {cs_rows['sub_field'].nunique()}")
print(f"  Year range       : {cs_rows['year'].min()} – {cs_rows['year'].max()}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — OVERALL CS TIME SERIES
# What: plots all 40 CS subcategories combined as one series
# Why:  gives us the big picture of CS growth before drilling
#       into individual subcategories
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─"*60)
print("  PART 1 — Overall CS Time Series")
print("─"*60)

cs_ts, cs_monthly = prepare_monthly_series(df, field="cs")

print(f"  Date range  : {cs_monthly['date'].min().strftime('%b %Y')} "
      f"– {cs_monthly['date'].max().strftime('%b %Y')}")
print(f"  Total months: {len(cs_monthly)}")
print(f"  Total papers: {cs_monthly['paper_count'].sum():,}")

# Plot raw time series
fig, ax = plt.subplots(figsize=(18, 6))

ax.plot(cs_monthly["date"], cs_monthly["paper_count"],
        color="#90CAF9", linewidth=1.2, alpha=0.6,
        label="Monthly papers")

rolling = cs_ts.rolling(window=12, center=True).mean()
ax.plot(rolling.index, rolling.values,
        color="#2196F3", linewidth=2.5,
        linestyle="--", label="12-month rolling average")

ax.set_title("Computer Science — Monthly Paper Submissions (1991–2025)",
             fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("Date", fontsize=13)
ax.set_ylabel("Papers per Month", fontsize=13)
ax.legend(fontsize=11)
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
sns.despine()
plt.tight_layout()
plt.savefig("plots/timeseries/cs/cs_overall_raw.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Raw time series saved")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — DECOMPOSITION OF OVERALL CS
# What: splits the CS series into trend, seasonality and noise
# Why:  helps us understand the structure before changepoint detection
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─"*60)
print("  PART 2 — Decomposition")
print("─"*60)
print("  Splitting CS series into: Trend + Seasonality + Noise")

decomp = seasonal_decompose(cs_ts, model="additive", period=12)

fig, axes = plt.subplots(4, 1, figsize=(18, 14))
fig.suptitle("Computer Science — Time Series Decomposition\n"
             "Splitting into: Trend | Seasonality | Noise",
             fontsize=15, fontweight="bold", y=1.01)

components = [
    (cs_ts,            "#90CAF9", "Observed\n(raw monthly data)"),
    (decomp.trend,     "#2196F3", "Trend\n(long-term growth direction)"),
    (decomp.seasonal,  "#FF9800", "Seasonality\n(yearly repeating pattern)"),
    (decomp.resid,     "#F44336", "Residual\n(unexplained noise)"),
]

for ax, (data, color, label) in zip(axes, components):
    ax.plot(data.index, data.values, color=color, linewidth=1.5)
    ax.set_ylabel(label, fontsize=10, labelpad=10)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    sns.despine(ax=ax)

axes[-1].set_xlabel("Date", fontsize=12)
plt.tight_layout()
plt.savefig("plots/timeseries/cs/cs_overall_decomposition.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Decomposition plot saved")


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — CHANGEPOINT DETECTION ON OVERALL CS
# What: finds where CS submission patterns suddenly changed
# Why:  identifies key historical moments in CS research history
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─"*60)
print("  PART 3 — Changepoint Detection (Overall CS)")
print("─"*60)

cp_dates, _ = detect_changepoints(cs_ts, pen=10)

print(f"  Changepoints detected: {len(cp_dates)}")
for cp in cp_dates:
    event = find_closest_event(cp.year, KNOWN_EVENTS)
    print(f"    → {cp.strftime('%B %Y')}  |  {event or 'No known event nearby'}")

plot_changepoints(
    cs_monthly, cp_dates,
    title="Computer Science — Changepoint Detection (1991–2025)\n"
          "Red dashed lines show detected structural breaks",
    color="#2196F3",
    save_path="plots/timeseries/cs/cs_overall_changepoints.png"
)

# Print summary table
print("\n  CHANGEPOINT SUMMARY — OVERALL CS")
print("  " + "─"*55)
for i, cp in enumerate(cp_dates, 1):
    before = cs_monthly[cs_monthly["date"] < cp]["paper_count"].mean()
    after  = cs_monthly[cs_monthly["date"] >= cp]["paper_count"].mean()
    change = ((after - before) / before * 100)
    event  = find_closest_event(cp.year, KNOWN_EVENTS)
    print(f"\n  Changepoint {i}: {cp.strftime('%B %Y')}")
    print(f"    Avg before   : {before:>8,.0f} papers/month")
    print(f"    Avg after    : {after:>8,.0f} papers/month")
    print(f"    Change       : {change:>+.1f}%")
    print(f"    Likely cause : {event or 'Unknown'}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 4 — TIME SERIES FOR IMPORTANT CS SUBCATEGORIES ONLY
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─"*60)
print("  PART 4 — Important CS Subcategories")
print("─"*60)

# Only analyse these 8 most important subcategories
IMPORTANT_SUBCATS = {
    "cs.LG": "Machine Learning",
    "cs.AI": "Artificial Intelligence",
    "cs.CV": "Computer Vision",
    "cs.CL": "Computation & Language (NLP)",
    "cs.CR": "Cryptography & Security",
    "cs.RO": "Robotics",
    "cs.NE": "Neural & Evolutionary Computing",
    "cs.SI": "Social & Information Networks",
}

print(f"  Analysing {len(IMPORTANT_SUBCATS)} key subcategories:")
for sub, name in IMPORTANT_SUBCATS.items():
    print(f"    → {sub} — {name}")

all_results = []

for subcat, name in IMPORTANT_SUBCATS.items():
    print(f"\n  {'─'*50}")
    print(f"  Processing: {subcat} — {name}")
    print(f"  {'─'*50}")

    try:
        ts, monthly = prepare_monthly_series(df, subfield=subcat)

        if len(monthly) < 24 or monthly["paper_count"].sum() < 100:
            print(f"    ⚠ Not enough data — skipping")
            continue

        # ── Decomposition plot ────────────────────────────
        decomp_sub = seasonal_decompose(
            ts, model="additive", period=12)

        fig, axes = plt.subplots(4, 1, figsize=(18, 12))
        fig.suptitle(
            f"{subcat} — {name}\nTime Series Decomposition",
            fontsize=14, fontweight="bold")

        components = [
            (ts,                  "#90CAF9", "Observed\n(raw data)"),
            (decomp_sub.trend,    "#2196F3", "Trend\n(long-term direction)"),
            (decomp_sub.seasonal, "#FF9800", "Seasonality\n(yearly pattern)"),
            (decomp_sub.resid,    "#F44336", "Residual\n(noise)"),
        ]
        for ax, (data, color, label) in zip(axes, components):
            ax.plot(data.index, data.values,
                    color=color, linewidth=1.5)
            ax.set_ylabel(label, fontsize=9)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
            sns.despine(ax=ax)
        axes[-1].set_xlabel("Date", fontsize=11)
        plt.tight_layout()
        decomp_path = (f"plots/timeseries/cs/"
                       f"{subcat.replace('.','_')}_decomposition.png")
        plt.savefig(decomp_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    ✓ Decomposition saved")

        # ── Changepoint detection ─────────────────────────
        cp_dates_sub, _ = detect_changepoints(ts, pen=8)

        print(f"    Changepoints found: {len(cp_dates_sub)}")
        for cp in cp_dates_sub:
            event = find_closest_event(cp.year, KNOWN_EVENTS)
            print(f"      → {cp.strftime('%B %Y')} | "
                  f"{event or 'No known event nearby'}")

        # ── Changepoint plot ──────────────────────────────
        plot_changepoints(
            monthly, cp_dates_sub,
            title=f"{subcat} — {name}\n"
                  f"Changepoint Detection (1991–2025)",
            color="#2196F3",
            save_path=(f"plots/timeseries/cs/"
                       f"{subcat.replace('.','_')}_changepoints.png")
        )

        # ── Store results ─────────────────────────────────
        for cp in cp_dates_sub:
            before = monthly[monthly["date"] < cp]["paper_count"].mean()
            after  = monthly[monthly["date"] >= cp]["paper_count"].mean()
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

    except Exception as e:
        print(f"    ⚠ Error: {e} — skipping")
        continue

# ─────────────────────────────────────────────────────────────────────────────
# PART 5 — SAVE SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "─"*60)
print("  PART 5 — Summary Table")
print("─"*60)

results_df = pd.DataFrame(all_results)
summary_path = (r"C:\Users\riyas\OneDrive\ARXIV project"
                r"\plots\timeseries\cs\cs_changepoints_summary.csv")
results_df.to_csv(summary_path, index=False)

print(f"\n  Total changepoints found : {len(results_df)}")
print(f"\n  Full summary:")
print(results_df[["sub_field","name","changepoint",
                  "change_pct","likely_cause"]]
      .to_string(index=False))

print("\n" + "="*60)
print("  CS TIME SERIES COMPLETE!")
print("="*60)
print("\n  Plots saved in: plots/timeseries/cs/")
print(f"\n  Files generated:")
print(f"    📊 cs_overall_raw.png")
print(f"    📊 cs_overall_decomposition.png")
print(f"    📊 cs_overall_changepoints.png")
for sub in IMPORTANT_SUBCATS:
    sub_clean = sub.replace('.','_')
    print(f"    📊 {sub_clean}_decomposition.png")
    print(f"    📊 {sub_clean}_changepoints.png")
print(f"    📄 cs_changepoints_summary.csv")
print("\n  Done! ✅")