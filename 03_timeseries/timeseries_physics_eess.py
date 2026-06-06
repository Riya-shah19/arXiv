"""
timeseries_physics_eess.py
==========================
Time Series Analysis + Changepoint Detection
Fields: Physics and Electrical Engineering (EESS)
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

os.chdir(r"C:\Users\riyas\OneDrive\ARXIV project")
os.makedirs("plots/timeseries/physics", exist_ok=True)
os.makedirs("plots/timeseries/eess",    exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# KNOWN REAL WORLD EVENTS
# ─────────────────────────────────────────────────────────────────────────────

PHYSICS_EVENTS = {
    1991: "arXiv launched - physics was first field",
    2008: "Large Hadron Collider operations begin",
    2012: "Higgs Boson discovered at CERN",
    2015: "LIGO detects gravitational waves",
    2016: "Gravitational waves confirmed - Nobel Prize",
    2019: "First image of a black hole released",
    2020: "COVID-19 research surge",
    2022: "James Webb Space Telescope launched",
    2023: "AI applied to physics simulations surge",
}

EESS_EVENTS = {
    1991: "arXiv launched",
    2012: "Deep learning revolution - image processing",
    2014: "Deep learning applied to speech recognition",
    2017: "Transformer - NLP and signal processing",
    2018: "WaveNet and neural audio synthesis",
    2020: "COVID - remote communication research surge",
    2022: "Diffusion models - image and audio generation",
    2023: "Large multimodal models boom",
}

# ─────────────────────────────────────────────────────────────────────────────
# KEY SUBCATEGORIES
# Physics - 3 most relevant to data science theme
# EESS    - all 4 (only 4 exist)
# ─────────────────────────────────────────────────────────────────────────────

PHYSICS_SUBCATS = {
    "physics.data-an" : "Data Analysis and Statistics",
    "physics.comp-ph" : "Computational Physics",
    "physics.app-ph"  : "Applied Physics",
}

EESS_SUBCATS = {
    "eess.IV": "Image and Video Processing",
    "eess.SP": "Signal Processing",
    "eess.AS": "Audio and Speech Processing",
    "eess.SY": "Systems and Control",
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def prepare_monthly_series(df, field=None, subfield=None):
    """
    Prepares clean monthly time series.
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
    ts      = monthly.set_index("date")["paper_count"]
    ts      = ts.asfreq("MS", fill_value=0)

    return ts, monthly


def find_closest_event(year, events, tolerance=2):
    """
    Matches changepoint year to closest known event.
    """
    closest = min(events.keys(), key=lambda y: abs(y - year))
    if abs(closest - year) <= tolerance:
        return events[closest]
    return None


def detect_changepoints(ts, pen=10):
    """
    PELT changepoint detection on trend component.
    """
    decomp   = seasonal_decompose(ts, model="additive", period=12)
    trend    = decomp.trend.dropna()
    model    = rpt.Pelt(model="rbf").fit(trend.values)
    breaks   = model.predict(pen=pen)
    cp_dates = [trend.index[i-1] for i in breaks[:-1]]
    return cp_dates, decomp


def plot_decomposition(ts, title, color, save_path):
    """
    4-panel decomposition: Observed, Trend, Seasonal, Residual.
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
    print(f" Decomposition saved -> {save_path}")


def plot_changepoints(ts_monthly, changepoint_dates,
                      title, color, label_color,
                      events, save_path):
    """
    Time series with changepoint lines and event labels.
    Top panel    -> series + date labels
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
        event = find_closest_event(cp.year, events)
        label = event if event else "Unknown shift"
        y_pos = 0.75 if i % 2 == 0 else 0.15

        ax_labels.text(
            mdates.date2num(cp), y_pos,
            label,
            ha="center", va="center",
            fontsize=8.5, color="#1A237E",
            style="italic",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor=label_color,
                      edgecolor="#555555",
                      alpha=0.95)
        )

    ax_labels.set_xlabel("Date", fontsize=12)
    plt.tight_layout(h_pad=0)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Changepoint plot saved -> {save_path}")


def run_full_analysis(df, field, field_label,
                      color, label_color,
                      events, subcats,
                      output_folder, pen_overall=10, pen_sub=8):
    """
    Runs the full analysis for one field:
        1. Raw time series
        2. Decomposition
        3. Overall changepoints
        4. Key subcategory changepoints

    Parameters
    ----------
    df           : full dataframe
    field        : field code e.g. "physics"
    field_label  : display name e.g. "Physics"
    color        : line colour for plots
    label_color  : background colour for event labels
    events       : known events dictionary
    subcats      : important subcategories dictionary
    output_folder: where to save plots
    pen_overall  : PELT penalty for overall series
    pen_sub      : PELT penalty for subcategories
    """
    all_results = []

    # ── Data check ────────────────────────────────────────
    rows = df[df["field"] == field]
    print(f"\n  {field_label} rows   : {len(rows):,}")
    print(f"  Subcategories      : {rows['sub_field'].nunique()}")
    print(f"  Total papers       : {rows['paper_count'].sum():,}")

    # ── Raw time series ───────────────────────────────────
    print(f"\n  PART 1 - Raw Time Series")
    ts, monthly = prepare_monthly_series(df, field=field)

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(monthly["date"], monthly["paper_count"],
            color=color, linewidth=1.2, alpha=0.6,
            label="Monthly papers")
    rolling = ts.rolling(window=12, center=True).mean()
    ax.plot(rolling.index, rolling.values,
            color=color, linewidth=2.5, linestyle="--",
            label="12-month rolling average")
    ax.set_title(
        f"{field_label} - Monthly Paper Submissions (1991-2025)",
        fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=13)
    ax.set_ylabel("Papers per Month", fontsize=13)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{field}_overall_raw.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Raw time series saved")

    # ── Decomposition ─────────────────────────────────────
    print(f"\n  PART 2 - Decomposition")
    plot_decomposition(
        ts,
        title=f"{field_label} - Time Series Decomposition\n"
              f"Observed | Trend | Seasonality | Residual",
        color=color,
        save_path=f"{output_folder}/{field}_overall_decomposition.png"
    )

    # ── Overall changepoints ───────────────────────────────
    print(f"\n  PART 3 - Changepoint Detection (Overall)")
    cp_dates, _ = detect_changepoints(ts, pen=pen_overall)

    print(f"  Changepoints detected: {len(cp_dates)}")
    for cp in cp_dates:
        event = find_closest_event(cp.year, events)
        print(f"    -> {cp.strftime('%B %Y')}  "
              f"|  {event or 'No known event nearby'}")

    plot_changepoints(
        monthly, cp_dates,
        title=f"{field_label} - Changepoint Detection (1991-2025)\n"
              f"Red dashed lines show detected structural breaks",
        color=color,
        label_color=label_color,
        events=events,
        save_path=f"{output_folder}/{field}_overall_changepoints.png"
    )

    # Summary
    print(f"\n  CHANGEPOINT SUMMARY - {field_label.upper()}")
    print("  " + "-"*55)
    for i, cp in enumerate(cp_dates, 1):
        before = monthly[monthly["date"] < cp]["paper_count"].mean()
        after  = monthly[monthly["date"] >= cp]["paper_count"].mean()
        change = ((after - before) / before * 100)
        event  = find_closest_event(cp.year, events)
        print(f"\n  Changepoint {i}: {cp.strftime('%B %Y')}")
        print(f"    Avg before   : {before:>8,.0f} papers/month")
        print(f"    Avg after    : {after:>8,.0f} papers/month")
        print(f"    Change       : {change:>+.1f}%")
        print(f"    Likely cause : {event or 'Unknown'}")

    # ── Key subcategories ──────────────────────────────────
    print(f"\n  PART 4 - Key Subcategories")
    print(f"  Analysing {len(subcats)} subcategories:")
    for sub, name in subcats.items():
        print(f"    -> {sub} - {name}")

    for subcat, name in subcats.items():
        print(f"\n  Processing: {subcat} - {name}")

        try:
            ts_sub, monthly_sub = prepare_monthly_series(
                df, subfield=subcat)

            if (len(monthly_sub) < 24 or
                    monthly_sub["paper_count"].sum() < 50):
                print(f"  Not enough data - skipping")
                continue

            # Decomposition
            plot_decomposition(
                ts_sub,
                title=f"{subcat} - {name}\n"
                      f"Time Series Decomposition",
                color=color,
                save_path=(f"{output_folder}/"
                           f"{subcat.replace('.','_')}"
                           f"_decomposition.png")
            )

            # Changepoints
            cp_sub, _ = detect_changepoints(ts_sub, pen=pen_sub)

            print(f"    Changepoints: {len(cp_sub)}")
            for cp in cp_sub:
                event = find_closest_event(cp.year, events)
                print(f"     -> {cp.strftime('%B %Y')} | "
                      f"{event or 'No known event'}")

            plot_changepoints(
                monthly_sub, cp_sub,
                title=f"{subcat} - {name}\n"
                      f"Changepoint Detection (1991-2025)",
                color=color,
                label_color=label_color,
                events=events,
                save_path=(f"{output_folder}/"
                           f"{subcat.replace('.','_')}"
                           f"_changepoints.png")
            )

            # Store results
            for cp in cp_sub:
                before = monthly_sub[
                    monthly_sub["date"] < cp]["paper_count"].mean()
                after  = monthly_sub[
                    monthly_sub["date"] >= cp]["paper_count"].mean()
                change = ((after - before) / before * 100)
                event  = find_closest_event(cp.year, events)
                all_results.append({
                    "sub_field"   : subcat,
                    "name"        : name,
                    "changepoint" : cp.strftime("%B %Y"),
                    "avg_before"  : round(before, 1),
                    "avg_after"   : round(after, 1),
                    "change_pct"  : round(change, 1),
                    "likely_cause": event or "Unknown",
                })

            print(f"   Done")

        except Exception as e:
            print(f"    Error: {e} - skipping")
            continue

    # ── Save summary CSV ───────────────────────────────────
    results_df = pd.DataFrame(all_results)
    csv_path   = f"{output_folder}/{field}_changepoints_summary.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Summary CSV saved -> {csv_path}")
    print(f"  Total changepoints : {len(results_df)}")

    if not results_df.empty:
        print(results_df[["sub_field","name",
                           "changepoint","change_pct",
                           "likely_cause"]]
              .to_string(index=False))

    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("  PHYSICS + EESS - TIME SERIES ANALYSIS")
print("=" * 60)

df = pd.read_csv(r"C:\Users\riyas\OneDrive\ARXIV project\01_data_collection\arxiv_monthly_counts.csv")


# ─────────────────────────────────────────────────────────────────────────────
# RUN PHYSICS ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  PHYSICS")
print("=" * 60)

run_full_analysis(
    df            = df,
    field         = "physics",
    field_label   = "Physics",
    color         = "#4CAF50",
    label_color   = "#E8F5E9",
    events        = PHYSICS_EVENTS,
    subcats       = PHYSICS_SUBCATS,
    output_folder = "plots/timeseries/physics",
    pen_overall   = 10,
    pen_sub       = 8,
)


# ─────────────────────────────────────────────────────────────────────────────
# RUN EESS ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  ELECTRICAL ENGINEERING & SYSTEMS SCIENCE (EESS)")
print("=" * 60)

run_full_analysis(
    df            = df,
    field         = "eess",
    field_label   = "Electrical Engineering (EESS)",
    color         = "#F44336",
    label_color   = "#FFEBEE",
    events        = EESS_EVENTS,
    subcats       = EESS_SUBCATS,
    output_folder = "plots/timeseries/eess",
    pen_overall   = 10,
    pen_sub       = 8,
)


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  ALL DONE!")
print("=" * 60)
print("\n   plots/timeseries/physics/")
print("     physics_overall_raw.png")
print("     physics_overall_decomposition.png")
print("     physics_overall_changepoints.png")
for sub in PHYSICS_SUBCATS:
    s = sub.replace(".","_")
    print(f"     {s}_decomposition.png")
    print(f"     {s}_changepoints.png")
print("     physics_changepoints_summary.csv")
print("\n   plots/timeseries/eess/")
print("     eess_overall_raw.png")
print("     eess_overall_decomposition.png")
print("     eess_overall_changepoints.png")
for sub in EESS_SUBCATS:
    s = sub.replace(".","_")
    print(f"     {s}_decomposition.png")
    print(f"     {s}_changepoints.png")
print("     eess_changepoints_summary.csv")