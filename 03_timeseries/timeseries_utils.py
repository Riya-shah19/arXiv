import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import ruptures as rpt

def resolve_data_path(script_file):
    # Find the CSV file relative to where this script is running
    script_dir = os.path.dirname(os.path.abspath(script_file))
    
    # Try different relative paths depending on how/where we run the script
    candidate_1 = os.path.join(script_dir, "..", "01_data_collection", "arxiv_monthly_counts.csv")
    if os.path.exists(candidate_1):
        return os.path.abspath(candidate_1)
        
    candidate_2 = os.path.join(script_dir, "..", "..", "01_data_collection", "arxiv_monthly_counts.csv")
    if os.path.exists(candidate_2):
        return os.path.abspath(candidate_2)
        
    candidate_3 = os.path.join(script_dir, "01_data_collection", "arxiv_monthly_counts.csv")
    if os.path.exists(candidate_3):
        return os.path.abspath(candidate_3)
        
    # Hardcoded fallback just in case
    return r"C:\Users\riyas\OneDrive\ARXIV project\01_data_collection\arxiv_monthly_counts.csv"

def prepare_monthly_series(df, field=None, subfield=None):
    # Filter by main field or subcategory and keep it <= 2025
    if subfield:
        filtered = df[(df["sub_field"] == subfield) & (df["year"] <= 2025)].copy()
    else:
        filtered = df[(df["field"] == field) & (df["year"] <= 2025)].copy()

    # Group by year and month to get the monthly sum
    monthly = (filtered
               .groupby(["year", "month"])["paper_count"]
               .sum()
               .reset_index())

    # Build datetime index
    monthly["date"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" +
        monthly["month"].astype(str).str.zfill(2) + "-01"
    )

    monthly = monthly.sort_values("date").reset_index(drop=True)
    ts = monthly.set_index("date")["paper_count"]
    ts = ts.asfreq("MS", fill_value=0)

    return ts, monthly

def find_closest_event(year, events, tolerance=2):
    # Look up historic events that happened near the changepoint year
    if not events:
        return None
    closest = min(events.keys(), key=lambda y: abs(y - year))
    if abs(closest - year) <= tolerance:
        return events[closest]
    return None

class DetectionResult(tuple):
    # Custom tuple so we don't break unpacking in existing scripts
    def __new__(cls, cp_dates, decomp, cost):
        return tuple.__new__(cls, (cp_dates, decomp))
    
    def __init__(self, cp_dates, decomp, cost):
        self.cost = cost

def detect_changepoints(ts, pen=10, model_type="rbf"):
    # Decompose into trend/seasonal/residual
    decomp = seasonal_decompose(ts, model="additive", period=12)
    
    # Remove seasonality (just trend + noise)
    trend_plus_resid = (decomp.trend + decomp.resid).dropna()
    
    # Run PELT search on the deseasonalized series
    model = rpt.Pelt(model=model_type).fit(trend_plus_resid.values)
    breaks = model.predict(pen=pen)
    
    # Calculate sum of segment costs
    total_cost = model.cost.sum_of_costs(breaks)
    
    cp_dates = [trend_plus_resid.index[i-1] for i in breaks[:-1]]
    return DetectionResult(cp_dates, decomp, total_cost)

def plot_decomposition(ts, title, color, save_path):
    # Subplot for each component of the timeseries
    decomp = seasonal_decompose(ts, model="additive", period=12)

    fig, axes = plt.subplots(4, 1, figsize=(18, 12))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    components = [
        (ts, color, "Observed\n(raw data)"),
        (decomp.trend, "#2196F3", "Trend\n(long-term direction)"),
        (decomp.seasonal, "#FF9800", "Seasonality\n(yearly pattern)"),
        (decomp.resid, "#F44336", "Residual\n(noise)"),
    ]

    for ax, (data, col, label) in zip(axes, components):
        ax.plot(data.index, data.values, color=col, linewidth=1.5)
        ax.set_ylabel(label, fontsize=9, labelpad=10)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        sns.despine(ax=ax)

    axes[-1].set_xlabel("Date", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Decomposition saved -> {save_path}")

def plot_changepoints(ts_monthly, changepoint_dates, title, color, save_path, events=None, label_bg_color="#E8EAF6", show_event_labels=True):
    # Setup plot with or without bottom labels
    if show_event_labels and events:
        fig, (ax, ax_labels) = plt.subplots(
            2, 1, figsize=(20, 9),
            gridspec_kw={"height_ratios": [5, 1]}
        )
    else:
        fig, ax = plt.subplots(figsize=(20, 7.5))
        ax_labels = None

    # Plot monthly raw count
    ax.plot(ts_monthly["date"], ts_monthly["paper_count"],
            color=color, linewidth=1.2, alpha=0.5,
            label="Monthly papers")

    # Overlay rolling average
    rolling = (ts_monthly.set_index("date")["paper_count"]
               .rolling(window=12, center=True).mean())
    ax.plot(rolling.index, rolling.values,
            color=color, linewidth=2.5,
            label="12-month rolling average")

    y_min = ts_monthly["paper_count"].min()
    y_max = ts_monthly["paper_count"].max()
    y_range = y_max - y_min

    # Alternating heights for label text to prevent overlapping
    date_heights = [0.92, 0.78, 0.92, 0.78, 0.92, 0.78]

    # Only draw date badges on lines if we have a clean number of changepoints
    show_line_labels = len(changepoint_dates) <= 15

    for i, cp in enumerate(changepoint_dates):
        ax.axvline(x=cp, color="#F44336", linewidth=2,
                   linestyle="--", alpha=0.8,
                   label="Changepoint" if i == 0 else "")

        if show_line_labels:
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
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlim(ts_monthly["date"].min(), ts_monthly["date"].max())
    
    # Tick every 2 years
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.tick_params(axis='x', labelsize=10)
    
    sns.despine(ax=ax)

    # Put matching event boxes in bottom timeline
    if show_event_labels and ax_labels is not None:
        ax_labels.set_xlim(ax.get_xlim())
        ax_labels.set_ylim(0, 1)
        ax_labels.axis("off")

        for i, cp in enumerate(changepoint_dates):
            event = find_closest_event(cp.year, events) if events else None
            label = event if event else "Unknown shift"
            y_pos = 0.75 if i % 2 == 0 else 0.15

            ax_labels.text(
                mdates.date2num(cp), y_pos,
                label.replace("\n", " "),
                ha="center", va="center",
                fontsize=8.5, color="#1A237E",
                style="italic",
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor=label_bg_color,
                          edgecolor="#555555",
                          alpha=0.95)
            )
        ax.set_xlabel("")
        ax_labels.set_xlabel("Date", fontsize=12)
    else:
        ax.set_xlabel("Date", fontsize=12)

    plt.tight_layout(h_pad=0)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Changepoint plot saved -> {save_path}")
