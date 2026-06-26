import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import timeseries_utils as tu

# Historic events relevant to mathematics
KNOWN_EVENTS = {
    1991: "arXiv launched",
    2000: "arXiv math section created",
    2008: "Global financial crisis (math finance)",
    2012: "Deep learning revolution begins",
    2015: "ML + math convergence",
    2020: "COVID-19 (online research surge)",
    2022: "ChatGPT released",
}

def main():
    sns.set_theme(style="whitegrid")
    os.makedirs("plots/timeseries", exist_ok=True)

    data_path = tu.resolve_data_path(__file__)
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    # Filter math data up to 2025
    math_df = df[(df["field"] == "math") & (df["year"] <= 2025)].copy()

    print("=" * 60)
    print("  MATHEMATICS TIME SERIES ANALYSIS")
    print("=" * 60)
    print(f"  Math rows found: {len(math_df):,}")

    ts, math_monthly = tu.prepare_monthly_series(df, field="math")

    print(f"  Date range  : {math_monthly['date'].min().strftime('%b %Y')} - {math_monthly['date'].max().strftime('%b %Y')}")
    print(f"  Total months: {len(math_monthly)}")
    print(f"  Total papers: {math_monthly['paper_count'].sum():,}")

    # Plot monthly raw count + rolling average
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(math_monthly["date"], math_monthly["paper_count"],
            color="#9C27B0", linewidth=1.5, alpha=0.8, label="Monthly papers")

    rolling_avg = math_monthly["paper_count"].rolling(window=12, center=True).mean()
    ax.plot(math_monthly["date"], rolling_avg,
            color="#4A148C", linewidth=2.5, linestyle="--", label="12-month rolling average")

    ax.set_title("Mathematics - Monthly Paper Submissions (1991-2025)", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=13)
    ax.set_ylabel("Papers per Month", fontsize=13)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    sns.despine()
    plt.tight_layout()
    plt.savefig("plots/timeseries/math_ts_raw.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(" Raw time series plot saved")

    # Time series decomposition
    tu.plot_decomposition(
        ts,
        title="Mathematics - Time Series Decomposition\nObserved | Trend | Seasonality | Residual",
        color="#9C27B0",
        save_path="plots/timeseries/math_ts_decomposition.png"
    )

    # Detect structural breaks using PELT
    detection_res = tu.detect_changepoints(ts, pen=10)
    cp_dates = detection_res[0]
    print(f"\n  Changepoints detected: {len(cp_dates)} | Segmentation Cost: {detection_res.cost:,.2f}")
    for cp in cp_dates:
        event = tu.find_closest_event(cp.year, KNOWN_EVENTS)
        print(f"   -> {cp.strftime('%b %Y')} | Likely cause: {event or 'Unknown'}")

    tu.plot_changepoints(
        math_monthly, cp_dates,
        title="Mathematics - Changepoint Detection (1991-2025)\nRed dashed lines show detected structural breaks",
        color="#9C27B0",
        save_path="plots/timeseries/math_ts_changepoints.png",
        events=KNOWN_EVENTS,
        label_bg_color="#E8EAF6"
    )

    # Output stats before/after each change point
    print("\n" + "=" * 55)
    print("  CHANGEPOINT SUMMARY - MATHEMATICS")
    print("=" * 55)
    for i, cp in enumerate(cp_dates, 1):
        before = math_monthly[math_monthly["date"] < cp]["paper_count"].mean()
        after = math_monthly[math_monthly["date"] >= cp]["paper_count"].mean()
        change = ((after - before) / before * 100)
        event = tu.find_closest_event(cp.year, KNOWN_EVENTS)

        print(f"\n  Changepoint {i}: {cp.strftime('%B %Y')}")
        print(f"    Avg before : {before:>8,.0f} papers/month")
        print(f"    Avg after  : {after:>8,.0f} papers/month")
        print(f"    Change     : {change:>+.1f}%")
        print(f"    Likely cause: {event or 'No known event nearby'}")

if __name__ == "__main__":
    main()