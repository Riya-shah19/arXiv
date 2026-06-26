import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import timeseries_utils as tu

# Historic developments in statistics/data science
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

STAT_NAMES = {
    "stat.AP": "Applications",
    "stat.CO": "Computation",
    "stat.ME": "Methodology",
    "stat.ML": "Machine Learning",
    "stat.OT": "Other Statistics",
    "stat.TH": "Theory",
}

IMPORTANT_SUBCATS = {
    "stat.ML": "Machine Learning",
    "stat.ME": "Methodology",
    "stat.TH": "Theory",
    "stat.AP": "Applications",
}

def analyze_overall_stat(df):
    print("\n" + "-" * 60)
    print("  Overall Statistics Raw Time Series Analysis")
    print("-" * 60)
    
    ts, monthly = tu.prepare_monthly_series(df, field="stat")
    
    # Plot monthly raw count + rolling average
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(monthly["date"], monthly["paper_count"], color="#FFCC80", linewidth=1.2, alpha=0.6, label="Monthly papers")
    
    rolling = ts.rolling(window=12, center=True).mean()
    ax.plot(rolling.index, rolling.values, color="#FF9800", linewidth=2.5, linestyle="--", label="12-month rolling average")
    
    ax.set_title("Statistics - Monthly Paper Submissions (1991-2025)", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=13)
    ax.set_ylabel("Papers per Month", fontsize=13)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    sns.despine()
    plt.tight_layout()
    plt.savefig("plots/timeseries/stat/stat_overall_raw.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Raw time series saved.")
    
    # Run and save decomposition
    tu.plot_decomposition(
        ts,
        title="Statistics - Time Series Decomposition\nObserved | Trend | Seasonality | Residual",
        color="#FF9800",
        save_path="plots/timeseries/stat/stat_overall_decomposition.png"
    )
    
    # Detect changepoints
    detection_res = tu.detect_changepoints(ts, pen=10)
    cp_dates = detection_res[0]
    print(f"  Changepoints detected: {len(cp_dates)} | Segmentation Cost: {detection_res.cost:,.2f}")
    for cp in cp_dates:
        event = tu.find_closest_event(cp.year, KNOWN_EVENTS)
        print(f"    -> {cp.strftime('%B %Y')}  |  {event or 'No known event nearby'}")
        
    tu.plot_changepoints(
        monthly, cp_dates,
        title="Statistics - Changepoint Detection (1991-2025)\nRed dashed lines show detected structural breaks",
        color="#FF9800",
        save_path="plots/timeseries/stat/stat_overall_changepoints.png",
        events=KNOWN_EVENTS,
        label_bg_color="#E8EAF6"
    )
    
    # Print stats summary
    print("\n  CHANGEPOINT SUMMARY - OVERALL STATISTICS")
    print("  " + "-" * 55)
    for i, cp in enumerate(cp_dates, 1):
        before = monthly[monthly["date"] < cp]["paper_count"].mean()
        after = monthly[monthly["date"] >= cp]["paper_count"].mean()
        change = ((after - before) / before * 100)
        event = tu.find_closest_event(cp.year, KNOWN_EVENTS)
        print(f"\n  Changepoint {i}: {cp.strftime('%B %Y')}")
        print(f"    Avg before   : {before:>8,.0f} papers/month")
        print(f"    Avg after    : {after:>8,.0f} papers/month")
        print(f"    Change       : {change:>+.1f}%")
        print(f"    Likely cause : {event or 'Unknown'}")

def analyze_stat_subcategories(df):
    print("\n" + "-" * 60)
    print("  Statistics Subcategories Analysis")
    print("-" * 60)
    
    all_results = []
    
    for subcat, name in IMPORTANT_SUBCATS.items():
        print(f"\n  Processing: {subcat} - {name}")
        
        try:
            ts, monthly = tu.prepare_monthly_series(df, subfield=subcat)
            if len(monthly) < 24 or monthly["paper_count"].sum() < 50:
                print("    Insufficient data - skipping.")
                continue
                
            sub_clean = subcat.replace(".", "_")
            
            # Subfield decomposition
            tu.plot_decomposition(
                ts,
                title=f"{subcat} - {name}\nTime Series Decomposition",
                color="#FF9800",
                save_path=f"plots/timeseries/stat/{sub_clean}_decomposition.png"
            )
            
            # Subfield PELT changepoints
            detection_res = tu.detect_changepoints(ts, pen=8)
            cp_dates = detection_res[0]
            print(f"    Changepoints found: {len(cp_dates)} | Segmentation Cost: {detection_res.cost:,.2f}")
            
            tu.plot_changepoints(
                monthly, cp_dates,
                title=f"{subcat} - {name}\nChangepoint Detection (1991-2025)",
                color="#FF9800",
                save_path=f"plots/timeseries/stat/{sub_clean}_changepoints.png",
                events=KNOWN_EVENTS,
                label_bg_color="#E8EAF6"
            )
            
            for cp in cp_dates:
                before = monthly[monthly["date"] < cp]["paper_count"].mean()
                after = monthly[monthly["date"] >= cp]["paper_count"].mean()
                change = ((after - before) / before * 100)
                event = tu.find_closest_event(cp.year, KNOWN_EVENTS)
                
                all_results.append({
                    "sub_field": subcat,
                    "name": name,
                    "changepoint": cp.strftime("%B %Y"),
                    "avg_before": round(before, 1),
                    "avg_after": round(after, 1),
                    "change_pct": round(change, 1),
                    "likely_cause": event or "Unknown",
                })
        except Exception as e:
            print(f"    Error processing {subcat}: {e}")
            continue
            
    if all_results:
        results_df = pd.DataFrame(all_results)
        summary_path = "plots/timeseries/stat/stat_changepoints_summary.csv"
        results_df.to_csv(summary_path, index=False)
        print(f"\n  Saved summary table -> {summary_path}")
        print(results_df[["sub_field", "name", "changepoint", "change_pct", "likely_cause"]].to_string(index=False))

def main():
    sns.set_theme(style="whitegrid")
    os.makedirs("plots/timeseries/stat", exist_ok=True)
    
    data_path = tu.resolve_data_path(__file__)
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    stat_rows = df[df["field"] == "stat"]
    print(f"Statistics rows found: {len(stat_rows):,}")
    print(f"Statistics subcategories: {stat_rows['sub_field'].nunique()}")
    print(f"Year range: {stat_rows['year'].min()} - {stat_rows['year'].max()}")
    print(f"Total papers: {stat_rows['paper_count'].sum():,}")
    
    # Show subcategory distribution
    print("\n  Papers by subcategory:")
    subcat_counts = stat_rows.groupby("sub_field")["paper_count"].sum().sort_values(ascending=False)
    for sub, total in subcat_counts.items():
        name = STAT_NAMES.get(sub, sub)
        print(f"    {sub:<12} {name:<30} -> {total:>8,}")
        
    analyze_overall_stat(df)
    analyze_stat_subcategories(df)
    
    print("\n============================================================")
    print("  STATISTICS TIME SERIES COMPLETE!")
    print("============================================================")

if __name__ == "__main__":
    main()