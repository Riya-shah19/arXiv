import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import timeseries_utils as tu

# Historic events that might explain big shifts
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

CS_NAMES = {
    "cs.AI": "Artificial Intelligence",
    "cs.AR": "Hardware Architecture",
    "cs.CC": "Computational Complexity",
    "cs.CE": "Computational Engineering",
    "cs.CG": "Computational Geometry",
    "cs.CL": "Computation & Language (NLP)",
    "cs.CR": "Cryptography & Security",
    "cs.CV": "Computer Vision",
    "cs.CY": "Computers & Society",
    "cs.DB": "Databases",
    "cs.DC": "Distributed Computing",
    "cs.DL": "Digital Libraries",
    "cs.DM": "Discrete Mathematics",
    "cs.DS": "Data Structures & Algorithms",
    "cs.ET": "Emerging Technologies",
    "cs.FL": "Formal Languages",
    "cs.GL": "General Literature",
    "cs.GR": "Graphics",
    "cs.GT": "Game Theory",
    "cs.HC": "Human-Computer Interaction",
    "cs.IR": "Information Retrieval",
    "cs.IT": "Information Theory",
    "cs.LG": "Machine Learning",
    "cs.LO": "Logic in CS",
    "cs.MA": "Multiagent Systems",
    "cs.MM": "Multimedia",
    "cs.MS": "Mathematical Software",
    "cs.NA": "Numerical Analysis",
    "cs.NE": "Neural & Evolutionary Computing",
    "cs.NI": "Networking & Internet",
    "cs.OH": "Other Computer Science",
    "cs.OS": "Operating Systems",
    "cs.PF": "Performance",
    "cs.PL": "Programming Languages",
    "cs.RO": "Robotics",
    "cs.SC": "Symbolic Computation",
    "cs.SD": "Sound",
    "cs.SE": "Software Engineering",
    "cs.SI": "Social & Information Networks",
    "cs.SY": "Systems & Control",
}

# The subcategories we want to focus on
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

def analyze_overall_cs(df):
    print("\n" + "-" * 60)
    print("  Overall CS Time Series Analysis")
    print("-" * 60)
    
    ts, monthly = tu.prepare_monthly_series(df, field="cs")
    
    print(f"  Date range  : {monthly['date'].min().strftime('%b %Y')} - {monthly['date'].max().strftime('%b %Y')}")
    print(f"  Total months: {len(monthly)}")
    print(f"  Total papers: {monthly['paper_count'].sum():,}")
    
    # Plot raw time series
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(monthly["date"], monthly["paper_count"], color="#90CAF9", linewidth=1.2, alpha=0.6, label="Monthly papers")
    
    rolling = ts.rolling(window=12, center=True).mean()
    ax.plot(rolling.index, rolling.values, color="#2196F3", linewidth=2.5, linestyle="--", label="12-month rolling average")
    
    ax.set_title("Computer Science — Monthly Paper Submissions (1991-2025)", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=13)
    ax.set_ylabel("Papers per Month", fontsize=13)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    sns.despine()
    plt.tight_layout()
    plt.savefig("plots/timeseries/cs/cs_overall_raw.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Raw time series plot saved.")
    
    # Run decomposition plot
    tu.plot_decomposition(
        ts, 
        title="Computer Science - Time Series Decomposition\nObserved | Trend | Seasonality | Residual",
        color="#90CAF9",
        save_path="plots/timeseries/cs/cs_overall_decomposition.png"
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
        title="Computer Science - Changepoint Detection (1991-2025)\nRed dashed lines show detected structural breaks",
        color="#2196F3",
        save_path="plots/timeseries/cs/cs_overall_changepoints.png",
        events=KNOWN_EVENTS
    )
    
    # Print out summary statistics for each shift
    print("\n  CHANGEPOINT SUMMARY - OVERALL CS")
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

def analyze_cs_subcategories(df):
    print("\n" + "-" * 60)
    print("  CS Subcategories Analysis")
    print("-" * 60)
    
    all_results = []
    
    for subcat, name in IMPORTANT_SUBCATS.items():
        print(f"\n  Processing: {subcat} - {name}")
        
        try:
            ts, monthly = tu.prepare_monthly_series(df, subfield=subcat)
            if len(monthly) < 24 or monthly["paper_count"].sum() < 100:
                print("    Insufficient data - skipping.")
                continue
                
            sub_clean = subcat.replace(".", "_")
            
            # Save decomposition
            tu.plot_decomposition(
                ts,
                title=f"{subcat} - {name}\nTime Series Decomposition",
                color="#90CAF9",
                save_path=f"plots/timeseries/cs/{sub_clean}_decomposition.png"
            )
            
            # Run PELT search
            detection_res = tu.detect_changepoints(ts, pen=8)
            cp_dates = detection_res[0]
            print(f"    Changepoints found: {len(cp_dates)} | Segmentation Cost: {detection_res.cost:,.2f}")
            
            tu.plot_changepoints(
                monthly, cp_dates,
                title=f"{subcat} - {name}\nChangepoint Detection (1991-2025)",
                color="#2196F3",
                save_path=f"plots/timeseries/cs/{sub_clean}_changepoints.png",
                events=KNOWN_EVENTS
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
        summary_path = "plots/timeseries/cs/cs_changepoints_summary.csv"
        results_df.to_csv(summary_path, index=False)
        print(f"\n  Saved summary table -> {summary_path}")
        print(results_df[["sub_field", "name", "changepoint", "change_pct", "likely_cause"]].to_string(index=False))

def main():
    sns.set_theme(style="whitegrid")
    os.makedirs("plots/timeseries/cs", exist_ok=True)
    
    data_path = tu.resolve_data_path(__file__)
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    cs_rows = df[df["field"] == "cs"]
    print(f"CS rows found: {len(cs_rows):,}")
    print(f"CS subcategories: {cs_rows['sub_field'].nunique()}")
    print(f"Year range: {cs_rows['year'].min()} - {cs_rows['year'].max()}")
    
    analyze_overall_cs(df)
    analyze_cs_subcategories(df)
    
    print("\n============================================================")
    print("  CS TIME SERIES ANALYSIS COMPLETE!")
    print("============================================================")

if __name__ == "__main__":
    main()
