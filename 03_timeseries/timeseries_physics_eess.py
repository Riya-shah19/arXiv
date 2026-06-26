import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import timeseries_utils as tu

# Major historic milestones in Physics and EESS
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

PHYSICS_SUBCATS = {
    "physics.data-an": "Data Analysis and Statistics",
    "physics.comp-ph": "Computational Physics",
    "physics.app-ph": "Applied Physics",
}

EESS_SUBCATS = {
    "eess.IV": "Image and Video Processing",
    "eess.SP": "Signal Processing",
    "eess.AS": "Audio and Speech Processing",
    "eess.SY": "Systems and Control",
}

def run_full_analysis(df, field, field_label, color, label_bg_color, events, subcats, output_folder, pen_overall=10, pen_sub=8):
    all_results = []
    
    rows = df[df["field"] == field]
    print(f"\n  {field_label} rows   : {len(rows):,}")
    print(f"  Subcategories      : {rows['sub_field'].nunique()}")
    print(f"  Total papers       : {rows['paper_count'].sum():,}")
    
    # 1. Raw time series plot
    print(f"\n  PART 1 - Raw Time Series")
    ts, monthly = tu.prepare_monthly_series(df, field=field)
    
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(monthly["date"], monthly["paper_count"], color=color, linewidth=1.2, alpha=0.6, label="Monthly papers")
    
    rolling = ts.rolling(window=12, center=True).mean()
    ax.plot(rolling.index, rolling.values, color=color, linewidth=2.5, linestyle="--", label="12-month rolling average")
    
    ax.set_title(f"{field_label} - Monthly Paper Submissions (1991-2025)", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=13)
    ax.set_ylabel("Papers per Month", fontsize=13)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{field}_overall_raw.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Raw time series saved")
    
    # 2. Plot seasonal decomposition
    print(f"\n  PART 2 - Decomposition")
    tu.plot_decomposition(
        ts,
        title=f"{field_label} - Time Series Decomposition\nObserved | Trend | Seasonality | Residual",
        color=color,
        save_path=f"{output_folder}/{field}_overall_decomposition.png"
    )
    
    # 3. Overall changepoint detection
    print(f"\n  PART 3 - Changepoint Detection (Overall)")
    detection_res = tu.detect_changepoints(ts, pen=pen_overall)
    cp_dates = detection_res[0]
    print(f"  Changepoints detected: {len(cp_dates)} | Segmentation Cost: {detection_res.cost:,.2f}")
    for cp in cp_dates:
        event = tu.find_closest_event(cp.year, events)
        print(f"    -> {cp.strftime('%B %Y')}  |  {event or 'No known event nearby'}")
        
    tu.plot_changepoints(
        monthly, cp_dates,
        title=f"{field_label} - Changepoint Detection (1991-2025)\nRed dashed lines show detected structural breaks",
        color=color,
        save_path=f"{output_folder}/{field}_overall_changepoints.png",
        events=events,
        label_bg_color=label_bg_color
    )
    
    # Print stats before/after each change
    print(f"\n  CHANGEPOINT SUMMARY - {field_label.upper()}")
    print("  " + "-" * 55)
    for i, cp in enumerate(cp_dates, 1):
        before = monthly[monthly["date"] < cp]["paper_count"].mean()
        after = monthly[monthly["date"] >= cp]["paper_count"].mean()
        change = ((after - before) / before * 100)
        event = tu.find_closest_event(cp.year, events)
        print(f"\n  Changepoint {i}: {cp.strftime('%B %Y')}")
        print(f"    Avg before   : {before:>8,.0f} papers/month")
        print(f"    Avg after    : {after:>8,.0f} papers/month")
        print(f"    Change       : {change:>+.1f}%")
        print(f"    Likely cause : {event or 'Unknown'}")
        
    # 4. Process individual subcategories
    print(f"\n  PART 4 - Key Subcategories")
    for subcat, name in subcats.items():
        print(f"\n  Processing: {subcat} - {name}")
        
        try:
            ts_sub, monthly_sub = tu.prepare_monthly_series(df, subfield=subcat)
            if len(monthly_sub) < 24 or monthly_sub["paper_count"].sum() < 50:
                print("    Insufficient data - skipping.")
                continue
                
            sub_clean = subcat.replace(".", "_")
            
            # Decompose subcategory series
            tu.plot_decomposition(
                ts_sub,
                title=f"{subcat} - {name}\nTime Series Decomposition",
                color=color,
                save_path=f"{output_folder}/{sub_clean}_decomposition.png"
            )
            
            # Subcategory PELT search
            detection_res = tu.detect_changepoints(ts_sub, pen=pen_sub)
            cp_sub = detection_res[0]
            print(f"    Changepoints found: {len(cp_sub)} | Segmentation Cost: {detection_res.cost:,.2f}")
            for cp in cp_sub:
                event = tu.find_closest_event(cp.year, events)
                print(f"     -> {cp.strftime('%B %Y')} | {event or 'No known event'}")
                
            tu.plot_changepoints(
                monthly_sub, cp_sub,
                title=f"{subcat} - {name}\nChangepoint Detection (1991-2025)",
                color=color,
                save_path=f"{output_folder}/{sub_clean}_changepoints.png",
                events=events,
                label_bg_color=label_bg_color
            )
            
            for cp in cp_sub:
                before = monthly_sub[monthly_sub["date"] < cp]["paper_count"].mean()
                after = monthly_sub[monthly_sub["date"] >= cp]["paper_count"].mean()
                change = ((after - before) / before * 100)
                event = tu.find_closest_event(cp.year, events)
                
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
        csv_path = f"{output_folder}/{field}_changepoints_summary.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\n  Summary CSV saved -> {csv_path}")
        print(results_df[["sub_field", "name", "changepoint", "change_pct", "likely_cause"]].to_string(index=False))

def main():
    sns.set_theme(style="whitegrid")
    os.makedirs("plots/timeseries/physics", exist_ok=True)
    os.makedirs("plots/timeseries/eess", exist_ok=True)
    
    data_path = tu.resolve_data_path(__file__)
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    print("=" * 60)
    print("  PHYSICS + EESS - TIME SERIES ANALYSIS")
    print("=" * 60)
    
    # Physics Run
    print("\n" + "=" * 60)
    print("  PHYSICS ANALYSIS")
    print("=" * 60)
    run_full_analysis(
        df=df,
        field="physics",
        field_label="Physics",
        color="#4CAF50",
        label_bg_color="#E8F5E9",
        events=PHYSICS_EVENTS,
        subcats=PHYSICS_SUBCATS,
        output_folder="plots/timeseries/physics",
        pen_overall=10,
        pen_sub=8
    )
    
    # EESS Run
    print("\n" + "=" * 60)
    print("  ELECTRICAL ENGINEERING & SYSTEMS SCIENCE (EESS) ANALYSIS")
    print("=" * 60)
    run_full_analysis(
        df=df,
        field="eess",
        field_label="Electrical Engineering (EESS)",
        color="#F44336",
        label_bg_color="#FFEBEE",
        events=EESS_EVENTS,
        subcats=EESS_SUBCATS,
        output_folder="plots/timeseries/eess",
        pen_overall=10,
        pen_sub=8
    )
    
    print("\n============================================================")
    print("  PHYSICS & EESS TIME SERIES COMPLETE!")
    print("============================================================")

if __name__ == "__main__":
    main()