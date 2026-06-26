import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import timeseries_utils as tu

# Historic events that might align with shifts in subfields
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

IMPORTANT_SUBCATS = {
    "math.OC": "Optimisation and Control",
    "math.PR": "Probability",
    "math.ST": "Statistics Theory",
    "math.NA": "Numerical Analysis",
    "math.CO": "Combinatorics",
}

def analyze_overall_math(df):
    print("\n" + "-" * 60)
    print("  Overall Mathematics Raw Time Series Analysis")
    print("-" * 60)
    
    ts, monthly = tu.prepare_monthly_series(df, field="math")
    
    # Plot monthly raw series + rolling average
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.plot(monthly["date"], monthly["paper_count"], color="#CE93D8", linewidth=1.2, alpha=0.6, label="Monthly papers")
    
    rolling = ts.rolling(window=12, center=True).mean()
    ax.plot(rolling.index, rolling.values, color="#9C27B0", linewidth=2.5, linestyle="--", label="12-month rolling average")
    
    ax.set_title("Mathematics - Monthly Paper Submissions (1991-2025)", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=13)
    ax.set_ylabel("Papers per Month", fontsize=13)
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    sns.despine()
    plt.tight_layout()
    plt.savefig("plots/timeseries/math/math_overall_raw.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Raw time series saved.")
    
    # Plot seasonal decomposition components
    tu.plot_decomposition(
        ts,
        title="Mathematics - Time Series Decomposition\nObserved | Trend | Seasonality | Residual",
        color="#9C27B0",
        save_path="plots/timeseries/math/math_overall_decomposition.png"
    )
    
    # Detect structural changepoints
    detection_res = tu.detect_changepoints(ts, pen=10)
    cp_dates = detection_res[0]
    print(f"  Changepoints detected: {len(cp_dates)} | Segmentation Cost: {detection_res.cost:,.2f}")
    for cp in cp_dates:
        event = tu.find_closest_event(cp.year, KNOWN_EVENTS)
        print(f"    -> {cp.strftime('%B %Y')}  |  {event or 'No known event nearby'}")
        
    tu.plot_changepoints(
        monthly, cp_dates,
        title="Mathematics - Changepoint Detection (1991-2025)\nRed dashed lines show detected structural breaks",
        color="#9C27B0",
        save_path="plots/timeseries/math/math_overall_changepoints.png",
        events=KNOWN_EVENTS,
        label_bg_color="#F3E5F5"
    )
    
    # Summary of changes
    print("\n  CHANGEPOINT SUMMARY - OVERALL MATHEMATICS")
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

def analyze_math_subcategories(df):
    print("\n" + "-" * 60)
    print("  Mathematics Subcategories Analysis")
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
            
            # Save seasonal decomposition plots
            tu.plot_decomposition(
                ts,
                title=f"{subcat} - {name}\nTime Series Decomposition",
                color="#9C27B0",
                save_path=f"plots/timeseries/math/{sub_clean}_decomposition.png"
            )
            
            # Run PELT search on subcategories
            detection_res = tu.detect_changepoints(ts, pen=8)
            cp_dates = detection_res[0]
            print(f"    Changepoints found: {len(cp_dates)} | Segmentation Cost: {detection_res.cost:,.2f}")
            
            tu.plot_changepoints(
                monthly, cp_dates,
                title=f"{subcat} - {name}\nChangepoint Detection (1991-2025)",
                color="#9C27B0",
                save_path=f"plots/timeseries/math/{sub_clean}_changepoints.png",
                events=KNOWN_EVENTS,
                label_bg_color="#F3E5F5"
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
        summary_path = "plots/timeseries/math/math_changepoints_summary.csv"
        results_df.to_csv(summary_path, index=False)
        print(f"\n  Saved summary table -> {summary_path}")
        print(results_df[["sub_field", "name", "changepoint", "change_pct", "likely_cause"]].to_string(index=False))

def main():
    sns.set_theme(style="whitegrid")
    os.makedirs("plots/timeseries/math", exist_ok=True)
    
    data_path = tu.resolve_data_path(__file__)
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    math_rows = df[df["field"] == "math"]
    print(f"Math rows found: {len(math_rows):,}")
    print(f"Math subcategories: {math_rows['sub_field'].nunique()}")
    print(f"Year range: {math_rows['year'].min()} - {math_rows['year'].max()}")
    print(f"Total papers: {math_rows['paper_count'].sum():,}")
    
    # Print subcategory list
    print("\n  Papers by subcategory:")
    subcat_counts = math_rows.groupby("sub_field")["paper_count"].sum().sort_values(ascending=False)
    for sub, total in subcat_counts.items():
        name = MATH_NAMES.get(sub, sub)
        print(f"    {sub:<12} {name:<30} -> {total:>8,}")
        
    analyze_overall_math(df)
    analyze_math_subcategories(df)
    
    print("\n============================================================")
    print("  MATHEMATICS TIME SERIES COMPLETE!")
    print("============================================================")

if __name__ == "__main__":
    main()
