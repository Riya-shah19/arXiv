import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir(r"C:\Users\riyas\OneDrive\ARXIV project")

# Resolve path if running inside the 03_timeseries folder
prefix = "../" if not os.path.exists("plots") and os.path.exists("../plots") else ""

csv_files = {
    'cs': f'{prefix}plots/timeseries/cs/cs_changepoints_summary.csv',
    'stat': f'{prefix}plots/timeseries/stat/stat_changepoints_summary.csv',
    'math': f'{prefix}plots/timeseries/math/math_changepoints_summary.csv',
    'physics': f'{prefix}plots/timeseries/physics/physics_changepoints_summary.csv',
    'eess': f'{prefix}plots/timeseries/eess/eess_changepoints_summary.csv'
}

# Load summaries for all fields
dataframes = []
for field, path in csv_files.items():
    if os.path.exists(path):
        df_temp = pd.read_csv(path)
        df_temp['field'] = field
        dataframes.append(df_temp)
    else:
        print(f"File not found: {path}")

# Combine them into one dataframe
combined_df = pd.concat(dataframes, ignore_index=True)

# Parse changepoint string to datetime
combined_df['date'] = pd.to_datetime(combined_df['changepoint'], format='%B %Y')
combined_df = combined_df.sort_values('date').reset_index(drop=True)

# Find shared changepoints (shifting within 6 months of each other)
shared_groups = []
current_group = []

for idx, row in combined_df.iterrows():
    if not current_group:
        current_group.append(row)
    else:
        dt_diff = (row['date'].year - current_group[0]['date'].year) * 12 + (row['date'].month - current_group[0]['date'].month)
        if abs(dt_diff) <= 6:
            current_group.append(row)
        else:
            if len(current_group) >= 2:
                shared_groups.append(current_group)
            current_group = [row]

if len(current_group) >= 2:
    shared_groups.append(current_group)

# Print a report on transition waves
print("SHARED CHANGEPOINTS / TRANSITION WAVES:")
print("=" * 60)
for idx, grp in enumerate(shared_groups, 1):
    dates = [r['date'].strftime('%b %Y') for r in grp]
    fields = sorted(list(set([r['field'].upper() for r in grp])))
    subfields = [r['sub_field'] for r in grp]
    causes = list(set([r['likely_cause'] for r in grp if str(r['likely_cause']).lower() != 'unknown']))
    
    print(f"Wave {idx}: Centered around {dates[0]} to {dates[-1]}")
    print(f"  Fields: {', '.join(fields)}")
    print(f"  Subcategories: {', '.join(subfields)}")
    if causes:
        print(f"  Likely Drivers: {', '.join(causes)}")
    print("-" * 60)

# Save combined results
combined_df.to_csv(f"{prefix}plots/timeseries/cross_field_changepoints_master.csv", index=False)

# Setup timeline plot
plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid")

fields_order = ['cs', 'stat', 'math', 'physics', 'eess']
colors = {
    'cs': '#1f77b4',
    'stat': '#ff7f0e',
    'math': '#9467bd',
    'physics': '#2ca02c',
    'eess': '#d62728'
}

# Draw timeline tracks
for y_idx, field in enumerate(fields_order):
    y_pos = len(fields_order) - 1 - y_idx
    plt.axhline(y=y_pos, color='#e5e7eb', linestyle='-', linewidth=1.5, zorder=1)
    
    field_data = combined_df[combined_df['field'] == field]
    plt.scatter(field_data['date'], [y_pos] * len(field_data), 
                color=colors[field], s=120, edgecolors='white', linewidths=1.5, zorder=3)
    
    # Alternating labels to avoid overlapping text
    for i, (_, row) in enumerate(field_data.iterrows()):
        offset = 0.15 if i % 2 == 0 else -0.25
        va = 'bottom' if i % 2 == 0 else 'top'
        plt.text(row['date'], y_pos + offset, row['sub_field'], 
                 ha='center', va=va, fontsize=8, color='#4b5563', alpha=0.85, rotation=15)

# Shade transition windows on the timeline
for grp in shared_groups:
    min_date = min([r['date'] for r in grp]) - pd.DateOffset(months=3)
    max_date = max([r['date'] for r in grp]) + pd.DateOffset(months=3)
    plt.axvspan(min_date, max_date, color='#3b82f6', alpha=0.08, zorder=0)
    
    causes = [r['likely_cause'] for r in grp if str(r['likely_cause']).lower() != 'unknown']
    if causes:
        center_date = min_date + (max_date - min_date) / 2
        label = causes[0].split(" - ")[0].split(" - ")[0]
        if len(label) > 28:
            label = label[:25] + "..."
        plt.text(center_date, -0.4, label, ha='center', va='bottom',
                 fontsize=8.5, color='#1e3a8a', fontweight='semibold', rotation=90, alpha=0.8)

# Format timeline axes
plt.yticks(range(len(fields_order)), [f.upper() for f in reversed(fields_order)], fontweight='bold', fontsize=11)
plt.ylim(-0.5, len(fields_order) - 0.3)
plt.xlim(pd.Timestamp('2000-01-01'), pd.Timestamp('2026-01-01'))

plt.title("Cross-Field arXiv Changepoint Comparison (2000 - 2025)", fontsize=14, pad=20, fontweight='bold')
plt.xlabel("Year", fontsize=12)

sns.despine(left=True, bottom=True)
plt.gca().tick_params(left=False)
plt.tight_layout()

# Save final comparison image
plt.savefig(f"{prefix}plots/timeseries/cross_field_comparison.png", dpi=180, bbox_inches='tight')
plt.close()
print(f"Timeline plot saved -> {prefix}plots/timeseries/cross_field_comparison.png")
