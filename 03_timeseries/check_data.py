import pandas as pd
import os

# Check which folder we are running from
print("Running from:", os.getcwd())

# Try to find the CSV file
possible_paths = [
    "arxiv_monthly_counts.csv",
    "../arxiv_monthly_counts.csv",
    "C:/Users/riyas/OneDrive/ARXIV project/arxiv_monthly_counts.csv",
]

for path in possible_paths:
    if os.path.exists(path):
        print(f"\n✅ Found CSV at: {path}")
        df = pd.read_csv(path)
        print(f"   Shape  : {df.shape}")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"   Fields : {df['field'].unique()}")
        print(f"\n   Sample:")
        print(df.head())
        break
else:
    print("\n❌ CSV not found in any location!")
    print("   Files in current folder:")
    for f in os.listdir():
        print(f"     {f}")