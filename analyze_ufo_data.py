import pandas as pd
import json

df = pd.read_excel(r'd:\UFO files\Data files\UFO.xlsx', header=None)

# We want to see the first few rows to understand the structure
print("--- First 10 rows of the dataset ---")
for i, row in df.head(10).iterrows():
    print(f"Row {i}: {row.tolist()}")

# Basic info
print("\n--- Basic Info ---")
print(f"Total Rows: {len(df)}")
print(f"Total Columns: {len(df.columns)}")
