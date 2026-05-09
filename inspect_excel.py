import pandas as pd
import sys

EXCEL_PATH = r'd:\UFO files\Data_files\UFO.xlsx'

df = pd.read_excel(EXCEL_PATH, header=1)

print("=== COLUMNS ===", flush=True)
print(df.columns.tolist(), flush=True)

print("\n=== SHAPE ===", flush=True)
print(df.shape, flush=True)

print("\n=== name_file non-null ===", flush=True)
name_files = df['name_file'].dropna().tolist()
print(f"Count: {len(name_files)}", flush=True)
for n in name_files[:25]:
    print(f"  {repr(n)}", flush=True)

print("\n=== media_file non-null ===", flush=True)
media_files = df['media_file'].dropna().tolist()
print(f"Count: {len(media_files)}", flush=True)
for n in media_files[:10]:
    print(f"  {repr(n)}", flush=True)

print("\n=== img_file non-null ===", flush=True)
img_files = df['img_file'].dropna().tolist()
print(f"Count: {len(img_files)}", flush=True)
for n in img_files[:10]:
    print(f"  {repr(n)}", flush=True)

print("\n=== Full sample row (first with name_file) ===", flush=True)
row = df[df['name_file'].notna()].iloc[0]
for col, val in row.items():
    print(f"  {col}: {repr(val)}", flush=True)

sys.stdout.flush()
