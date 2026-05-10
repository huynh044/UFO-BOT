"""Quick pre-flight: verify all name_file entries resolve to real PDFs on disk."""
import os
import sys
from pathlib import Path
import pandas as pd

BASE_DIR   = Path(r"D:\Project\UFO_BOT\UFO-BOT")
EXCEL_PATH = BASE_DIR / "Data_files" / "UFO.xlsx"
PDF_DIR    = BASE_DIR / "Pdf_files"

df = pd.read_excel(EXCEL_PATH, header=1)
df.columns = df.columns.str.strip()
df_pdf = df[df["name_file"].notna()].copy()

print(f"Total name_file rows: {len(df_pdf)}\n")

found, missing = [], []
for _, row in df_pdf.iterrows():
    name = str(row["name_file"]).strip()
    path = PDF_DIR / f"{name}.pdf"
    if path.exists():
        found.append(name)
    else:
        missing.append(name)

print(f"[FOUND]   {len(found)} files")
print(f"[MISSING] {len(missing)} files")

if missing:
    print("\nMissing files:")
    for m in missing:
        print(f"  - {m}.pdf")

# Also list actual PDFs on disk that are NOT in Excel
actual_pdfs = {p.stem for p in PDF_DIR.glob("*.pdf")}
excel_names = set(str(r["name_file"]).strip() for _, r in df_pdf.iterrows())
untracked = actual_pdfs - excel_names
if untracked:
    print(f"\n[UNTRACKED] {len(untracked)} PDFs on disk not in Excel:")
    for u in sorted(untracked):
        print(f"  - {u}.pdf")
else:
    print("\n[OK] All PDFs on disk are tracked in Excel.")
sys.stdout.flush()
