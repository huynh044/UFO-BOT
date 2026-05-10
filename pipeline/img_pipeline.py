"""
UFO RAG Pipeline - Step 3: Image Processing (img_file column)
=============================================================
This pipeline:
  1. Reads UFO.xlsx and extracts all rows where `img_file` is not null
  2. Does NOT read the physical image file — content comes from `description`
  3. Generates a structured Markdown document from the row's metadata
  4. Saves output as <img_file>.md in the output directory
  5. Generates a manifest JSON with full metadata for each processed file

Note: img_file can refer to .png, .jpg, or .pdf (image-based PDF) files.
The `source_type` field in metadata distinguishes these from text PDFs.
"""

import sys
import json
import logging
import math
from pathlib import Path
from datetime import datetime

import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(r"D:\UFO files")
EXCEL_PATH    = BASE_DIR / "Data_files" / "UFO.xlsx"
OUTPUT_DIR    = BASE_DIR / "output" / "img_markdown"
MANIFEST_PATH = BASE_DIR / "output" / "img_manifest.json"
LOG_PATH      = BASE_DIR / "output" / "img_pipeline.log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def safe_str(val) -> str:
    """Convert pandas value to clean string; return empty string for NaN."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    return str(val).strip()


def load_excel(path: Path) -> pd.DataFrame:
    """Load the Excel file and return rows where img_file is not null."""
    log.info(f"Loading Excel: {path}")
    df = pd.read_excel(path, header=1)
    df.columns = df.columns.str.strip()
    df_img = df[df["img_file"].notna()].copy()
    df_img.reset_index(drop=True, inplace=True)
    log.info(f"Found {len(df_img)} rows with img_file (image entries)")
    return df_img


def build_metadata(row: pd.Series) -> dict:
    """Extract structured metadata from an Excel row."""
    return {
        "id":                safe_str(row.get("ID")),
        "img_file":          safe_str(row.get("img_file")),
        "agency":            safe_str(row.get("agency")),
        "description":       safe_str(row.get("description")),
        "release_date":      safe_str(row.get("release_date")),
        "incident_date":     safe_str(row.get("incident_date")),
        "incident_location": safe_str(row.get("incident_location")),
        "file_location":     safe_str(row.get("file_location")),
        "source_type":       "image_file",
    }


def generate_markdown(metadata: dict) -> str:
    """
    Build a structured Markdown document from metadata fields.
    The `description` column is the main content body.
    `file_location` is embedded as a clickable link for the chatbot to surface.
    """
    # ── Optional fields (only render if not empty) ──────────────────────────
    optional_rows = []
    if metadata["incident_date"]:
        optional_rows.append(f"| **Incident Date**     | {metadata['incident_date']} |")
    if metadata["incident_location"]:
        optional_rows.append(f"| **Incident Location** | {metadata['incident_location']} |")
    if metadata["release_date"]:
        optional_rows.append(f"| **Release Date**      | {metadata['release_date']} |")

    optional_block = "\n".join(optional_rows) if optional_rows else "| *(not available)* | |"

    # ── File link block ─────────────────────────────────────────────────────
    if metadata["file_location"]:
        link_block = f"[🖼 View / Download Image File]({metadata['file_location']})"
    else:
        link_block = "*Link not yet available.*"

    markdown = f"""# UAP Image Record — {metadata['img_file']}

## Overview

| Field | Value |
|---|---|
| **Agency / Source** | {metadata['agency'] or 'Unknown'} |
| **Source Type**     | Image / Photo |
| **File Name**       | `{metadata['img_file']}` |
{optional_block}

## Description

{metadata['description'] or '*No description available.*'}

## Image Source

{link_block}
"""
    return markdown.strip()


def save_markdown(img_file: str, markdown: str, metadata: dict) -> Path:
    """
    Save the generated Markdown to a .md file.
    Prepend a YAML-like comment header block for traceability.
    """
    out_path = OUTPUT_DIR / f"{img_file}.md"

    header_lines = ["<!--- METADATA"]
    for k, v in metadata.items():
        header_lines.append(f"{k}: {v}")
    header_lines.append("--->\n")
    header = "\n".join(header_lines)

    out_path.write_text(header + "\n" + markdown, encoding="utf-8")
    return out_path


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_pipeline():
    log.info("=" * 60)
    log.info("UFO RAG Pipeline - Image Processing (img_file)")
    log.info("=" * 60)

    df_img = load_excel(EXCEL_PATH)

    all_records   = []
    success_count = 0
    fail_count    = 0

    for i, row in df_img.iterrows():
        img_file = safe_str(row["img_file"])
        metadata = build_metadata(row)

        log.info(f"\n[{i+1}/{len(df_img)}] Processing: {img_file}")

        # ── Skip if already generated ──────────────────────────────────────
        out_path = OUTPUT_DIR / f"{img_file}.md"
        if out_path.exists():
            log.info(f"  -> Already exists, skipping. ({out_path.name})")
            all_records.append({
                "img_file":    img_file,
                "metadata":    metadata,
                "status":      "skipped",
                "output_path": str(out_path),
                "error":       None,
            })
            success_count += 1
            continue

        # ── Generate and save ──────────────────────────────────────────────
        try:
            markdown = generate_markdown(metadata)
            out_path = save_markdown(img_file, markdown, metadata)
            log.info(f"  -> Saved: {out_path}")

            all_records.append({
                "img_file":    img_file,
                "metadata":    metadata,
                "status":      "success",
                "output_path": str(out_path),
                "error":       None,
            })
            success_count += 1

        except Exception as e:
            log.error(f"  -> FAILED: {e}")
            all_records.append({
                "img_file":    img_file,
                "metadata":    metadata,
                "status":      "error",
                "output_path": None,
                "error":       str(e),
            })
            fail_count += 1

    # ── Save manifest ────────────────────────────────────────────────────────
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "total":        len(all_records),
        "success":      success_count,
        "failed":       fail_count,
        "records":      all_records,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(f"\n[DONE] Manifest saved: {MANIFEST_PATH}")

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info(f"  Total images : {len(all_records)}")
    log.info(f"  Success      : {success_count}")
    log.info(f"  Failed       : {fail_count}")
    log.info(f"  Output dir   : {OUTPUT_DIR}")
    log.info(f"  Manifest     : {MANIFEST_PATH}")
    log.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
