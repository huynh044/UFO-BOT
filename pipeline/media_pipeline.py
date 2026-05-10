"""
UFO RAG Pipeline - Step 2: Media Processing (media_file column)
===============================================================
This pipeline:
  1. Reads UFO.xlsx and extracts all rows where `media_file` is not null
  2. Does NOT read the physical .mp4 file — content comes from `description`
  3. Generates a structured Markdown document from the row's metadata
  4. Saves output as <media_file>.md in the output directory
  5. Generates a manifest JSON with full metadata for each processed file

Markdown format per file:
  - YAML-like comment header (for traceability)
  - Human-readable body built from description + all metadata fields
  - file_location (Google Drive link) embedded so the chatbot can surface it
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
OUTPUT_DIR    = BASE_DIR / "output" / "media_markdown"
MANIFEST_PATH = BASE_DIR / "output" / "media_manifest.json"
LOG_PATH      = BASE_DIR / "output" / "media_pipeline.log"

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
    """Load the Excel file and return rows where media_file is not null."""
    log.info(f"Loading Excel: {path}")
    df = pd.read_excel(path, header=1)
    df.columns = df.columns.str.strip()
    df_media = df[df["media_file"].notna()].copy()
    df_media.reset_index(drop=True, inplace=True)
    log.info(f"Found {len(df_media)} rows with media_file (video entries)")
    return df_media


def build_metadata(row: pd.Series) -> dict:
    """Extract structured metadata from an Excel row."""
    return {
        "id":                safe_str(row.get("ID")),
        "media_file":        safe_str(row.get("media_file")),
        "agency":            safe_str(row.get("agency")),
        "description":       safe_str(row.get("description")),
        "release_date":      safe_str(row.get("release_date")),
        "incident_date":     safe_str(row.get("incident_date")),
        "incident_location": safe_str(row.get("incident_location")),
        "file_location":     safe_str(row.get("file_location")),
        "source_type":       "media_video",
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
        link_block = f"[▶ View / Download Media File]({metadata['file_location']})"
    else:
        link_block = "*Link not yet available.*"

    markdown = f"""# UAP Media Record — {metadata['media_file']}

## Overview

| Field | Value |
|---|---|
| **Agency / Source** | {metadata['agency'] or 'Unknown'} |
| **Source Type**     | Video / Media |
| **File Name**       | `{metadata['media_file']}.mp4` |
{optional_block}

## Description

{metadata['description'] or '*No description available.*'}

## Media Source

{link_block}
"""
    return markdown.strip()


def save_markdown(media_file: str, markdown: str, metadata: dict) -> Path:
    """
    Save the generated Markdown to a .md file.
    Prepend a YAML-like comment header block for traceability.
    """
    out_path = OUTPUT_DIR / f"{media_file}.md"

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
    log.info("UFO RAG Pipeline - Media Processing (media_file)")
    log.info("=" * 60)

    df_media = load_excel(EXCEL_PATH)

    all_records   = []
    success_count = 0
    fail_count    = 0

    for i, row in df_media.iterrows():
        media_file = safe_str(row["media_file"])
        metadata   = build_metadata(row)

        log.info(f"\n[{i+1}/{len(df_media)}] Processing: {media_file}")

        # ── Skip if already generated ──────────────────────────────────────
        out_path = OUTPUT_DIR / f"{media_file}.md"
        if out_path.exists():
            log.info(f"  -> Already exists, skipping. ({out_path.name})")
            all_records.append({
                "media_file":  media_file,
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
            out_path = save_markdown(media_file, markdown, metadata)
            log.info(f"  -> Saved: {out_path}")

            all_records.append({
                "media_file":  media_file,
                "metadata":    metadata,
                "status":      "success",
                "output_path": str(out_path),
                "error":       None,
            })
            success_count += 1

        except Exception as e:
            log.error(f"  -> FAILED: {e}")
            all_records.append({
                "media_file":  media_file,
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
    log.info(f"  Total media : {len(all_records)}")
    log.info(f"  Success     : {success_count}")
    log.info(f"  Failed      : {fail_count}")
    log.info(f"  Output dir  : {OUTPUT_DIR}")
    log.info(f"  Manifest    : {MANIFEST_PATH}")
    log.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
