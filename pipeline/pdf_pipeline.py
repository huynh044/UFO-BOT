"""
UFO RAG Pipeline - Step 1: PDF Processing (name_file column)
============================================================
This pipeline:
  1. Reads UFO.xlsx and extracts all rows where `name_file` is not null
  2. Resolves the physical PDF path from d:/UFO files/Pdf files/<name_file>.pdf
  3. Converts each PDF to Markdown using docling
  4. Saves output as <name_file>.md in the output directory
  5. Generates a manifest JSON with full metadata for each processed file
"""

import os
import sys
import json
import logging
import traceback
from pathlib import Path
from datetime import datetime

import pandas as pd

# ─── Configure HuggingFace / Torch cache before importing docling ─────────────
# Aggressively redirect all possible caches to D drive to protect C drive
CACHE_BASE = Path(r"D:\cache")
os.environ["USERPROFILE"]    = str(CACHE_BASE / "user")
os.environ["LOCALAPPDATA"]   = str(CACHE_BASE / "user" / "Local")
os.environ["APPDATA"]        = str(CACHE_BASE / "user" / "Roaming")
os.environ["TMP"]            = str(CACHE_BASE / "temp")
os.environ["TEMP"]           = str(CACHE_BASE / "temp")

os.environ["HF_HOME"]                    = str(CACHE_BASE / "huggingface")
os.environ["TRANSFORMERS_CACHE"]         = str(CACHE_BASE / "huggingface" / "transformers")
os.environ["HUGGINGFACE_HUB_CACHE"]     = str(CACHE_BASE / "huggingface" / "hub")
os.environ["TORCH_HOME"]                 = str(CACHE_BASE / "torch")
os.environ["TORCH_EXTENSIONS_DIR"]       = str(CACHE_BASE / "torch_extensions")
os.environ["NLTK_DATA"]                 = str(CACHE_BASE / "nltk_data")

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"]   = "1"

# Ensure all cache directories exist
for p in [
    CACHE_BASE / "huggingface", 
    CACHE_BASE / "torch", 
    CACHE_BASE / "temp", 
    CACHE_BASE / "user" / "Local", 
    CACHE_BASE / "user" / "Roaming",
    CACHE_BASE / "torch_extensions",
    CACHE_BASE / "nltk_data"
]:
    p.mkdir(parents=True, exist_ok=True)


import fitz  # PyMuPDF - for counting pages before converting
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.document import InputFormat
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(r"D:\Project\UFO_BOT\UFO-BOT")
EXCEL_PATH   = BASE_DIR / "Data_files" / "UFO.xlsx"
PDF_DIR      = BASE_DIR / "Pdf_files"
OUTPUT_DIR   = BASE_DIR / "output" / "pdf_markdown"
MANIFEST_PATH = BASE_DIR / "output" / "pdf_manifest.json"
LOG_PATH     = BASE_DIR / "output" / "pdf_pipeline.log"

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
    if val is None or (isinstance(val, float) and __import__("math").isnan(val)):
        return ""
    return str(val).strip()


def load_excel(path: Path) -> pd.DataFrame:
    """Load the Excel file and return rows where name_file is not null."""
    log.info(f"Loading Excel: {path}")
    df = pd.read_excel(path, header=1)
    df.columns = df.columns.str.strip()          # strip whitespace from column names
    df_pdf = df[df["name_file"].notna()].copy()
    df_pdf.reset_index(drop=True, inplace=True)
    log.info(f"Found {len(df_pdf)} rows with name_file (PDF entries)")
    return df_pdf


def resolve_pdf_path(name_file: str) -> Path:
    """Resolve the absolute PDF file path from a bare name_file value."""
    return PDF_DIR / f"{name_file}.pdf"


def build_metadata(row: pd.Series) -> dict:
    """Extract structured metadata from an Excel row."""
    return {
        "id":                safe_str(row.get("ID")),
        "name_file":         safe_str(row.get("name_file")),
        "agency":            safe_str(row.get("agency")),
        "description":       safe_str(row.get("description")),
        "release_date":      safe_str(row.get("release_date")),
        "incident_date":     safe_str(row.get("incident_date")),
        "incident_location": safe_str(row.get("incident_location")),
        "file_location":     safe_str(row.get("file_location")),
        "source_type":       "pdf_document",
    }


PAGE_CHUNK_SIZE = 10  # Keep small: docling-parse C++ accumulates memory per page within a single convert() call


def convert_pdf_to_markdown(pdf_path: Path, converter: DocumentConverter) -> str:
    """
    Convert a single PDF to Markdown, processing in chunks of PAGE_CHUNK_SIZE pages.
    This avoids the std::bad_alloc bug in docling-parse's C++ backend which leaks
    memory across pages and crashes on large documents.
    """
    # Count total pages using PyMuPDF (fast, no full parse)
    with fitz.open(str(pdf_path)) as doc:
        total_pages = len(doc)

    log.info(f"    PDF has {total_pages} pages — processing in chunks of {PAGE_CHUNK_SIZE}")

    markdown_parts = []
    for start in range(1, total_pages + 1, PAGE_CHUNK_SIZE):
        end = min(start + PAGE_CHUNK_SIZE - 1, total_pages)
        log.info(f"    -> Chunk pages {start}–{end} ...")
        try:
            result = converter.convert(str(pdf_path), page_range=(start, end))
            markdown_parts.append(result.document.export_to_markdown())
            # CRITICAL: unload C++ backend to free memory after each chunk
            try:
                result.input._backend.unload()
            except Exception:
                pass
        except Exception as e:
            log.warning(f"    -> Chunk {start}–{end} failed: {e}. Skipping chunk.")

    return "\n\n".join(markdown_parts)


def save_markdown(name_file: str, markdown: str, metadata: dict) -> Path:
    """
    Save the Markdown content to a .md file.
    Prepend a YAML-like metadata header comment block for traceability.
    """
    out_path = OUTPUT_DIR / f"{name_file}.md"

    header_lines = ["<!--- METADATA"]
    for k, v in metadata.items():
        header_lines.append(f"{k}: {v}")
    header_lines.append("--->\n")
    header = "\n".join(header_lines)

    out_path.write_text(header + markdown, encoding="utf-8")
    return out_path


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_pipeline():
    log.info("=" * 60)
    log.info("UFO RAG Pipeline - PDF Processing (name_file)")
    log.info("=" * 60)

    df_pdf = load_excel(EXCEL_PATH)

    # ── Pre-flight: scan which files exist on disk ──────────────────────────
    log.info("\n[PRE-FLIGHT] Scanning PDF files on disk...")
    all_records = []
    missing_files = []

    for _, row in df_pdf.iterrows():
        name_file = safe_str(row["name_file"])
        pdf_path  = resolve_pdf_path(name_file)
        exists    = pdf_path.exists()

        record = {
            "name_file": name_file,
            "pdf_path":  str(pdf_path),
            "exists":    exists,
            "metadata":  build_metadata(row),
            "status":    "pending",
            "output_path": None,
            "error": None,
        }
        all_records.append(record)

        if not exists:
            missing_files.append(name_file)
            log.warning(f"  [MISSING] {pdf_path}")
        else:
            log.info(f"  [OK]      {pdf_path.name}")

    log.info(f"\nPre-flight summary: {len(df_pdf) - len(missing_files)}/{len(df_pdf)} files found")
    if missing_files:
        log.warning(f"Missing files ({len(missing_files)}): {missing_files}")

    # ── Initialize docling converter ────────────────────────────────────────
    log.info("\n[INIT] Initializing docling DocumentConverter...")
    pipeline_options = PdfPipelineOptions()

    # Use GPU accelerator
    pipeline_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CUDA)

    # Limit OCR batch size
    pipeline_options.ocr_batch_size = 1

    # CRITICAL: force_backend_text skips C++ image rendering for text-based PDF pages,
    # which is the root cause of the std::bad_alloc preprocess crash
    pipeline_options.force_backend_text       = True

    # Disable all heavy enrichment features to minimize memory usage
    pipeline_options.generate_page_images     = False
    pipeline_options.generate_picture_images  = False
    pipeline_options.generate_table_images    = False
    pipeline_options.do_picture_classification = False
    pipeline_options.do_picture_description   = False
    pipeline_options.do_formula_enrichment    = False
    pipeline_options.do_code_enrichment       = False
    pipeline_options.images_scale             = 1.0

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    log.info("Docling ready.")

    # ── Process each PDF ────────────────────────────────────────────────────
    success_count = 0
    fail_count    = 0

    for i, record in enumerate(all_records):
        name_file = record["name_file"]
        pdf_path  = Path(record["pdf_path"])
        metadata  = record["metadata"]

        log.info(f"\n[{i+1}/{len(all_records)}] Processing: {name_file}")

        # Skip if already converted
        out_path = OUTPUT_DIR / f"{name_file}.md"
        if out_path.exists():
            log.info(f"  -> Already converted, skipping. ({out_path.name})")
            record["status"]      = "skipped"
            record["output_path"] = str(out_path)
            success_count += 1
            continue

        # Skip if file not on disk
        if not record["exists"]:
            log.warning(f"  -> File not found on disk, skipping.")
            record["status"] = "missing"
            fail_count += 1
            continue

        # Convert
        try:
            log.info(f"  -> Converting PDF to Markdown...")
            markdown = convert_pdf_to_markdown(pdf_path, converter)
            out_path = save_markdown(name_file, markdown, metadata)
            log.info(f"  -> Saved: {out_path}")

            record["status"]      = "success"
            record["output_path"] = str(out_path)
            success_count += 1

        except Exception as e:
            log.error(f"  -> FAILED: {e}")
            log.debug(traceback.format_exc())
            record["status"] = "error"
            record["error"]  = str(e)
            fail_count += 1

    # ── Save manifest ───────────────────────────────────────────────────────
    manifest = {
        "generated_at":  datetime.now().isoformat(),
        "total":         len(all_records),
        "success":       success_count,
        "failed":        fail_count,
        "records":       all_records,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"\n[DONE] Manifest saved: {MANIFEST_PATH}")

    # ── Summary ─────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info(f"PIPELINE COMPLETE")
    log.info(f"  Total PDFs  : {len(all_records)}")
    log.info(f"  Success     : {success_count}")
    log.info(f"  Failed/Miss : {fail_count}")
    log.info(f"  Output dir  : {OUTPUT_DIR}")
    log.info(f"  Manifest    : {MANIFEST_PATH}")
    log.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
