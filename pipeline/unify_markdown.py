import shutil
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(r"D:\Project\UFO_BOT\UFO-BOT")
OUTPUT_DIR  = BASE_DIR / "output"
UNIFIED_DIR = BASE_DIR / "output" / "unified_inputs"

SOURCE_DIRS = [
    OUTPUT_DIR / "pdf_markdown",
    OUTPUT_DIR / "img_markdown",
    OUTPUT_DIR / "media_markdown"
]

def unify():
    print(f"--- Unifying Markdown files into: {UNIFIED_DIR} ---")
    
    # Create or clean unified directory
    if UNIFIED_DIR.exists():
        shutil.rmtree(UNIFIED_DIR)
    UNIFIED_DIR.mkdir(parents=True, exist_ok=True)
    
    total_copied = 0
    
    for src in SOURCE_DIRS:
        if not src.exists():
            print(f"Skipping missing directory: {src.name}")
            continue
            
        print(f"Processing {src.name}...")
        files = list(src.glob("*.md"))
        for f in files:
            # Copy file to unified folder
            shutil.copy2(f, UNIFIED_DIR / f.name)
            total_copied += 1
            
    print(f"\n[DONE] Successfully unified {total_copied} markdown files.")
    print(f"LightRAG INPUT_DIR should be set to: {UNIFIED_DIR}")

if __name__ == "__main__":
    unify()
