import os

# HuggingFace cache
os.environ["HF_HOME"] = r"D:\cache\huggingface"

# Transformers cache
os.environ["TRANSFORMERS_CACHE"] = r"D:\cache\huggingface\transformers"

# HuggingFace Hub cache
os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\cache\huggingface\hub"

# Torch cache
os.environ["TORCH_HOME"] = r"D:\cache\torch"

# Disable symlink warnings in Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from docling.document_converter import DocumentConverter

source = r"D:\\UFO files\\Pdf files\\18_100754_ general 1946-7_vol_2.pdf"  # document per local path or URL
converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export_to_markdown())  # output: "## Docling Technical Report[...]"