from __future__ import annotations
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple

from .config import RAW_DIR, PROCESSED_DIR, SMALL_CHUNK_CHARS, LARGE_CHUNK_CHARS, CHUNK_OVERLAP_CHARS, ROOT_DIR
from .enhanced_data_processing import DocumentProcessor

# Use enhanced document processor
MERGED_TXT_PATH = PROCESSED_DIR / "merged_financial_reports.txt"


def build_chunk_corpus() -> List[Dict]:
    """Main function to build chunks from processed financial documents."""
    # Use enhanced document processor
    processor = DocumentProcessor()
    
    # Process all documents (PDF, Excel, HTML, DOCX)
    processed_texts, qa_pairs = processor.process_all_documents()
    
    # Save processed data
    processor.save_processed_data()
    
    # Build chunks from merged text
    return build_chunks_from_merged_text()


def build_chunks_from_merged_text() -> List[Dict]:
    """Build chunks from the merged text file."""
    if not MERGED_TXT_PATH.exists():
        raise FileNotFoundError(f"Merged text file not found: {MERGED_TXT_PATH}")
    
    text = MERGED_TXT_PATH.read_text(encoding="utf-8")
    
    all_chunks: List[Dict] = []
    for size_label, size in (("small", SMALL_CHUNK_CHARS), ("large", LARGE_CHUNK_CHARS)):
        for local_id, (start, chunk) in enumerate(chunk_text(text, size, CHUNK_OVERLAP_CHARS)):
            all_chunks.append({
                "doc_id": 0,  # Single document
                "doc_name": "merged_financial_reports.txt",
                "size": size_label,
                "chunk_id": f"merged_{size_label}_{local_id}",
                "start": start,
                "text": chunk,
            })
    
    return all_chunks


def chunk_text(text: str, max_chars: int, overlap: int = 0) -> List[Tuple[int, str]]:
    """Split text into overlapping chunks."""
    chunks: List[Tuple[int, str]] = []
    start = 0
    
    while start < len(text):
        end = min(start + max_chars, len(text))
        
        # Try to break at sentence boundaries
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            search_start = max(start, end - 100)
            sentence_end = text.rfind('.', search_start, end)
            if sentence_end > start and sentence_end > end - 100:
                end = sentence_end + 1
        
        chunks.append((start, text[start:end]))
        
        if end == len(text):
            break
            
        start = end - overlap
        if start < 0:
            start = 0
    
    return chunks


def save_processed_texts(out_dir: Path = PROCESSED_DIR) -> List[Path]:
    """Legacy function - now uses enhanced processor."""
    processor = DocumentProcessor()
    processor.process_all_documents()
    processor.save_processed_data(out_dir)
    
    # Return the merged file path
    return [MERGED_TXT_PATH]


def export_chunks_jsonl(chunks: List[Dict], out_path: Path):
    """Export chunks to JSONL format."""
    with out_path.open("w", encoding="utf-8") as f:
        for row in chunks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def preview_enhanced_processing():
    """Preview the enhanced document processing capabilities."""
    processor = DocumentProcessor()
    
    # Discover documents
    documents = processor.discover_documents()
    
    print("=== Enhanced Document Processing Preview ===\n")
    
    if not documents:
        print("❌ No documents found!")
        print("Please add documents to the following locations:")
        print("- PDF files: financial_data/*.pdf")
        print("- Excel files: financial_data/*.xlsx")
        print("- HTML files: financial_data/*.html")
        print("- Word files: financial_data/*.docx")
        return
    
    print("📁 Found documents:")
    for doc_type, file_paths in documents.items():
        print(f"  {doc_type.upper()}: {len(file_paths)} files")
        for path in file_paths[:3]:  # Show first 3
            print(f"    - {path.name}")
        if len(file_paths) > 3:
            print(f"    ... and {len(file_paths) - 3} more")
    
    print(f"\n🔧 Processing capabilities:")
    print("  ✅ PDF text extraction with OCR fallback")
    print("  ✅ Excel spreadsheet parsing")
    print("  ✅ HTML document cleaning")
    print("  ✅ Word document extraction")
    print("  ✅ Text cleaning and noise removal")
    print("  ✅ Logical section segmentation")
    print("  ✅ Automatic Q/A pair generation")
    
    print(f"\n📊 Expected output:")
    print("  - Processed financial data (JSON)")
    print("  - Q/A pairs for fine-tuning (JSONL)")
    print("  - Merged text for RAG (TXT)")
    print("  - Chunked data for retrieval")


if __name__ == "__main__":
    preview_enhanced_processing()
