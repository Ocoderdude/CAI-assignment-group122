#!/usr/bin/env python3
"""
Utility script to preview the merged financial data and test chunking.
Run this to see what the consolidated text looks like before building indexes.
"""

from pathlib import Path
from financial_qa.data_processing import merge_all_reports_to_single_file, build_chunk_corpus
from financial_qa.config import MERGED_TXT_PATH

def main():
    print("=== Financial Data Processing Preview ===\n")
    
    # Merge all HTML reports into single file
    print("1. Merging HTML reports...")
    try:
        merged_path = merge_all_reports_to_single_file()
        print(f"✓ Successfully merged reports to: {merged_path}")
        
        # Show file size
        size_mb = merged_path.stat().st_size / (1024 * 1024)
        print(f"   File size: {size_mb:.2f} MB")
        
    except Exception as e:
        print(f"✗ Error merging reports: {e}")
        return
    
    # Preview the merged content
    print("\n2. Previewing merged content (first 1000 chars):")
    print("-" * 60)
    content = merged_path.read_text(encoding="utf-8")
    preview = content[:1000] + "..." if len(content) > 1000 else content
    print(preview)
    print("-" * 60)
    
    # Test chunking
    print("\n3. Testing chunk generation...")
    try:
        chunks = build_chunk_corpus()
        print(f"✓ Generated {len(chunks)} chunks")
        
        # Show chunk distribution
        small_chunks = [c for c in chunks if c["size"] == "small"]
        large_chunks = [c for c in chunks if c["size"] == "large"]
        print(f"   Small chunks: {len(small_chunks)}")
        print(f"   Large chunks: {len(large_chunks)}")
        
        # Show sample chunks
        print("\n4. Sample chunks:")
        print("-" * 60)
        for i, chunk in enumerate(chunks[:3]):
            print(f"Chunk {i+1} ({chunk['size']}, score: {chunk['chunk_id']}):")
            print(f"Text preview: {chunk['text'][:200]}...")
            print()
            
    except Exception as e:
        print(f"✗ Error generating chunks: {e}")
        return
    
    print("=== Preview Complete ===")
    print(f"\nNext steps:")
    print(f"1. Review the merged file: {merged_path}")
    print(f"2. Run the Streamlit app: streamlit run app_streamlit.py")
    print(f"3. The app will automatically build indexes from this merged data")

if __name__ == "__main__":
    main()
