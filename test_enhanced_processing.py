#!/usr/bin/env python3
"""
Test script for enhanced document processing capabilities
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from financial_qa.enhanced_data_processing import DocumentProcessor
from financial_qa.data_processing import preview_enhanced_processing

def test_enhanced_processing():
    """Test the enhanced document processing capabilities"""
    print("🚀 Testing Enhanced Document Processing\n")
    
    # Preview capabilities
    preview_enhanced_processing()
    
    print("\n" + "="*60 + "\n")
    
    # Test document processor
    processor = DocumentProcessor()
    
    # Discover documents
    documents = processor.discover_documents()
    
    if not documents:
        print("❌ No documents found to process!")
        print("\n📁 Please add documents to the following locations:")
        print("  - financial_data/*.pdf")
        print("  - financial_data/*.xlsx") 
        print("  - financial_data/*.html")
        print("  - financial_data/*.docx")
        print("\n💡 You can copy your existing HTML files to financial_data/")
        return
    
    print("🔍 Testing document processing...")
    
    # Process a sample document to test extraction
    for doc_type, file_paths in documents.items():
        if file_paths:
            sample_file = file_paths[0]
            print(f"\n📄 Testing {doc_type.upper()} extraction: {sample_file.name}")
            
            try:
                if doc_type == "pdf":
                    text = processor.extract_text_from_pdf(sample_file)
                elif doc_type == "excel":
                    text = processor.extract_text_from_excel(sample_file)
                elif doc_type == "html":
                    text = processor.extract_text_from_html(sample_file)
                elif doc_type == "docx":
                    text = processor.extract_text_from_docx(sample_file)
                else:
                    continue
                
                if text:
                    print(f"  ✅ Extraction successful: {len(text)} characters")
                    
                    # Test text cleaning
                    cleaned = processor.clean_text(text)
                    print(f"  🧹 Cleaning: {len(text)} → {len(cleaned)} characters")
                    
                    # Test sectioning
                    sections = processor.segment_into_sections(cleaned)
                    print(f"  📊 Sections: {len(sections)} logical sections")
                    
                    # Show section types
                    section_types = [s["type"] for s in sections]
                    print(f"  📋 Section types: {', '.join(set(section_types))}")
                    
                else:
                    print(f"  ❌ Extraction failed")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
    
    print(f"\n🎯 Next steps:")
    print("  1. Install new dependencies: pip install -r requirements.txt")
    print("  2. Run full processing: python -c 'from financial_qa.enhanced_data_processing import main; main()'")
    print("  3. Test the system: streamlit run app_streamlit.py")


if __name__ == "__main__":
    test_enhanced_processing()
