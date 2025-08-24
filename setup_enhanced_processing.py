#!/usr/bin/env python3
"""
Setup script for enhanced document processing
Helps migrate from HTML-only to multi-format processing
"""

import shutil
from pathlib import Path

def setup_enhanced_processing():
    """Setup the enhanced document processing system"""
    print("🚀 Setting up Enhanced Document Processing System\n")
    
    # Create financial_data directory
    financial_data_dir = Path("financial_data")
    financial_data_dir.mkdir(exist_ok=True)
    print(f"✅ Created directory: {financial_data_dir}")
    
    # Move existing HTML files to financial_data
    html_files = list(Path(".").glob("Rail Vision*Financial Results*.html"))
    
    if html_files:
        print(f"\n📁 Found {len(html_files)} existing HTML files")
        print("🔄 Moving them to financial_data/ directory...")
        
        for html_file in html_files:
            dest_path = financial_data_dir / html_file.name
            shutil.move(str(html_file), str(dest_path))
            print(f"  ✅ Moved: {html_file.name}")
        
        print(f"\n💡 HTML files are now in {financial_data_dir}/")
        print("   They will be processed alongside any new PDF/Excel files")
    
    # Create sample directory structure
    print(f"\n📋 Expected directory structure:")
    print(f"  {financial_data_dir}/")
    print(f"    ├── *.pdf          # PDF financial reports")
    print(f"    ├── *.xlsx         # Excel spreadsheets")
    print(f"    ├── *.html         # HTML reports (moved)")
    print(f"    └── *.docx         # Word documents")
    
    # Show next steps
    print(f"\n🎯 Next Steps:")
    print(f"  1. Add any additional documents to {financial_data_dir}/")
    print(f"  2. Install new dependencies: pip install -r requirements.txt")
    print(f"  3. Test processing: python test_enhanced_processing.py")
    print(f"  4. Process documents: python -c 'from financial_qa.enhanced_data_processing import main; main()'")
    print(f"  5. Run the app: streamlit run app_streamlit.py")
    
    print(f"\n🔧 New Capabilities:")
    print(f"  ✅ Multi-format document support (PDF, Excel, HTML, Word)")
    print(f"  ✅ OCR for image-based PDFs")
    print(f"  ✅ Advanced text cleaning and noise removal")
    print(f"  ✅ Logical financial section segmentation")
    print(f"  ✅ Automatic Q/A pair generation (50+ pairs)")
    print(f"  ✅ Better text quality and formatting")
    
    print(f"\n✨ Enhanced processing will automatically:")
    print(f"  - Handle multiple document types")
    print(f"  - Clean and standardize text")
    print(f"  - Segment into logical sections")
    print(f"  - Generate training data for fine-tuning")
    print(f"  - Create high-quality chunks for RAG")


if __name__ == "__main__":
    setup_enhanced_processing()
