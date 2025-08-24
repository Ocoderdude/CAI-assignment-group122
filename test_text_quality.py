#!/usr/bin/env python3
"""
Test script to check PDF text quality and identify formatting issues.
"""

from pathlib import Path
from financial_qa.data_processing import read_pdf_to_text, discover_pdfs

def test_text_quality():
    print("=== Testing PDF Text Quality ===\n")
    
    # Find PDF files
    pdfs = discover_pdfs()
    print(f"Found {len(pdfs)} PDF reports")
    
    # Test first PDF
    if pdfs:
        pdf_path = pdfs[0]
        print(f"\nTesting: {pdf_path.name}")
        print("-" * 50)
        
        # Parse with current method
        parsed_text = read_pdf_to_text(pdf_path)
        print(f"Parsed text size: {len(parsed_text)} characters")
        
        # Show first 500 chars of parsed text
        print("\nFirst 500 characters of parsed text:")
        print("=" * 50)
        preview = parsed_text[:500] + "..." if len(parsed_text) > 500 else parsed_text
        print(preview)
        print("=" * 50)
        
        # Check for character spacing issues
        print("\nChecking for character spacing issues...")
        words = parsed_text.split()
        problematic_words = []
        
        for word in words[:20]:  # Check first 20 words
            if len(word) == 1 and word.isalpha():
                problematic_words.append(word)
        
        if problematic_words:
            print(f"Found {len(problematic_words)} single-character words: {problematic_words}")
        else:
            print("No obvious single-character words found")
        
        # Check for excessive spaces
        double_spaces = parsed_text.count('  ')
        if double_spaces > 0:
            print(f"Found {double_spaces} instances of double spaces")
        
        # Show sample of problematic text
        print("\nSample text around 'Revenue':")
        revenue_idx = parsed_text.lower().find('revenue')
        if revenue_idx != -1:
            sample = parsed_text[max(0, revenue_idx-50):revenue_idx+100]
            print(f"Context: ...{sample}...")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_text_quality()
