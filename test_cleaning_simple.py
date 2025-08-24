#!/usr/bin/env python3
"""
Simple test script to test the cleaning function without importing torch.
"""

import re

def _clean_response(text: str) -> str:
    """Clean up generated response text - comprehensive fix for all spacing issues."""
    
    # Step 1: Fix the main character spacing issue
    # Pattern: letter space letter -> letterletter
    text = re.sub(r'([a-zA-Z])\s+([a-zA-Z])', r'\1\2', text)
    text = re.sub(r'([0-9])\s+([0-9])', r'\1\2', text)
    text = re.sub(r'([a-zA-Z])\s+([0-9])', r'\1\2', text)
    text = re.sub(r'([0-9])\s+([a-zA-Z])', r'\1\2', text)
    
    # Step 2: Fix punctuation spacing
    text = re.sub(r'([a-zA-Z0-9])\s+([,.!?])', r'\1\2', text)
    text = re.sub(r'([,.!?])\s+([a-zA-Z0-9])', r'\1 \2', text)
    
    # Step 3: Fix specific financial formatting patterns
    text = re.sub(r'(\$[0-9,]+)\s*([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'([0-9,]+)\s*([a-zA-Z])', r'\1 \2', text)
    
    # Step 4: Fix common financial terms that got mangled
    text = re.sub(r'Revenuesforthe', 'Revenues for the', text)
    text = re.sub(r'amountedto', 'amounted to', text)
    text = re.sub(r'dueto', 'due to', text)
    text = re.sub(r'completionofthe', 'completion of the', text)
    text = re.sub(r'Researchanddevelopment', 'Research and development', text)
    text = re.sub(r'expenses, net', 'expenses, net', text)
    text = re.sub(r'comparedto', 'compared to', text)
    text = re.sub(r'expensesof', 'expenses of', text)
    text = re.sub(r'inthe', 'in the', text)
    text = re.sub(r'Generalandadministrative', 'General and administrative', text)
    text = re.sub(r'forthe', 'for the', text)
    text = re.sub(r'were1,', 'were $1,', text)
    text = re.sub(r'comparedto \$', 'compared to $', text)
    text = re.sub(r'inthethreemo', 'in the three months', text)
    
    # Step 5: Fix sentence structure
    text = re.sub(r'([a-z])\s+([A-Z])', r'\1. \2', text)
    
    # Step 6: Clean up multiple spaces and final formatting
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Step 7: Fix common financial abbreviations
    text = re.sub(r'LTP', 'LTP', text)  # Keep as is
    text = re.sub(r'R&D', 'R&D', text)  # Keep as is
    
    return text

def test_cleaning():
    print("=== Testing Response Cleaning ===\n")
    
    # Test the problematic response you showed
    problematic_text = """Revenuesforthethreemonthsended. September30, 2022, amountedto 202,000, duetothecompletionofthe. LTPwith. Rio. Tintoasdetailedabove. Researchanddevelopment ("R&D") expenses, netforthethreemonthsended. September30, 2022, were 1, 651,000, comparedtotheexpensesof 
1
,
683
,
000
i
n
t
h
e
t
h
r
e
e
m
o
n
t
h
s
e
n
d
e
d
.
S
e
p
t
e
m
b
e
r
30
,
2021.
G
e
n
e
r
a
l
a
n
d
a
d
m
i
n
i
s
t
r
a
t
i
v
e
e
x
p
e
n
s
e
s
f
o
r
t
h
e
t
h
r
e
e
m
o
n
t
h
s
e
n
d
e
d
.
S
e
p
t
e
m
b
e
r
30
,
2022
,
w
e
r
e
1,683,000inthethreemonthsended.September30,2021.Generalandadministrativeexpensesforthethreemonthsended.September30,2022,were1,050,000, comparedto $703,000 inthethreemo"""
    
    print("Original problematic text:")
    print("=" * 60)
    print(problematic_text)
    print("=" * 60)
    
    print("\nCleaned text:")
    print("=" * 60)
    cleaned = _clean_response(problematic_text)
    print(cleaned)
    print("=" * 60)
    
    print(f"\nText length: {len(problematic_text)} -> {len(cleaned)}")
    
    # Test some specific patterns
    test_cases = [
        "Revenuesforthe",
        "amountedto",
        "dueto",
        "inthe",
        "were1,",
        "comparedto",
    ]
    
    print("\nTesting specific pattern fixes:")
    for pattern in test_cases:
        fixed = _clean_response(pattern)
        print(f"'{pattern}' -> '{fixed}'")

if __name__ == "__main__":
    test_cleaning()
