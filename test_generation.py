#!/usr/bin/env python3
"""
Test script to check raw model generation and test the cleaning function.
"""

from financial_qa.generate import _clean_response

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
