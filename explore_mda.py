"""
Exploratory script to find the correct edgartools API for Part I Item 2 (MD&A).
Run: python explore_mda.py
"""
from edgar import Company, set_identity

set_identity("sarthak.malla@mbzuai.ac.ae")

# Use AAPL as a known, well-structured filer
company = Company("AAPL")
filings = company.get_filings(form="10-Q", filing_date="2023-01-01:2023-12-31")
filing = filings[0]
print(f"Filing: {filing.filing_date} | {filing.accession_number}\n")

doc = filing.obj()
report = doc.document

# ── Approach 1: direct get_section with various name variants ─────────────────
print("=" * 60)
print("Approach 1: get_section() with name variants")
print("=" * 60)
for name in ["Item 2", "item2", "item 2", "Part I Item 2",
             "Management's Discussion", "MD&A", "mda", "Part I, Item 2", "part_ii_item_1a", "part_ii_item_1", "part_i_item_2", "part_i_item_3"]:
    try:
        s = report.get_section(name)
        text = s.text(clean=True)[:200] if s else None
        print(f"  {name!r:35s} -> {repr(text[:80]) if text else None}")
    except Exception as e:
        print(f"  {name!r:35s} -> ERROR: {e}")

# ── Approach 2: inspect available sections / parts ────────────────────────────
print("\n" + "=" * 60)
print("Approach 2: list available attributes on doc / report")
print("=" * 60)
for attr in ["parts", "sections", "items", "toc", "table_of_contents"]:
    val = getattr(report, attr, "NOT FOUND")
    display = repr(val) if not callable(val) else '<method>'
    print(f"  report.{attr} = {display}")

# ── Approach 3: try doc.parts["Part I"] ──────────────────────────────────────
print("\n" + "=" * 60)
print("Approach 3: doc.parts")
print("=" * 60)
try:
    parts = report.parts
    print(f"  parts type: {type(parts)}")
    print(f"  parts keys/values: {parts}")
    if hasattr(parts, "__iter__"):
        for p in parts:
            print(f"    part: {p!r}")
            if hasattr(p, "get_section"):
                s = p.get_section("Item 2")
                text = s.text(clean=True)[:200] if s else None
                print(f"      -> get_section('Item 2'): {repr(text[:80]) if text else None}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── Approach 4: inspect the TenQ object itself ────────────────────────────────
print("\n" + "=" * 60)
print("Approach 4: TenQ object attributes")
print("=" * 60)
try:
    tenq = filing.obj()
    print(f"  tenq type: {type(tenq)}")
    interesting = [a for a in dir(tenq) if not a.startswith("_") and
                   any(k in a.lower() for k in ["mda","manage","discussion","item","part","section"])]
    for attr in interesting:
        val = getattr(tenq, attr, None)
        if callable(val):
            try:
                result = val()
                print(f"  tenq.{attr}() = {repr(str(result)[:120])}")
            except Exception as e:
                print(f"  tenq.{attr}() -> ERROR: {e}")
        else:
            print(f"  tenq.{attr} = {repr(str(val)[:120])}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── Approach 5: full text search for MD&A header ──────────────────────────────
print("\n" + "=" * 60)
print("Approach 5: locate MD&A via full text keyword search")
print("=" * 60)
try:
    full = report.text(clean=True, include_tables=False, table_max_col_width=200)
    keywords = [
        "ITEM 2.", "Item 2.", "MANAGEMENT'S DISCUSSION",
        "Management's Discussion and Analysis",
    ]
    for kw in keywords:
        idx = full.find(kw)
        if idx != -1:
            print(f"  Found {kw!r} at index {idx}")
            print(f"  Preview: {repr(full[idx:idx+200])}")
            break
    else:
        print("  None of the keywords found in full text.")
except Exception as e:
    print(f"  ERROR: {e}")
