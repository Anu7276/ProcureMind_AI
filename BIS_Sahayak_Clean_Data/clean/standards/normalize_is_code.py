"""Turn any spelling of an Indian Standard code into the canonical `key` used in standards_master.json.

    normalize_is_code("IS 1239-1")            -> "IS 1239 (Part 1)"
    normalize_is_code("IS/IEC60947-4-1:2002") -> "IS/IEC 60947 (Part 4-1)"
    normalize_is_code("IS 10322 (Part 5 Section 3):2010") -> "IS 10322 (Part 5/Sec 3)"

The edition year is NOT part of the key (one standard = one record; edition_year is a field).
Anti-hallucination check:  normalize_is_code(llm_code) in set(json.load(open("is_code_whitelist.json")))
"""
import re

def parse_is_code(raw):
    s = re.sub(r"\s+", " ", (raw or "").strip())
    m = re.match(r"^(IS(?:/IEC|/ISO|/ISO/IEC)?)\s*(\d+)(.*)$", s, re.I)
    if not m:
        return None
    prefix, num, rest = m.group(1).upper(), m.group(2), m.group(3).strip()
    part = year = None
    pm = re.search(r"\(\s*PART\s*([^)]*)\)", rest, re.I)
    if pm:
        part = re.sub(r"\s+", " ", pm.group(1).strip())
        rest = rest.replace(pm.group(0), "")
        part = re.sub(r"\s*[/,]?\s*Sec(?:tion)?\.?\s*(\d+)", r"/Sec \1", part, flags=re.I)
    else:
        pm = re.match(r"^-\s*(\d+(?:-\d+)*)\b(?!\d{3})", rest)          # IS 1239-1 style
        if pm and not re.match(r"^-\s*\d{4}$", rest):
            part, rest = pm.group(1), rest[pm.end():]
    ym = re.search(r"(?<!\d)(1[89]\d\d|20\d\d)(?!\d)", rest)
    if ym:
        year = int(ym.group(1))
    key = f"{prefix} {num}" + (f" (Part {part})" if part else "")
    return dict(key=key, prefix=prefix, number=num, part=part, year=year)

def normalize_is_code(raw):
    p = parse_is_code(raw)
    return p["key"] if p else None

if __name__ == "__main__":
    tests = {"IS 269:2015": "IS 269", "IS 10019: 1981": "IS 10019", "IS 1239-1": "IS 1239 (Part 1)",
             "IS/IEC60947-4-1:2002": "IS/IEC 60947 (Part 4-1)", "IS 1003 (Part 1): 2003": "IS 1003 (Part 1)",
             "IS10124(PART10):1988": "IS 10124 (Part 10)", "IS 10322 (Part 5 Section 3)": "IS 10322 (Part 5/Sec 3)"}
    for raw, want in tests.items():
        assert normalize_is_code(raw) == want, (raw, normalize_is_code(raw), want)
    print("ok")
