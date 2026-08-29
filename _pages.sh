#!/usr/bin/env bash
cd /home/t-amitalfasi/glot
pdftotext paper/short.pdf /tmp/s.txt 2>/dev/null
python3 - <<'EOF'
pages = open('/tmp/s.txt', encoding='utf-8', errors='replace').read().split('\f')
print(f"total pages: {len(pages)}")
for i, p in enumerate(pages, 1):
    for marker in ("REFERENCES", "References"):
        if marker in p:
            print(f"  references begin on page {i}  ->  BODY = {i-1} full pages + part of {i}")
            break
    else:
        continue
    break
for i, p in enumerate(pages, 1):
    if "Experimental protocol" in p:
        print(f"  appendix A begins on page {i}")
        break
print("\n  section order:")
import re
for i, p in enumerate(pages, 1):
    for line in p.splitlines():
        s = line.strip()
        if re.match(r'^\d+(\.\d+)?\s+[A-Z]', s) and len(s) < 70:
            print(f"   p{i}: {s}")
EOF
