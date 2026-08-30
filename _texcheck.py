"""Structural check for the tables in short.tex.

No pdflatex available here, and the rows in tab:absolute / tab:robarms / tab:fix
were written by hand, so the realistic failure is a row whose ampersand count
does not match its tabular column spec. That is a hard compile error and is
worth catching mechanically rather than by eye.
"""
import re
import sys

SRC = "paper/short.tex"
text = open(SRC, encoding="utf-8").read()
lines = text.split("\n")

bad = 0

# ---- environment balance -------------------------------------------------
for env in ("table", "tabular", "center", "document", "itemize", "enumerate"):
    o = len(re.findall(r"\\begin\{" + env + r"\}", text))
    c = len(re.findall(r"\\end\{" + env + r"\}", text))
    status = "ok" if o == c else "MISMATCH"
    if o != c:
        bad += 1
    print(f"{env:10s} begin={o:3d} end={c:3d}  {status}")

# ---- column counts per tabular -------------------------------------------
print("\ntabular row widths:")
i = 0
while i < len(lines):
    m = re.search(r"\\begin\{tabular\}\{((?:[^{}]|\{[^{}]*\})*)\}", lines[i])
    if not m:
        i += 1
        continue
    spec = m.group(1)
    ncol = len(re.findall(r"p\{[^}]*\}|[lcr]", spec))
    start = i
    body = []
    i += 1
    while i < len(lines) and "\\end{tabular}" not in lines[i]:
        body.append((i, lines[i]))
        i += 1
    # rows may span several source lines; join on the \\ terminator
    widths = {}
    buf, first = "", None
    for ln, row in body:
        s = row.strip()
        if not s or s.startswith("%"):
            continue
        if first is None:
            first = ln
        buf += " " + s
        if not s.endswith("\\\\"):
            continue
        if "\\multicolumn" not in buf and "\\cmidrule" not in buf:
            n = len(re.findall(r"(?<!\\)&", buf)) + 1
            widths.setdefault(n, []).append(first + 1)
        buf, first = "", None
    ok = set(widths) <= {ncol}
    if not ok:
        bad += 1
    print(f"  line {start+1:5d}  spec={spec!r:30s} cols={ncol}  "
          f"row widths={ {k: len(v) for k, v in sorted(widths.items())} }"
          f"  {'ok' if ok else 'MISMATCH'}")
    if not ok:
        for w, lns in sorted(widths.items()):
            if w != ncol:
                print(f"      width {w} at lines {lns[:8]}")

# ---- labels and refs -----------------------------------------------------
labels = set(re.findall(r"\\label\{([^}]*)\}", text))
refs = set(re.findall(r"\\ref\{([^}]*)\}", text))
missing = sorted(refs - labels)
print(f"\nlabels={len(labels)} refs={len(refs)}")
if missing:
    bad += 1
    print("  UNDEFINED REFS:", missing)
else:
    print("  every \\ref resolves")

print("\nOK" if not bad else f"\n{bad} PROBLEM(S)")
sys.exit(1 if bad else 0)
