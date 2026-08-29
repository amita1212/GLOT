#!/usr/bin/env bash
# Install a minimal TeX toolchain and compile the paper for the first time.
# The paper has never been built, so undefined references, broken tables and
# missing macros have never been surfaced.
set -u
cd /home/t-amitalfasi/glot

if ! command -v pdflatex > /dev/null; then
    echo "=== installing texlive (this takes a few minutes) ==="
    sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
        texlive-latex-base texlive-latex-recommended texlive-fonts-recommended \
        texlive-latex-extra texlive-bibtex-extra > /tmp/texinstall.log 2>&1
    echo "apt rc=$?"
fi
command -v pdflatex || { echo "NO PDFLATEX -- see /tmp/texinstall.log"; tail -20 /tmp/texinstall.log; exit 1; }

cd paper || exit 1
echo
echo "=== pass 1 ==="
pdflatex -interaction=nonstopmode -halt-on-error short.tex > /tmp/tex1.log 2>&1
RC=$?
echo "rc=$RC"
if [ $RC -ne 0 ]; then
    echo "--- FIRST ERRORS ---"
    grep -n -A4 '^!' /tmp/tex1.log | head -60
    exit 1
fi
bibtex short > /tmp/bib.log 2>&1; echo "bibtex rc=$?"
pdflatex -interaction=nonstopmode short.tex > /tmp/tex2.log 2>&1
pdflatex -interaction=nonstopmode short.tex > /tmp/tex3.log 2>&1

echo
echo "=== PAGE COUNT ==="
pdfinfo short.pdf 2>/dev/null | grep -i pages || \
    grep -o 'Output written on short.pdf ([0-9]* pages' /tmp/tex3.log

echo
echo "=== UNDEFINED REFERENCES / CITATIONS ==="
grep -i 'undefined' /tmp/tex3.log | sort -u | head -30
echo "(none above = clean)"

echo
echo "=== OVERFULL BOXES > 10pt ==="
grep 'Overfull \\hbox' /tmp/tex3.log | awk -F'[()]' '{print $2, $0}' \
    | awk '$1+0 > 10' | head -20
echo "(none above = clean)"

echo
echo "=== BIBTEX WARNINGS ==="
grep -i 'warning\|error' /tmp/bib.log | head -20
echo "(none above = clean)"
