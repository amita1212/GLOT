cd /home/t-amitalfasi/glot/paper
for p in $(seq 1 11); do
  pdftotext -f $p -l $p short.pdf /tmp/pg.txt 2>/dev/null
  hits=$(grep -i -o -e "what we changed" -e "changed the answer" -e "what we found" \
                    -e "conclusion" -e "references" -e "hyperbolic preliminaries" \
                    -e "reproducibility statement" /tmp/pg.txt | sort -u | tr '\n' ' ')
  echo "page $p: $hits"
done
