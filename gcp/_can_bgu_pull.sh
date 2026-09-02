#!/usr/bin/env bash
# Can the BGU checkout fast-forward to the parity fix without conflict?
cd ~/glot || exit 1
echo "=== BGU checkout ==="
echo "  branch : $(git rev-parse --abbrev-ref HEAD)"
echo "  head   : $(git log --oneline -1)"
echo "  remotes:"; git remote -v | sed 's/^/    /'
echo
echo "=== untracked files that a pull might need to overwrite ==="
git fetch --all --quiet 2>&1 | tail -2
UP=$(git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null || echo "none")
echo "  upstream: $UP"
if [ "$UP" != "none" ]; then
  echo "  commits behind: $(git rev-list --count HEAD..$UP 2>/dev/null)"
  echo "  incoming files:"
  git diff --name-only HEAD.."$UP" 2>/dev/null | sed 's/^/    /'
  echo
  echo "=== collision check: incoming files that exist here but are UNTRACKED ==="
  hit=0
  while read -r f; do
    [ -z "$f" ] && continue
    if [ -e "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
      echo "    COLLISION: $f is untracked here and incoming"
      hit=1
    fi
  done < <(git diff --name-only HEAD.."$UP" 2>/dev/null)
  [ "$hit" = 0 ] && echo "    none -- pull --ff-only will succeed"
fi
echo
echo "=== dry-run the pull ==="
git merge --ff-only "$UP" --no-commit 2>&1 | head -5 | sed 's/^/    /' || true
git merge --abort 2>/dev/null
echo "  (above is advisory; chain does the real pull after the queue finishes)"
