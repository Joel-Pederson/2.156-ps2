#!/usr/bin/env bash
# Fetch the course-provided data, assets, and utils_public.py.
#
# These are ~70 MB and belong to the course, not to us, so they are NOT
# committed here. They land in data/cp2-repo/, which the .gitignore `data/`
# rule already covers.
#
# The clone is pinned to the commit in course_commit.txt. That matters because
# utils_public.py defines diversity_score -- the objective we are graded on. If
# the staff amend it, an unpinned clone would pull the change silently and our
# earlier results would quietly stop being comparable.
#
#   ./setup_data.sh            # fetch/restore the clone at the pinned commit
#   ./setup_data.sh --update   # move the pin to the course repo's latest
set -euo pipefail

COURSE_REPO="https://github.com/decode-mit/2155-CP2-2026.git"
PS2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$PS2_ROOT/data/cp2-repo"
PIN_FILE="$PS2_ROOT/course_commit.txt"

UPDATE=0
case "${1:-}" in
  --update) UPDATE=1 ;;
  "")       ;;
  *)        echo "usage: $0 [--update]" >&2; exit 2 ;;
esac

read_pin() {
  grep -vE '^[[:space:]]*(#|$)' "$PIN_FILE" | head -1 | tr -d '[:space:]'
}

write_pin() { # $1 = sha, $2 = date, $3 = subject
  { sed -n '1,/^#[[:space:]]*[0-9a-f]\{7\}/p' "$PIN_FILE" | sed '$d'
    echo "# ${1:0:7}  $2  \"$3\""
    echo "$1"
  } > "$PIN_FILE.tmp"
  mv "$PIN_FILE.tmp" "$PIN_FILE"
}

[ -f "$PIN_FILE" ] || { echo "ERROR: $PIN_FILE missing" >&2; exit 1; }
PINNED="$(read_pin)"
[ -n "$PINNED" ] || { echo "ERROR: no commit found in $PIN_FILE" >&2; exit 1; }

# A bare `git init` + targeted fetch keeps this shallow: we never download the
# course repo's history, just the one tree we want.
if [ ! -d "$DEST/.git" ]; then
  echo "Creating clone at $DEST"
  mkdir -p "$(dirname "$DEST")"
  git init -q "$DEST"
  git -C "$DEST" remote add origin "$COURSE_REPO"
fi

if [ -n "$(git -C "$DEST" status --porcelain 2>/dev/null)" ]; then
  echo "WARNING: local edits in $DEST will be left alone, but that clone is" >&2
  echo "         meant to be disposable -- our code belongs in the ps2 repo." >&2
fi

if [ "$UPDATE" -eq 1 ]; then
  echo "Checking $COURSE_REPO for a newer commit..."
  git -C "$DEST" fetch -q --depth 1 origin HEAD
  TARGET="$(git -C "$DEST" rev-parse FETCH_HEAD)"
else
  TARGET="$PINNED"
  if ! git -C "$DEST" cat-file -e "${TARGET}^{commit}" 2>/dev/null; then
    echo "Fetching pinned commit ${TARGET:0:7}..."
    git -C "$DEST" fetch -q --depth 1 origin "$TARGET"
  fi
fi

git -C "$DEST" checkout -q --detach "$TARGET"

# Fail loudly now rather than three cells into the notebook.
for f in utils_public.py datasets/scores.npy datasets/grids_0.npy assets/Park.jpg; do
  [ -e "$DEST/$f" ] || { echo "ERROR: expected $f missing from $DEST" >&2; exit 1; }
done

if [ "$UPDATE" -eq 1 ]; then
  if [ "$TARGET" = "$PINNED" ]; then
    echo "Already at the latest course commit (${TARGET:0:7}). Pin unchanged."
  else
    echo
    echo "Course repo moved: ${PINNED:0:7} -> ${TARGET:0:7}"
    echo "--- files changed ---"
    git -C "$DEST" diff --stat "$PINNED" "$TARGET" || true
    if ! git -C "$DEST" diff --quiet "$PINNED" "$TARGET" -- utils_public.py; then
      echo
      echo "!!! utils_public.py CHANGED -- check whether diversity_score moved:"
      git -C "$DEST" diff "$PINNED" "$TARGET" -- utils_public.py || true
      echo "!!! If the objective changed, earlier results are not comparable."
    fi
    write_pin "$TARGET" \
      "$(git -C "$DEST" log -1 --format=%ad --date=short "$TARGET")" \
      "$(git -C "$DEST" log -1 --format=%s "$TARGET")"
    echo
    echo "Pin updated. Commit course_commit.txt on its own so the move is traceable."
  fi
fi

echo "Course data ready at $DEST (commit ${TARGET:0:7})"
