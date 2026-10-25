#!/usr/bin/env bash
# Fetch the course-provided data, assets, and utils_public.py.
#
# These are ~70 MB and belong to the course, not to us, so they are NOT
# committed here. They land in data/cp2-repo/, which the .gitignore `data/`
# rule already covers. Re-running this updates an existing clone.
#
#   ./setup_data.sh
set -euo pipefail

COURSE_REPO="https://github.com/decode-mit/2155-CP2-2026.git"
PS2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$PS2_ROOT/data/cp2-repo"

if [ -d "$DEST/.git" ]; then
  echo "Updating existing clone at $DEST"
  git -C "$DEST" pull --ff-only
else
  echo "Cloning $COURSE_REPO -> $DEST"
  mkdir -p "$(dirname "$DEST")"
  git clone --depth 1 "$COURSE_REPO" "$DEST"
fi

# Fail loudly now rather than three cells into the notebook.
for f in utils_public.py datasets/scores.npy datasets/grids_0.npy assets/Park.jpg; do
  [ -e "$DEST/$f" ] || { echo "ERROR: expected $f missing from $DEST" >&2; exit 1; }
done

echo "Course data ready at $DEST"
