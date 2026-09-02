#!/usr/bin/env bash
# Clone the reference implementation (MIT) so its example data + R code are available locally.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/external/SCOUT"
if [ -d "$DEST/.git" ]; then
  echo "already present: $DEST"; exit 0
fi
mkdir -p "$ROOT/external"
git clone --depth 1 https://github.com/hrstuart/SCOUT.git "$DEST"
echo "example data: $DEST/examples/sim_example/"
