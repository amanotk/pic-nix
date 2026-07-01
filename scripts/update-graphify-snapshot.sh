#!/usr/bin/env bash
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

OUT_DIR="graphify-out"
SNAPSHOT_DIR="docs/graphify"

if ! command -v graphify >/dev/null 2>&1; then
  echo "error: graphify command not found" >&2
  exit 1
fi

mkdir -p "$SNAPSHOT_DIR"
mkdir -p "$OUT_DIR"

if [[ ! -f "$OUT_DIR/graph.json" && -f "$SNAPSHOT_DIR/graph.json" ]]; then
  cp "$SNAPSHOT_DIR/graph.json" "$OUT_DIR/graph.json"
fi

if [[ ! -f "$OUT_DIR/GRAPH_REPORT.md" && -f "$SNAPSHOT_DIR/GRAPH_REPORT.md" ]]; then
  cp "$SNAPSHOT_DIR/GRAPH_REPORT.md" "$OUT_DIR/GRAPH_REPORT.md"
fi

if [[ ! -f "$OUT_DIR/graph.html" && -f "$SNAPSHOT_DIR/graph.html" ]]; then
  cp "$SNAPSHOT_DIR/graph.html" "$OUT_DIR/graph.html"
fi

graphify update .

for file in GRAPH_REPORT.md graph.json graph.html; do
  if [[ ! -f "$OUT_DIR/$file" ]]; then
    echo "error: missing $OUT_DIR/$file after graphify run" >&2
    exit 1
  fi
  cp "$OUT_DIR/$file" "$SNAPSHOT_DIR/$file"
done

# Keep the committed snapshot portable across machines and independent of any
# local scratch corpus created during graphify experimentation.
perl -0pi -e "s#\Q$ROOT\E/\.graphify-scope/##g; s#\Q$ROOT\E/##g; s#\Q$ROOT\E#.#g" \
  "$SNAPSHOT_DIR/GRAPH_REPORT.md" \
  "$SNAPSHOT_DIR/graph.json" \
  "$SNAPSHOT_DIR/graph.html"

echo "Updated graphify snapshot in $SNAPSHOT_DIR/"
