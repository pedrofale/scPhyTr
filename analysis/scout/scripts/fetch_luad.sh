#!/usr/bin/env bash
# SCOUT's lung adenocarcinoma dataset: Quinn et al. 2021 (Science), M5K mouse (~5000 engineered
# A549 implanted in the left lung). GEO GSE161363, samples GSM4905334 (lineage) + GSM4905335 (RNA).
# NOTE: this is NOT the KP-Tracer (Yang et al. 2022) data — a different xenograft experiment.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/luad_quinn2021"
mkdir -p "$DEST"; cd "$DEST"
B=https://ftp.ncbi.nlm.nih.gov/geo/samples
for f in \
  GSM4905nnn/GSM4905334/suppl/GSM4905334_alleleTable.5k.txt.gz \
  GSM4905nnn/GSM4905334/suppl/GSM4905334_character_matrices_5k.tar.gz \
  GSM4905nnn/GSM4905334/suppl/GSM4905334_trees_5k.tar.gz \
  GSM4905nnn/GSM4905335/suppl/GSM4905335_barcodes.5k.tsv.gz \
  GSM4905nnn/GSM4905335/suppl/GSM4905335_genes.5k.tsv.gz \
  GSM4905nnn/GSM4905335/suppl/GSM4905335_matrix.5k.mtx.gz \
  GSM4905nnn/GSM4905335/suppl/GSM4905335_meta.5k.tsv.gz ; do
  out=$(basename "$f")
  if [ -s "$out" ]; then echo "have $out"; else echo "fetching $out"; curl -sSL --max-time 1800 -o "$out" "$B/$f"; fi
done
[ -d trees_5k ] || { mkdir -p trees_5k && tar xzf GSM4905334_trees_5k.tar.gz -C trees_5k; }
ls -la
echo "trees:"; find trees_5k -name "*.nwk" -o -name "*.txt" -o -name "*.tree" | head
