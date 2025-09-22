#!/usr/bin/env bash
set -euo pipefail
set -x

USER_NAME="${SUDO_USER:-${USER:-${LOGNAME:-$(id -un 2>/dev/null || whoami 2>/dev/null)}}}"

# Input
STEP_NAME="$1"
MODEL_NAME="$2"
SLHA_FILE="$3"

# Paths
PROSPINO_DIR="/u/$USER_NAME/al_pmssmwithgp/prospino-pmssm"
SCRIPT_DIR="$(dirname "$0")"
SCAN_DIR="$(dirname "$SLHA_FILE")"
WORK_BASE="${SCAN_DIR}/.prospino_temp"
OUT_CSV="${WORK_BASE}/crosssections.csv"

# Setup
mkdir -p "$WORK_BASE"
workdir="${WORK_BASE}/${MODEL_NAME}"
rm -rf "$workdir"
mkdir -p "$workdir"

echo "[INFO] Running Prospino for $MODEL_NAME ..."

cd "$workdir"

echo "[INFO] Copy Prospino to temporary directory ..."
cp -a "$PROSPINO_DIR/." "$workdir"

# Clean up old output
rm -f prospino.in.les_houches prospino.dat

# Create new les_houches file
/u/dvoss/al_pmssmwithgp/prospino-pmssm/prospino_normalize.py $SLHA_FILE $MODEL_NAME "$workdir/prospino.in.les_houches"

# Run Prospino

echo "[INFO] Run Prospino ..."
"$PROSPINO_DIR/prospino" > "prospino_${MODEL_NAME}.log" 2>&1

wait

# Parse and append cross section to CSV
python3 "$SCRIPT_DIR/prospino_cross.py" prospino.dat "$OUT_CSV" "$MODEL_NAME" "$STEP_NAME" "$SLHA_FILE"
