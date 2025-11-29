#!/usr/bin/env bash
set -euo pipefail

SOURCE="/u/almik/LiquidReg/data"
TARGET="/u/almik/REG/data"
MODE="symlink"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$SOURCE" || -z "$TARGET" ]]; then
  echo "Usage: setup_data.sh --source <old_data_dir> --target <fmfgwreg_repo> [--mode symlink|copy]"
  exit 1
fi

DATA_DIR="$TARGET/data"
OASIS_TRAIN_SRC="$SOURCE/OASIS_train"
OASIS_VAL_SRC="$SOURCE/OASIS_val"
OASIS_TEST_SRC="$SOURCE/OASIS_test"
L2R_SRC="$SOURCE/gen_L2R"
IXI_EVAL_SRC="$SOURCE/gen_IXI_eval"

OASIS_TRAIN_DST="$DATA_DIR/oasis/train"
OASIS_VAL_DST="$DATA_DIR/oasis/val"
OASIS_TEST_DST="$DATA_DIR/oasis/test"
L2R_DST="$DATA_DIR/l2r"
IXI_EVAL_DST="$DATA_DIR/ixi_eval"

mkdir -p "$OASIS_TRAIN_DST" "$OASIS_VAL_DST" "$OASIS_TEST_DST" "$L2R_DST" "$IXI_EVAL_DST"

copy_or_link() {
  local src="$1"
  local dst="$2"

  if [[ "$MODE" == "symlink" ]]; then
    [[ -e "$dst" ]] || ln -s "$src" "$dst"
  else
    if [[ -d "$src" ]]; then
      [[ -e "$dst" ]] || cp -r "$src" "$dst"
    else
      [[ -e "$dst" ]] || cp "$src" "$dst"
    fi
  fi
}

sync_dir() {
  local SRC_DIR="$1"
  local DST_DIR="$2"

  if [[ ! -d "$SRC_DIR" ]]; then
    echo "Skipping missing dir: $SRC_DIR"
    return
  fi

  for item in "$SRC_DIR"/*; do
    base=$(basename "$item")
    copy_or_link "$item" "$DST_DIR/$base"
  done
}

sync_dir "$OASIS_TRAIN_SRC" "$OASIS_TRAIN_DST"
sync_dir "$OASIS_VAL_SRC" "$OASIS_VAL_DST"
sync_dir "$OASIS_TEST_SRC" "$OASIS_TEST_DST"
sync_dir "$L2R_SRC" "$L2R_DST"
sync_dir "$IXI_EVAL_SRC" "$IXI_EVAL_DST"

echo "Done. Data organized under: $DATA_DIR"
