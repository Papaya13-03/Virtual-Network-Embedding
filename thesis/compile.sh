#!/usr/bin/env bash
# Compile DoAn.tex inside the thesis-tex Docker image.
# All intermediate files are kept in .build/; only DoAn.pdf appears in this dir.
#
# Usage:
#   ./compile.sh             # build PDF (incremental)
#   ./compile.sh --force     # force full rebuild (no incremental cache)
#   ./compile.sh --watch     # rebuild continuously on file changes
#   ./compile.sh --clean     # remove .build/ and DoAn.pdf
#   ./compile.sh --shell     # drop into an interactive shell inside the image

set -euo pipefail

IMAGE="thesis-tex:latest"
MAIN="DoAn.tex"
OUTDIR=".build"
PDF="${MAIN%.tex}.pdf"

cd "$(dirname "$0")"

ensure_image() {
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[compile.sh] image '$IMAGE' not found — building from Dockerfile..." >&2
    if [[ ! -f Dockerfile ]]; then
      echo "[compile.sh] ERROR: Dockerfile is missing — cannot build image." >&2
      exit 1
    fi
    docker build --platform linux/amd64 -t "$IMAGE" .
  fi
}

run() {
  docker run --rm --platform linux/amd64 \
    --entrypoint "" \
    -v "$PWD":/work -w /work \
    "$IMAGE" "$@"
}

run_it() {
  docker run --rm -it --platform linux/amd64 \
    --entrypoint "" \
    -v "$PWD":/work -w /work \
    "$IMAGE" "$@"
}

case "${1:-}" in
  --clean)
    rm -rf "$OUTDIR" "$PDF"
    echo "[compile.sh] cleaned ($OUTDIR, $PDF)"
    exit 0
    ;;
  --shell)
    ensure_image
    run_it bash
    exit 0
    ;;
  --watch)
    ensure_image
    mkdir -p "$OUTDIR"
    echo "[compile.sh] entering watch mode (Ctrl-C to stop)"
    run latexmk -pdf -bibtex -pvc -view=none \
        -interaction=nonstopmode -halt-on-error -file-line-error \
        -outdir="$OUTDIR" "$MAIN"
    exit 0
    ;;
  --force)
    ensure_image
    rm -rf "$OUTDIR"
    ;;
  "" )
    ensure_image
    ;;
  *)
    echo "Unknown option: $1" >&2
    echo "Usage: $0 [--force | --watch | --clean | --shell]" >&2
    exit 2
    ;;
esac

mkdir -p "$OUTDIR"
run latexmk -pdf -bibtex \
    -interaction=nonstopmode -halt-on-error -file-line-error \
    -outdir="$OUTDIR" "$MAIN"
cp "$OUTDIR/$PDF" "$PDF"
echo "[compile.sh] built $PDF"
