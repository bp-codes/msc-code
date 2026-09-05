#!/usr/bin/env bash
# build-docs.sh — Build Sphinx HTML and PDF (LaTeX) outputs.
# Usage:
#   ./build-docs.sh                 # normal build
#   ./build-docs.sh clean           # remove _build
#   DOCS_DIR=docs ./build-docs.sh   # override source dir

set -euo pipefail

DOCS_DIR="${DOCS_DIR:-docs}"              # Sphinx source dir (contains conf.py)
BUILD_DIR="${BUILD_DIR:-$DOCS_DIR/_build}"
HTML_DIR="$BUILD_DIR/html"
LATEX_DIR="$BUILD_DIR/latex"
PDF_NAME="${PDF_NAME:-}"                  # Set to override output PDF filename

# -------- helpers --------
die() { echo "Error: $*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# -------- cleaning --------
if [[ "${1:-}" == "clean" ]]; then
  rm -rf "$BUILD_DIR"
  echo "Cleaned $BUILD_DIR"
  exit 0
fi

# -------- sanity checks --------
[[ -f "$DOCS_DIR/conf.py" ]] || die "No conf.py in $DOCS_DIR (is this a Sphinx project?)"

have sphinx-build || die $'sphinx-build not found.\nHint (one-time): pip install sphinx'

# We prefer latexmk for reliable PDF builds
if ! have latexmk; then
  echo "latexmk not found — PDF build may fail."
  echo "Install TeX (+latexmk) first, e.g.:"
  echo "  Ubuntu: sudo apt-get install -y texlive-full latexmk"
  echo "  macOS:  brew install --cask mactex-no-gui && tlmgr update --self --all"
  echo "Continuing…"
fi

# -------- HTML build --------
echo "==> Building HTML to $HTML_DIR"
mkdir -p "$HTML_DIR"
if [[ -f "$DOCS_DIR/Makefile" ]]; then
  # use Sphinx quickstart Makefile if present
  ( cd "$DOCS_DIR" && make -s html )
else
  sphinx-build -b html "$DOCS_DIR" "$HTML_DIR"
fi
echo "HTML index: $HTML_DIR/index.html"

# -------- LaTeX/PDF build --------
echo "==> Building LaTeX sources to $LATEX_DIR"
mkdir -p "$LATEX_DIR"
if [[ -f "$DOCS_DIR/Makefile" ]]; then
  # Makefile knows how to run latexpdf; this is the simplest path
  ( cd "$DOCS_DIR" && make -s latexpdf )
  # Find the produced PDF
  pdf=$(ls -1 "$LATEX_DIR"/*.pdf 2>/dev/null | head -n1 || true)
else
  # Manual: generate LaTeX, then compile with latexmk (fallback to pdflatex if needed)
  sphinx-build -b latex "$DOCS_DIR" "$LATEX_DIR"
  pushd "$LATEX_DIR" >/dev/null
  main_tex=$(ls -1 *.tex 2>/dev/null | head -n1 || true)
  [[ -n "${main_tex:-}" ]] || die "No .tex produced by Sphinx"
  if have latexmk; then
    latexmk -pdf -interaction=nonstopmode -halt-on-error "$main_tex"
  elif have pdflatex; then
    # minimal fallback (run twice for TOC/refs)
    pdflatex -interaction=nonstopmode "$main_tex" || true
    pdflatex -interaction=nonstopmode "$main_tex"
  else
    die $'No LaTeX engine found. Install latexmk or pdflatex.'
  fi
  pdf=$(ls -1 *.pdf 2>/dev/null | head -n1 || true)
  popd >/dev/null
fi

[[ -n "${pdf:-}" && -f "$pdf" ]] || die "PDF was not produced. Check LaTeX logs in $LATEX_DIR."

# Optional: rename PDF to a stable name
if [[ -n "$PDF_NAME" ]]; then
  target="$LATEX_DIR/$PDF_NAME.pdf"
  mv -f "$pdf" "$target"
  pdf="$target"
fi

echo "PDF output: $pdf"
echo "✅ Done."
 
