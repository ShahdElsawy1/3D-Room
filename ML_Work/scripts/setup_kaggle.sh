#!/usr/bin/env bash
# One-time setup for a Kaggle GPU notebook (2xT4 or P100). Run from the repo root:
#     bash scripts/setup_kaggle.sh            # everything
#     bash scripts/setup_kaggle.sh vggt       # only VGGT   (also: colmap | gsplat | base)
# Needs "Internet: On" in the notebook settings. Safe to re-run.
set -uo pipefail
: > "${WORK:-/kaggle/working}/env.sh" 2>/dev/null || true
WORK=${WORK:-/kaggle/working}
what=${1:-all}
log() { echo -e "\n=== $* ==="; }

base() {
  log "base python deps"
  pip install -q numpy opencv-python-headless Pillow requests matplotlib
  apt-get -qq install -y ffmpeg >/dev/null 2>&1 || true
  ffmpeg -version | head -1
}

vggt() {
  log "VGGT"
  [ -d "$WORK/vggt" ] || git clone --depth 1 https://github.com/facebookresearch/vggt "$WORK/vggt"
  # VGGT pins torch==2.3.1 / numpy==1.26.1, which do not exist for Kaggle's Python. Keep the preinstalled torch.
  grep -viE '^(torch|torchvision|numpy)([=<>~ ]|$)' "$WORK/vggt/requirements.txt" > /tmp/vggt_req.txt
  pip install -q -r /tmp/vggt_req.txt einops safetensors huggingface_hub opencv-python-headless
  echo "export VGGT_DIR=$WORK/vggt" >> "$WORK/env.sh"
  python -c "import sys; sys.path.insert(0,'$WORK/vggt'); import torch; from vggt.models.vggt import VGGT; print('VGGT import OK, torch', torch.__version__)" \
    || echo "!! VGGT import FAILED - see error above"
}

colmap_glomap() {
  log "COLMAP (+ GLOMAP / global_mapper)"
  # Kaggle's `mamba` is an unrelated python package, so use micromamba.
  # IMPORTANT: the conda env ships its own bare `python`. Never put its bin/ on PATH - it would shadow the system
  # python (no pip, no torch, no cv2). Expose only `colmap` and `glomap`, via tiny wrapper scripts.
  PREFIX="$WORK/cm"; TOOLS="$WORK/tools"; mkdir -p "$TOOLS"
  if [ ! -x "$PREFIX/bin/glomap" ]; then
    if [ ! -x "$WORK/micromamba" ]; then
      curl -sL https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$WORK" bin/micromamba && mv "$WORK/bin/micromamba" "$WORK/micromamba"
    fi
    "$WORK/micromamba" create -y -q -p "$PREFIX" -c conda-forge colmap glomap libfaiss || echo "micromamba install failed"
  fi
  # the conda colmap build can miss libfaiss; add it to an existing env too
  [ -x "$PREFIX/bin/colmap" ] && ! ls "$PREFIX/lib" | grep -qi libfaiss && "$WORK/micromamba" install -y -q -p "$PREFIX" -c conda-forge libfaiss || true
  for tool in colmap glomap; do
    if [ -x "$PREFIX/bin/$tool" ]; then
      # conda binaries expect the env's lib/ on the loader path (normally set by `activate`); set it for these two commands only
      printf '#!/bin/sh\nexport LD_LIBRARY_PATH=%s/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\nexec %s/bin/%s "$@"\n' "$PREFIX" "$PREFIX" "$tool" > "$TOOLS/$tool"; chmod +x "$TOOLS/$tool"
    fi
  done
  echo "export PATH=$TOOLS:\$PATH" >> "$WORK/env.sh"
  export PATH="$TOOLS:$PATH"
  command -v glomap >/dev/null && echo "glomap: $(command -v glomap)"
  command -v colmap >/dev/null && echo "colmap: $(command -v colmap)  ($(colmap help 2>&1 | grep -c global_mapper) global_mapper entries)"
  colmap help >/dev/null 2>&1 || echo "!! colmap does not start - run: ldd $PREFIX/bin/colmap | grep 'not found'"
  glomap -h >/dev/null 2>&1 || echo "!! glomap does not start - run: ldd $PREFIX/bin/glomap | grep 'not found'"
  echo "python (must stay the system one): $(command -v python)"
  if ! command -v glomap >/dev/null && ! (command -v colmap >/dev/null && colmap help 2>&1 | grep -q global_mapper); then
    echo "!! No global mapper installed. Run the pipeline with --method vggt for now (see README troubleshooting)."
  fi
}

gsplat() {
  log "gsplat + examples"
  export TORCH_CUDA_ARCH_LIST="7.5"          # T4 = compute capability 7.5
  echo "python: $(command -v python)"
  python -m pip install -q ninja
  python -m pip install -q gsplat
  VER=$(python -m pip show gsplat | awk '/^Version:/{print $2}' | cut -d+ -f1)
  echo "gsplat version: $VER"
  if [ ! -d "$WORK/gsplat" ]; then
    git clone -q --depth 1 --branch "v$VER" https://github.com/nerfstudio-project/gsplat "$WORK/gsplat" \
      || git clone -q --depth 1 https://github.com/nerfstudio-project/gsplat "$WORK/gsplat"
  fi
  REQ="$WORK/gsplat/examples/requirements.txt"
  # plain packages first (one failing git package must not abort the rest) ...
  # drop git packages, comment lines, and the numpy pin (Kaggle's opencv/jax/etc. need numpy>=2)
  grep -v 'git+' "$REQ" | grep -viE '^ *(#|numpy[=<>~ ]|numpy$)' > /tmp/gs_req.txt
  python -m pip install -q -r /tmp/gs_req.txt || echo "!! some gsplat example requirements failed"
  # ... then the CUDA-compiled git packages, which must see the installed torch
  grep 'git+' "$REQ" | grep -v '^ *#' | sed 's/^ *//' | while read -r line; do
    python -m pip install -q --no-build-isolation "$line" || echo "!! failed to build: $line"
  done
  echo "export GSPLAT_DIR=$WORK/gsplat" >> "$WORK/env.sh"
  python -c "import gsplat, torch; print('gsplat', gsplat.__version__, 'cuda', torch.cuda.is_available())" || echo "!! gsplat import failed"
  python -c "import fused_ssim; print('fused_ssim OK')" 2>/dev/null || echo "!! fused_ssim missing"
  python -c "import fused_bilagrid; print('fused_bilagrid OK')" 2>/dev/null || echo "(fused_bilagrid missing - only needed for bilateral-grid options)"
}

case "$what" in
  base) base ;;
  vggt) vggt ;;
  colmap) colmap_glomap ;;
  gsplat) gsplat ;;
  all) base; vggt; colmap_glomap; gsplat ;;
  *) echo "usage: $0 [all|base|vggt|colmap|gsplat]"; exit 1 ;;
esac
log "done. Next: source $WORK/env.sh"
