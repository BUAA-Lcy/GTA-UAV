#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${ROOT_DIR}/Game4Loc/pretrained/lightglue"

SP_PATH="${TARGET_DIR}/superpoint_v1.pth"
LG_PATH="${TARGET_DIR}/superpoint_lightglue.pth"

ONLY_LG=0
if [[ "${1:-}" == "--only-lg" ]]; then
  ONLY_LG=1
fi

mkdir -p "${TARGET_DIR}"

download_with_fallback() {
  local out_path="$1"
  shift
  local urls=("$@")
  local tmp_path="${out_path}.tmp"

  rm -f "${tmp_path}"
  for url in "${urls[@]}"; do
    echo "[INFO] Downloading: ${url}"
    if curl -fL --retry 5 --retry-delay 2 --connect-timeout 20 -o "${tmp_path}" "${url}"; then
      mv -f "${tmp_path}" "${out_path}"
      echo "[INFO] Saved to: ${out_path}"
      return 0
    fi
    echo "[WARN] Failed: ${url}"
  done

  rm -f "${tmp_path}"
  echo "[ERROR] All download URLs failed for ${out_path}"
  return 1
}

validate_torch_checkpoint() {
  local ckpt_path="$1"
  python - "$ckpt_path" <<'PY'
import sys
import os
import zipfile

path = sys.argv[1]
if not os.path.isfile(path):
    raise RuntimeError(f"Checkpoint does not exist: {path}")
if os.path.getsize(path) < 1024 * 1024:
    raise RuntimeError(f"Checkpoint looks too small: {path}")

try:
    import torch
    obj = torch.load(path, map_location="cpu")
    if obj is None:
        raise RuntimeError(f"Checkpoint is empty: {path}")
    print(f"[INFO] Checkpoint OK (torch.load): {path} ({type(obj).__name__})")
except ModuleNotFoundError:
    # Fallback: validate as zip archive (modern .pth format).
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                raise RuntimeError(f"Corrupted checkpoint entry: {bad}")
        print(f"[INFO] Checkpoint OK (zip integrity): {path}")
    else:
        print(f"[WARN] torch not available; only basic size check passed: {path}")
PY
}

echo "[INFO] Target dir: ${TARGET_DIR}"

if [[ "${ONLY_LG}" -eq 0 ]]; then
  download_with_fallback "${SP_PATH}" \
    "https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superpoint_v1.pth" \
    "https://huggingface.co/image-matching-models/gim-lightglue/resolve/main/superpoint_v1.pth"
  validate_torch_checkpoint "${SP_PATH}"
fi

download_with_fallback "${LG_PATH}" \
  "https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth" \
  "https://huggingface.co/image-matching-models/gim-lightglue/resolve/main/superpoint_lightglue.pth"
validate_torch_checkpoint "${LG_PATH}"

echo "[INFO] Done."
