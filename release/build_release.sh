#!/usr/bin/env bash
# Build the macOS arm64 MFQ command-line sidecar for MFQ Studio.
#
# This build intentionally omits MiniCPM-o's Python and realtime stacks.  The
# resulting executable exposes the main `mfq` CLI (build, serve, quantize,
# solve-ew, calibrate, and tpq), but not `mfq-minicpmo-realtime`.

set -euo pipefail

mfq_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mfq_project_dir="$(cd -- "${mfq_script_dir}/.." && pwd)"
mfq_python_version="${MFQ_RELEASE_PYTHON:-3.12}"
mfq_venv_dir="${MFQ_RELEASE_VENV:-${mfq_script_dir}/.venv}"
mfq_build_dir="${mfq_script_dir}/build/pyinstaller"
mfq_spec_dir="${mfq_script_dir}/build/spec"
mfq_dist_dir="${mfq_script_dir}/sidecars"
mfq_artifact_name="mfq-aarch64-apple-darwin"
mfq_artifact_path="${mfq_dist_dir}/${mfq_artifact_name}"

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

if [[ "$(uname -s)" != "Darwin" ]]; then
  fail "this release target must be built on macOS"
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  fail "this release target must be built by native Apple Silicon (arm64)"
fi
command -v uv >/dev/null 2>&1 || fail "uv is required; install it before building"
command -v codesign >/dev/null 2>&1 || fail "Xcode command-line tools are required"

mkdir -p "${mfq_build_dir}" "${mfq_spec_dir}" "${mfq_dist_dir}"

cd "${mfq_project_dir}"

# Keep the release environment separate from the developer environment.  Do
# not install either MiniCPM-o extra: their pinned Python dependencies are not
# part of this desktop distribution.
uv venv --python "${mfq_python_version}" "${mfq_venv_dir}"
uv pip install \
  --python "${mfq_venv_dir}/bin/python" \
  -e '.[daemon,train,calibration,tpq,metal]' \
  'pyinstaller==6.21.0'

mfq_codesign_args=()
if [[ -n "${MFQ_RELEASE_SIGNING_IDENTITY:-}" ]]; then
  mfq_codesign_args=(--codesign-identity "${MFQ_RELEASE_SIGNING_IDENTITY}")
fi

"${mfq_venv_dir}/bin/pyinstaller" \
  --noconfirm \
  --clean \
  --onefile \
  --name "${mfq_artifact_name}" \
  --target-architecture arm64 \
  --paths "${mfq_project_dir}" \
  --collect-data mfq \
  --collect-all fastapi \
  --collect-all uvicorn \
  --collect-all pydantic \
  --collect-all torch \
  --collect-all transformers \
  --collect-all tokenizers \
  --collect-all safetensors \
  --collect-all mlx \
  --exclude-module mfq.runtime.minicpmo45 \
  --exclude-module mfq.runtime.minicpmo45_realtime \
  --exclude-module minicpmo_utils \
  --exclude-module stepaudio2_minicpmo \
  --exclude-module torchaudio \
  --exclude-module onnxruntime \
  --exclude-module s3tokenizer \
  --exclude-module hyperpyyaml \
  --exclude-module librosa \
  --distpath "${mfq_dist_dir}" \
  --workpath "${mfq_build_dir}" \
  --specpath "${mfq_spec_dir}" \
  "${mfq_codesign_args[@]}" \
  "${mfq_project_dir}/mfq/cli.py"

[[ -f "${mfq_artifact_path}" ]] || fail "PyInstaller did not create ${mfq_artifact_path}"
[[ -x "${mfq_artifact_path}" ]] || fail "PyInstaller output is not executable"
lipo -archs "${mfq_artifact_path}" | tr ' ' '\n' | grep -qx 'arm64' \
  || fail "PyInstaller output does not contain an arm64 slice"
codesign --verify --strict --verbose=2 "${mfq_artifact_path}"

"${mfq_artifact_path}" --version
"${mfq_artifact_path}" serve --help >/dev/null
"${mfq_artifact_path}" quantize --help >/dev/null
"${mfq_artifact_path}" solve-ew --help >/dev/null
"${mfq_artifact_path}" calibrate --help >/dev/null
"${mfq_artifact_path}" tpq --help >/dev/null

printf 'built %s\n' "${mfq_artifact_path}"
