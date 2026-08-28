#!/usr/bin/env bash
# Build the self-contained Apple Silicon MFQ Studio DMG.

set -euo pipefail

mfq_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mfq_project_dir="$(cd -- "${mfq_script_dir}/.." && pwd)"
mfq_python_version="${MFQ_RELEASE_PYTHON:-3.12}"
mfq_venv_dir="${MFQ_RELEASE_VENV:-${mfq_script_dir}/.venv}"
mfq_build_root="${mfq_script_dir}/build"
mfq_native_build_dir="${mfq_build_root}/runtime"
mfq_pyinstaller_build_dir="${mfq_build_root}/pyinstaller"
mfq_spec_dir="${mfq_build_root}/spec"
mfq_sidecar_dir="${mfq_script_dir}/sidecars"
mfq_framework_dir="${mfq_script_dir}/Frameworks"
mfq_resource_dir="${mfq_script_dir}/Resources"
mfq_studio_dir="${mfq_project_dir}/MFQStudio"
mfq_tauri_target="aarch64-apple-darwin"
mfq_dmg_dir="${mfq_studio_dir}/src-tauri/target/${mfq_tauri_target}/release/bundle/dmg"
mfq_output_dir="${MFQ_RELEASE_OUTPUT_DIR:-${mfq_script_dir}/dist}"
mfq_cli_name="mfq-cli"
mfq_runtime_name="mfq-decode-metal-aarch64-apple-darwin"
mfq_perplexity_name="mfq-perplexity-aarch64-apple-darwin"
mfq_macos_deployment_target="${MFQ_RELEASE_MACOS_DEPLOYMENT_TARGET:-26.2}"
mfq_release_rustflags="${RUSTFLAGS:-}"
mfq_release_rustflags="${mfq_release_rustflags:+${mfq_release_rustflags} }--remap-path-prefix=${HOME}=/mfq-build/home --remap-path-prefix=${mfq_project_dir}=/mfq-src"

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

[[ "$(uname -s)" == "Darwin" ]] || fail "this release target requires macOS"
[[ "$(uname -m)" == "arm64" ]] || fail "this release target requires native Apple Silicon"
for mfq_command in uv cmake codesign hdiutil install_name_tool lipo npm cargo strings; do
  command -v "${mfq_command}" >/dev/null 2>&1 || fail "${mfq_command} is required"
done

mkdir -p \
  "${mfq_native_build_dir}" \
  "${mfq_pyinstaller_build_dir}" \
  "${mfq_spec_dir}" \
  "${mfq_sidecar_dir}" \
  "${mfq_framework_dir}" \
  "${mfq_resource_dir}" \
  "${mfq_output_dir}"

cd "${mfq_project_dir}"
if [[ "${MFQ_RELEASE_REUSE_VENV:-0}" == "1" ]]; then
  [[ -x "${mfq_venv_dir}/bin/python" ]] || fail "reused release environment has no Python"
  [[ -x "${mfq_venv_dir}/bin/pyinstaller" ]] || fail "reused release environment has no PyInstaller"
else
  uv venv --clear --python "${mfq_python_version}" "${mfq_venv_dir}"
fi
source "${mfq_venv_dir}/bin/activate"
if [[ "${MFQ_RELEASE_REUSE_VENV:-0}" != "1" ]]; then
  uv sync \
    --locked \
    --active \
    --no-default-groups \
    --no-editable \
    --group release \
    --extra daemon \
    --extra train \
    --extra calibration \
    --extra minicpmo45-realtime \
    --extra tpq \
    --extra metal
fi

command -v ninja >/dev/null 2>&1 || fail "ninja is required"
mfq_mlx_root="$("${mfq_venv_dir}/bin/python" -c 'import mlx; print(next(iter(mlx.__path__)))')"
for mfq_mlx_file in \
  "${mfq_mlx_root}/lib/libmlx.dylib" \
  "${mfq_mlx_root}/lib/libjaccl.dylib" \
  "${mfq_mlx_root}/lib/mlx.metallib"; do
  [[ -f "${mfq_mlx_file}" ]] || fail "missing MLX runtime file: ${mfq_mlx_file}"
done

cmake -S "${mfq_project_dir}/cpp_runtime" -B "${mfq_native_build_dir}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="${mfq_macos_deployment_target}" \
  -DGGML_NATIVE=OFF \
  -DMFQ_BUILD_CPP_SERVER=ON \
  -DMFQ_BUILD_METAL_RUNTIME=ON \
  -DMFQ_MLX_ROOT="${mfq_mlx_root}" \
  -DMFQ_MLX_METALLIB_DEFAULT:STRING=
cmake --build "${mfq_native_build_dir}" \
  --target mfq-decode-metal mfq-perplexity \
  --parallel "${MFQ_RELEASE_JOBS:-$(sysctl -n hw.ncpu)}"

mfq_runtime_source="${mfq_native_build_dir}/metal/mfq-decode-metal"
mfq_perplexity_source="${mfq_native_build_dir}/metal/mfq-perplexity"
mfq_video_source="${mfq_native_build_dir}/metal/libmfq_avfoundation_video.dylib"
[[ -x "${mfq_runtime_source}" ]] || fail "native build did not create mfq-decode-metal"
[[ -x "${mfq_perplexity_source}" ]] || fail "native build did not create mfq-perplexity"
[[ -f "${mfq_video_source}" ]] || fail "native build did not create the AVFoundation video library"

install -m 755 "${mfq_runtime_source}" "${mfq_sidecar_dir}/${mfq_runtime_name}"
install -m 755 "${mfq_perplexity_source}" "${mfq_sidecar_dir}/${mfq_perplexity_name}"
install -m 644 "${mfq_mlx_root}/lib/libmlx.dylib" "${mfq_framework_dir}/libmlx.dylib"
install -m 644 "${mfq_mlx_root}/lib/libjaccl.dylib" "${mfq_framework_dir}/libjaccl.dylib"
install -m 755 "${mfq_video_source}" "${mfq_framework_dir}/libmfq_avfoundation_video.dylib"
install -m 644 "${mfq_mlx_root}/lib/mlx.metallib" "${mfq_resource_dir}/mlx.metallib"

for mfq_native_binary in \
  "${mfq_sidecar_dir}/${mfq_runtime_name}" \
  "${mfq_sidecar_dir}/${mfq_perplexity_name}"; do
  install_name_tool -add_rpath "@executable_path/../Frameworks" "${mfq_native_binary}"
done

mfq_signing_identity="${MFQ_RELEASE_SIGNING_IDENTITY:-${APPLE_SIGNING_IDENTITY:--}}"
for mfq_signed_file in \
  "${mfq_framework_dir}/libjaccl.dylib" \
  "${mfq_framework_dir}/libmlx.dylib" \
  "${mfq_framework_dir}/libmfq_avfoundation_video.dylib" \
  "${mfq_sidecar_dir}/${mfq_runtime_name}" \
  "${mfq_sidecar_dir}/${mfq_perplexity_name}"; do
  codesign --force --sign "${mfq_signing_identity}" "${mfq_signed_file}"
  codesign --verify --strict --verbose=2 "${mfq_signed_file}"
done

MFQ_MLX_METALLIB="${mfq_resource_dir}/mlx.metallib" \
DYLD_LIBRARY_PATH="${mfq_framework_dir}" \
  "${mfq_sidecar_dir}/${mfq_runtime_name}" --self-test-metal

mfq_pyinstaller_args=(
  --noconfirm
  --clean
  --onedir
  --name "${mfq_cli_name}"
  --target-architecture arm64
  --paths "${mfq_project_dir}"
  --collect-data mfq
  --collect-all fastapi
  --collect-all uvicorn
  --collect-all pydantic
  --collect-all torch
  --collect-all transformers
  --collect-all tokenizers
  --collect-all safetensors
  --collect-all mlx
  --collect-all hyperpyyaml
  --collect-all librosa
  --collect-all onnxruntime
  --collect-all s3tokenizer
  --collect-all scipy
  --collect-all soundfile
  --collect-all stepaudio2
  --collect-all torchaudio
  --copy-metadata requests
  --copy-metadata torchcodec
  --hidden-import mfq.runtime.minicpmo45_realtime
  --exclude-module mfq.runtime.minicpmo45
  --exclude-module minicpmo_utils
  --distpath "${mfq_resource_dir}"
  --workpath "${mfq_pyinstaller_build_dir}"
  --specpath "${mfq_spec_dir}"
)
if [[ "${mfq_signing_identity}" != "-" ]]; then
  mfq_pyinstaller_args+=(--codesign-identity "${mfq_signing_identity}")
fi
mfq_pyinstaller_args+=("${mfq_project_dir}/mfq/cli.py")
"${mfq_venv_dir}/bin/pyinstaller" "${mfq_pyinstaller_args[@]}"

mfq_cli_path="${mfq_resource_dir}/${mfq_cli_name}/${mfq_cli_name}"
[[ -x "${mfq_cli_path}" ]] || fail "PyInstaller did not create the unified mfq CLI"
for mfq_arm64_file in \
  "${mfq_cli_path}" \
  "${mfq_sidecar_dir}/${mfq_runtime_name}" \
  "${mfq_sidecar_dir}/${mfq_perplexity_name}"; do
  lipo -archs "${mfq_arm64_file}" | tr ' ' '\n' | grep -qx arm64 \
    || fail "${mfq_arm64_file} is not arm64"
done
codesign --verify --strict --verbose=2 "${mfq_cli_path}"

"${mfq_cli_path}" --version
"${mfq_cli_path}" serve --help >/dev/null
"${mfq_cli_path}" quantize --help >/dev/null
"${mfq_cli_path}" solve-ew --help >/dev/null
"${mfq_cli_path}" calibrate --help >/dev/null
"${mfq_cli_path}" tpq --help >/dev/null
"${mfq_cli_path}" voice-runtime-check >/dev/null

cd "${mfq_studio_dir}"
npm ci
npm run check
if [[ "${mfq_signing_identity}" != "-" ]]; then
  CI=true \
  APPLE_SIGNING_IDENTITY="${APPLE_SIGNING_IDENTITY:-${mfq_signing_identity}}" \
  RUSTFLAGS="${mfq_release_rustflags}" \
    npm run tauri -- build \
      --config src-tauri/tauri.release-macos.conf.json \
      --config '{"bundle":{"macOS":{"hardenedRuntime":true}}}' \
      --target "${mfq_tauri_target}"
else
  CI=true \
  APPLE_SIGNING_IDENTITY="${APPLE_SIGNING_IDENTITY:-${mfq_signing_identity}}" \
  RUSTFLAGS="${mfq_release_rustflags}" \
    npm run tauri -- build \
      --config src-tauri/tauri.release-macos.conf.json \
      --target "${mfq_tauri_target}"
fi

[[ -d "${mfq_dmg_dir}" ]] || fail "Tauri did not create a DMG directory"
mfq_dmg_path="$(find "${mfq_dmg_dir}" -maxdepth 1 -type f -name '*.dmg' -print -quit)"
[[ -n "${mfq_dmg_path}" ]] || fail "Tauri did not create a DMG"
mfq_release_dmg="${mfq_output_dir}/$(basename "${mfq_dmg_path}")"
install -m 644 "${mfq_dmg_path}" "${mfq_release_dmg}"
hdiutil verify "${mfq_release_dmg}"
mfq_mount_dir="$(mktemp -d "${TMPDIR:-/tmp}/mfq-release-verify.XXXXXX")"
mfq_mounted=false
cleanup_release_mount() {
  if [[ "${mfq_mounted}" == true ]]; then
    hdiutil detach "${mfq_mount_dir}" >/dev/null
  fi
  rmdir "${mfq_mount_dir}" 2>/dev/null || true
}
trap cleanup_release_mount EXIT
hdiutil attach -readonly -nobrowse -mountpoint "${mfq_mount_dir}" "${mfq_release_dmg}" >/dev/null
mfq_mounted=true
mfq_packaged_app="${mfq_mount_dir}/MFQ Studio.app"
codesign --verify --deep --strict --verbose=2 "${mfq_packaged_app}"
for mfq_private_prefix in "${HOME}/" "${mfq_project_dir}/"; do
  if strings -a "${mfq_packaged_app}/Contents/MacOS/mfq-studio" \
      | grep -F "${mfq_private_prefix}" >/dev/null; then
    fail "packaged Studio contains a private build path"
  fi
done
MFQ_MLX_METALLIB="${mfq_packaged_app}/Contents/Resources/mlx.metallib" \
  "${mfq_packaged_app}/Contents/MacOS/mfq-decode-metal" --self-test-metal
hdiutil detach "${mfq_mount_dir}" >/dev/null
mfq_mounted=false
rmdir "${mfq_mount_dir}"
trap - EXIT
printf 'built %s\n' "${mfq_release_dmg}"
