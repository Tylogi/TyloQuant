#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "usage: $0 MODEL_DIR BUILD_DIR block|nax [OUTPUT_DIR]" >&2
    exit 2
fi

model_dir=$1
build_dir=$2
mode=$3
output_dir=${4:-"$(pwd)/dsv4f-prefill-72g-results"}

case "$mode" in
    block) nax=0 ;;
    nax) nax=1 ;;
    *)
        echo "mode must be block or nax" >&2
        exit 2
        ;;
esac

runner="$build_dir/metal/mfq-metal-ssd-hf-prefill-smoke"
if [[ ! -x "$runner" ]]; then
    echo "missing benchmark executable: $runner" >&2
    exit 2
fi

mkdir -p "$output_dir"

# Fixed production experiment configuration. The mode above is the only A/B
# variable: 72 GiB cache, 4096 input tokens, eight I/O workers in the smoke
# runner, double-buffered prefill, no cache prewarm, and a fresh process for
# every trial. Keep the cooldown fixed so sustained GPU power state is not a
# hidden variable.
for trial in 1 2 3; do
    sleep 60
    env \
        -u http_proxy \
        -u https_proxy \
        -u HTTP_PROXY \
        -u HTTPS_PROXY \
        -u ALL_PROXY \
        -u all_proxy \
        -u MFQ_METAL_PROFILE_COMPONENTS \
        NO_PROXY='*' \
        no_proxy='*' \
        MFQ_METAL_NINTM_PREFILL_NAX="$nax" \
        "$runner" \
        "$model_dir" \
        73728 \
        4096 \
        1 \
        "$output_dir/$mode-$trial.f32" \
        | tee "$output_dir/$mode-$trial.log"
done
