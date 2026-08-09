"""MFQ — Mixed Format Quantization.

Tensor-wise mixed-precision quantization toolkit for Blackwell and Apple Silicon.

Subpackage overview
--------
formats
    MFQ native-format definitions (NINT neuron-anchored INT codec, precision schemes, headers, and serialization).
quantize
    Quantization logic (NINT tensor quantization, sensitivity analysis and precision allocation, and Hadamard rotation).
calibration
    Per-layer calibration (collect full-precision and quantized hidden states and allocate precision by layer).
kernels
    Hardware backend kernels (CUDA / Metal; see runtime for the torch GPU reference implementation).
runtime
    Inference engine.
utils
    Common utilities (logging and tensor helpers).
"""

from mfq._version import __version__

__all__ = ["__version__"]
