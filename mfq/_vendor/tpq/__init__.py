"""TPQ inference runtime for GLM, DeepSeek-V4 and Kimi K3.

The model directory's canonical ``tpq.json`` or legacy ``cccp.json`` selects
an architecture configuration;
CPU/CUDA VQ, MoE, Attention and tensor-parallel kernels are selected through
the shared ``tpq.ops`` capability registry.  Packed expert indices stay
compact in storage, RAM and VRAM.

Use ``python -m tpq launch chat|serve --model <directory>``.  Per-model chat
entry points were removed after the unified launcher became the only public
runtime, so model differences remain configuration rather than duplicate CLI
systems.
"""

__version__ = "1.2.0"

__all__ = ["__version__"]
