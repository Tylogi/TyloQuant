"""Operator capability descriptions independent of model names."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperatorRequest:
    """Capabilities required by one operator call.

    Differences among Kimi, GLM, and DeepSeek appear only in these fields and model configuration,
    never in implementation names within registry keys.
    """

    operation: str
    device_type: str
    packed_formats: tuple[str, ...] = ()
    code_dims: tuple[int, ...] = ()
    codebook_sizes: tuple[int, ...] = ()
    activation: str = "none"
    top_k: int = 1
    batch_size: int = 1

    def normalized(self) -> "OperatorRequest":
        return OperatorRequest(
            operation=self.operation.strip().lower(),
            device_type=self.device_type.strip().lower(),
            packed_formats=tuple(sorted(set(self.packed_formats))),
            code_dims=tuple(sorted(set(int(v) for v in self.code_dims))),
            codebook_sizes=tuple(
                sorted(set(int(v) for v in self.codebook_sizes))
            ),
            activation=self.activation.strip().lower(),
            top_k=int(self.top_k),
            batch_size=int(self.batch_size),
        )


@dataclass(frozen=True)
class OperatorCapability:
    operation: str
    device_types: tuple[str, ...]
    packed_formats: tuple[str, ...] = ()
    code_dims: tuple[int, ...] = ()
    codebook_sizes: tuple[int, ...] = ()
    activations: tuple[str, ...] = ()
    max_top_k: int = 1
    batch_sizes: tuple[int, ...] = (1,)

    def supports(self, request: OperatorRequest) -> bool:
        request = request.normalized()
        return (
            request.operation == self.operation
            and request.device_type in self.device_types
            and set(request.packed_formats).issubset(self.packed_formats)
            and set(request.code_dims).issubset(self.code_dims)
            and set(request.codebook_sizes).issubset(self.codebook_sizes)
            and request.activation in self.activations
            and 0 < request.top_k <= self.max_top_k
            and request.batch_size in self.batch_sizes
        )

