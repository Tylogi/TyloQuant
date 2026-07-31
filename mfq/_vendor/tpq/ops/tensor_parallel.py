"""Model-independent tensor-parallel decode operators.

The executors in this module are keyed by tensor shapes and mathematical
capabilities.  Model runtimes only provide weights plus configuration values;
no model family name participates in dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable

import torch
import torch.nn.functional as F


def _new_cuda_graph() -> torch.cuda.CUDAGraph:
    """Retain graph topology only for fixed-address layer composition."""
    return torch.cuda.CUDAGraph(
        keep_graph=os.environ.get("TPQ_TP_LAYER_GRAPH", "0") != "0"
    )


def _instantiate_retained_graph(graph: torch.cuda.CUDAGraph) -> None:
    if os.environ.get("TPQ_TP_LAYER_GRAPH", "0") != "0":
        graph.instantiate()


def _no_owner_rank_order(executor, state) -> tuple[int, ...]:
    """Return canonical rank order for formal TP, legacy order otherwise.

    ``state.owner`` is permitted to describe where an unsplit source weight
    was staged during construction.  It must not choose the launch or
    reduction order once the all-rank TPHidden data flow is enabled.
    """
    ranks = len(executor.devices)
    if (
        getattr(executor, "hidden_mode", False)
        and os.environ.get("TPQ_TP_NO_OWNER", "1") != "0"
    ):
        return tuple(range(ranks))
    owner = int(state.owner)
    return (
        owner,
        *(rank for rank in range(ranks) if rank != owner),
    )


def _compose_normalize_prelude(
    executor,
    layer: int,
    source,
    residual,
    active_rows: int,
    projections,
    norm_weights,
    post_norm_weights,
    workspaces,
    eps: float,
) -> None:
    """Prefix a retained Attention graph with fixed rank-local normalization."""
    if os.environ.get("TPQ_TP_LAYER_GRAPH", "0") == "0":
        raise RuntimeError("TP layer Graph composition is disabled")
    from ..fusedext import make_tp_graph_sequence_batch

    state = executor.layers[layer]
    if (
        state.graphs is None
        or state.events is None
        or state.source_event is None
    ):
        raise RuntimeError("TP rank graphs are not captured")
    local_source = (
        source
        if tuple(source.devices) == executor.devices
        else source.subset(executor.devices)
    )
    target = executor.input_hidden(layer)
    preludes = local_source.capture_normalize_graphs(
        target,
        executor.streams,
        post_norm_weights,
        float(eps),
        residual=residual,
        active_rows=int(active_rows),
        projections=projections,
        norm_weights=norm_weights,
        workspaces=workspaces,
    )
    rank_order = _no_owner_rank_order(executor, state)
    state.graph_batch = make_tp_graph_sequence_batch(
        [
            int(executor.devices[rank].index)
            for rank in rank_order
        ],
        [
            [preludes[rank], state.graphs[ordered_rank]]
            for ordered_rank, rank in enumerate(rank_order)
        ],
        [executor.streams[rank] for rank in rank_order],
        list(state.events),
        state.source_event,
    )
    state.composed_input_addresses = local_source.fixed_addresses


def _compose_mlp_prelude(
    executor,
    layer: int,
    source,
    attention,
    prefix_output,
    residual,
    active_rows: int,
    projections,
    norm_weights,
    post_norm_weights,
    workspaces,
    eps: float,
    boundary: bool,
) -> None:
    """Prefix a retained gated-MLP graph with fixed residual preparation."""
    if os.environ.get("TPQ_TP_LAYER_GRAPH", "0") == "0":
        raise RuntimeError("TP layer Graph composition is disabled")
    from ..fusedext import make_tp_graph_sequence_batch

    state = executor.layers[layer]
    if (
        state.graphs is None
        or state.events is None
        or state.source_event is None
    ):
        raise RuntimeError("TP gated MLP rank graphs are not captured")
    local_source = (
        source
        if tuple(source.devices) == executor.devices
        else source.subset(executor.devices)
    )
    local_attention = (
        attention
        if tuple(attention.devices) == executor.devices
        else attention.subset(executor.devices)
    )
    local_prefix = (
        prefix_output
        if tuple(prefix_output.devices) == executor.devices
        else prefix_output.subset(executor.devices)
    )
    target = executor.input_hidden(layer)
    preludes = local_source.capture_mlp_prelude_graphs(
        local_attention,
        local_prefix,
        target,
        executor.streams,
        residual,
        int(active_rows),
        projections,
        norm_weights,
        post_norm_weights,
        workspaces,
        float(eps),
        boundary=bool(boundary),
    )
    rank_order = _no_owner_rank_order(executor, state)
    state.graph_batch = make_tp_graph_sequence_batch(
        [
            int(executor.devices[rank].index)
            for rank in rank_order
        ],
        [
            [preludes[rank], state.graphs[ordered_rank]]
            for ordered_rank, rank in enumerate(rank_order)
        ],
        [executor.streams[rank] for rank in rank_order],
        list(state.events),
        state.source_event,
    )
    state.composed_input_addresses = local_attention.fixed_addresses
    state.composed_prefix_graphs = preludes


class OwnerGroupedTensorParallel:
    """Dispatch layers to the TP subgroup containing their owner rank.

    Large packed operators may still use the complete device tuple, while
    latency-bound projections use smaller contiguous subgroups.  The wrapper
    preserves one executor interface and keeps group selection out of model
    names and operator registry keys.
    """

    def __init__(
        self,
        devices: tuple[torch.device, ...],
        group_size: int,
        factory: Callable[
            [tuple[torch.device, ...]],
            object,
        ],
    ) -> None:
        group_size = int(group_size)
        if (
            group_size <= 1
            or group_size > len(devices)
            or len(devices) % group_size
        ):
            raise ValueError(
                "TP subgroup size must divide the visible device count"
            )
        self.devices = devices
        self.group_size = group_size
        self.groups = tuple(
            devices[start:start + group_size]
            for start in range(0, len(devices), group_size)
        )
        self.executors = tuple(factory(group) for group in self.groups)
        self.layer_groups: dict[int, int] = {}
        self._global_outputs: dict[int, object] = {}

    def add_layer(
        self,
        layer: int,
        owner: int,
        *args,
        **kwargs,
    ) -> None:
        if layer in self.layer_groups:
            raise ValueError(
                f"TP subgroup layer {layer} is already registered"
            )
        group_index = int(owner) // self.group_size
        local_owner = int(owner) % self.group_size
        self.executors[group_index].add_layer(
            layer,
            local_owner,
            *args,
            **kwargs,
        )
        self.layer_groups[layer] = group_index

    def capture(self) -> None:
        for executor in self.executors:
            executor.capture()

    def _executor(self, layer: int):
        return self.executors[self.layer_groups[layer]]

    def run(self, layer: int, *args, **kwargs):
        return self._executor(layer).run(layer, *args, **kwargs)

    def input_buffer(self, layer: int) -> torch.Tensor:
        return self._executor(layer).input_buffer(layer)

    def input_hidden(self, layer: int):
        return self._executor(layer).input_hidden(layer)

    def output_hidden(self, layer: int):
        local = self._executor(layer).output_hidden(layer)
        if self.group_size == len(self.devices):
            return local
        output = self._global_outputs.get(layer)
        if output is None:
            from .hidden import TPHidden

            output = TPHidden.empty(
                self.devices,
                tuple(local.shape),
                dtype=local.dtype,
            )
            self._global_outputs[layer] = output
        return output

    def input_sharded(self, layer: int):
        return self._executor(layer).input_sharded(layer)

    def composed_input_sharded(self, layer: int, *args, **kwargs):
        return self._executor(layer).composed_input_sharded(
            layer,
            *args,
            **kwargs,
        )

    def run_prepared(self, layer: int, *args, **kwargs):
        return self._executor(layer).run_prepared(
            layer,
            *args,
            **kwargs,
        )

    def run_hidden(self, layer: int, *args, **kwargs):
        executor = self._executor(layer)
        if (
            args
            and hasattr(args[0], "subset")
            and tuple(args[0].devices) != executor.devices
        ):
            args = (args[0].subset(executor.devices), *args[1:])
        kwargs.setdefault("output", self.output_hidden(layer))
        return executor.run_hidden(
            layer,
            *args,
            **kwargs,
        )

    def compose_normalize_prelude(
        self,
        layer: int,
        *args,
        **kwargs,
    ) -> None:
        return self._executor(layer).compose_normalize_prelude(
            layer,
            *args,
            **kwargs,
        )

    def compose_mlp_prelude(
        self,
        layer: int,
        *args,
        **kwargs,
    ) -> None:
        return self._executor(layer).compose_mlp_prelude(
            layer,
            *args,
            **kwargs,
        )

    def compose_owner_branch(
        self,
        layer: int,
        owner_graph: torch.cuda.CUDAGraph,
    ) -> None:
        return self._executor(layer).compose_owner_branch(
            layer,
            owner_graph,
        )

    def run_sharded(self, layer: int, *args, **kwargs):
        kwargs.setdefault("output", self.output_hidden(layer))
        return self._executor(layer).run_sharded(
            layer,
            *args,
            **kwargs,
        )

    def launch_partials(self, layer: int, *args, **kwargs):
        executor = self._executor(layer)
        if (
            args
            and hasattr(args[0], "subset")
            and tuple(args[0].devices) != executor.devices
        ):
            args = (args[0].subset(executor.devices), *args[1:])
        return executor.launch_partials(
            layer,
            *args,
            **kwargs,
        )

    def last_partials(self, layer: int):
        return self._executor(layer).last_partials(layer)

    def finalize_moe(self, layer: int, *args, **kwargs):
        kwargs.setdefault("output", self.output_hidden(layer))
        return self._executor(layer).finalize_moe(
            layer,
            *args,
            **kwargs,
        )

    def finalize_moe_full(self, layer: int, *args, **kwargs):
        kwargs.setdefault("output", self.output_hidden(layer))
        return self._executor(layer).finalize_moe_full(
            layer,
            *args,
            **kwargs,
        )

    def start(self, layer: int, *args, **kwargs):
        return self._executor(layer).start(layer, *args, **kwargs)

    def start_prepared(self, layer: int, *args, **kwargs):
        return self._executor(layer).start_prepared(
            layer,
            *args,
            **kwargs,
        )

    def finish(self, layer: int, *args, **kwargs):
        return self._executor(layer).finish(layer, *args, **kwargs)

    def reset(self) -> None:
        for executor in self.executors:
            reset = getattr(executor, "reset", None)
            if reset is not None:
                reset()


@dataclass(frozen=True)
class GatedMLPSpec:
    hidden_size: int
    intermediate_size: int
    activation: str
    activation_beta: float
    activation_linear_beta: float | None


@dataclass
class _GatedMLPLayer:
    owner: int
    source: torch.Tensor
    local_inputs: list[torch.Tensor]
    gate_up: list[torch.Tensor]
    down: list[torch.Tensor]
    contributions: list[torch.Tensor]
    zero: torch.Tensor
    graphs: list[torch.cuda.CUDAGraph] | None = None
    events: list[torch.cuda.Event] | None = None
    source_event: torch.cuda.Event | None = None
    input_events: list[torch.cuda.Event] | None = None
    output_replicas: list[torch.Tensor] | None = None
    output_events: list[torch.cuda.Event] | None = None
    graph_batch: object | None = None
    composed_input_addresses: tuple[int, ...] | None = None
    composed_prefix_graphs: tuple[torch.cuda.CUDAGraph, ...] | None = None
    launch_stream: torch.cuda.Stream | None = None
    ready_event: torch.cuda.Event | None = None
    pending_output: torch.Tensor | None = None


class TensorParallelGatedMLP:
    """Row-TP gated MLP with one persistent CUDA graph per rank.

    Gate/Up output rows and matching Down input columns are sharded.  Every
    rank reads one fixed input, computes its local intermediate slice, and
    returns a FP32 partial output.  Formal TPHidden execution reduces directly
    to every rank, so the next layer consumes its local replica without an
    owner broadcast.  Weights remain in their original BF16 representation.
    """

    def __init__(
        self,
        devices: tuple[torch.device, ...],
        spec: GatedMLPSpec,
    ) -> None:
        if len(devices) <= 1:
            raise ValueError("tensor-parallel MLP requires at least 2 ranks")
        if spec.intermediate_size % len(devices):
            raise ValueError(
                "intermediate size must divide the tensor-parallel size"
            )
        self.devices = devices
        self.spec = spec
        self.hidden_mode = (
            os.environ.get("TPQ_TP_HIDDEN", "0") != "0"
        )
        self.streams = [
            torch.cuda.Stream(device=device) for device in devices
        ]
        self.layers: dict[int, _GatedMLPLayer] = {}

    def add_layer(
        self,
        layer: int,
        owner: int,
        combined_gate_up: torch.Tensor,
        down: torch.Tensor,
    ) -> None:
        if layer in self.layers:
            raise ValueError(f"TP MLP layer {layer} is already registered")
        if (
            combined_gate_up.dtype != torch.bfloat16
            or down.dtype != torch.bfloat16
            or combined_gate_up.shape
            != (
                2 * self.spec.intermediate_size,
                self.spec.hidden_size,
            )
            or down.shape
            != (
                self.spec.hidden_size,
                self.spec.intermediate_size,
            )
        ):
            raise ValueError(
                f"TP MLP layer {layer} weight shape/dtype mismatch"
            )
        gate, up = combined_gate_up.chunk(2, dim=0)
        gate_parts = gate.chunk(len(self.devices), dim=0)
        up_parts = up.chunk(len(self.devices), dim=0)
        down_parts = down.chunk(len(self.devices), dim=1)
        owner_device = self.devices[owner]
        gate_up_shards: list[torch.Tensor] = []
        down_shards: list[torch.Tensor] = []
        local_inputs: list[torch.Tensor] = []
        for rank, device in enumerate(self.devices):
            with torch.cuda.device(device):
                gate_up_shards.append(
                    torch.cat(
                        (
                            gate_parts[rank].to(device),
                            up_parts[rank].to(device),
                        ),
                        dim=0,
                    ).contiguous()
                )
                down_shards.append(
                    down_parts[rank].to(device).contiguous()
                )
                local_inputs.append(
                    torch.empty(
                        1,
                        self.spec.hidden_size,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                )
        with torch.cuda.device(owner_device):
            source = torch.empty(
                1,
                self.spec.hidden_size,
                dtype=torch.bfloat16,
                device=owner_device,
            )
            zero = torch.zeros(
                1,
                self.spec.hidden_size,
                dtype=torch.float32,
                device=owner_device,
            )
            launch_stream = torch.cuda.Stream(device=owner_device)
            ready_event = torch.cuda.Event()
        self.layers[layer] = _GatedMLPLayer(
            owner=owner,
            source=source,
            local_inputs=local_inputs,
            gate_up=gate_up_shards,
            down=down_shards,
            contributions=[],
            zero=zero,
            launch_stream=launch_stream,
            ready_event=ready_event,
        )

    def capture(self) -> None:
        from ..fusedext import (
            make_tp_graph_launch_batch,
            tp_peer_copy_fused,
        )
        from .api import gated_activation

        for device in self.devices:
            torch.cuda.synchronize(device)
        for layer, state in self.layers.items():
            owner_device = self.devices[state.owner]
            with torch.cuda.device(owner_device):
                state.source.zero_()
            for rank, device in enumerate(self.devices):
                with torch.cuda.device(device):
                    state.local_inputs[rank].zero_()
            rank_order = _no_owner_rank_order(self, state)
            graphs: list[torch.cuda.CUDAGraph] = []
            contributions: list[torch.Tensor] = []
            events: list[torch.cuda.Event] = []
            ordered_streams: list[torch.cuda.Stream] = []
            source_event = torch.cuda.Event()
            with torch.cuda.device(owner_device):
                source_event.record(torch.cuda.current_stream(owner_device))
                torch.cuda.synchronize(owner_device)
            for rank in rank_order:
                device = self.devices[rank]
                stream = self.streams[rank]

                def execute_rank() -> torch.Tensor:
                    if (
                        not self.hidden_mode
                        and not tp_peer_copy_fused(
                            state.source,
                            state.local_inputs[rank],
                        )
                    ):
                        raise RuntimeError(
                            "TP gated MLP input dispatch was rejected"
                        )
                    projected = F.linear(
                        state.local_inputs[rank],
                        state.gate_up[rank],
                    )
                    gate, up = projected.chunk(2, dim=-1)
                    activated = gated_activation(
                        gate,
                        up,
                        activation=self.spec.activation,
                        beta=self.spec.activation_beta,
                        linear_beta=self.spec.activation_linear_beta,
                        output=gate,
                    )
                    if activated is None:
                        raise RuntimeError(
                            "TP gated MLP activation was rejected"
                        )
                    return F.linear(
                        activated,
                        state.down[rank],
                    ).float()

                with (
                    torch.cuda.device(device),
                    torch.cuda.stream(stream),
                ):
                    execute_rank()
                    stream.synchronize()
                    event = torch.cuda.Event()
                    graph = _new_cuda_graph()
                    with torch.cuda.graph(graph, stream=stream):
                        contribution = execute_rank()
                    _instantiate_retained_graph(graph)
                    event.record(stream)
                    stream.synchronize()
                graphs.append(graph)
                contributions.append(contribution)
                events.append(event)
                ordered_streams.append(stream)
            state.contributions = contributions
            state.graphs = graphs
            state.events = events
            state.source_event = source_event
            state.input_events = []
            for device in self.devices:
                with torch.cuda.device(device):
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(device))
                    state.input_events.append(event)
            state.output_replicas = []
            state.output_events = []
            for device in self.devices:
                with torch.cuda.device(device):
                    state.output_replicas.append(
                        torch.empty(
                            1,
                            self.spec.hidden_size,
                            dtype=torch.bfloat16,
                            device=device,
                        )
                    )
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(device))
                    state.output_events.append(event)
            state.graph_batch = make_tp_graph_launch_batch(
                [
                    int(self.devices[rank].index)
                    for rank in rank_order
                ],
                graphs,
                ordered_streams,
                events,
                source_event,
            )

    def run(self, layer: int, value: torch.Tensor) -> torch.Tensor:
        if self.hidden_mode:
            hidden = self.input_hidden(layer)
            state = self.layers[layer]
            hidden.copy_from_owner(value, state.owner)
            output = self.run_hidden(layer, hidden)
            owner_output = output.local(state.owner)
            with torch.cuda.device(self.devices[state.owner]):
                torch.cuda.current_stream().wait_event(
                    output.ready_events[state.owner]
                )
            return owner_output
        output = self.start(layer, value)
        return self.finish(layer, output)

    def input_buffer(self, layer: int) -> torch.Tensor:
        return self.layers[layer].source

    def input_hidden(self, layer: int):
        from .hidden import TPHidden

        state = self.layers[layer]
        if state.input_events is None:
            raise RuntimeError("TP gated MLP graphs are not captured")
        return TPHidden(
            self.devices,
            tuple(state.local_inputs),
            tuple(state.input_events),
        )

    def compose_mlp_prelude(
        self,
        layer: int,
        source,
        attention,
        prefix_output,
        residual,
        active_rows: int,
        projections,
        norm_weights,
        post_norm_weights,
        workspaces,
        eps: float,
        *,
        boundary: bool,
    ) -> None:
        _compose_mlp_prelude(
            self,
            layer,
            source,
            attention,
            prefix_output,
            residual,
            active_rows,
            projections,
            norm_weights,
            post_norm_weights,
            workspaces,
            eps,
            boundary,
        )

    def compose_owner_branch(
        self,
        layer: int,
        owner_graph: torch.cuda.CUDAGraph,
    ) -> None:
        """Run shared MLP and one owner-local child in parallel."""
        from ..fusedext import make_tp_graph_dag_batch

        state = self.layers[layer]
        if (
            state.graphs is None
            or state.events is None
            or state.source_event is None
            or state.composed_prefix_graphs is None
        ):
            raise RuntimeError(
                "owner branch requires a composed gated-MLP prelude"
            )
        rank_order = _no_owner_rank_order(self, state)
        graph_stages = []
        for ordered_rank, rank in enumerate(rank_order):
            parallel = [state.graphs[ordered_rank]]
            if rank == state.owner:
                parallel.append(owner_graph)
            graph_stages.append(
                [
                    [state.composed_prefix_graphs[rank]],
                    parallel,
                ]
            )
        state.graph_batch = make_tp_graph_dag_batch(
            [
                int(self.devices[rank].index)
                for rank in rank_order
            ],
            graph_stages,
            [self.streams[rank] for rank in rank_order],
            list(state.events),
            state.source_event,
        )

    def compose_rank_parallel_branch(
        self,
        layer: int,
        branch_graphs: tuple[torch.cuda.CUDAGraph, ...],
    ) -> None:
        """Run one additional fixed-address graph beside each rank MLP.

        The normalization/residual prelude remains the sole first stage.
        Afterwards both branches consume the same rank-local hidden and run
        concurrently.  This is a graph capability, not a model-specific MoE
        path.
        """
        from ..fusedext import make_tp_graph_dag_batch

        state = self.layers[layer]
        if (
            state.graphs is None
            or state.events is None
            or state.source_event is None
            or state.composed_prefix_graphs is None
            or len(branch_graphs) != len(self.devices)
        ):
            raise RuntimeError(
                "rank-parallel branch requires one retained graph per rank"
            )
        rank_order = _no_owner_rank_order(self, state)
        state.graph_batch = make_tp_graph_dag_batch(
            [
                int(self.devices[rank].index)
                for rank in rank_order
            ],
            [
                [
                    [state.composed_prefix_graphs[rank]],
                    [
                        state.graphs[ordered_rank],
                        branch_graphs[rank],
                    ],
                ]
                for ordered_rank, rank in enumerate(rank_order)
            ],
            [self.streams[rank] for rank in rank_order],
            list(state.events),
            state.source_event,
        )

    def output_hidden(self, layer: int):
        from .hidden import TPHidden

        state = self.layers[layer]
        if state.output_replicas is None or state.output_events is None:
            raise RuntimeError("TP gated MLP outputs are unavailable")
        return TPHidden(
            self.devices,
            tuple(state.output_replicas),
            tuple(state.output_events),
        )

    def run_hidden(self, layer: int, hidden, output=None):
        state = self.layers[layer]
        if state.graph_batch is None or not self.hidden_mode:
            raise RuntimeError(
                "TP gated MLP TPHidden graph is not captured"
            )
        if output is None:
            output = self.output_hidden(layer)
        if (
            tuple(hidden.devices) != self.devices
            or hidden.shape != torch.Size((1, self.spec.hidden_size))
            or output.shape != hidden.shape
            or hidden.dtype != torch.bfloat16
            or output.dtype != torch.bfloat16
            or hidden.ready_events is None
            or output.ready_events is None
        ):
            raise ValueError("TP gated MLP TPHidden layout mismatch")
        expected_addresses = (
            state.composed_input_addresses
            if state.composed_input_addresses is not None
            else tuple(
                item.data_ptr() for item in state.local_inputs
            )
        )
        if hidden.fixed_addresses != expected_addresses:
            raise ValueError(
                "TP gated MLP input must use captured fixed addresses"
            )
        rank_order = _no_owner_rank_order(self, state)
        state.graph_batch.launch_all_rank_from_events(
            [
                hidden.ready_events[rank].cuda_event
                for rank in rank_order
            ],
            state.contributions,
            list(output.replicas),
            [
                event.cuda_event
                for event in output.ready_events
            ],
        )
        return output

    def launch_partials(self, layer: int, hidden):
        """Launch a shared/Dense branch without a hidden collective."""
        from .hidden import TPPartials

        state = self.layers[layer]
        if state.graph_batch is None or not self.hidden_mode:
            raise RuntimeError(
                "TP gated MLP partial graph is not captured"
            )
        if (
            tuple(hidden.devices) != self.devices
            or hidden.shape != torch.Size((1, self.spec.hidden_size))
            or hidden.dtype != torch.bfloat16
            or hidden.ready_events is None
            or state.events is None
        ):
            raise ValueError("TP gated MLP partial input mismatch")
        expected_addresses = (
            state.composed_input_addresses
            if state.composed_input_addresses is not None
            else tuple(
                item.data_ptr() for item in state.local_inputs
            )
        )
        if hidden.fixed_addresses != expected_addresses:
            raise ValueError(
                "TP gated MLP partial input must use fixed addresses"
            )
        rank_order = _no_owner_rank_order(self, state)
        state.graph_batch.launch_from_events(
            [
                hidden.ready_events[rank].cuda_event
                for rank in rank_order
            ]
        )
        return TPPartials(
            tuple(self.devices[rank] for rank in rank_order),
            tuple(state.contributions),
            tuple(state.events),
        )

    def run_prepared(self, layer: int) -> torch.Tensor:
        if self.hidden_mode:
            state = self.layers[layer]
            return self.run(layer, state.source)
        output = self.start_prepared(layer)
        return self.finish(layer, output)

    def start(self, layer: int, value: torch.Tensor) -> torch.Tensor:
        """Launch one MLP branch on its persistent auxiliary stream.

        The caller may execute an independent branch on the owner default
        stream before calling :meth:`finish`.  Inputs, graphs and result
        buffers remain fixed-size; this only changes scheduling.
        """
        state = self.layers[layer]
        if self.hidden_mode:
            if state.pending_output is not None:
                raise RuntimeError(
                    "TP gated MLP layer already has pending work"
                )
            hidden = self.input_hidden(layer)
            hidden.copy_from_owner(value, state.owner)
            output = self.run_hidden(layer, hidden)
            state.pending_output = output.local(state.owner)
            return state.pending_output
        if state.graph_batch is None:
            raise RuntimeError("TP gated MLP graphs are not captured")
        if (
            state.launch_stream is None
            or state.ready_event is None
        ):
            raise RuntimeError("TP gated MLP async state is not initialized")
        if state.pending_output is not None:
            raise RuntimeError("TP gated MLP layer already has pending work")
        owner_device = self.devices[state.owner]
        if value.device != owner_device:
            raise ValueError("TP gated MLP input is not on its owner rank")
        with torch.cuda.device(owner_device):
            state.source.copy_(value)
        return self.start_prepared(layer)

    def start_prepared(self, layer: int) -> torch.Tensor:
        """Launch using data already written into the fixed source buffer."""
        state = self.layers[layer]
        if self.hidden_mode:
            return self.start(layer, state.source)
        if state.graph_batch is None:
            raise RuntimeError("TP gated MLP graphs are not captured")
        if (
            state.launch_stream is None
            or state.ready_event is None
        ):
            raise RuntimeError("TP gated MLP async state is not initialized")
        if state.pending_output is not None:
            raise RuntimeError("TP gated MLP layer already has pending work")
        owner_device = self.devices[state.owner]
        with torch.cuda.device(owner_device):
            current = torch.cuda.current_stream(owner_device)
            state.launch_stream.wait_stream(current)
            with torch.cuda.stream(state.launch_stream):
                output = state.graph_batch.launch_reduce(
                    state.contributions,
                    state.zero,
                )
                state.ready_event.record(state.launch_stream)
        state.pending_output = output
        return output

    def finish(
        self,
        layer: int,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Join an MLP branch previously launched by :meth:`start`."""
        state = self.layers[layer]
        if self.hidden_mode:
            if (
                state.pending_output is None
                or state.pending_output.data_ptr() != output.data_ptr()
                or state.output_events is None
            ):
                raise RuntimeError(
                    "TP gated MLP layer has no matching hidden work"
                )
            owner_device = self.devices[state.owner]
            with torch.cuda.device(owner_device):
                torch.cuda.current_stream(owner_device).wait_event(
                    state.output_events[state.owner]
                )
            state.pending_output = None
            return output
        if (
            state.ready_event is None
            or state.pending_output is None
            or state.pending_output.data_ptr() != output.data_ptr()
        ):
            raise RuntimeError("TP gated MLP layer has no matching work")
        owner_device = self.devices[state.owner]
        with torch.cuda.device(owner_device):
            torch.cuda.current_stream(owner_device).wait_event(
                state.ready_event
            )
        state.pending_output = None
        return output


@dataclass(frozen=True)
class RowParallelLinearSpec:
    in_features: int
    out_features: int
    input_dtype: torch.dtype = torch.bfloat16
    weight_dtype: torch.dtype = torch.bfloat16
    capture_owner_dispatch: bool = False


@dataclass
class _RowParallelLinearLayer:
    owner: int
    source: torch.Tensor
    source_parts: list[torch.Tensor]
    local_inputs: list[torch.Tensor]
    weights: list[torch.Tensor]
    contributions: list[torch.Tensor]
    zero: torch.Tensor
    graphs: list[torch.cuda.CUDAGraph] | None = None
    events: list[torch.cuda.Event] | None = None
    source_event: torch.cuda.Event | None = None
    input_events: list[torch.cuda.Event] | None = None
    output_replicas: list[torch.Tensor] | None = None
    output_events: list[torch.cuda.Event] | None = None
    routed_workspaces: list[torch.Tensor] | None = None
    shared_workspaces: list[torch.Tensor] | None = None
    global_workspaces: dict[
        tuple[int, ...],
        tuple[list[torch.Tensor], list[torch.Tensor]],
    ] | None = None
    graph_batch: object | None = None
    bound_input_addresses: tuple[int, ...] | None = None
    bound_input_hidden: object | None = None
    composed_input_addresses: tuple[int, ...] | None = None


class TensorParallelRowLinear:
    """Model-independent row-parallel BF16 linear projection.

    Input columns and matching weight columns are sharded across ranks.  Each
    rank receives only its input slice and produces one FP32 partial output.
    Formal TPHidden execution publishes the sole mathematical reduction to
    every rank without selecting a hidden owner.  No rank retains a complete
    copy of the weight.
    """

    def __init__(
        self,
        devices: tuple[torch.device, ...],
        spec: RowParallelLinearSpec,
    ) -> None:
        if len(devices) <= 1:
            raise ValueError("row-parallel linear requires at least 2 ranks")
        if spec.in_features % len(devices):
            raise ValueError(
                "linear input width must divide the tensor-parallel size"
            )
        self.devices = devices
        self.spec = spec
        self.local_width = spec.in_features // len(devices)
        self.hidden_mode = (
            os.environ.get("TPQ_TP_HIDDEN", "0") != "0"
        )
        self.streams = [
            torch.cuda.Stream(device=device) for device in devices
        ]
        self.layers: dict[int, _RowParallelLinearLayer] = {}

    def add_layer(
        self,
        layer: int,
        owner: int,
        weight: torch.Tensor,
    ) -> None:
        if layer in self.layers:
            raise ValueError(
                f"row-parallel linear layer {layer} is already registered"
            )
        if (
            weight.dtype != self.spec.weight_dtype
            or weight.shape
            != (self.spec.out_features, self.spec.in_features)
        ):
            raise ValueError(
                f"row-parallel linear layer {layer} weight shape/dtype "
                "mismatch"
            )
        owner_device = self.devices[owner]
        with torch.cuda.device(owner_device):
            source = torch.empty(
                1,
                self.spec.in_features,
                dtype=self.spec.input_dtype,
                device=owner_device,
            )
            source_parts = list(
                source.split(self.local_width, dim=-1)
            )
            zero = torch.zeros(
                1,
                self.spec.out_features,
                dtype=torch.float32,
                device=owner_device,
            )
        local_inputs = []
        weights = []
        for rank, device in enumerate(self.devices):
            with torch.cuda.device(device):
                local_inputs.append(
                    torch.empty(
                        1,
                        self.local_width,
                        dtype=self.spec.input_dtype,
                        device=device,
                    )
                )
                weights.append(
                    weight[
                        :,
                        rank * self.local_width:
                        (rank + 1) * self.local_width,
                    ]
                    .to(device)
                    .contiguous()
                )
        self.layers[layer] = _RowParallelLinearLayer(
            owner=owner,
            source=source,
            source_parts=source_parts,
            local_inputs=local_inputs,
            weights=weights,
            contributions=[],
            zero=zero,
        )

    def bind_input_hidden(self, layer: int, hidden) -> None:
        """Bind local slices of a fixed all-rank producer before capture.

        The producer remains replicated because the previous Row-TP
        collective publishes onto every rank.  This operator reads only the
        rank-local column slice directly from that replica, so there is no
        owner dispatch and no intermediate shard copy.
        """
        state = self.layers[layer]
        if state.graph_batch is not None:
            raise RuntimeError(
                "row-parallel input must be bound before graph capture"
            )
        if (
            tuple(hidden.devices) != self.devices
            or hidden.shape != torch.Size((1, self.spec.in_features))
            or hidden.dtype != self.spec.input_dtype
            or hidden.ready_events is None
        ):
            raise ValueError("row-parallel bound TPHidden layout mismatch")
        state.local_inputs = [
            hidden.replicas[rank][
                :,
                rank * self.local_width:
                (rank + 1) * self.local_width,
            ]
            for rank in range(len(self.devices))
        ]
        state.bound_input_addresses = hidden.fixed_addresses
        state.bound_input_hidden = hidden

    def input_hidden(self, layer: int):
        """Return the fixed full replicas backing local Row-TP slices."""
        hidden = self.layers[layer].bound_input_hidden
        if hidden is None:
            raise RuntimeError(
                "row-parallel linear has no bound TPHidden input"
            )
        return hidden

    def compose_normalize_prelude(
        self,
        layer: int,
        source,
        post_norm_weights,
        eps: float,
    ) -> None:
        """Fuse rank-local RMSNorm ahead of the retained Row-TP graphs."""
        _compose_normalize_prelude(
            self,
            layer,
            source,
            None,
            0,
            (),
            (),
            post_norm_weights,
            (),
            float(eps),
        )

    def capture(self) -> None:
        from ..fusedext import (
            make_tp_graph_launch_batch,
            tp_peer_copy_fused,
        )

        for device in self.devices:
            torch.cuda.synchronize(device)
        for state in self.layers.values():
            owner_device = self.devices[state.owner]
            with torch.cuda.device(owner_device):
                state.source.zero_()
            for rank, device in enumerate(self.devices):
                with torch.cuda.device(device):
                    state.local_inputs[rank].zero_()
            rank_order = _no_owner_rank_order(self, state)
            graphs = []
            events = []
            contributions = []
            ordered_streams = []
            source_event = torch.cuda.Event()
            with torch.cuda.device(owner_device):
                source_event.record(torch.cuda.current_stream(owner_device))
                torch.cuda.synchronize(owner_device)
            for rank in rank_order:
                device = self.devices[rank]
                stream = self.streams[rank]

                def execute_rank() -> torch.Tensor:
                    if (
                        (
                            not self.hidden_mode
                            or self.spec.capture_owner_dispatch
                        )
                        and not tp_peer_copy_fused(
                            state.source_parts[rank],
                            state.local_inputs[rank],
                        )
                    ):
                        raise RuntimeError(
                            "row-parallel linear input dispatch was rejected"
                        )
                    local = state.local_inputs[rank]
                    if local.dtype != state.weights[rank].dtype:
                        local = local.to(state.weights[rank].dtype)
                    return F.linear(
                        local,
                        state.weights[rank],
                    ).float()

                with (
                    torch.cuda.device(device),
                    torch.cuda.stream(stream),
                ):
                    execute_rank()
                    stream.synchronize()
                    event = torch.cuda.Event()
                    graph = _new_cuda_graph()
                    with torch.cuda.graph(graph, stream=stream):
                        contribution = execute_rank()
                    _instantiate_retained_graph(graph)
                    event.record(stream)
                    stream.synchronize()
                graphs.append(graph)
                events.append(event)
                contributions.append(contribution)
                ordered_streams.append(stream)
            state.graphs = graphs
            state.events = events
            state.contributions = contributions
            state.source_event = source_event
            state.input_events = []
            for device in self.devices:
                with torch.cuda.device(device):
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(device))
                    state.input_events.append(event)
            state.output_replicas = []
            state.output_events = []
            state.routed_workspaces = []
            state.shared_workspaces = []
            state.global_workspaces = {}
            for device in self.devices:
                with torch.cuda.device(device):
                    state.output_replicas.append(
                        torch.empty(
                            1,
                            self.spec.out_features,
                            dtype=self.spec.input_dtype,
                            device=device,
                        )
                    )
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(device))
                    state.output_events.append(event)
                    state.routed_workspaces.append(
                        torch.empty(
                            1,
                            self.spec.out_features,
                            dtype=self.spec.input_dtype,
                            device=device,
                        )
                    )
                    state.shared_workspaces.append(
                        torch.empty(
                            1,
                            self.spec.out_features,
                            dtype=self.spec.input_dtype,
                            device=device,
                        )
                    )
            state.graph_batch = make_tp_graph_launch_batch(
                [
                    int(self.devices[rank].index)
                    for rank in rank_order
                ],
                graphs,
                ordered_streams,
                events,
                source_event,
            )

    def run(self, layer: int, value: torch.Tensor) -> torch.Tensor:
        state = self.layers[layer]
        if state.graph_batch is None:
            raise RuntimeError(
                "row-parallel linear graphs are not captured"
            )
        owner_device = self.devices[state.owner]
        if (
            value.device != owner_device
            or value.dtype != self.spec.input_dtype
            or value.shape != (1, self.spec.in_features)
        ):
            raise ValueError(
                "row-parallel linear input shape/dtype/device mismatch"
            )
        with torch.cuda.device(owner_device):
            if self.hidden_mode:
                sharded = self.input_sharded(layer)
                sharded.copy_from_full(value)
                output = self.run_sharded(layer, sharded)
                owner_output = output.local(state.owner)
                torch.cuda.current_stream(owner_device).wait_event(
                    output.ready_events[state.owner]
                )
                return owner_output
            state.source.copy_(value)
            return state.graph_batch.launch_reduce(
                state.contributions,
                state.zero,
            )

    def input_buffer(self, layer: int) -> torch.Tensor:
        return self.layers[layer].source

    def input_sharded(self, layer: int):
        from .hidden import TPSharded

        state = self.layers[layer]
        if state.input_events is None:
            raise RuntimeError(
                "row-parallel linear graphs are not captured"
            )
        return TPSharded(
            self.devices,
            tuple(state.local_inputs),
            self.spec.in_features,
            tuple(state.input_events),
        )

    def bound_input_sharded(self, layer: int, hidden):
        """Expose bound input views with the producer's current events."""
        from .hidden import TPSharded

        state = self.layers[layer]
        if (
            state.bound_input_addresses is None
            or hidden.fixed_addresses != state.bound_input_addresses
            or tuple(hidden.devices) != self.devices
            or hidden.shape != torch.Size((1, self.spec.in_features))
            or hidden.dtype != self.spec.input_dtype
            or hidden.ready_events is None
        ):
            raise ValueError(
                "row-parallel input does not match its bound producer"
            )
        return TPSharded(
            self.devices,
            tuple(state.local_inputs),
            self.spec.in_features,
            tuple(hidden.ready_events),
        )

    def composed_input_sharded(self, layer: int, source):
        """Use producer events to launch a normalize→Row-TP parent graph."""
        from .hidden import TPSharded

        state = self.layers[layer]
        if (
            state.composed_input_addresses is None
            or source.fixed_addresses
            != state.composed_input_addresses
            or tuple(source.devices) != self.devices
            or source.shape
            != torch.Size((1, self.spec.in_features))
            or source.dtype != self.spec.input_dtype
            or source.ready_events is None
        ):
            raise ValueError(
                "row-parallel composed source layout mismatch"
            )
        return TPSharded(
            self.devices,
            tuple(state.local_inputs),
            self.spec.in_features,
            tuple(source.ready_events),
        )

    def output_hidden(self, layer: int):
        from .hidden import TPHidden

        state = self.layers[layer]
        if state.output_replicas is None or state.output_events is None:
            raise RuntimeError(
                "row-parallel linear outputs are unavailable"
            )
        return TPHidden(
            self.devices,
            tuple(state.output_replicas),
            tuple(state.output_events),
        )

    def run_sharded(self, layer: int, sharded, output=None):
        state = self.layers[layer]
        if state.graph_batch is None or not self.hidden_mode:
            raise RuntimeError(
                "row-parallel sharded-input graph is not captured"
            )
        if output is None:
            output = self.output_hidden(layer)
        if (
            tuple(sharded.devices) != self.devices
            or sharded.shape
            != torch.Size((1, self.spec.in_features))
            or sharded.dtype != self.spec.input_dtype
            or output.shape
            != torch.Size((1, self.spec.out_features))
            or output.dtype != self.spec.input_dtype
            or sharded.ready_events is None
            or output.ready_events is None
        ):
            raise ValueError("row-parallel hidden layout mismatch")
        if any(
            sharded.shards[rank].data_ptr()
            != state.local_inputs[rank].data_ptr()
            for rank in range(len(self.devices))
        ):
            raise ValueError(
                "row-parallel input must use captured fixed addresses"
            )
        rank_order = _no_owner_rank_order(self, state)
        state.graph_batch.launch_all_rank_from_events(
            [
                sharded.ready_events[rank].cuda_event
                for rank in rank_order
            ],
            state.contributions,
            list(output.replicas),
            [
                event.cuda_event
                for event in output.ready_events
            ],
        )
        return output

    def finalize_moe_full(
        self,
        layer: int,
        value: torch.Tensor,
        shared_partials,
        residual,
        output=None,
    ):
        """Dispatch a fixed owner row through the captured Row-TP graph."""
        from .hidden import TPSharded

        state = self.layers[layer]
        if (
            not self.spec.capture_owner_dispatch
            or state.source_event is None
            or value.device != self.devices[state.owner]
            or value.shape != state.source.shape
            or value.dtype != state.source.dtype
        ):
            raise ValueError(
                "full-owner MoE finalizer layout/capability mismatch"
            )
        owner_device = self.devices[state.owner]
        with torch.cuda.device(owner_device):
            state.source.copy_(value)
            state.source_event.record(
                torch.cuda.current_stream(owner_device)
            )
        dispatched = TPSharded(
            self.devices,
            tuple(state.local_inputs),
            self.spec.in_features,
            tuple(state.source_event for _ in self.devices),
        )
        return self.finalize_moe(
            layer,
            dispatched,
            shared_partials,
            residual,
            output=output,
        )

    def launch_partials(self, layer: int, sharded):
        """Launch Row-TP and expose FP32 partials without reducing them."""
        from .hidden import TPPartials

        state = self.layers[layer]
        if (
            state.graph_batch is None
            or not self.hidden_mode
            or state.events is None
        ):
            raise RuntimeError(
                "row-parallel partial graph is not captured"
            )
        self._validate_sharded_input(state, sharded)
        rank_order = _no_owner_rank_order(self, state)
        state.graph_batch.launch_from_events(
            [
                sharded.ready_events[rank].cuda_event
                for rank in rank_order
            ]
        )
        return TPPartials(
            tuple(self.devices[rank] for rank in rank_order),
            tuple(state.contributions),
            tuple(state.events),
        )

    def last_partials(self, layer: int):
        """Expose the captured fixed-address partials for diagnostics."""
        from .hidden import TPPartials

        state = self.layers[layer]
        if state.events is None:
            raise RuntimeError(
                "row-parallel partial graph is not captured"
            )
        rank_order = _no_owner_rank_order(self, state)
        return TPPartials(
            tuple(self.devices[rank] for rank in rank_order),
            tuple(state.contributions),
            tuple(state.events),
        )

    def finalize_moe(
        self,
        layer: int,
        sharded,
        shared_partials,
        residual,
        output=None,
    ):
        """Launch routed Row-TP then perform the sole MoE hidden collective."""
        state = self.layers[layer]
        if state.graph_batch is None or not self.hidden_mode:
            raise RuntimeError("TP MoE finalizer graph is not captured")
        self._validate_sharded_input(state, sharded)
        if output is None:
            output = self.output_hidden(layer)
        if (
            shared_partials.shape
            != torch.Size((1, self.spec.out_features))
            or tuple(shared_partials.devices)
            != tuple(
                self.devices[rank]
                for rank in _no_owner_rank_order(self, state)
            )
            or tuple(residual.devices) != tuple(output.devices)
            or residual.shape
            != torch.Size((1, self.spec.out_features))
            or output.shape != residual.shape
            or residual.dtype != self.spec.input_dtype
            or output.dtype != self.spec.input_dtype
            or residual.ready_events is None
            or output.ready_events is None
            or state.global_workspaces is None
        ):
            raise ValueError("TP MoE finalizer hidden layout mismatch")
        output_key = tuple(
            int(device.index) for device in output.devices
        )
        if tuple(output.devices) == self.devices:
            if (
                state.routed_workspaces is None
                or state.shared_workspaces is None
            ):
                raise RuntimeError(
                    "TP MoE local workspaces are unavailable"
                )
            routed_workspaces = state.routed_workspaces
            shared_workspaces = state.shared_workspaces
        else:
            workspace_pair = state.global_workspaces.get(output_key)
            if workspace_pair is None:
                routed_workspaces = []
                shared_workspaces = []
                for device in output.devices:
                    with torch.cuda.device(device):
                        routed_workspaces.append(
                            torch.empty_like(output.on_device(device))
                        )
                        shared_workspaces.append(
                            torch.empty_like(output.on_device(device))
                        )
                workspace_pair = (
                    routed_workspaces,
                    shared_workspaces,
                )
                state.global_workspaces[output_key] = workspace_pair
            routed_workspaces, shared_workspaces = workspace_pair
        rank_order = _no_owner_rank_order(self, state)
        state.graph_batch.launch_moe_all_rank_from_events(
            [
                sharded.ready_events[rank].cuda_event
                for rank in rank_order
            ],
            state.contributions,
            list(shared_partials.contributions),
            [
                event.cuda_event
                for event in shared_partials.ready_events
            ],
            list(residual.replicas),
            [
                event.cuda_event
                for event in residual.ready_events
            ],
            routed_workspaces,
            shared_workspaces,
            list(output.replicas),
            [
                event.cuda_event
                for event in output.ready_events
            ],
        )
        return output

    def _validate_sharded_input(self, state, sharded) -> None:
        if (
            tuple(sharded.devices) != self.devices
            or sharded.shape
            != torch.Size((1, self.spec.in_features))
            or sharded.dtype != self.spec.input_dtype
            or sharded.ready_events is None
        ):
            raise ValueError("row-parallel sharded input mismatch")
        if any(
            sharded.shards[rank].data_ptr()
            != state.local_inputs[rank].data_ptr()
            for rank in range(len(self.devices))
        ):
            raise ValueError(
                "row-parallel input must use captured fixed addresses"
            )

    def run_prepared(self, layer: int) -> torch.Tensor:
        state = self.layers[layer]
        if state.graph_batch is None:
            raise RuntimeError(
                "row-parallel linear graphs are not captured"
            )
        owner_device = self.devices[state.owner]
        with torch.cuda.device(owner_device):
            if self.hidden_mode:
                return self.run(layer, state.source)
            return state.graph_batch.launch_reduce(
                state.contributions,
                state.zero,
            )


@dataclass(frozen=True)
class RouteDownSpec:
    hidden_size: int
    routed_hidden_size: int
    expert_count: int


@dataclass
class _RouteDownLayer:
    owner: int
    source: torch.Tensor
    source_parts: list[torch.Tensor]
    router_inputs: list[torch.Tensor]
    down_inputs: list[torch.Tensor]
    router: list[torch.Tensor]
    routed_down: list[torch.Tensor]
    router_contributions: list[torch.Tensor]
    latent_contributions: list[torch.Tensor]
    zeros: list[torch.Tensor]
    graphs: list[torch.cuda.CUDAGraph] | None = None
    events: list[torch.cuda.Event] | None = None
    source_event: torch.cuda.Event | None = None
    input_events: list[torch.cuda.Event] | None = None
    router_output_replicas: list[torch.Tensor] | None = None
    latent_output_replicas: list[torch.Tensor] | None = None
    output_events: list[torch.cuda.Event] | None = None
    graph_batch: object | None = None
    bound_input_addresses: tuple[int, ...] | None = None


class TensorParallelRouteDown:
    """Fused Column-TP Router and Row-TP routed Down projection.

    Both projections consume the same normalized hidden state.  The input is
    kept locally replicated for the Router, whose expert-output rows are
    sharded.  Routed Down consumes a hidden-width shard and produces one FP32
    partial per rank.  Router logits are assembled by one small all-rank
    reduction while routed Down is summed in the same collective launch.  In
    TPHidden mode both results are published directly onto every rank; no
    hidden owner exists in the steady-state data flow.  The implementation is
    keyed by tensor shapes rather than a model family.
    """

    def __init__(
        self,
        devices: tuple[torch.device, ...],
        spec: RouteDownSpec,
    ) -> None:
        ranks = len(devices)
        if ranks <= 1 or spec.hidden_size % ranks:
            raise ValueError(
                "route/down hidden width must divide the TP size"
            )
        if spec.expert_count % ranks:
            raise ValueError(
                "Router expert rows must divide the TP size"
            )
        self.devices = devices
        self.spec = spec
        self.local_hidden = spec.hidden_size // ranks
        self.local_experts = spec.expert_count // ranks
        self.hidden_mode = (
            os.environ.get("TPQ_TP_HIDDEN", "0") != "0"
        )
        self.streams = [
            torch.cuda.Stream(device=device) for device in devices
        ]
        self.layers: dict[int, _RouteDownLayer] = {}

    def add_layer(
        self,
        layer: int,
        owner: int,
        router: torch.Tensor,
        routed_down: torch.Tensor,
    ) -> None:
        spec = self.spec
        if layer in self.layers:
            raise ValueError(f"route/down layer {layer} already exists")
        if (
            router.dtype != torch.float32
            or router.shape != (spec.expert_count, spec.hidden_size)
            or routed_down.dtype != torch.bfloat16
            or routed_down.shape
            != (spec.routed_hidden_size, spec.hidden_size)
        ):
            raise ValueError(
                f"route/down layer {layer} weight shape/dtype mismatch"
            )
        owner_device = self.devices[owner]
        with torch.cuda.device(owner_device):
            source = torch.empty(
                1,
                spec.hidden_size,
                dtype=torch.bfloat16,
                device=owner_device,
            )
            source_parts = list(
                source.split(self.local_hidden, dim=-1)
            )
            zeros = [
                torch.zeros(
                    1,
                    width,
                    dtype=torch.float32,
                    device=owner_device,
                )
                for width in (
                    spec.expert_count,
                    spec.routed_hidden_size,
                )
            ]
        router_inputs = []
        down_inputs = []
        router_weights = []
        routed_down_weights = []
        for rank, device in enumerate(self.devices):
            hidden_start = rank * self.local_hidden
            hidden_end = hidden_start + self.local_hidden
            expert_start = rank * self.local_experts
            expert_end = expert_start + self.local_experts
            with torch.cuda.device(device):
                router_inputs.append(
                    torch.empty(
                        1,
                        spec.hidden_size,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                )
                down_inputs.append(
                    torch.empty(
                        1,
                        self.local_hidden,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                )
                router_weights.append(
                    router[expert_start:expert_end]
                    .to(device)
                    .contiguous()
                )
                routed_down_weights.append(
                    routed_down[:, hidden_start:hidden_end]
                    .to(device)
                    .contiguous()
                )
        self.layers[layer] = _RouteDownLayer(
            owner=owner,
            source=source,
            source_parts=source_parts,
            router_inputs=router_inputs,
            down_inputs=down_inputs,
            router=router_weights,
            routed_down=routed_down_weights,
            router_contributions=[],
            latent_contributions=[],
            zeros=zeros,
        )

    def bind_input_hidden(self, layer: int, hidden) -> None:
        """Bind one fixed all-rank producer directly before graph capture."""
        state = self.layers[layer]
        if (
            tuple(hidden.devices) != self.devices
            or hidden.shape != torch.Size((1, self.spec.hidden_size))
            or hidden.dtype != torch.bfloat16
            or hidden.ready_events is None
        ):
            raise ValueError("route/down bound TPHidden layout mismatch")
        state.router_inputs = list(hidden.replicas)
        state.down_inputs = [
            hidden.replicas[rank][
                :,
                rank * self.local_hidden:
                (rank + 1) * self.local_hidden,
            ]
            for rank in range(len(self.devices))
        ]
        state.bound_input_addresses = hidden.fixed_addresses

    def capture(self) -> None:
        from ..fusedext import (
            make_tp_graph_launch_batch,
            tp_peer_copy_fused,
        )

        for device in self.devices:
            torch.cuda.synchronize(device)
        for state in self.layers.values():
            owner_device = self.devices[state.owner]
            with torch.cuda.device(owner_device):
                state.source.zero_()
            for rank, device in enumerate(self.devices):
                with torch.cuda.device(device):
                    state.router_inputs[rank].zero_()
                    state.down_inputs[rank].zero_()
            rank_order = (
                tuple(range(len(self.devices)))
                if self.hidden_mode
                else (
                    state.owner,
                    *(
                        rank
                        for rank in range(len(self.devices))
                        if rank != state.owner
                    ),
                )
            )
            graphs = []
            events = []
            ordered_streams = []
            router_contributions = []
            latent_contributions = []
            source_event = torch.cuda.Event()
            with torch.cuda.device(owner_device):
                source_event.record(torch.cuda.current_stream(owner_device))
                torch.cuda.synchronize(owner_device)
            for rank in rank_order:
                device = self.devices[rank]
                stream = self.streams[rank]
                expert_start = rank * self.local_experts
                expert_end = expert_start + self.local_experts
                with torch.cuda.device(device):
                    router_contribution = torch.zeros(
                        1,
                        self.spec.expert_count,
                        dtype=torch.float32,
                        device=device,
                    )

                def execute_rank():
                    if (
                        not self.hidden_mode
                        and (
                            not tp_peer_copy_fused(
                                state.source,
                                state.router_inputs[rank],
                            )
                            or not tp_peer_copy_fused(
                                state.source_parts[rank],
                                state.down_inputs[rank],
                            )
                        )
                    ):
                        raise RuntimeError(
                            "route/down input dispatch was rejected"
                        )
                    torch.mm(
                        state.router_inputs[rank].float(),
                        state.router[rank].t(),
                        out=router_contribution[
                            :, expert_start:expert_end
                        ],
                    )
                    latent_contribution = torch.mm(
                        state.down_inputs[rank],
                        state.routed_down[rank].t(),
                        out_dtype=torch.float32,
                    )
                    return router_contribution, latent_contribution

                with (
                    torch.cuda.device(device),
                    torch.cuda.stream(stream),
                ):
                    execute_rank()
                    stream.synchronize()
                    event = torch.cuda.Event()
                    graph = _new_cuda_graph()
                    with torch.cuda.graph(graph, stream=stream):
                        outputs = execute_rank()
                    _instantiate_retained_graph(graph)
                    event.record(stream)
                    stream.synchronize()
                graphs.append(graph)
                events.append(event)
                ordered_streams.append(stream)
                router_contributions.append(outputs[0])
                latent_contributions.append(outputs[1])
            state.graphs = graphs
            state.events = events
            state.source_event = source_event
            state.router_contributions = router_contributions
            state.latent_contributions = latent_contributions
            if self.hidden_mode:
                state.input_events = []
                state.router_output_replicas = []
                state.latent_output_replicas = []
                state.output_events = []
                for device in self.devices:
                    with torch.cuda.device(device):
                        input_event = torch.cuda.Event()
                        input_event.record(
                            torch.cuda.current_stream(device)
                        )
                        state.input_events.append(input_event)
                        state.router_output_replicas.append(
                            torch.empty(
                                1,
                                self.spec.expert_count,
                                dtype=torch.float32,
                                device=device,
                            )
                        )
                        state.latent_output_replicas.append(
                            torch.empty(
                                1,
                                self.spec.routed_hidden_size,
                                dtype=torch.bfloat16,
                                device=device,
                            )
                        )
                        output_event = torch.cuda.Event()
                        output_event.record(
                            torch.cuda.current_stream(device)
                        )
                        state.output_events.append(output_event)
            state.graph_batch = make_tp_graph_launch_batch(
                [
                    int(self.devices[rank].index)
                    for rank in rank_order
                ],
                graphs,
                ordered_streams,
                events,
                source_event,
            )

    def run(
        self,
        layer: int,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.layers[layer]
        if state.graph_batch is None:
            raise RuntimeError("route/down graphs are not captured")
        owner_device = self.devices[state.owner]
        if (
            value.device != owner_device
            or value.dtype != torch.bfloat16
            or value.shape != (1, self.spec.hidden_size)
        ):
            raise ValueError(
                "route/down input shape/dtype/device mismatch"
            )
        with torch.cuda.device(owner_device):
            state.source.copy_(value)
            outputs = state.graph_batch.launch_reduce_many(
                [
                    state.router_contributions,
                    state.latent_contributions,
                ],
                state.zeros,
            )
        return outputs[0], outputs[1]

    def input_sharded(self, layer: int):
        """Return the fixed per-rank hidden slices consumed by this graph."""
        from .hidden import TPSharded

        state = self.layers[layer]
        if state.input_events is None:
            raise RuntimeError(
                "route/down TPHidden input buffers are unavailable"
            )
        return TPSharded(
            self.devices,
            tuple(state.down_inputs),
            self.spec.hidden_size,
            tuple(state.input_events),
        )

    def output_hidden(self, layer: int):
        """Return replicated Router logits and routed latent on every rank."""
        from .hidden import TPHidden

        state = self.layers[layer]
        if (
            state.router_output_replicas is None
            or state.latent_output_replicas is None
            or state.output_events is None
        ):
            raise RuntimeError(
                "route/down TPHidden outputs are unavailable"
            )
        events = tuple(state.output_events)
        return (
            TPHidden(
                self.devices,
                tuple(state.router_output_replicas),
                events,
            ),
            TPHidden(
                self.devices,
                tuple(state.latent_output_replicas),
                events,
            ),
        )

    def retained_rank_graphs(
        self,
        layer: int,
    ) -> tuple[torch.cuda.CUDAGraph, ...]:
        """Return the fixed graph mapped to canonical rank order."""
        state = self.layers[layer]
        if state.graphs is None:
            raise RuntimeError("route/down retained graphs are unavailable")
        rank_order = _no_owner_rank_order(self, state)
        by_rank: list[torch.cuda.CUDAGraph | None] = [
            None
            for _ in self.devices
        ]
        for ordered_rank, rank in enumerate(rank_order):
            by_rank[rank] = state.graphs[ordered_rank]
        if any(graph is None for graph in by_rank):
            raise RuntimeError("route/down retained graph mapping is invalid")
        return tuple(by_rank)  # type: ignore[arg-type]

    def reduce_hidden_from_events(self, layer: int, ready_events):
        """Publish already-computed rank partials without relaunching graphs."""
        state = self.layers[layer]
        if (
            not self.hidden_mode
            or state.graph_batch is None
            or state.output_events is None
            or len(ready_events) != len(self.devices)
        ):
            raise RuntimeError(
                "route/down collective-only path is unavailable"
            )
        router_output, latent_output = self.output_hidden(layer)
        state.graph_batch.reduce_all_rank_many_from_events(
            [event.cuda_event for event in ready_events],
            [
                state.router_contributions,
                state.latent_contributions,
            ],
            [
                list(router_output.replicas),
                list(latent_output.replicas),
            ],
            [event.cuda_event for event in state.output_events],
        )
        return router_output, latent_output

    def run_hidden(self, layer: int, hidden):
        """Run sharded Router/Down and publish both reductions to all ranks."""
        state = self.layers[layer]
        if (
            not self.hidden_mode
            or state.graph_batch is None
            or state.input_events is None
            or state.output_events is None
        ):
            raise RuntimeError(
                "route/down all-rank graph is not captured"
            )
        sharded = self.input_sharded(layer)
        if (
            state.bound_input_addresses is not None
            and hidden.fixed_addresses == state.bound_input_addresses
        ):
            input_events = hidden.ready_events
        else:
            sharded.copy_from_replicated(hidden)
            for rank, device in enumerate(self.devices):
                hidden_rank = hidden.devices.index(device)
                with torch.cuda.device(device):
                    torch.cuda.current_stream(device).wait_event(
                        hidden.ready_events[hidden_rank]
                    )
                    state.router_inputs[rank].copy_(
                        hidden.replicas[hidden_rank]
                    )
                    state.input_events[rank].record(
                        torch.cuda.current_stream(device)
                    )
            input_events = tuple(state.input_events)
        router_output, latent_output = self.output_hidden(layer)
        state.graph_batch.launch_all_rank_many_from_events(
            [event.cuda_event for event in input_events],
            [
                state.router_contributions,
                state.latent_contributions,
            ],
            [
                list(router_output.replicas),
                list(latent_output.replicas),
            ],
            [event.cuda_event for event in state.output_events],
        )
        return router_output, latent_output


@dataclass(frozen=True)
class MoEPreludeSpec:
    hidden_size: int
    routed_hidden_size: int
    shared_intermediate_size: int
    expert_count: int
    activation: str
    activation_beta: float
    activation_linear_beta: float | None


@dataclass
class _MoEPreludeLayer:
    owner: int
    source: torch.Tensor
    local_inputs: list[torch.Tensor]
    router: list[torch.Tensor]
    routed_down: list[torch.Tensor]
    shared_gate_up: list[torch.Tensor]
    shared_down: list[torch.Tensor]
    router_contributions: list[torch.Tensor]
    latent_contributions: list[torch.Tensor]
    shared_contributions: list[torch.Tensor]
    zeros: list[torch.Tensor]
    graphs: list[torch.cuda.CUDAGraph] | None = None
    events: list[torch.cuda.Event] | None = None
    source_event: torch.cuda.Event | None = None
    graph_batch: object | None = None


class TensorParallelMoEPrelude:
    """One-broadcast TP prelude for Router, routed Down and shared MLP.

    All three branches consume the same normalized hidden state.  Capturing
    them in one rank-local Graph removes duplicate peer broadcasts and host
    launches.  The caller receives reduced FP32 router logits, routed latent
    and shared contribution; packed expert execution remains an independent
    capability and keeps its compact indices.
    """

    def __init__(
        self,
        devices: tuple[torch.device, ...],
        spec: MoEPreludeSpec,
    ) -> None:
        ranks = len(devices)
        if ranks <= 1:
            raise ValueError("MoE prelude requires at least 2 TP ranks")
        if (
            spec.hidden_size % ranks
            or spec.shared_intermediate_size % ranks
        ):
            raise ValueError(
                "MoE prelude hidden/intermediate widths must divide TP"
            )
        self.devices = devices
        self.spec = spec
        self.local_hidden = spec.hidden_size // ranks
        self.streams = [
            torch.cuda.Stream(device=device) for device in devices
        ]
        self.layers: dict[int, _MoEPreludeLayer] = {}

    def add_layer(
        self,
        layer: int,
        owner: int,
        router: torch.Tensor,
        routed_down: torch.Tensor,
        shared_gate_up: torch.Tensor,
        shared_down: torch.Tensor,
    ) -> None:
        spec = self.spec
        if layer in self.layers:
            raise ValueError(f"MoE prelude layer {layer} already exists")
        expected = (
            router.dtype == torch.float32
            and router.shape
            == (spec.expert_count, spec.hidden_size)
            and routed_down.dtype == torch.bfloat16
            and routed_down.shape
            == (spec.routed_hidden_size, spec.hidden_size)
            and shared_gate_up.dtype == torch.bfloat16
            and shared_gate_up.shape
            == (2 * spec.shared_intermediate_size, spec.hidden_size)
            and shared_down.dtype == torch.bfloat16
            and shared_down.shape
            == (spec.hidden_size, spec.shared_intermediate_size)
        )
        if not expected:
            raise ValueError(
                f"MoE prelude layer {layer} weight shape/dtype mismatch"
            )
        gate, up = shared_gate_up.chunk(2, dim=0)
        gate_parts = gate.chunk(len(self.devices), dim=0)
        up_parts = up.chunk(len(self.devices), dim=0)
        shared_down_parts = shared_down.chunk(
            len(self.devices),
            dim=1,
        )
        router_parts = router.split(self.local_hidden, dim=1)
        routed_down_parts = routed_down.split(
            self.local_hidden,
            dim=1,
        )
        local_inputs = []
        router_weights = []
        routed_down_weights = []
        shared_gate_up_weights = []
        shared_down_weights = []
        for rank, device in enumerate(self.devices):
            with torch.cuda.device(device):
                local_inputs.append(
                    torch.empty(
                        1,
                        spec.hidden_size,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                )
                router_weights.append(
                    router_parts[rank].to(device).contiguous()
                )
                routed_down_weights.append(
                    routed_down_parts[rank].to(device).contiguous()
                )
                shared_gate_up_weights.append(
                    torch.cat(
                        (
                            gate_parts[rank].to(device),
                            up_parts[rank].to(device),
                        ),
                        dim=0,
                    ).contiguous()
                )
                shared_down_weights.append(
                    shared_down_parts[rank].to(device).contiguous()
                )
        owner_device = self.devices[owner]
        with torch.cuda.device(owner_device):
            source = torch.empty(
                1,
                spec.hidden_size,
                dtype=torch.bfloat16,
                device=owner_device,
            )
            zeros = [
                torch.zeros(
                    1,
                    width,
                    dtype=torch.float32,
                    device=owner_device,
                )
                for width in (
                    spec.expert_count,
                    spec.routed_hidden_size,
                    spec.hidden_size,
                )
            ]
        self.layers[layer] = _MoEPreludeLayer(
            owner=owner,
            source=source,
            local_inputs=local_inputs,
            router=router_weights,
            routed_down=routed_down_weights,
            shared_gate_up=shared_gate_up_weights,
            shared_down=shared_down_weights,
            router_contributions=[],
            latent_contributions=[],
            shared_contributions=[],
            zeros=zeros,
        )

    def capture(self) -> None:
        from ..fusedext import (
            make_tp_graph_launch_batch,
            tp_peer_copy_fused,
        )
        from .api import gated_activation

        for device in self.devices:
            torch.cuda.synchronize(device)
        spec = self.spec
        for state in self.layers.values():
            owner_device = self.devices[state.owner]
            with torch.cuda.device(owner_device):
                state.source.zero_()
            rank_order = _no_owner_rank_order(self, state)
            graphs = []
            events = []
            ordered_streams = []
            router_contributions = []
            latent_contributions = []
            shared_contributions = []
            source_event = torch.cuda.Event()
            with torch.cuda.device(owner_device):
                source_event.record(torch.cuda.current_stream(owner_device))
                torch.cuda.synchronize(owner_device)
            for rank in rank_order:
                device = self.devices[rank]
                stream = self.streams[rank]
                start = rank * self.local_hidden
                end = start + self.local_hidden

                def execute_rank():
                    if not tp_peer_copy_fused(
                        state.source,
                        state.local_inputs[rank],
                    ):
                        raise RuntimeError(
                            "MoE prelude input dispatch was rejected"
                        )
                    local = state.local_inputs[rank]
                    local_slice = local[:, start:end]
                    router_partial = F.linear(
                        local_slice.float(),
                        state.router[rank],
                    )
                    latent_partial = F.linear(
                        local_slice,
                        state.routed_down[rank],
                    ).float()
                    projected = F.linear(
                        local,
                        state.shared_gate_up[rank],
                    )
                    gate, up = projected.chunk(2, dim=-1)
                    activated = gated_activation(
                        gate,
                        up,
                        activation=spec.activation,
                        beta=spec.activation_beta,
                        linear_beta=spec.activation_linear_beta,
                        output=gate,
                    )
                    if activated is None:
                        raise RuntimeError(
                            "MoE prelude gated activation was rejected"
                        )
                    shared_partial = F.linear(
                        activated,
                        state.shared_down[rank],
                    ).float()
                    return (
                        router_partial,
                        latent_partial,
                        shared_partial,
                    )

                with (
                    torch.cuda.device(device),
                    torch.cuda.stream(stream),
                ):
                    execute_rank()
                    stream.synchronize()
                    event = torch.cuda.Event()
                    graph = _new_cuda_graph()
                    with torch.cuda.graph(graph, stream=stream):
                        outputs = execute_rank()
                    _instantiate_retained_graph(graph)
                    event.record(stream)
                    stream.synchronize()
                graphs.append(graph)
                events.append(event)
                ordered_streams.append(stream)
                router_contributions.append(outputs[0])
                latent_contributions.append(outputs[1])
                shared_contributions.append(outputs[2])
            state.graphs = graphs
            state.events = events
            state.source_event = source_event
            state.router_contributions = router_contributions
            state.latent_contributions = latent_contributions
            state.shared_contributions = shared_contributions
            state.graph_batch = make_tp_graph_launch_batch(
                [
                    int(self.devices[rank].index)
                    for rank in rank_order
                ],
                graphs,
                ordered_streams,
                events,
                source_event,
            )

    def run(
        self,
        layer: int,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.layers[layer]
        if state.graph_batch is None:
            raise RuntimeError("MoE prelude graphs are not captured")
        owner_device = self.devices[state.owner]
        if (
            value.device != owner_device
            or value.dtype != torch.bfloat16
            or value.shape != (1, self.spec.hidden_size)
        ):
            raise ValueError("MoE prelude input shape/dtype/device mismatch")
        with torch.cuda.device(owner_device):
            state.source.copy_(value)
            outputs = state.graph_batch.launch_reduce_many(
                [
                    state.router_contributions,
                    state.latent_contributions,
                    state.shared_contributions,
                ],
                state.zeros,
            )
        return outputs[0], outputs[1], outputs[2]


@dataclass(frozen=True)
class KDASpec:
    hidden_size: int
    heads: int
    head_dim: int
    gate_rank: int
    rms_eps: float
    gate_lower_bound: float
    conv_history: int


@dataclass
class _KDALayer:
    owner: int
    source: torch.Tensor
    local_inputs: list[torch.Tensor]
    input_projection: list[torch.Tensor]
    gate_projection: list[torch.Tensor]
    conv_weights: list[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]
    a_log: list[torch.Tensor]
    dt_bias: list[torch.Tensor]
    norm_weight: list[torch.Tensor]
    output_projection: list[torch.Tensor]
    recurrent_state: list[torch.Tensor]
    conv_state: list[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]
    workspaces: list[torch.Tensor]
    recurrent_outputs: list[torch.Tensor]
    contributions: list[torch.Tensor]
    zero: torch.Tensor
    graphs: list[torch.cuda.CUDAGraph] | None = None
    events: list[torch.cuda.Event] | None = None
    source_event: torch.cuda.Event | None = None
    input_events: list[torch.cuda.Event] | None = None
    output_replicas: list[torch.Tensor] | None = None
    output_events: list[torch.cuda.Event] | None = None
    graph_batch: object | None = None
    composed_input_addresses: tuple[int, ...] | None = None


class TensorParallelKDA:
    """Head-parallel recurrent attention selected by a KDA config."""

    def __init__(
        self,
        devices: tuple[torch.device, ...],
        spec: KDASpec,
    ) -> None:
        if len(devices) <= 1 or spec.heads % len(devices):
            raise ValueError("KDA heads must divide the TP size")
        self.devices = devices
        self.spec = spec
        self.local_heads = spec.heads // len(devices)
        self.local_width = self.local_heads * spec.head_dim
        self.hidden_mode = (
            os.environ.get("TPQ_TP_HIDDEN", "0") != "0"
        )
        self.streams = [
            torch.cuda.Stream(device=device) for device in devices
        ]
        self.layers: dict[int, _KDALayer] = {}

    def add_layer(
        self,
        layer: int,
        owner: int,
        combined_input: torch.Tensor,
        gate_projection: torch.Tensor,
        conv_weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm_weight: torch.Tensor,
        output_projection: torch.Tensor,
    ) -> None:
        spec = self.spec
        total_width = spec.heads * spec.head_dim
        q, k, v, g, gate_a, beta = combined_input.split(
            (
                total_width,
                total_width,
                total_width,
                total_width,
                spec.gate_rank,
                spec.heads,
            ),
            dim=0,
        )
        head_parts = [
            value.chunk(len(self.devices), dim=0)
            for value in (q, k, v, g)
        ]
        beta_parts = beta.chunk(len(self.devices), dim=0)
        gate_b_parts = gate_projection.chunk(
            len(self.devices),
            dim=0,
        )
        conv_parts = [
            weight.chunk(len(self.devices), dim=0)
            for weight in conv_weights
        ]
        dt_parts = dt_bias.chunk(len(self.devices), dim=0)
        output_parts = output_projection.chunk(
            len(self.devices),
            dim=1,
        )
        local_inputs: list[torch.Tensor] = []
        input_weights: list[torch.Tensor] = []
        gate_weights: list[torch.Tensor] = []
        local_conv_weights = []
        local_a_log = []
        local_dt_bias = []
        local_norm_weight = []
        local_output_projection = []
        recurrent_state = []
        conv_state = []
        workspaces = []
        recurrent_outputs = []
        for rank, device in enumerate(self.devices):
            with torch.cuda.device(device):
                local_inputs.append(
                    torch.empty(
                        1,
                        spec.hidden_size,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                )
                input_weights.append(
                    torch.cat(
                        (
                            head_parts[0][rank].to(device),
                            head_parts[1][rank].to(device),
                            head_parts[2][rank].to(device),
                            head_parts[3][rank].to(device),
                            gate_a.to(device),
                            beta_parts[rank].to(device),
                        ),
                        dim=0,
                    ).contiguous()
                )
                gate_weights.append(
                    gate_b_parts[rank].to(device).contiguous()
                )
                local_conv_weights.append(
                    tuple(
                        parts[rank].to(device).contiguous()
                        for parts in conv_parts
                    )
                )
                local_a_log.append(a_log.to(device).contiguous())
                local_dt_bias.append(
                    dt_parts[rank].to(device).contiguous()
                )
                local_norm_weight.append(
                    norm_weight.to(device).contiguous()
                )
                local_output_projection.append(
                    output_parts[rank].to(device).contiguous()
                )
                recurrent_state.append(
                    torch.zeros(
                        self.local_heads,
                        spec.head_dim,
                        spec.head_dim,
                        dtype=torch.float32,
                        device=device,
                    )
                )
                conv_state.append(
                    tuple(
                        torch.zeros(
                            self.local_width,
                            spec.conv_history,
                            dtype=torch.bfloat16,
                            device=device,
                        )
                        for _ in range(3)
                    )
                )
                workspaces.append(
                    torch.empty(
                        3 * self.local_width,
                        dtype=torch.float32,
                        device=device,
                    )
                )
                recurrent_outputs.append(
                    torch.empty(
                        self.local_heads,
                        spec.head_dim,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                )
        owner_device = self.devices[owner]
        with torch.cuda.device(owner_device):
            source = torch.empty(
                1,
                spec.hidden_size,
                dtype=torch.bfloat16,
                device=owner_device,
            )
            zero = torch.zeros(
                1,
                spec.hidden_size,
                dtype=torch.float32,
                device=owner_device,
            )
        self.layers[layer] = _KDALayer(
            owner=owner,
            source=source,
            local_inputs=local_inputs,
            input_projection=input_weights,
            gate_projection=gate_weights,
            conv_weights=local_conv_weights,
            a_log=local_a_log,
            dt_bias=local_dt_bias,
            norm_weight=local_norm_weight,
            output_projection=local_output_projection,
            recurrent_state=recurrent_state,
            conv_state=conv_state,
            workspaces=workspaces,
            recurrent_outputs=recurrent_outputs,
            contributions=[],
            zero=zero,
        )

    def capture(self) -> None:
        from ..fusedext import (
            make_tp_graph_launch_batch,
            tp_peer_copy_fused,
        )
        from .api import attention_step

        for device in self.devices:
            torch.cuda.synchronize(device)
        spec = self.spec
        split = (
            self.local_width,
            self.local_width,
            self.local_width,
            self.local_width,
            spec.gate_rank,
            self.local_heads,
        )
        for state in self.layers.values():
            owner_device = self.devices[state.owner]
            with torch.cuda.device(owner_device):
                state.source.zero_()
            for rank, device in enumerate(self.devices):
                with torch.cuda.device(device):
                    state.local_inputs[rank].zero_()
            rank_order = _no_owner_rank_order(self, state)
            graphs = []
            events = []
            contributions = []
            ordered_streams = []
            source_event = torch.cuda.Event()
            with torch.cuda.device(owner_device):
                source_event.record(torch.cuda.current_stream(owner_device))
                torch.cuda.synchronize(owner_device)
            for rank in rank_order:
                device = self.devices[rank]
                stream = self.streams[rank]

                def execute_rank() -> torch.Tensor:
                    if (
                        not self.hidden_mode
                        and not tp_peer_copy_fused(
                            state.source,
                            state.local_inputs[rank],
                        )
                    ):
                        raise RuntimeError(
                            "TP KDA input dispatch was rejected"
                        )
                    projected = F.linear(
                        state.local_inputs[rank],
                        state.input_projection[rank],
                    ).split(split, dim=-1)
                    query, key, value, output_gate = (
                        item.reshape(
                            self.local_heads,
                            spec.head_dim,
                        )
                        for item in projected[:4]
                    )
                    if not attention_step(
                        "short_conv3",
                        "cuda",
                        query=query.reshape(-1),
                        key=key.reshape(-1),
                        value=value.reshape(-1),
                        states=state.conv_state[rank],
                        weights=state.conv_weights[rank],
                    ):
                        raise RuntimeError(
                            "TP KDA short convolution was rejected"
                        )
                    recurrent_gate = F.linear(
                        projected[4],
                        state.gate_projection[rank],
                    ).view(self.local_heads, spec.head_dim)
                    recurrent = attention_step(
                        "kda_recurrent",
                        "cuda",
                        query=query,
                        key=key,
                        value=value,
                        gate=recurrent_gate,
                        beta=projected[5].reshape(
                            self.local_heads
                        ).float(),
                        a_log=state.a_log[rank],
                        dt_bias=state.dt_bias[rank],
                        state=state.recurrent_state[rank],
                        workspace=state.workspaces[rank],
                        output=state.recurrent_outputs[rank],
                        lower_bound=spec.gate_lower_bound,
                    )
                    normalized = attention_step(
                        "gated_rmsnorm",
                        "cuda",
                        value=recurrent,
                        gate=output_gate.reshape(
                            self.local_heads,
                            spec.head_dim,
                        ),
                        weight=state.norm_weight[rank],
                        output=recurrent,
                        eps=spec.rms_eps,
                    )
                    if normalized is None:
                        raise RuntimeError(
                            "TP KDA gated RMSNorm was rejected"
                        )
                    return F.linear(
                        normalized.reshape(1, -1),
                        state.output_projection[rank],
                    ).float()

                with (
                    torch.cuda.device(device),
                    torch.cuda.stream(stream),
                ):
                    execute_rank()
                    stream.synchronize()
                    event = torch.cuda.Event()
                    graph = _new_cuda_graph()
                    with torch.cuda.graph(graph, stream=stream):
                        contribution = execute_rank()
                    _instantiate_retained_graph(graph)
                    event.record(stream)
                    stream.synchronize()
                graphs.append(graph)
                events.append(event)
                contributions.append(contribution)
                ordered_streams.append(stream)
            state.graphs = graphs
            state.events = events
            state.source_event = source_event
            state.input_events = []
            for device in self.devices:
                with torch.cuda.device(device):
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(device))
                    state.input_events.append(event)
            state.output_replicas = []
            state.output_events = []
            for device in self.devices:
                with torch.cuda.device(device):
                    state.output_replicas.append(
                        torch.empty(
                            1,
                            spec.hidden_size,
                            dtype=torch.bfloat16,
                            device=device,
                        )
                    )
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(device))
                    state.output_events.append(event)
            state.contributions = contributions
            state.graph_batch = make_tp_graph_launch_batch(
                [
                    int(self.devices[rank].index)
                    for rank in rank_order
                ],
                graphs,
                ordered_streams,
                events,
                source_event,
            )

    def run(self, layer: int, value: torch.Tensor) -> torch.Tensor:
        state = self.layers[layer]
        if state.graph_batch is None:
            raise RuntimeError("TP KDA graphs are not captured")
        owner_device = self.devices[state.owner]
        with torch.cuda.device(owner_device):
            state.source.copy_(value)
            return state.graph_batch.launch_reduce(
                state.contributions,
                state.zero,
            )

    def input_buffer(self, layer: int) -> torch.Tensor:
        return self.layers[layer].source

    def input_hidden(self, layer: int):
        """Return the fixed per-rank buffers captured by this executor."""
        from .hidden import TPHidden

        state = self.layers[layer]
        if state.input_events is None:
            raise RuntimeError("TP KDA graphs are not captured")
        return TPHidden(
            self.devices,
            tuple(state.local_inputs),
            tuple(state.input_events),
        )

    def compose_normalize_prelude(
        self,
        layer: int,
        source,
        residual,
        active_rows: int,
        projections,
        norm_weights,
        post_norm_weights,
        workspaces,
        eps: float,
    ) -> None:
        _compose_normalize_prelude(
            self,
            layer,
            source,
            residual,
            active_rows,
            projections,
            norm_weights,
            post_norm_weights,
            workspaces,
            eps,
        )

    def run_hidden(
        self,
        layer: int,
        hidden,
        output=None,
    ):
        """Run Column→Row attention and publish the result on every rank."""
        state = self.layers[layer]
        if state.graph_batch is None or not self.hidden_mode:
            raise RuntimeError("TP KDA TPHidden graph is not captured")
        if output is None:
            output = self.output_hidden(layer)
        input_events = self.prepare_hidden_events(
            layer,
            hidden,
            output=output,
        )
        rank_order = _no_owner_rank_order(self, state)
        state.graph_batch.launch_all_rank_from_events(
            [
                input_events[rank].cuda_event
                for rank in rank_order
            ],
            state.contributions,
            list(output.replicas),
            [
                event.cuda_event
                for event in output.ready_events
            ],
        )
        return output

    def prepare_hidden_events(
        self,
        layer: int,
        hidden,
        position: int | None = None,
        *,
        output=None,
    ):
        """Validate a fixed KDA input for a larger all-rank layer plan."""
        del position
        state = self.layers[layer]
        if output is None:
            output = self.output_hidden(layer)
        self._validate_hidden_pair(state, hidden, output)
        if hidden.ready_events is None or output.ready_events is None:
            raise ValueError("CUDA TPHidden requires ready events")
        return hidden.ready_events

    def output_hidden(self, layer: int):
        """Return this layer's stable all-rank Row-TP output buffers."""
        from .hidden import TPHidden

        state = self.layers[layer]
        if state.output_replicas is None or state.output_events is None:
            raise RuntimeError("TP KDA output buffers are unavailable")
        return TPHidden(
            self.devices,
            tuple(state.output_replicas),
            tuple(state.output_events),
        )

    def _validate_hidden_pair(self, state, hidden, output) -> None:
        if (
            tuple(hidden.devices) != self.devices
            or hidden.shape != torch.Size((1, self.spec.hidden_size))
            or output.shape != hidden.shape
            or hidden.dtype != torch.bfloat16
            or output.dtype != torch.bfloat16
        ):
            raise ValueError("TP KDA TPHidden layout mismatch")
        expected_addresses = (
            state.composed_input_addresses
            if state.composed_input_addresses is not None
            else tuple(
                item.data_ptr() for item in state.local_inputs
            )
        )
        if hidden.fixed_addresses != expected_addresses:
            raise ValueError(
                "TP KDA input must use its captured fixed addresses"
            )

    def run_prepared(self, layer: int) -> torch.Tensor:
        state = self.layers[layer]
        if state.graph_batch is None:
            raise RuntimeError("TP KDA graphs are not captured")
        owner_device = self.devices[state.owner]
        with torch.cuda.device(owner_device):
            return state.graph_batch.launch_reduce(
                state.contributions,
                state.zero,
            )

    def reset(self) -> None:
        for state in self.layers.values():
            for rank, device in enumerate(self.devices):
                with torch.cuda.device(device):
                    state.recurrent_state[rank].zero_()
                    for conv in state.conv_state[rank]:
                        conv.zero_()


@dataclass(frozen=True)
class MLASpec:
    hidden_size: int
    heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    max_ctx: int
    rms_eps: float


@dataclass
class _MLALayer:
    owner: int
    source: torch.Tensor
    source_position: torch.Tensor
    local_inputs: list[torch.Tensor]
    local_positions: list[torch.Tensor]
    input_projection: list[torch.Tensor]
    query_norm: list[torch.Tensor]
    query_projection: list[torch.Tensor]
    kv_norm: list[torch.Tensor]
    key_absorb: list[torch.Tensor]
    value_absorb: list[torch.Tensor]
    output_projection: list[torch.Tensor]
    latent_cache: list[torch.Tensor]
    rope_cache: list[torch.Tensor]
    score_workspace: list[torch.Tensor]
    attention_output: list[torch.Tensor]
    contributions: list[torch.Tensor]
    zero: torch.Tensor
    graphs: list[torch.cuda.CUDAGraph] | None = None
    events: list[torch.cuda.Event] | None = None
    source_event: torch.cuda.Event | None = None
    input_events: list[torch.cuda.Event] | None = None
    output_replicas: list[torch.Tensor] | None = None
    output_events: list[torch.cuda.Event] | None = None
    graph_batch: object | None = None
    composed_input_addresses: tuple[int, ...] | None = None


class TensorParallelMLA:
    """Head-parallel latent attention with dynamic device-side length.

    Head-dependent Q-B/G/O and absorbed KV factors are sharded.  The small
    MQA low-rank projections are replicated so every rank can keep its local
    KV state without returning to a layer-owner bottleneck.  Each rank is one
    fixed CUDA Graph; only the source hidden state and device position change.
    """

    def __init__(
        self,
        devices: tuple[torch.device, ...],
        spec: MLASpec,
    ) -> None:
        if len(devices) <= 1 or spec.heads % len(devices):
            raise ValueError("MLA heads must divide the TP size")
        if spec.max_ctx <= 0:
            raise ValueError("MLA max_ctx must be positive")
        self.devices = devices
        self.spec = spec
        self.local_heads = spec.heads // len(devices)
        self.hidden_mode = (
            os.environ.get("TPQ_TP_HIDDEN", "0") != "0"
        )
        self.streams = [
            torch.cuda.Stream(device=device) for device in devices
        ]
        self.layers: dict[int, _MLALayer] = {}

    def add_layer(
        self,
        layer: int,
        owner: int,
        combined_input: torch.Tensor,
        query_norm: torch.Tensor,
        query_projection: torch.Tensor,
        kv_norm: torch.Tensor,
        key_absorb: torch.Tensor,
        value_absorb: torch.Tensor,
        output_projection: torch.Tensor,
    ) -> None:
        spec = self.spec
        q_width = spec.qk_nope_head_dim + spec.qk_rope_head_dim
        expected_input_rows = (
            spec.q_lora_rank
            + spec.kv_lora_rank
            + spec.qk_rope_head_dim
            + spec.heads * spec.v_head_dim
        )
        if (
            combined_input.dtype != torch.bfloat16
            or combined_input.shape
            != (expected_input_rows, spec.hidden_size)
            or query_projection.shape
            != (spec.heads * q_width, spec.q_lora_rank)
            or key_absorb.shape
            != (
                spec.heads,
                spec.qk_nope_head_dim,
                spec.kv_lora_rank,
            )
            or value_absorb.shape
            != (
                spec.heads,
                spec.v_head_dim,
                spec.kv_lora_rank,
            )
            or output_projection.shape
            != (
                spec.hidden_size,
                spec.heads * spec.v_head_dim,
            )
        ):
            raise ValueError(
                f"TP MLA layer {layer} weight shape/dtype mismatch"
            )
        query_a, kv_a, gate = combined_input.split(
            (
                spec.q_lora_rank,
                spec.kv_lora_rank + spec.qk_rope_head_dim,
                spec.heads * spec.v_head_dim,
            ),
            dim=0,
        )
        gate_parts = gate.chunk(len(self.devices), dim=0)
        query_parts = (
            query_projection.view(
                spec.heads,
                q_width,
                spec.q_lora_rank,
            )
            .chunk(len(self.devices), dim=0)
        )
        key_parts = key_absorb.chunk(len(self.devices), dim=0)
        value_parts = value_absorb.chunk(len(self.devices), dim=0)
        output_parts = output_projection.chunk(
            len(self.devices),
            dim=1,
        )
        local_inputs = []
        local_positions = []
        input_weights = []
        query_norms = []
        query_weights = []
        kv_norms = []
        key_weights = []
        value_weights = []
        output_weights = []
        latent_cache = []
        rope_cache = []
        score_workspace = []
        attention_output = []
        for rank, device in enumerate(self.devices):
            with torch.cuda.device(device):
                local_inputs.append(
                    torch.empty(
                        1,
                        spec.hidden_size,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                )
                local_positions.append(
                    torch.zeros(1, dtype=torch.long, device=device)
                )
                input_weights.append(
                    torch.cat(
                        (
                            query_a.to(device),
                            kv_a.to(device),
                            gate_parts[rank].to(device),
                        ),
                        dim=0,
                    ).contiguous()
                )
                query_norms.append(query_norm.to(device).contiguous())
                query_weights.append(
                    query_parts[rank]
                    .reshape(
                        self.local_heads * q_width,
                        spec.q_lora_rank,
                    )
                    .to(device)
                    .contiguous()
                )
                kv_norms.append(kv_norm.to(device).contiguous())
                key_weights.append(
                    key_parts[rank].to(device).contiguous()
                )
                value_weights.append(
                    value_parts[rank].to(device).contiguous()
                )
                output_weights.append(
                    output_parts[rank].to(device).contiguous()
                )
                latent_cache.append(
                    torch.zeros(
                        spec.max_ctx,
                        spec.kv_lora_rank,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                )
                rope_cache.append(
                    torch.zeros(
                        spec.max_ctx,
                        spec.qk_rope_head_dim,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                )
                score_workspace.append(
                    torch.empty(
                        self.local_heads,
                        spec.max_ctx,
                        dtype=torch.float32,
                        device=device,
                    )
                )
                attention_output.append(
                    torch.empty(
                        self.local_heads,
                        1,
                        spec.kv_lora_rank,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                )
        owner_device = self.devices[owner]
        with torch.cuda.device(owner_device):
            source = torch.empty(
                1,
                spec.hidden_size,
                dtype=torch.bfloat16,
                device=owner_device,
            )
            source_position = torch.zeros(
                1,
                dtype=torch.long,
                device=owner_device,
            )
            zero = torch.zeros(
                1,
                spec.hidden_size,
                dtype=torch.float32,
                device=owner_device,
            )
        self.layers[layer] = _MLALayer(
            owner=owner,
            source=source,
            source_position=source_position,
            local_inputs=local_inputs,
            local_positions=local_positions,
            input_projection=input_weights,
            query_norm=query_norms,
            query_projection=query_weights,
            kv_norm=kv_norms,
            key_absorb=key_weights,
            value_absorb=value_weights,
            output_projection=output_weights,
            latent_cache=latent_cache,
            rope_cache=rope_cache,
            score_workspace=score_workspace,
            attention_output=attention_output,
            contributions=[],
            zero=zero,
        )

    def capture(self) -> None:
        from ..fusedext import (
            make_tp_graph_launch_batch,
            tp_peer_copy_fused,
        )
        from .api import attention_step, rmsnorm

        for device in self.devices:
            torch.cuda.synchronize(device)
        spec = self.spec
        q_width = spec.qk_nope_head_dim + spec.qk_rope_head_dim
        split = (
            spec.q_lora_rank,
            spec.kv_lora_rank + spec.qk_rope_head_dim,
            self.local_heads * spec.v_head_dim,
        )
        scale_denominator = float(q_width**0.5)
        for state in self.layers.values():
            owner_device = self.devices[state.owner]
            with torch.cuda.device(owner_device):
                state.source.zero_()
                state.source_position.zero_()
            for rank, device in enumerate(self.devices):
                with torch.cuda.device(device):
                    state.local_inputs[rank].zero_()
                    state.local_positions[rank].zero_()
            rank_order = _no_owner_rank_order(self, state)
            graphs = []
            events = []
            contributions = []
            ordered_streams = []
            source_event = torch.cuda.Event()
            with torch.cuda.device(owner_device):
                source_event.record(torch.cuda.current_stream(owner_device))
                torch.cuda.synchronize(owner_device)
            for rank in rank_order:
                device = self.devices[rank]
                stream = self.streams[rank]

                def execute_rank() -> torch.Tensor:
                    if (
                        not self.hidden_mode
                        and (
                            not tp_peer_copy_fused(
                                state.source,
                                state.local_inputs[rank],
                            )
                            or not tp_peer_copy_fused(
                                state.source_position,
                                state.local_positions[rank],
                            )
                        )
                    ):
                        raise RuntimeError(
                            "TP MLA input dispatch was rejected"
                        )
                    query_source, compressed, output_gate = F.linear(
                        state.local_inputs[rank],
                        state.input_projection[rank],
                    ).split(split, dim=-1)
                    query_source = rmsnorm(
                        query_source,
                        state.query_norm[rank],
                        1e-6,
                    )
                    if query_source is None:
                        raise RuntimeError("TP MLA query RMSNorm unavailable")
                    query = F.linear(
                        query_source,
                        state.query_projection[rank],
                    ).view(self.local_heads, q_width)
                    query_nope, query_rope = query.split(
                        (
                            spec.qk_nope_head_dim,
                            spec.qk_rope_head_dim,
                        ),
                        dim=-1,
                    )
                    latent, key_rope = compressed.split(
                        (
                            spec.kv_lora_rank,
                            spec.qk_rope_head_dim,
                        ),
                        dim=-1,
                    )
                    latent = rmsnorm(
                        latent,
                        state.kv_norm[rank],
                        1e-6,
                    )
                    if latent is None:
                        raise RuntimeError("TP MLA KV RMSNorm unavailable")
                    state.latent_cache[rank].index_copy_(
                        0,
                        state.local_positions[rank],
                        latent,
                    )
                    state.rope_cache[rank].index_copy_(
                        0,
                        state.local_positions[rank],
                        key_rope,
                    )
                    absorbed_query = torch.bmm(
                        query_nope[:, None, :],
                        state.key_absorb[rank],
                    )
                    context = attention_step(
                        "compressed_kv_decode",
                        "cuda",
                        query_nope=absorbed_query,
                        query_rope=query_rope[:, None, :],
                        latent_cache=state.latent_cache[rank],
                        rope_cache=state.rope_cache[rank],
                        position=state.local_positions[rank],
                        scale_denominator=scale_denominator,
                        score_workspace=state.score_workspace[rank],
                        output=state.attention_output[rank],
                    )
                    if context is None:
                        raise RuntimeError("TP MLA decode core unavailable")
                    output = torch.bmm(
                        context,
                        state.value_absorb[rank].transpose(1, 2),
                    ).reshape(1, -1)
                    output.mul_(output_gate.sigmoid())
                    return F.linear(
                        output,
                        state.output_projection[rank],
                    ).float()

                with (
                    torch.cuda.device(device),
                    torch.cuda.stream(stream),
                ):
                    execute_rank()
                    stream.synchronize()
                    event = torch.cuda.Event()
                    graph = _new_cuda_graph()
                    with torch.cuda.graph(graph, stream=stream):
                        contribution = execute_rank()
                    _instantiate_retained_graph(graph)
                    event.record(stream)
                    stream.synchronize()
                graphs.append(graph)
                events.append(event)
                contributions.append(contribution)
                ordered_streams.append(stream)
            state.graphs = graphs
            state.events = events
            state.source_event = source_event
            state.input_events = []
            for device in self.devices:
                with torch.cuda.device(device):
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(device))
                    state.input_events.append(event)
            state.output_replicas = []
            state.output_events = []
            for device in self.devices:
                with torch.cuda.device(device):
                    state.output_replicas.append(
                        torch.empty(
                            1,
                            spec.hidden_size,
                            dtype=torch.bfloat16,
                            device=device,
                        )
                    )
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(device))
                    state.output_events.append(event)
            state.contributions = contributions
            state.graph_batch = make_tp_graph_launch_batch(
                [
                    int(self.devices[rank].index)
                    for rank in rank_order
                ],
                graphs,
                ordered_streams,
                events,
                source_event,
            )

    def run(
        self,
        layer: int,
        value: torch.Tensor,
        position: int,
    ) -> torch.Tensor:
        state = self.layers[layer]
        if state.graph_batch is None:
            raise RuntimeError("TP MLA graphs are not captured")
        if not 0 <= int(position) < self.spec.max_ctx:
            raise ValueError("TP MLA position exceeds max_ctx")
        owner_device = self.devices[state.owner]
        if value.device != owner_device:
            raise ValueError("TP MLA input is not on its owner rank")
        with torch.cuda.device(owner_device):
            state.source.copy_(value)
            state.source_position.fill_(int(position))
            return state.graph_batch.launch_reduce(
                state.contributions,
                state.zero,
            )

    def input_buffer(self, layer: int) -> torch.Tensor:
        return self.layers[layer].source

    def input_hidden(self, layer: int):
        """Return the fixed per-rank buffers captured by this executor."""
        from .hidden import TPHidden

        state = self.layers[layer]
        if state.input_events is None:
            raise RuntimeError("TP MLA graphs are not captured")
        return TPHidden(
            self.devices,
            tuple(state.local_inputs),
            tuple(state.input_events),
        )

    def run_hidden(
        self,
        layer: int,
        hidden,
        position: int,
        output=None,
    ):
        """Run Column→Row MLA and publish the result on every rank."""
        state = self.layers[layer]
        if state.graph_batch is None or not self.hidden_mode:
            raise RuntimeError("TP MLA TPHidden graph is not captured")
        if not 0 <= int(position) < self.spec.max_ctx:
            raise ValueError("TP MLA position exceeds max_ctx")
        if output is None:
            output = self.output_hidden(layer)
        input_events = self.prepare_hidden_events(
            layer,
            hidden,
            position,
            output=output,
        )
        rank_order = _no_owner_rank_order(self, state)
        state.graph_batch.launch_all_rank_from_events(
            [
                input_events[rank].cuda_event
                for rank in rank_order
            ],
            state.contributions,
            list(output.replicas),
            [
                event.cuda_event
                for event in output.ready_events
            ],
        )
        return output

    def prepare_hidden_events(
        self,
        layer: int,
        hidden,
        position: int | None = None,
        *,
        output=None,
    ):
        """Prepare fixed MLA position events without launching its Graph."""
        if position is None:
            raise ValueError("TP MLA position is required")
        state = self.layers[layer]
        if not 0 <= int(position) < self.spec.max_ctx:
            raise ValueError("TP MLA position exceeds max_ctx")
        if output is None:
            output = self.output_hidden(layer)
        self._validate_hidden_pair(state, hidden, output)
        if hidden.ready_events is None or output.ready_events is None:
            raise ValueError("CUDA TPHidden requires ready events")
        if state.input_events is None:
            raise RuntimeError("TP MLA input events are unavailable")
        for rank, device in enumerate(self.devices):
            with torch.cuda.device(device):
                stream = torch.cuda.current_stream(device)
                stream.wait_event(hidden.ready_events[rank])
                state.local_positions[rank].fill_(int(position))
                state.input_events[rank].record(stream)
        return tuple(state.input_events)

    def compose_normalize_prelude(
        self,
        layer: int,
        source,
        residual,
        active_rows: int,
        projections,
        norm_weights,
        post_norm_weights,
        workspaces,
        eps: float,
    ) -> None:
        _compose_normalize_prelude(
            self,
            layer,
            source,
            residual,
            active_rows,
            projections,
            norm_weights,
            post_norm_weights,
            workspaces,
            eps,
        )

    def output_hidden(self, layer: int):
        """Return this layer's stable all-rank Row-TP output buffers."""
        from .hidden import TPHidden

        state = self.layers[layer]
        if state.output_replicas is None or state.output_events is None:
            raise RuntimeError("TP MLA output buffers are unavailable")
        return TPHidden(
            self.devices,
            tuple(state.output_replicas),
            tuple(state.output_events),
        )

    def _validate_hidden_pair(self, state, hidden, output) -> None:
        if (
            tuple(hidden.devices) != self.devices
            or hidden.shape != torch.Size((1, self.spec.hidden_size))
            or output.shape != hidden.shape
            or hidden.dtype != torch.bfloat16
            or output.dtype != torch.bfloat16
        ):
            raise ValueError("TP MLA TPHidden layout mismatch")
        expected_addresses = (
            state.composed_input_addresses
            if state.composed_input_addresses is not None
            else tuple(
                item.data_ptr() for item in state.local_inputs
            )
        )
        if hidden.fixed_addresses != expected_addresses:
            raise ValueError(
                "TP MLA input must use its captured fixed addresses"
            )

    def run_prepared(
        self,
        layer: int,
        position: int,
    ) -> torch.Tensor:
        state = self.layers[layer]
        if state.graph_batch is None:
            raise RuntimeError("TP MLA graphs are not captured")
        if not 0 <= int(position) < self.spec.max_ctx:
            raise ValueError("TP MLA position exceeds max_ctx")
        owner_device = self.devices[state.owner]
        with torch.cuda.device(owner_device):
            state.source_position.fill_(int(position))
            return state.graph_batch.launch_reduce(
                state.contributions,
                state.zero,
            )

    def reset(self) -> None:
        for state in self.layers.values():
            with torch.cuda.device(self.devices[state.owner]):
                state.source_position.zero_()


class TensorParallelMoELayerPlan:
    """Submit one fixed-address no-owner MoE layer in one host call.

    This is a scheduling primitive, not a model-specific mathematical
    operator.  All four phases retain their existing all-rank collectives and
    compact packed-expert kernels; the plan only removes repeated
    Python→C++ transitions between them.
    """

    def __init__(
        self,
        layer: int,
        input_hidden,
        residual,
        shared_executor,
        route_executor,
        expert_executor,
        final_executor,
    ) -> None:
        from ..fusedext import make_tp_no_owner_moe_layer_plan

        layer = int(layer)
        devices = tuple(input_hidden.devices)
        shared_state = shared_executor.layers[layer]
        route_state = route_executor.layers[layer]
        final_state = final_executor.layers[layer]
        if (
            devices != tuple(shared_executor.devices)
            or devices != tuple(route_executor.devices)
            or devices != tuple(final_executor.devices)
            or tuple(residual.devices) != devices
            or input_hidden.ready_events is None
            or residual.ready_events is None
            or shared_state.graph_batch is None
            or shared_state.events is None
            or route_state.graph_batch is None
            or route_state.output_events is None
            or final_state.graph_batch is None
            or final_state.routed_workspaces is None
            or final_state.shared_workspaces is None
        ):
            raise RuntimeError(
                "fixed no-owner MoE plan requires complete all-rank state"
            )
        expected_input_addresses = (
            shared_state.composed_input_addresses
            if shared_state.composed_input_addresses is not None
            else tuple(
                item.data_ptr() for item in shared_state.local_inputs
            )
        )
        if input_hidden.fixed_addresses != expected_input_addresses:
            raise ValueError(
                "fixed no-owner MoE plan input addresses do not match"
            )
        router_output, latent_output = route_executor.output_hidden(layer)
        (
            expert_batch,
            expert_contributions,
            packed_output,
        ) = expert_executor.fixed_layer_plan(layer)
        output = final_executor.output_hidden(layer)
        plan = make_tp_no_owner_moe_layer_plan(
            shared_state.graph_batch,
            route_state.graph_batch,
            expert_batch,
            final_state.graph_batch,
            input_hidden.ready_events,
            (
                tuple(route_state.router_contributions),
                tuple(route_state.latent_contributions),
            ),
            (
                tuple(router_output.replicas),
                tuple(latent_output.replicas),
            ),
            route_state.output_events,
            tuple(expert_contributions),
            tuple(packed_output.replicas),
            packed_output.ready_events,
            tuple(final_state.contributions),
            tuple(shared_state.contributions),
            tuple(shared_state.events),
            tuple(residual.replicas),
            # The MLP parent graph produces both the normalized input and
            # prefix residual.  Its all-rank done events therefore guard both.
            tuple(shared_state.events),
            tuple(final_state.routed_workspaces),
            tuple(final_state.shared_workspaces),
            tuple(output.replicas),
            output.ready_events,
        )
        if plan is None:
            raise RuntimeError(
                "fixed no-owner MoE extension plan is unavailable"
            )
        self.layer = layer
        self.devices = devices
        self.output = output
        self._plan = plan
        # Retain Python owners for CUDA Graph and event handles cached by the
        # extension plan.  Tensors are additionally retained in C++.
        self._dependencies = (
            input_hidden,
            residual,
            shared_executor,
            route_executor,
            expert_executor,
            final_executor,
            router_output,
            latent_output,
            packed_output,
        )

    def launch(self, input_events=None):
        if input_events is None:
            self._plan.launch()
        else:
            if len(input_events) != len(self.devices):
                raise ValueError(
                    "profiled no-owner MoE events must match TP ranks"
                )
            self._plan.launch_from_events(
                [event.cuda_event for event in input_events]
            )
        return self.output


class TensorParallelDecodeLayerPlan:
    """Submit Attention→routed MoE for one layer in one host call.

    This plan only composes already-captured generic all-rank operators.
    Attention still performs Column/Head-TP→Row-TP and the packed expert
    remains tensor-sharded across every rank.  The fixed attention output
    events directly trigger MoE, so no owner, hidden broadcast, or new
    collective is introduced.
    """

    def __init__(
        self,
        layer: int,
        attention_executor,
        moe_plan: TensorParallelMoELayerPlan,
    ) -> None:
        from ..fusedext import make_tp_no_owner_decode_layer_plan

        layer = int(layer)
        state = attention_executor.layers[layer]
        attention_output = attention_executor.output_hidden(layer)
        if (
            state.graph_batch is None
            or not state.contributions
            or attention_output.ready_events is None
            or tuple(attention_output.devices) != tuple(moe_plan.devices)
        ):
            raise RuntimeError(
                "decode layer plan requires captured all-rank attention"
            )
        plan = make_tp_no_owner_decode_layer_plan(
            state.graph_batch,
            moe_plan._plan,
            list(state.contributions),
            list(attention_output.replicas),
            list(attention_output.ready_events),
        )
        if plan is None:
            raise RuntimeError(
                "fixed no-owner decode layer plan is unavailable"
            )
        self.layer = layer
        self.devices = tuple(attention_output.devices)
        self.attention_executor = attention_executor
        self.attention_output = attention_output
        self.output = moe_plan.output
        self._plan = plan
        self._dependencies = (
            attention_executor,
            moe_plan,
            attention_output,
        )

    @property
    def persistent_enabled(self) -> bool:
        capability = getattr(self._plan, "persistent_enabled", None)
        return bool(capability is not None and capability())

    def launch(self, hidden, position: int | None = None):
        input_events = self.attention_executor.prepare_hidden_events(
            self.layer,
            hidden,
            position,
            output=self.attention_output,
        )
        self._plan.launch_from_events(
            [event.cuda_event for event in input_events]
        )
        return self.output


__all__ = [
    "GatedMLPSpec",
    "KDASpec",
    "MLASpec",
    "MoEPreludeSpec",
    "OwnerGroupedTensorParallel",
    "RouteDownSpec",
    "RowParallelLinearSpec",
    "TensorParallelGatedMLP",
    "TensorParallelKDA",
    "TensorParallelMLA",
    "TensorParallelMoELayerPlan",
    "TensorParallelDecodeLayerPlan",
    "TensorParallelMoEPrelude",
    "TensorParallelRouteDown",
    "TensorParallelRowLinear",
]
