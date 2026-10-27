from __future__ import annotations

import functools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mfq.architectures.hf_config import text_section
from mfq.formats.assets import MODEL_CONFIG_ASSET
from mfq.formats.io import open_mmap
from mfq.server.protocol.models import (
    ModelCacheComponent,
    ModelCacheFixedComponent,
    ModelCacheProfile,
)

if TYPE_CHECKING:
    from mfq.server.state.catalog import DiscoveredModel


@functools.lru_cache(maxsize=16)
def _mfq_config(path: str, size: int, modified_ns: int) -> tuple[Any, bool]:
    # Load planning prices KV several times per load; read each file version
    # once. Callers only read the returned config.
    with open_mmap(path) as store:
        record = store.records.get(MODEL_CONFIG_ASSET)
        if record is not None:
            if record.nbytes > 4 << 20:
                raise ValueError('model configuration exceeds 4 MiB')
            config = json.loads(store.read_blob(MODEL_CONFIG_ASSET))
        else:
            config = store.header.extra.get('hf_config', {})
        has_predictor = any(name.startswith(('predictor.block.', 'mtp.')) for name in store.records)
    return config, has_predictor


def _checkpoint_config(artifact: DiscoveredModel) -> tuple[Any, bool]:
    """Return the checkpoint's model config and whether it stores a predictor."""
    if artifact.resource.format == 'mfq':
        path = Path(artifact.path).resolve()
        stat = path.stat()
        return _mfq_config(str(path), stat.st_size, stat.st_mtime_ns)
    path = artifact.path / 'config.json'
    if path.stat().st_size > 4 << 20:
        raise ValueError('model configuration exceeds 4 MiB')
    config = json.loads(path.read_text(encoding='utf-8'))
    index = artifact.path / 'model.safetensors.index.json'
    if index.is_file():
        names = json.loads(index.read_text(encoding='utf-8')).get('weight_map', {})
    else:
        from mfq.server.state.catalog import ModelCatalog
        names = {name for shard in artifact.path.glob('*.safetensors')
            for name in ModelCatalog._safetensors_header(shard)}
    return config, any(name.startswith(('predictor.block.', 'mtp.')) for name in names)


def checkpoint_cache_profile(artifact: DiscoveredModel, *, cuda: bool = False,
    include_recurrent: bool = False) -> ModelCacheProfile | None:
    if not artifact.resource.complete:
        raise ValueError('checkpoint shards are incomplete')
    config, has_predictor = _checkpoint_config(artifact)
    if not isinstance(config, dict):
        return None
    text = config.get('text_config', config)
    predictor_layers = text.get('mtp_num_hidden_layers', 1) if isinstance(text, dict) and has_predictor else 0
    if type(predictor_layers) is not int or predictor_layers < 0:
        return None
    return cache_profile(config, include_recurrent=include_recurrent, predictor_layers=predictor_layers, cuda=cuda)


def checkpoint_attention_kv_bytes_per_token(artifact: DiscoveredModel) -> int | None:
    """FP16 K/V bytes per token of the backbone attention layers, from config fields alone.

    A coarse estimate for checkpoints without a cache profile; None when the
    config does not state the attention geometry.
    """
    try:
        config, _ = _checkpoint_config(artifact)
    except (OSError, ValueError):
        return None
    if not isinstance(config, Mapping):
        return None
    text = text_section(config)

    def positive(key: str) -> int | None:
        value = text.get(key)
        return value if type(value) is int and value > 0 else None

    # Hybrid models keep per-token K/V only in their attention mixers; sliding
    # windows are priced at full context.
    schedule = text.get('layers_block_type') or text.get('layer_types')
    pattern = text.get('hybrid_override_pattern')
    if isinstance(schedule, list) and schedule:
        layers = sum(isinstance(kind, str) and 'attention' in kind and 'linear' not in kind for kind in schedule)
    elif isinstance(pattern, str) and pattern:
        layers = pattern.count('*')
    else:
        layers = positive('num_hidden_layers')
        interval = positive('full_attention_interval')
        if layers is not None and interval is not None:
            layers //= interval
    if layers == 0:
        return 0
    heads = positive('num_attention_heads')
    kv_heads = positive('num_key_value_heads') or heads
    hidden = positive('hidden_size')
    head_dim = positive('head_dim') or (hidden // heads if hidden and heads else None)
    if layers is None or kv_heads is None or not head_dim:
        return None
    return layers * kv_heads * head_dim * 2 * 2


def cache_profile(config: dict[str, Any], *, include_recurrent: bool = True, predictor_layers: int = 0,
    cuda: bool = False) -> ModelCacheProfile | None:
    text = config.get('text_config', config)
    if not isinstance(text, dict):
        return None
    kind = text.get('model_type', config.get('model_type'))

    def integer(key: str, default: int | None = None) -> int:
        value = text.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(key)
        return value

    try:
        maximum, layers = integer('max_position_embeddings'), integer('num_hidden_layers')
        if min(maximum, layers) <= 0:
            return None
        fixed_components = []
        components = []

        def state(group: str, name: str, count: int, size: int) -> None:
            if count and size:
                fixed_components.append(ModelCacheFixedComponent(group=group, name=name, layers=count, bytes=size))

        if kind == 'deepseek_v4':
            dim, index, window = integer('head_dim'), integer('index_head_dim'), integer('sliding_window')
            ratios = text.get('compress_ratios')
            if min(dim, index, window) <= 0 or not isinstance(ratios, list) or len(ratios) < layers:
                return None
            state('MLA', 'sliding_window', layers, layers * window * dim * 2)
            for ratio in (4, 128):
                count = ratios[:layers].count(ratio)
                width = dim + index if ratio == 4 else dim
                state('MLA', f'compression_state_{ratio}', count, count * ratio * width * (12 if ratio == 4 else 4))
                if count:
                    components.append(ModelCacheComponent(group='MLA', name='compressed_kv', layers=count,
                        bytes_per_row=count * dim * 2, tokens_per_row=ratio))
                    if ratio == 4:
                        components.append(ModelCacheComponent(group='MLA', name='indexer_key', subgroup='indexer', layers=count,
                            bytes_per_row=count * index * 2, tokens_per_row=ratio))
            if any(value not in (0, 4, 128) for value in ratios[:layers]):
                return None
        elif kind in {'qwen4_exp', 'qwen4_exp_text', 'qwen3_5', 'qwen3_5_text', 'qwen3_5_moe', 'qwen3_5_moe_text', 'qwen3_next'}:
            types = text.get('layer_types')
            if not isinstance(types, list):
                interval = integer('full_attention_interval', 4)
                if interval <= 0:
                    return None
                types = ['full_attention' if (i + 1) % interval == 0 else 'linear_attention' for i in range(layers)]
            if len(types) != layers or any(value not in ('linear_attention', 'full_attention') for value in types):
                return None
            linear = types.count('linear_attention')
            if linear and include_recurrent:
                keys, key_dim = integer('linear_num_key_heads'), integer('linear_key_head_dim')
                heads, value_dim = integer('linear_num_value_heads'), integer('linear_value_head_dim')
                kernel = integer('linear_conv_kernel_dim')
                if min(keys, key_dim, heads, value_dim, kernel) <= 0:
                    return None
                channels = 2 * keys * key_dim + heads * value_dim
                if key_dim != value_dim:
                    return None
                state_copies = 1 if kind in {'qwen4_exp', 'qwen4_exp_text'} else 2
                recurrent = linear * 4 * heads * value_dim * value_dim
                convolution = linear * 4 * (kernel - 1) * channels
                state('GDN', 'recurrent_state', linear, recurrent)
                state('GDN', 'convolution_state', linear, convolution)
                state('GDN', 'reset_template', linear, (state_copies - 1) * (recurrent + convolution))
            full = types.count('full_attention') + predictor_layers
            if full:
                heads, dim = integer('num_key_value_heads'), integer('head_dim')
                if min(heads, dim) <= 0:
                    return None
                group = 'QSA' if kind in {'qwen4_exp', 'qwen4_exp_text'} else 'GQA'
                read_rows = integer('indexer_budget') if group == 'QSA' and 'indexer_budget' in text else None
                components.append(ModelCacheComponent(group=group, name='raw_kv', layers=full,
                    bytes_per_row=full * heads * dim * 4, allocation='power_of_two', minimum_rows=16,
                    max_read_rows_per_token=read_rows, head_dimension=dim, kv_heads=heads))
            if kind in {'qwen4_exp', 'qwen4_exp_text'}:
                index, ratio = integer('indexer_head_dim'), integer('indexer_compress_ratio')
                if min(index, ratio) <= 0:
                    return None
                if full and cuda:
                    components.extend([
                        ModelCacheComponent(group='QSA', name='indexer_key', subgroup='indexer', layers=full,
                            bytes_per_row=full * index * 2),
                        ModelCacheComponent(group='QSA', name='indexer_pooled', subgroup='indexer', layers=full,
                            bytes_per_row=full * index * 2, tokens_per_row=ratio),
                    ])
                elif full:
                    state('QSA', 'indexer_tail', full, full * ((ratio + 5) * index * 2 + 2))
                    components.extend([
                        ModelCacheComponent(group='QSA', name='indexer_pooled', subgroup='indexer', layers=full,
                            bytes_per_row=full * index * 4, tokens_per_row=ratio, allocation='power_of_two', minimum_rows=16,
                            row_rounding='floor'),
                    ])
                ple_layers = text.get('ple_layer_ids', [])
                if not isinstance(ple_layers, list):
                    return None
                if ple_layers and include_recurrent:
                    kernel, ngram, hc, hidden = integer('ple_conv_kernel_size'), integer('ngram_size'), integer('hc_count'), integer('hidden_size')
                    if min(kernel, ngram, hc, hidden) <= 0:
                        return None
                    state('PLE', 'convolution_state', len(ple_layers), len(ple_layers) * 4 * (kernel - 1) * ngram * hc * hidden)
        elif kind in {'qwen2', 'qwen3', 'qwen3_moe', 'llama', 'mistral', 'minicpm', 'minicpmo'}:
            heads = integer('num_key_value_heads', integer('num_attention_heads'))
            dim = integer('head_dim', integer('hidden_size') // integer('num_attention_heads'))
            if min(heads, dim) <= 0 or text.get('sliding_window'):
                return None
            query_heads = integer('num_attention_heads')
            group = 'GQA' if query_heads > heads else 'MHA'
            components.append(ModelCacheComponent(group=group, name='raw_kv', layers=layers,
                bytes_per_row=layers * heads * dim * 4, allocation='power_of_two', minimum_rows=16,
                head_dimension=dim, kv_heads=heads))
        else:
            return None
        return ModelCacheProfile(max_context=maximum, fixed_bytes=sum(item.bytes for item in fixed_components),
            fixed_components=fixed_components, components=components)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError):
        return None
