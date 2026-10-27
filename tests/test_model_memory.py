from dataclasses import replace

import numpy as np
import pytest

from mfq.formats.header import FileHeader
from mfq.formats.io import save
from mfq.formats.shards import format_shard_path, shard_header
from mfq.server.services.model_memory import cache_profile, checkpoint_cache_profile
from mfq.server.state.catalog import ModelCatalog


def test_flash_next_profile_includes_gdn_ple_and_qsa_index_caches():
    config = {'model_type': 'qwen4_exp_text', 'max_position_embeddings': 1048576,
              'num_hidden_layers': 48, 'layer_types': ['linear_attention'] * 3 + ['full_attention'],
              'linear_num_key_heads': 16, 'linear_key_head_dim': 128,
              'linear_num_value_heads': 48, 'linear_value_head_dim': 128, 'linear_conv_kernel_dim': 4,
              'num_key_value_heads': 2, 'head_dim': 256, 'indexer_head_dim': 128,
              'indexer_compress_ratio': 4, 'ple_layer_ids': [2], 'ple_conv_kernel_size': 4,
              'ngram_size': 3, 'hc_count': 4, 'hidden_size': 2560}
    config['layer_types'] *= 12
    profile = cache_profile(config)
    tail = 12 * (9 * 128 * 2 + 2)
    assert profile.fixed_bytes == 36 * (48 * 128 * 128 + 3 * 10240) * 4 + 9 * 10240 * 4 + tail
    assert [component.bytes_per_row for component in profile.components] == [24576, 6144]
    assert [component.tokens_per_row for component in profile.components] == [1, 4]
    assert all(component.allocation == 'power_of_two' and component.minimum_rows == 16 for component in profile.components)
    assert sum(item.bytes for item in profile.fixed_components) == profile.fixed_bytes
    assert [item.name for item in profile.components] == ['raw_kv', 'indexer_pooled']
    assert profile.components[-1].row_rounding == 'floor'
    assert profile.components[-1].active_after == 0
    kv = cache_profile(config, include_recurrent=False)
    assert kv.fixed_bytes == tail and [item.name for item in kv.fixed_components] == ['indexer_tail']
    assert {item.group for item in kv.components} == {'QSA'}
    assert kv.components == profile.components
    assert cache_profile({'text_config': config}) == profile
    assert profile.components[0].max_read_rows_per_token is None
    for selected in (1024, 2048, 4096):
        selected_profile = cache_profile({**config, 'indexer_budget': selected}, predictor_layers=1)
        raw = selected_profile.components[0]
        assert raw.max_read_rows_per_token == selected
        assert raw.bytes_per_row * selected == 13 * 2048 * selected


def test_deepseek_compressed_latents_are_not_double_counted_as_full_k_and_v():
    config = {'model_type': 'deepseek_v4', 'max_position_embeddings': 1048576, 'num_hidden_layers': 43,
              'head_dim': 512, 'index_head_dim': 128, 'sliding_window': 128,
              'compress_ratios': [0, 0] + [r for _ in range(20) for r in (4, 128)] + [4, 0, 0, 0]}
    profile = cache_profile(config)
    assert profile.fixed_bytes == 43 * 128 * 512 * 2 + 21 * 4 * 640 * 12 + 20 * 128 * 512 * 4
    assert [(component.bytes_per_row, component.tokens_per_row, component.allocation) for component in profile.components] == [(21504, 4, 'exact'), (5376, 4, 'exact'), (20480, 128, 'exact')]
    assert sum(item.bytes for item in profile.fixed_components) == profile.fixed_bytes
    assert profile.components[1].subgroup == 'indexer'


def test_regular_gqa_uses_kv_heads_and_two_byte_storage():
    profile = cache_profile({'model_type': 'llama', 'num_hidden_layers': 32, 'num_attention_heads': 32,
                             'num_key_value_heads': 8, 'hidden_size': 4096, 'max_position_embeddings': 8192})
    assert profile.fixed_bytes == 0
    assert profile.components[0].bytes_per_row == 32 * 8 * 128 * 2 * 2
    assert profile.components[0].max_read_rows_per_token is None


def test_incomplete_unknown_and_invalid_configs_do_not_fabricate_cache_estimates():
    for config in ({}, {'model_type': 'llama'}, {'model_type': 'unknown', 'max_position_embeddings': 4096, 'num_hidden_layers': 32},
                   {'model_type': 'llama', 'max_position_embeddings': -1, 'num_hidden_layers': 32},
                   {'model_type': 'llama', 'max_position_embeddings': 4096, 'num_hidden_layers': True}):
        assert cache_profile(config) is None


def test_kv_profile_does_not_require_gdn_geometry_or_count_recurrent_state():
    profile = cache_profile({'model_type': 'qwen3_5_text', 'max_position_embeddings': 32768,
        'num_hidden_layers': 4, 'layer_types': ['linear_attention'] * 3 + ['full_attention'],
        'num_key_value_heads': 4, 'head_dim': 256}, include_recurrent=False)
    assert profile.fixed_bytes == 0 and profile.fixed_components == []
    assert len(profile.components) == 1
    assert profile.components[0].group == 'GQA' and profile.components[0].layers == 1


@pytest.mark.parametrize('changed', ['config', 'predictor'])
def test_checkpoint_profile_invalidates_when_another_shard_changes(tmp_path, changed):
    config = dict(model_type='qwen4_exp', max_position_embeddings=4096,
        num_hidden_layers=4, full_attention_interval=4, num_key_value_heads=2,
        head_dim=128, indexer_head_dim=128, indexer_compress_ratio=8)
    header = FileHeader(version=2, model_arch='qwen4_exp', extra={'hf_config': config})
    first, second = [format_shard_path(tmp_path / 'model.mfq', index, 2) for index in (1, 2)]
    primary = shard_header(header, split_no=0, split_count=2, tensor_count=2, record_count=2)
    companion = shard_header(header, split_no=1, split_count=2, tensor_count=2, record_count=2)
    weight = np.zeros((2, 2), dtype=np.float16)
    save(first, primary, {'model.embed_tokens.weight': weight})
    save(second, companion, {'model.norm.weight': weight})
    # Configuration lives in the primary shard; the predictor can be anywhere.
    entry = second if changed == 'config' else first
    model = replace(ModelCatalog._inspect(tmp_path, first), path=entry)
    before = checkpoint_cache_profile(model, cuda=True)
    assert before is not None and before.components[0].layers == 1
    entry_stat = entry.stat()

    if changed == 'config':
        primary.extra['hf_config'] = {**config, 'num_key_value_heads': 4}
        save(first, primary, {'model.embed_tokens.weight': weight})
    else:
        save(second, companion, {'mtp.block.0.norm.weight': weight})

    assert entry.stat().st_size == entry_stat.st_size
    assert entry.stat().st_mtime_ns == entry_stat.st_mtime_ns
    after = checkpoint_cache_profile(model, cuda=True)
    assert after is not None
    if changed == 'config':
        assert after.components[0].bytes_per_row == 2 * before.components[0].bytes_per_row
    else:
        assert after.components[0].layers == 2
