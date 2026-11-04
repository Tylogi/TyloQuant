"""Disk preflight must precede expensive tensor/calibration work."""

import errno
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from mfq import disk_space
from mfq.cli import main
from mfq.disk_space import DiskSpaceRequirement, check_disk_space
from mfq.formats.header import FileHeader
from mfq.formats.shards import write_blob_record_shards
from mfq.tools import quantize_hf_to_mfq as q


def free_space(monkeypatch, nbytes):
    monkeypatch.setattr(disk_space.shutil, 'disk_usage', lambda _path: SimpleNamespace(free=nbytes))


def forbidden(*_args, **_kwargs):
    raise AssertionError('heavy work started before disk preflight')


def test_same_filesystem_adds_cache_and_output_and_reports_both(monkeypatch, tmp_path):
    free_space(monkeypatch, 172_000_000_000)
    with pytest.raises(OSError) as raised:
        check_disk_space([
            DiskSpaceRequirement(tmp_path/'cache'/'fitted', 89_500_000_000, 'fitted cache'),
            DiskSpaceRequirement(tmp_path/'output.mfq', 96_070_000_000, 'final output'),
        ], stage='statistics')
    assert raised.value.errno == errno.ENOSPC
    message = str(raised.value)
    assert all(text in message for text in ('89.500 GB', '96.070 GB', '172.000 GB', '--alphaq-cache'))


def test_separate_filesystems_have_independent_budgets(monkeypatch, tmp_path):
    cache, output = tmp_path/'cache', tmp_path/'output'
    monkeypatch.setattr(disk_space, 'filesystem', lambda p: (1, cache) if p == cache else (2, output))
    monkeypatch.setattr(disk_space.shutil, 'disk_usage', lambda p: SimpleNamespace(
        free=100_000_000_000 if p == cache else 105_000_000_000))
    requirements = [DiskSpaceRequirement(cache, 89_500_000_000, 'cache'),
                    DiskSpaceRequirement(output, 96_070_000_000, 'output')]
    check_disk_space(requirements, stage='quantization')
    monkeypatch.setattr(disk_space.shutil, 'disk_usage', lambda p: SimpleNamespace(
        free=100_000_000_000 if p == cache else 90_000_000_000))
    with pytest.raises(OSError, match='output'):
        check_disk_space(requirements, stage='quantization')


def checkpoint(tmp_path):
    root = tmp_path/'model'
    root.mkdir()
    (root/'config.json').write_text(json.dumps({'model_type': 'qwen3_5', 'hidden_size': 128}))
    save_file({'model.language_model.layers.0.mlp.down_proj.weight':
               torch.ones((4, 128), dtype=torch.bfloat16)}, root/'model.safetensors')
    return root


@pytest.mark.parametrize('staged', [False, True])
def test_hf_fails_before_quantization_and_succeeds_with_room(monkeypatch, tmp_path, staged):
    root = checkpoint(tmp_path)
    output = tmp_path/'output.mfq'
    args = q.build_parser().parse_args(['--input', str(root), '--output', str(output),
                                      '--quant-backend', 'cpu', '--device', 'cpu'])
    args.staged_blobs = staged
    free_space(monkeypatch, 0)
    with monkeypatch.context() as patch:
        patch.setattr(q, '_write_nint_axis0_blob', forbidden)
        with pytest.raises(OSError, match='before HF/MFQ quantization'):
            q.convert(args)
    assert not output.exists()
    assert not (tmp_path/f'.{output.name}.tmp_blobs').exists()
    free_space(monkeypatch, 10**12)
    q.convert(args)
    assert output.stat().st_size > 0


def test_alphaq_fails_before_statistics_workers_or_training(monkeypatch, tmp_path):
    from tests.test_alphaq_workflow import fixture
    from mfq.calibration import alphaq_workflow, alphaq_parallel

    root, imatrix, _count = fixture(tmp_path)
    free_space(monkeypatch, 0)
    monkeypatch.setattr(alphaq_workflow, 'collect_statistics', forbidden)
    monkeypatch.setattr(alphaq_workflow, 'trained_tables', forbidden)
    monkeypatch.setattr(alphaq_parallel.DeviceWorkers, '__enter__', forbidden)
    with pytest.raises(OSError, match='before AlphaQ statistics'):
        main(['quantize', str(root), str(tmp_path/'output.mfq'), '--target-bpw', '12',
              '--imatrix', str(imatrix), '--backend', 'cpu'])


def test_final_writer_rechecks_space_before_copy(monkeypatch, tmp_path):
    blob = tmp_path/'tensor.blob'
    blob.write_bytes(b'payload')
    free_space(monkeypatch, 0)
    monkeypatch.setattr('mfq.formats.shards._copy_record', forbidden)
    with pytest.raises(OSError, match='before final MFQ assembly'):
        write_blob_record_shards(tmp_path/'output.mfq', FileHeader(model_arch='test'),
                                 [q.BlobRecord('tensor', 'BF16', 7, blob)])
    assert blob.exists()
    assert not (tmp_path/'output.mfq.tmp').exists()


def test_imatrix_rejects_before_backend_or_cuda_probe(monkeypatch, tmp_path):
    from mfq.calibration import imatrix
    from mfq.calibration.dataset import CalibrationBatch

    root = checkpoint(tmp_path)
    free_space(monkeypatch, 0)
    monkeypatch.setattr(imatrix, '_backend', forbidden)
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    batch = CalibrationBatch(np.ones((1, 4), dtype=np.int64), 'train', ('test',), (0,))
    corpus = SimpleNamespace(iter_batches=lambda *_args, **_kwargs: [batch])
    with pytest.raises(OSError, match='before imatrix model loading'):
        imatrix.collect_imatrix(root, corpus, tmp_path/'imatrix.npz')


def test_consume_fitted_exports_and_can_refit_after_consumption(monkeypatch, tmp_path):
    from tests.test_alphaq_workflow import fixture

    root, imatrix, _count = fixture(tmp_path)
    cache, output = tmp_path/'cache', tmp_path/'output.mfq'
    free_space(monkeypatch, 10**12)
    argv = ['quantize', str(root), str(output), '--target-bpw', '12', '--imatrix', str(imatrix),
            '--alphaq-cache', str(cache), '--alphaq-profiles', 'NINT4,NINT8', '--backend', 'cpu',
            '--alphaq-consume-fitted']
    assert main(argv) == 0
    assert output.stat().st_size > 0
    assert not list((cache/'fitted').glob('*/tensor.blob'))
    assert not list((cache/'fitted').glob('*/complete.json'))
    assert (cache/'statistics.json').exists()
    assert main([*argv, '--overwrite']) == 0


def test_final_alphaq_rechecks_after_fitting_and_retains_cache(monkeypatch, tmp_path):
    from tests.test_alphaq_workflow import fixture
    from mfq.calibration.alphaq_fit import FittedPayloadCache

    root, imatrix, _count = fixture(tmp_path)
    original = FittedPayloadCache.require
    free_space(monkeypatch, 10**12)

    def require_then_fill_disk(self, item):
        record = original(self, item)
        free_space(monkeypatch, 0)
        return record

    monkeypatch.setattr(FittedPayloadCache, 'require', require_then_fill_disk)
    with pytest.raises(OSError, match='before final AlphaQ assembly'):
        main(['quantize', str(root), str(tmp_path/'output.mfq'), '--target-bpw', '12',
              '--imatrix', str(imatrix), '--alphaq-cache', str(tmp_path/'cache'),
              '--alphaq-profiles', 'NINT4,NINT8', '--backend', 'cpu'])
    assert list((tmp_path/'cache'/'fitted').glob('*/tensor.blob'))
    assert not (tmp_path/'output.mfq').exists()


def test_consume_releases_each_payload_only_after_successful_copy(monkeypatch, tmp_path):
    from tests.test_alphaq_workflow import fixture
    from mfq.formats import shards
    from mfq.calibration.alphaq_fit import FittedPayloadCache

    root, imatrix, _count = fixture(tmp_path)
    cache, output = tmp_path/'cache', tmp_path/'output.mfq'
    free_space(monkeypatch, 10**12)
    copied, released = [], []
    copy_record, consume = shards._copy_record, FittedPayloadCache.consume

    def fail_second(record, target):
        if len(copied) == 1:
            raise OSError('injected copy failure')
        copy_record(record, target)
        copied.append(record.name)

    def track_consume(self, record):
        assert record.name in copied
        consume(self, record)
        assert not record.path.exists()
        released.append(record.name)

    monkeypatch.setattr(shards, '_copy_record', fail_second)
    monkeypatch.setattr(FittedPayloadCache, 'consume', track_consume)
    with pytest.raises(OSError, match='injected copy failure'):
        main(['quantize', str(root), str(output), '--target-bpw', '12', '--imatrix', str(imatrix),
              '--alphaq-cache', str(cache), '--alphaq-profiles', 'NINT4,NINT8', '--backend', 'cpu',
              '--alphaq-consume-fitted'])
    assert copied == released and len(released) == 1
    assert len(list((cache/'fitted').glob('*/tensor.blob'))) == 5
    assert not output.exists() and not (tmp_path/'output.mfq.tmp').exists()


def test_full_precision_preflight_rejects_before_blob_copy(monkeypatch, tmp_path):
    from mfq.tools import convert_hf_to_full_mfq as full

    root = checkpoint(tmp_path)
    free_space(monkeypatch, 0)
    monkeypatch.setattr(full, '_write_plan_blob', forbidden)
    args = full.build_parser().parse_args(['--input', str(root), '--output', str(tmp_path/'output.mfq')])
    with pytest.raises(OSError, match='before full-precision MFQ conversion'):
        full.convert(args)


def test_shared_encoded_cache_consume_is_rejected_before_statistics(monkeypatch, tmp_path):
    from tests.test_alphaq_workflow import fixture
    from mfq.calibration import alphaq_workflow

    root, imatrix, _count = fixture(tmp_path)
    cache = tmp_path/'cache'
    cache.mkdir()
    (cache/'encoded-candidates.json').write_text('{}')
    monkeypatch.setattr(alphaq_workflow, 'collect_statistics', forbidden)
    with pytest.raises(ValueError, match='shared encoded-candidates'):
        main(['quantize', str(root), str(tmp_path/'output.mfq'), '--target-bpw', '12',
              '--imatrix', str(imatrix), '--alphaq-cache', str(cache), '--backend', 'cpu',
              '--alphaq-consume-fitted'])


def test_gguf_preflight_precedes_tensor_reads_and_allows_success(monkeypatch, tmp_path):
    from mfq.tools import quantize_gguf_to_mfq as gguf
    from mfq.formats.assets import RuntimeAsset

    values = np.ones((4, 128), dtype=np.float32)
    tensor = SimpleNamespace(name='output.weight', tensor_type=SimpleNamespace(name='F32'),
                             shape=np.array([128, 4]), data=values)
    reader = SimpleNamespace(tensors=[tensor], fields={})
    monkeypatch.setattr(gguf, '_load_gguf', lambda: (lambda *_args: reader, lambda data, _dtype: data))
    monkeypatch.setattr(gguf, 'gguf_metadata_asset', lambda _reader:
                        RuntimeAsset('__mfq_asset__/test.json', 'application/json', b'{}'))
    source, recipe, output = tmp_path/'source.gguf', tmp_path/'recipe.gguf', tmp_path/'output.mfq'
    source.write_bytes(b'fake source')
    recipe.write_bytes(b'fake recipe')
    args = gguf.build_parser().parse_args(['--input-bf16-gguf', str(source), '--recipe-gguf', str(recipe),
        '--output', str(output), '--quant-backend', 'cpu', '--device', 'cpu'])
    free_space(monkeypatch, 0)
    with monkeypatch.context() as patch:
        patch.setattr(gguf, '_read_dense_tensor', forbidden)
        with pytest.raises(OSError, match='before GGUF quantization'):
            gguf.convert(args)
    assert not output.exists()
    # A generated IN baseline must leave room for the later output and its
    # fitted split blobs before spending time generating that baseline.
    args._following_in_output_bytes = 100 << 20
    free_space(monkeypatch, 128 << 20)
    with monkeypatch.context() as patch:
        patch.setattr(gguf, '_read_dense_tensor', forbidden)
        with pytest.raises(OSError, match='following important-neuron'):
            gguf.convert(args)
    args._following_in_output_bytes = 0
    free_space(monkeypatch, 10**12)
    gguf.convert(args)
    assert output.stat().st_size > 0


@pytest.mark.parametrize('final_space', [True, False])
def test_imatrix_with_room_uses_cpu_fakes_and_rechecks_final_write(monkeypatch, tmp_path, final_space):
    from mfq.calibration import imatrix
    from mfq.calibration.dataset import CalibrationBatch
    from mfq.calibration.loading import SourceLayers
    from mfq.quantize.imatrix import ImportanceEntry

    root = checkpoint(tmp_path)
    corpus_root = tmp_path/'corpus'
    corpus_root.mkdir()
    (corpus_root/'manifest.json').write_text('{}')
    batch = CalibrationBatch(np.ones((1, 4), dtype=np.int64), 'train', ('test',), (0,))
    corpus = SimpleNamespace(root=corpus_root, manifest={}, iter_batches=lambda *_args, **_kwargs: [batch])
    backend = SimpleNamespace(num_layers=0, hidden_size=128, teacher_dtype=torch.bfloat16,
                              device=torch.device('cpu'), release_initial_state=lambda: None)
    entries = {'test.weight': ImportanceEntry(np.ones((1, 128)), np.array([4]))}

    def collected():
        if not final_space:
            free_space(monkeypatch, 0)
        return entries

    collector = SimpleNamespace(entries=collected, naq_categories={}, close=lambda: None)
    free_space(monkeypatch, 10**12)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(imatrix, '_backend', lambda *_args: (backend, 'fake'))
    monkeypatch.setattr(imatrix, '_targets', lambda *_args: ({}, ()))
    monkeypatch.setattr(imatrix, 'ActivationImatrixCollector', lambda *_args, **_kwargs: collector)
    monkeypatch.setattr(imatrix, '_release', lambda *_args: None)
    monkeypatch.setattr(torch.cuda, 'synchronize', forbidden)
    monkeypatch.setattr(torch.cuda, 'empty_cache', forbidden)
    monkeypatch.setattr(SourceLayers, 'initial_hidden',
                        lambda _self, ids: torch.zeros((*ids.shape, 128), dtype=torch.bfloat16))
    output = tmp_path/'imatrix.npz'
    if final_space:
        result = imatrix.collect_imatrix(root, corpus, output)
        assert output.stat().st_size > 0 and set(result.entries) == {'test.weight'}
    else:
        with pytest.raises(OSError, match='before imatrix serialization'):
            imatrix.collect_imatrix(root, corpus, output)
        assert not output.exists()
