"""Freeze AlphaQ's runtime metadata and enforce its final serialized cap."""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

from mfq.architectures.hf_config import text_section
from mfq.formats.assets import (
    ASSET_MANIFEST_KEY, RuntimeAsset, hf_runtime_assets, model_config_asset,
    model_graph_asset, runtime_asset_manifest,
)
from mfq.formats.header import FileHeader
from mfq.formats.shards import _write_header_and_table


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def model_files(root: Path) -> tuple[tuple[str, int, int], ...]:
    index = root/'model.safetensors.index.json'
    if index.exists():
        shards = set(json.loads(index.read_text())['weight_map'].values())
    else:
        shards = {p.name for p in root.glob('*.safetensors')}
    if not shards:
        raise ValueError('AlphaQ requires safetensors model weights')
    names = sorted(shards | {p.name for p in (index, root/'config.json') if p.exists()})
    return tuple((name, (root/name).stat().st_size, (root/name).stat().st_mtime_ns) for name in names)


def runtime_metadata(root: Path, plans, *, imatrix, imatrix_sha256: str, policy: dict,
                     tokenizer: Path | None = None, sampling_profile=None):
    """Create all runtime assets before allocating the complete file budget."""
    from mfq.tools import quantize_hf_to_mfq as q

    config = q.load_hf_model_config(root)
    config = json.loads(json.dumps(config))
    text = text_section(config)
    mtp = any(p.name.startswith('predictor.') for p in plans)
    if not mtp:
        if 'num_nextn_predict_layers' in text:
            text['num_nextn_predict_layers'] = 0
        elif 'mtp_num_hidden_layers' in text:
            text['mtp_num_hidden_layers'] = 0
            text['mtp_use_dedicated_embeddings'] = False
    assets = [model_config_asset(config)]
    graph = q.graph_spec_for_plan(config, [p.name for p in plans])
    if graph is not None:
        assets.append(model_graph_asset(graph.as_dict()))
    assets.extend(q.source_runtime_assets(root, config))
    if (root/'tokenizer.json').exists() or (root/'tokenizer_config.json').exists():
        assets.extend(hf_runtime_assets(root))
    # HF tokenizer assets are self-contained; the runtime can derive its
    # tokenizer-only GGUF from them. Embed a second representation only when
    # explicitly requested, so allocation does not pay for duplicate metadata.
    if tokenizer is not None:
        assets.append(q.gguf_metadata_asset(q._gguf_reader(tokenizer)))
    assets = tuple({a.name: a for a in assets}.values())
    extra = dict(source=root.name, source_format='hf', policy='AlphaQ full-file allocation',
        alphaq=policy, hf_config=config, mtp={'included': mtp})
    if imatrix is not None:
        extra['imatrix'] = dict(file=imatrix.path.name, sha256=imatrix_sha256,
            entries=len(imatrix.entries), datasets=list(imatrix.datasets),
            chunk_count=imatrix.chunk_count, chunk_size=imatrix.chunk_size, legacy=imatrix.legacy)
    extra[ASSET_MANIFEST_KEY] = runtime_asset_manifest(assets)
    profile = q.profile_for_new_mfq(root, config, explicit_profile=sampling_profile)
    if profile is not None:
        extra[q.RUNTIME_SAMPLING_METADATA_KEY] = profile
    header = FileHeader(version=2, model_arch=f"{config.get('model_type','unknown')}-hf-mfq-alphaq",
                        num_tensors=len(plans)+len(assets), extra=extra)
    return header, assets


@dataclass(frozen=True)
class AlphaQOutput:
    """Internal normal-writer contract; every payload still uses that writer."""
    root: Path
    plans: tuple
    source_files: tuple[tuple[str, int, int], ...]
    header: FileHeader
    assets: tuple[RuntimeAsset, ...]
    maximum_file_bytes: int
    payload_cache: object | None = None
    excluded_components: tuple[str, ...] = ()
    consume_fitted: bool = False

    def validate_plan(self, plans):
        if tuple(plans) != self.plans:
            raise ValueError('AlphaQ allocation plan differs from the final converter plan')
        if model_files(self.root) != self.source_files:
            raise ValueError('AlphaQ model source changed after allocation')

    def validate_records(self, records):
        expected = {p.name for p in self.plans} | {a.name for a in self.assets}
        if len(records) != len(expected) or {r.name for r in records} != expected:
            raise ValueError('AlphaQ output tensor/asset scope differs from allocation')
        if model_files(self.root) != self.source_files:
            raise ValueError('AlphaQ source changed during conversion')
        buffer = io.BytesIO()
        _write_header_and_table(buffer, self.header, records)
        total = buffer.tell()+sum(r.nbytes for r in records)
        auxiliary = sum(r.nbytes for r in records
                        if any(r.name.startswith(c+'.') for c in self.excluded_components))
        if total-auxiliary > self.maximum_file_bytes:
            raise ValueError(f'AlphaQ serialized output exceeds file budget: {total-auxiliary} > {self.maximum_file_bytes}')
        return total

    def write_cached(self, output, *, overwrite=False):
        """One-pass normal container writing, with no full-model scratch copy."""
        if self.payload_cache is None:
            return False
        from mfq.calibration.alphaq_payloads import EncodedRecord
        from mfq.formats.assets import ASSET_DTYPE
        from mfq.formats.shards import write_blob_record_shards
        from mfq.disk_space import DiskSpaceRequirement, check_disk_space, filesystem
        from mfq.formats.nint import NintSpec
        from mfq.tools import quantize_hf_to_mfq as q
        consumer = getattr(self.payload_cache, 'consume', None) if self.consume_fitted else None
        if self.consume_fitted and not callable(consumer):
            raise ValueError('--alphaq-consume-fitted requires a fitted payload cache')
        sizes = [q._plan_blob_nbytes(p, NintSpec(), getattr(self.payload_cache, 'artifact_root', None))
                 for p in self.plans]
        pending = getattr(self.payload_cache, 'pending_bytes', None)
        if callable(pending):
            cache_bytes = pending(self.plans, sizes)
            file_bytes = sum(sizes) + sum(len(a.data) for a in self.assets)
            same_disk = filesystem(self.payload_cache.directory)[0] == filesystem(Path(output))[0]
            output_bytes = file_bytes
            if self.consume_fitted and same_disk:
                output_bytes = file_bytes-sum(sizes) + max(sizes, default=0)
            check_disk_space([
                DiskSpaceRequirement(self.payload_cache.directory, cache_bytes, 'remaining fitted payloads'),
                DiskSpaceRequirement(Path(output), output_bytes, 'final AlphaQ output and copy overlap'),
            ], stage='AlphaQ payload fitting')
        records = [self.payload_cache.require(p) for p in self.plans]
        records.extend(EncodedRecord(a.name, ASSET_DTYPE, len(a.data),
                       lambda target, data=a.data: target.write(data)) for a in self.assets)
        expected = self.validate_records(records)
        # Every payload now exists. For consume mode, simulate copy/delete in
        # writer order, crediting only payloads on the output filesystem.
        additional, peak = expected, expected
        if self.consume_fitted:
            additional = expected-sum(r.nbytes for r in records)
            peak = additional
            output_device = filesystem(Path(output))[0]
            for record in records[:len(self.plans)]:
                additional += record.nbytes
                peak = max(peak, additional)
                if filesystem(record.path)[0] == output_device:
                    additional -= record.nbytes
            peak = max(peak, additional + sum(len(a.data) for a in self.assets))
        check_disk_space([DiskSpaceRequirement(Path(output), peak, 'final AlphaQ output and copy overlap')],
                         stage='final AlphaQ assembly')
        def validate(paths):
            self.payload_cache.validate()
            self.validate_plan(self.plans)
            if len(paths) != 1 or paths[0].stat().st_size != expected:
                raise ValueError('cached AlphaQ file size differs from validated records')
        tensor_names = {p.name for p in self.plans}
        write_blob_record_shards(output, self.header, records, overwrite=overwrite,
                                 before_publish=validate,
                                 after_record=(lambda record: consumer(record)
                                     if record.name in tensor_names else None)
                                     if self.consume_fitted else None,
                                 preflight=not self.consume_fitted)
        return True
