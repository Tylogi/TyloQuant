"""Deterministic, domain-stratified calibration corpora.

The on-disk artifact stores the exact token ids used by calibration.  Model
statistics and validation therefore never depend on re-tokenizing mutable raw
text or on the iteration order of a remote dataset.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_FORMAT = "mfq.calibration-corpus.v1"
_SPLITS = {"train": 0, "validation": 1}
_SPLIT_NAMES = {value: key for key, value in _SPLITS.items()}


@dataclass(frozen=True)
class EaddarioSource:
    domain: str
    filename: str
    weight: float


@dataclass(frozen=True)
class TraceSource:
    filename: str
    expected_mode: str | None = None


@dataclass(frozen=True)
class _TraceCandidate:
    file_index: int
    offset: int
    length: int
    line_number: int
    split: str
    mode: str
    source_dataset: str
    prompt_sha256: str
    trace_sha256: str
    priority: int
    declared_tokens: int | None


EADDARIO_SOURCE_SIZES = ("micro", "tiny", "small", "medium", "large")


def eaddario_sources(size: str = "medium") -> tuple[EaddarioSource, ...]:
    if size not in EADDARIO_SOURCE_SIZES:
        raise ValueError(f"unknown eaddario source size {size!r}; expected {EADDARIO_SOURCE_SIZES}")
    return (
        EaddarioSource("language", f"text_all_{size}.parquet", 0.35),
        EaddarioSource("code", f"code_{size}.parquet", 0.25),
        EaddarioSource("math", f"math_{size}.parquet", 0.20),
        EaddarioSource("tools", f"tools_{size}.parquet", 0.20),
    )


# Medium is the smallest common preset with enough multilingual text for the
# formal 1.5 Mi-token train split plus a 256 Ki-token held-out split.
DEFAULT_EADDARIO_SOURCES = eaddario_sources("medium")


@dataclass(frozen=True)
class CalibrationBatch:
    """One unpadded batch of exact saved token ids."""

    input_ids: np.ndarray
    split: str
    domains: tuple[str, ...]
    chunk_indices: tuple[int, ...]

    @property
    def attention_mask(self) -> np.ndarray:
        return np.ones(self.input_ids.shape, dtype=np.int64)


@dataclass(frozen=True)
class CalibrationCorpus:
    root: Path
    manifest: dict[str, Any]
    tokens: np.ndarray
    offsets: np.ndarray
    split_ids: np.ndarray
    domain_ids: np.ndarray

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.manifest["domains"])

    @property
    def chunks(self) -> int:
        return int(self.split_ids.size)

    def close(self) -> None:
        for value in (self.tokens, self.offsets, self.split_ids, self.domain_ids):
            mapping = getattr(value, "_mmap", None)
            if mapping is not None and not mapping.closed:
                mapping.close()

    def __enter__(self) -> CalibrationCorpus:
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    def token_count(self, split: str | None = None) -> int:
        if split is None:
            return int(self.tokens.size)
        split_id = _split_id(split)
        selected = np.flatnonzero(self.split_ids == split_id)
        if not selected.size:
            return 0
        lengths = self.offsets[selected + 1] - self.offsets[selected]
        return int(lengths.sum())

    def chunk_tokens(self, index: int) -> np.ndarray:
        if index < 0 or index >= self.chunks:
            raise IndexError(f"calibration chunk {index} is outside [0, {self.chunks})")
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        return np.asarray(self.tokens[start:end], dtype=np.int64)

    def iter_batches(
        self,
        split: str,
        *,
        window_length: int,
        batch_size: int = 1,
        max_tokens: int | None = None,
        seed: int | None = None,
        drop_last: bool = False,
    ) -> Iterator[CalibrationBatch]:
        """Yield same-length batches without padding or crossing document chunks."""

        if window_length < 2:
            raise ValueError("window_length must be at least 2")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_tokens is not None and max_tokens < 2:
            raise ValueError("max_tokens must be at least 2")

        split_id = _split_id(split)
        windows: list[tuple[int, int, int]] = []
        for chunk_index in np.flatnonzero(self.split_ids == split_id):
            start = int(self.offsets[chunk_index])
            end = int(self.offsets[chunk_index + 1])
            cursor = start
            while cursor < end:
                stop = min(cursor + window_length, end)
                length = stop - cursor
                if length == window_length or (length >= 2 and not drop_last):
                    windows.append((int(chunk_index), cursor, stop))
                cursor = stop

        if seed is not None:
            rng = np.random.default_rng(seed)
            rng.shuffle(windows)

        if max_tokens is not None:
            kept: list[tuple[int, int, int]] = []
            used = 0
            for item in windows:
                length = item[2] - item[1]
                remaining = max_tokens - used
                if remaining < 2:
                    break
                if length > remaining:
                    if drop_last:
                        continue
                    kept.append((item[0], item[1], item[1] + remaining))
                    used += remaining
                    break
                kept.append(item)
                used += length
                if used == max_tokens:
                    break
            windows = kept

        buckets: dict[int, list[tuple[int, int, int]]] = {}
        for item in windows:
            buckets.setdefault(item[2] - item[1], []).append(item)

        for length in sorted(buckets, reverse=True):
            bucket = buckets[length]
            for start in range(0, len(bucket), batch_size):
                items = bucket[start : start + batch_size]
                if len(items) < batch_size and drop_last:
                    continue
                batch = np.stack(
                    [
                        np.asarray(self.tokens[left:right], dtype=np.int64)
                        for _, left, right in items
                    ]
                )
                yield CalibrationBatch(
                    input_ids=batch,
                    split=split,
                    domains=tuple(
                        self.domains[int(self.domain_ids[index])] for index, _, _ in items
                    ),
                    chunk_indices=tuple(index for index, _, _ in items),
                )


def _split_id(split: str) -> int:
    try:
        return _SPLITS[split]
    except KeyError as exc:
        raise ValueError(f"unknown calibration split {split!r}; expected {tuple(_SPLITS)}") from exc


def _largest_remainder(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    if total < 0:
        raise ValueError("token total cannot be negative")
    if not weights:
        raise ValueError("at least one domain weight is required")
    if any(not math.isfinite(value) or value <= 0 for value in weights.values()):
        raise ValueError("domain weights must be finite and positive")
    normalizer = float(sum(weights.values()))
    exact = {key: total * float(value) / normalizer for key, value in weights.items()}
    result = {key: int(math.floor(value)) for key, value in exact.items()}
    remaining = total - sum(result.values())
    order = sorted(weights, key=lambda key: (-(exact[key] - result[key]), key))
    for key in order[:remaining]:
        result[key] += 1
    return result


def _stable_domain_seed(seed: int, domain: str) -> int:
    digest = hashlib.sha256(f"{seed}:{domain}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _tokenizer_metadata(tokenizer: Any, render_mode: str) -> dict[str, Any]:
    template = str(getattr(tokenizer, "chat_template", "") or "")
    return {
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
        "eos_token_id": (
            None
            if getattr(tokenizer, "eos_token_id", None) is None
            else int(tokenizer.eos_token_id)
        ),
        "render_mode": render_mode,
        "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
    }


def _record_token_ids(
    tokenizer: Any,
    text: str,
    *,
    render_mode: str,
    chat_template_kwargs: Mapping[str, Any],
) -> list[int]:
    if render_mode == "chat":
        if not hasattr(tokenizer, "apply_chat_template"):
            raise TypeError("chat rendering requires tokenizer.apply_chat_template")
        value = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            **dict(chat_template_kwargs),
        )
        if isinstance(value, Mapping):
            value = value["input_ids"]
    elif render_mode == "plain":
        value = tokenizer.encode(text, add_special_tokens=False)
    else:
        raise ValueError("render_mode must be 'chat' or 'plain'")

    ids = [int(token) for token in value]
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None and (not ids or ids[-1] != int(eos)):
        ids.append(int(eos))
    return ids


def _trace_record(
    value: Any,
    *,
    expected_generator_model: str,
    expected_mode: str | None,
) -> tuple[list[dict[str, Any]], str, str, str, int | None]:
    if not isinstance(value, Mapping):
        raise TypeError("trace row must be a JSON object")
    model = str(value.get("model", "")).strip()
    if model.casefold() != expected_generator_model.casefold():
        raise ValueError(
            f"trace generator model {model!r} does not match {expected_generator_model!r}"
        )
    mode = str(value.get("mode", "")).strip()
    if mode not in {"thinking", "nonthinking"}:
        raise ValueError(f"unsupported trace mode {mode!r}")
    if expected_mode is not None and mode != expected_mode:
        raise ValueError(f"trace mode {mode!r} does not match source mode {expected_mode!r}")
    if str(value.get("finish_reason", "")).strip() != "stop":
        raise ValueError(f"trace finish reason is not stop: {value.get('finish_reason')!r}")

    raw_messages = value.get("messages")
    if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, (str, bytes)):
        raise TypeError("trace messages must be a non-empty sequence")
    messages: list[dict[str, Any]] = []
    for raw_message in raw_messages:
        if not isinstance(raw_message, Mapping):
            raise TypeError("every trace message must be an object")
        role = str(raw_message.get("role", "")).strip()
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported trace message role {role!r}")
        if "content" not in raw_message:
            raise ValueError("trace message has no content")
        messages.append(dict(raw_message))
    if not messages or messages[-1].get("role") not in {"user", "tool"}:
        raise ValueError("trace messages must end at a user query or tool response")

    output = value.get("output")
    if not isinstance(output, str) or not output.strip():
        raise ValueError("trace output must be a non-empty string")
    raw_reasoning = value.get("reasoning")
    if raw_reasoning is None:
        reasoning = ""
    elif isinstance(raw_reasoning, str):
        reasoning = raw_reasoning
    else:
        raise TypeError("trace reasoning must be a string or null")
    if mode == "thinking" and not reasoning.strip():
        raise ValueError("thinking trace has no reasoning content")
    if mode == "nonthinking" and reasoning.strip():
        raise ValueError("nonthinking trace unexpectedly contains reasoning content")

    usage = value.get("usage")
    declared_tokens: int | None = None
    if isinstance(usage, Mapping) and usage.get("total_tokens") is not None:
        declared_tokens = int(usage["total_tokens"])
        if declared_tokens < 2:
            raise ValueError("trace usage.total_tokens must be at least two")
    source_dataset = str(value.get("source_dataset", "unknown")).strip() or "unknown"
    messages.append(
        {
            "role": "assistant",
            "content": output,
            "reasoning_content": reasoning,
        }
    )
    return messages, mode, source_dataset, model, declared_tokens


def _trace_token_ids(
    tokenizer: Any,
    value: Any,
    *,
    expected_generator_model: str,
    expected_mode: str | None,
) -> tuple[list[int], str, str, int | None]:
    messages, mode, source_dataset, _model, declared_tokens = _trace_record(
        value,
        expected_generator_model=expected_generator_model,
        expected_mode=expected_mode,
    )
    if not hasattr(tokenizer, "apply_chat_template"):
        raise TypeError("trace rendering requires tokenizer.apply_chat_template")
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    if isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    ids = [int(token) for token in rendered]
    if len(ids) < 2:
        raise ValueError("rendered trace contains fewer than two tokens")
    return ids, mode, source_dataset, declared_tokens


def _pack_stream(tokens: Sequence[int], sequence_length: int) -> list[np.ndarray]:
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if len(tokens) < 2:
        raise ValueError("each domain/split quota must contain at least two tokens")
    if len(tokens) <= sequence_length + 1:
        return [np.asarray(tokens, dtype=np.int32)]
    chunks = []
    cursor = 0
    while cursor < len(tokens):
        remaining = len(tokens) - cursor
        take = (
            remaining
            if sequence_length == 2 and remaining == 3
            else (
                sequence_length - 1
                if remaining == sequence_length + 1
                else min(sequence_length, remaining)
            )
        )
        value = np.asarray(tokens[cursor : cursor + take], dtype=np.int32)
        chunks.append(value)
        cursor += take
    return chunks


def _write_corpus_artifact(
    output_path: Path,
    chunks: Sequence[np.ndarray],
    split_ids: Sequence[int],
    domain_ids: Sequence[int],
    manifest: dict[str, Any],
) -> CalibrationCorpus:
    if not chunks:
        raise ValueError("calibration corpus has no chunks")
    offsets = np.zeros(len(chunks) + 1, dtype=np.int64)
    for index, chunk in enumerate(chunks):
        if chunk.ndim != 1 or chunk.size < 2:
            raise ValueError("calibration chunks must be one-dimensional with at least two tokens")
        offsets[index + 1] = offsets[index] + chunk.size
    tokens = np.concatenate(chunks).astype(np.int32, copy=False)
    split_array = np.asarray(split_ids, dtype=np.uint8)
    domain_array = np.asarray(domain_ids, dtype=np.uint16)
    if split_array.size != len(chunks) or domain_array.size != len(chunks):
        raise ValueError("calibration chunk metadata length mismatch")

    temporary = output_path.with_name(output_path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary calibration directory already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        np.save(temporary / "tokens.npy", tokens, allow_pickle=False)
        np.save(temporary / "offsets.npy", offsets, allow_pickle=False)
        np.save(temporary / "split_ids.npy", split_array, allow_pickle=False)
        np.save(temporary / "domain_ids.npy", domain_array, allow_pickle=False)
        os.replace(temporary, output_path)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return load_corpus(output_path)


def build_corpus_from_records(
    records_by_domain: Mapping[str, Iterable[str]],
    tokenizer: Any,
    output: str | Path,
    *,
    train_tokens: int = 1_572_864,
    validation_tokens: int = 262_144,
    sequence_length: int = 2048,
    domain_weights: Mapping[str, float] | None = None,
    seed: int = 20260718,
    render_mode: str = "plain",
    chat_template_kwargs: Mapping[str, Any] | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> CalibrationCorpus:
    """Create an exact-token corpus with disjoint train/validation records."""

    if train_tokens < 2 or validation_tokens < 2:
        raise ValueError("train_tokens and validation_tokens must both be at least 2")
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"calibration corpus already exists: {output_path}")
    if not records_by_domain:
        raise ValueError("records_by_domain is empty")

    domains = tuple(sorted(records_by_domain))
    if domain_weights is None:
        domain_weights = {domain: 1.0 for domain in domains}
    if set(domain_weights) != set(domains):
        raise ValueError("domain_weights keys must exactly match records_by_domain")
    train_quota = _largest_remainder(train_tokens, domain_weights)
    validation_quota = _largest_remainder(validation_tokens, domain_weights)
    template_kwargs = dict(chat_template_kwargs or {})

    normalized_records: dict[str, list[str]] = {}
    seen_records: set[str] = set()
    for domain in domains:
        values = []
        for raw in records_by_domain[domain]:
            text = str(raw).strip()
            if not text or text in seen_records:
                continue
            seen_records.add(text)
            values.append(text)
        if not values:
            raise ValueError(f"eaddario domain {domain!r} contains no unique usable records")
        normalized_records[domain] = values

    streams: dict[tuple[str, str], list[int]] = {
        (split, domain): [] for split in _SPLITS for domain in domains
    }
    record_counts: dict[str, dict[str, int]] = {
        split: {domain: 0 for domain in domains} for split in _SPLITS
    }
    record_hashes: dict[str, dict[str, list[str]]] = {
        split: {domain: [] for domain in domains} for split in _SPLITS
    }

    for domain in domains:
        records = normalized_records[domain].copy()
        rng = np.random.default_rng(_stable_domain_seed(seed, domain))
        rng.shuffle(records)
        targets = {"train": train_quota[domain], "validation": validation_quota[domain]}
        remaining = dict(targets)

        for text in records:
            if all(value == 0 for value in remaining.values()):
                break
            ids = _record_token_ids(
                tokenizer,
                text,
                render_mode=render_mode,
                chat_template_kwargs=template_kwargs,
            )
            if len(ids) < 2:
                continue
            available = [split for split, value in remaining.items() if value > 0]
            split = max(available, key=lambda key: (remaining[key] / targets[key], key))
            take = min(len(ids), remaining[split])
            streams[(split, domain)].extend(ids[:take])
            remaining[split] -= take
            record_counts[split][domain] += 1
            record_hashes[split][domain].append(hashlib.sha256(text.encode("utf-8")).hexdigest())

        if any(value > 0 for value in remaining.values()):
            raise ValueError(
                f"domain {domain!r} did not provide enough tokens: remaining={remaining}"
            )

    train_record_hashes = {value for values in record_hashes["train"].values() for value in values}
    validation_record_hashes = {
        value for values in record_hashes["validation"].values() for value in values
    }
    if train_record_hashes & validation_record_hashes:
        raise RuntimeError("calibration train and validation record sets overlap")

    chunks: list[np.ndarray] = []
    split_ids: list[int] = []
    domain_ids: list[int] = []
    for split in ("train", "validation"):
        for domain_id, domain in enumerate(domains):
            packed = _pack_stream(streams[(split, domain)], sequence_length)
            chunks.extend(packed)
            split_ids.extend([_SPLITS[split]] * len(packed))
            domain_ids.extend([domain_id] * len(packed))

    manifest = {
        "format": _FORMAT,
        "seed": int(seed),
        "sequence_length": int(sequence_length),
        "domains": list(domains),
        "domain_weights": {key: float(domain_weights[key]) for key in domains},
        "target_tokens": {"train": int(train_tokens), "validation": int(validation_tokens)},
        "actual_tokens": {
            split: int(
                sum(
                    chunk.size
                    for chunk, split_id in zip(chunks, split_ids, strict=True)
                    if split_id == _SPLITS[split]
                )
            )
            for split in _SPLITS
        },
        "record_counts": record_counts,
        "record_sets": {
            split: {
                domain: {
                    "count": len(record_hashes[split][domain]),
                    "sha256": hashlib.sha256(
                        "\n".join(record_hashes[split][domain]).encode("ascii")
                    ).hexdigest(),
                }
                for domain in domains
            }
            for split in _SPLITS
        },
        "tokenizer": _tokenizer_metadata(tokenizer, render_mode),
        "chat_template_kwargs": template_kwargs,
        "sources": dict(source_metadata or {}),
    }
    if manifest["actual_tokens"] != manifest["target_tokens"]:
        raise RuntimeError(
            f"packed token counts differ from requested counts: {manifest['actual_tokens']}"
        )
    return _write_corpus_artifact(output_path, chunks, split_ids, domain_ids, manifest)


def _file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return {"path": str(path), "size": path.stat().st_size, "sha256": digest.hexdigest()}


def _hash_int(seed: int, label: str, digest: str) -> int:
    value = hashlib.sha256(f"{seed}:{label}:{digest}".encode("ascii")).digest()
    return int.from_bytes(value[:8], "big", signed=False)


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _number_summary(values: Sequence[int]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p99": None, "max": None}
    ordered = sorted(int(value) for value in values)

    def quantile(fraction: float) -> int:
        return ordered[int(math.floor((len(ordered) - 1) * fraction))]

    return {
        "count": len(ordered),
        "mean": float(sum(ordered) / len(ordered)),
        "min": ordered[0],
        "p50": quantile(0.50),
        "p90": quantile(0.90),
        "p99": quantile(0.99),
        "max": ordered[-1],
    }


def _trace_exclusion_reason(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    finish_reason = str(value.get("finish_reason", "")).strip()
    if finish_reason == "length":
        return "finish_reason_length"
    output = value.get("output")
    if finish_reason == "stop" and isinstance(output, str) and not output.strip():
        return "empty_output"
    return None


def _index_trace_sources(
    jsonl_sources: Sequence[tuple[Path, TraceSource]],
    *,
    expected_generator_model: str,
    validation_fraction: float,
    seed: int,
) -> tuple[list[_TraceCandidate], list[dict[str, Any]], dict[str, Any]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("trace validation fraction must be strictly between zero and one")
    threshold = int(validation_fraction * (1 << 64))
    candidates: list[_TraceCandidate] = []
    seen_traces: set[str] = set()
    files: list[dict[str, Any]] = []
    split_counts = {split: 0 for split in _SPLITS}
    mode_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    duplicate_records = 0
    excluded_records = 0
    exclusion_reasons: dict[str, int] = {}

    for file_index, (path, source) in enumerate(jsonl_sources):
        if not path.is_file():
            raise FileNotFoundError(f"trace JSONL file does not exist: {path}")
        file_digest = hashlib.sha256()
        records = 0
        empty_lines = 0
        file_exclusion_reasons: dict[str, int] = {}
        with path.open("rb") as stream:
            line_number = 0
            while True:
                offset = stream.tell()
                raw = stream.readline()
                if not raw:
                    break
                line_number += 1
                file_digest.update(raw)
                stripped = raw.strip()
                if not stripped:
                    empty_lines += 1
                    continue
                try:
                    value = json.loads(stripped)
                    exclusion_reason = _trace_exclusion_reason(value)
                    if exclusion_reason is not None:
                        excluded_records += 1
                        exclusion_reasons[exclusion_reason] = (
                            exclusion_reasons.get(exclusion_reason, 0) + 1
                        )
                        file_exclusion_reasons[exclusion_reason] = (
                            file_exclusion_reasons.get(exclusion_reason, 0) + 1
                        )
                        continue
                    messages, mode, source_dataset, _model, declared_tokens = _trace_record(
                        value,
                        expected_generator_model=expected_generator_model,
                        expected_mode=source.expected_mode,
                    )
                except Exception as exc:
                    raise ValueError(f"invalid trace row {path}:{line_number}: {exc}") from exc
                records += 1
                trace_sha256 = hashlib.sha256(stripped).hexdigest()
                if trace_sha256 in seen_traces:
                    duplicate_records += 1
                    continue
                seen_traces.add(trace_sha256)
                prompt_sha256 = _json_sha256(messages[:-1])
                split_value = _hash_int(seed, "split", prompt_sha256)
                split = "validation" if split_value < threshold else "train"
                candidates.append(
                    _TraceCandidate(
                        file_index=file_index,
                        offset=offset,
                        length=len(raw),
                        line_number=line_number,
                        split=split,
                        mode=mode,
                        source_dataset=source_dataset,
                        prompt_sha256=prompt_sha256,
                        trace_sha256=trace_sha256,
                        priority=_hash_int(seed, "priority", trace_sha256),
                        declared_tokens=declared_tokens,
                    )
                )
                split_counts[split] += 1
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
                source_counts[source_dataset] = source_counts.get(source_dataset, 0) + 1
        files.append(
            {
                "filename": source.filename,
                "expected_mode": source.expected_mode,
                "identity": {
                    "path": str(path.resolve()),
                    "size": path.stat().st_size,
                    "sha256": file_digest.hexdigest(),
                },
                "records": records,
                "empty_lines": empty_lines,
                "excluded_records": sum(file_exclusion_reasons.values()),
                "exclusion_reasons": dict(sorted(file_exclusion_reasons.items())),
            }
        )

    if not candidates:
        raise ValueError("trace sources contain no unique valid records")
    if any(split_counts[split] == 0 for split in _SPLITS):
        raise ValueError(f"trace split assignment produced an empty split: {split_counts}")
    audit = {
        "records": len(candidates),
        "duplicate_records": duplicate_records,
        "excluded_records": excluded_records,
        "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        "split_records": split_counts,
        "mode_records": dict(sorted(mode_counts.items())),
        "source_dataset_records": dict(sorted(source_counts.items())),
    }
    return candidates, files, audit


def _read_trace_candidate(
    stream: Any,
    candidate: _TraceCandidate,
    source: TraceSource,
    tokenizer: Any,
    expected_generator_model: str,
) -> tuple[list[int], str, str, int | None]:
    stream.seek(candidate.offset)
    raw = stream.read(candidate.length)
    if hashlib.sha256(raw.strip()).hexdigest() != candidate.trace_sha256:
        raise RuntimeError("trace source changed between indexing and tokenization")
    value = json.loads(raw)
    return _trace_token_ids(
        tokenizer,
        value,
        expected_generator_model=expected_generator_model,
        expected_mode=source.expected_mode,
    )


def build_trace_corpus_from_jsonl(
    jsonl_sources: Sequence[tuple[str | Path, TraceSource]],
    tokenizer: Any,
    output: str | Path,
    *,
    expected_generator_model: str,
    train_tokens: int = 1_572_864,
    validation_tokens: int = 262_144,
    sequence_length: int = 2048,
    seed: int = 20260718,
    source_metadata: Mapping[str, Any] | None = None,
) -> CalibrationCorpus:
    """Build an exact-token corpus while preserving every selected trace boundary."""

    if train_tokens < 2 or validation_tokens < 2:
        raise ValueError("train_tokens and validation_tokens must both be at least 2")
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if not expected_generator_model.strip():
        raise ValueError("expected_generator_model is required for trace calibration")
    if not jsonl_sources:
        raise ValueError("at least one trace JSONL source is required")
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"calibration corpus already exists: {output_path}")

    resolved_sources = tuple((Path(path).resolve(), source) for path, source in jsonl_sources)
    validation_fraction = validation_tokens / (train_tokens + validation_tokens)
    candidates, file_metadata, audit = _index_trace_sources(
        resolved_sources,
        expected_generator_model=expected_generator_model,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    pools = {
        split: sorted(
            (candidate for candidate in candidates if candidate.split == split),
            key=lambda candidate: (candidate.priority, candidate.trace_sha256),
        )
        for split in _SPLITS
    }
    domains = tuple(sorted({candidate.mode for candidate in candidates}))
    domain_indices = {domain: index for index, domain in enumerate(domains)}
    chunks: list[np.ndarray] = []
    chunk_splits: list[int] = []
    chunk_domains: list[int] = []
    selected_hashes = {split: {domain: [] for domain in domains} for split in _SPLITS}
    selected_prompts = {split: set() for split in _SPLITS}
    record_counts = {split: {domain: 0 for domain in domains} for split in _SPLITS}
    source_counts = {split: {} for split in _SPLITS}
    selected_lengths: list[int] = []
    usage_deltas: list[int] = []
    rejected_overlong = {split: 0 for split in _SPLITS}
    partial_records = {split: 0 for split in _SPLITS}
    targets = {"train": int(train_tokens), "validation": int(validation_tokens)}

    streams = [path.open("rb") for path, _source in resolved_sources]
    try:
        for split in ("train", "validation"):
            remaining = targets[split]
            for candidate in pools[split]:
                ids, mode, source_dataset, declared_tokens = _read_trace_candidate(
                    streams[candidate.file_index],
                    candidate,
                    resolved_sources[candidate.file_index][1],
                    tokenizer,
                    expected_generator_model,
                )
                if mode != candidate.mode or source_dataset != candidate.source_dataset:
                    raise RuntimeError("trace metadata changed between indexing and tokenization")
                if len(ids) > sequence_length:
                    rejected_overlong[split] += 1
                    continue
                if declared_tokens is not None:
                    usage_deltas.append(len(ids) - declared_tokens)
                if len(ids) < remaining - 1:
                    take = len(ids)
                elif len(ids) >= remaining:
                    take = remaining
                else:
                    continue
                if take < 2:
                    continue
                if take != len(ids):
                    partial_records[split] += 1
                chunk = np.asarray(ids[:take], dtype=np.int32)
                chunks.append(chunk)
                chunk_splits.append(_SPLITS[split])
                chunk_domains.append(domain_indices[mode])
                selected_hashes[split][mode].append(candidate.trace_sha256)
                selected_prompts[split].add(candidate.prompt_sha256)
                record_counts[split][mode] += 1
                counts = source_counts[split]
                counts[source_dataset] = counts.get(source_dataset, 0) + 1
                selected_lengths.append(take)
                remaining -= take
                if remaining == 0:
                    break
            if remaining:
                raise ValueError(
                    f"trace sources did not provide enough <= {sequence_length}-token records "
                    f"for {split}: missing {remaining} tokens"
                )
    finally:
        for stream in streams:
            stream.close()

    overlap = selected_prompts["train"] & selected_prompts["validation"]
    if overlap:
        raise RuntimeError("trace train and validation prompt sets overlap")
    actual_tokens = {
        split: int(
            sum(
                chunk.size
                for chunk, split_id in zip(chunks, chunk_splits, strict=True)
                if split_id == _SPLITS[split]
            )
        )
        for split in _SPLITS
    }
    if actual_tokens != targets:
        raise RuntimeError(f"trace token counts differ from requested counts: {actual_tokens}")
    token_digest = hashlib.sha256()
    for chunk in chunks:
        token_digest.update(chunk.astype("<i4", copy=False).tobytes())
    manifest = {
        "format": _FORMAT,
        "corpus_kind": "model_trace",
        "seed": int(seed),
        "sequence_length": int(sequence_length),
        "domains": list(domains),
        "domain_weights": {
            domain: float(
                sum(
                    chunk.size
                    for chunk, domain_id in zip(chunks, chunk_domains, strict=True)
                    if domain_id == domain_indices[domain]
                )
                / sum(chunk.size for chunk in chunks)
            )
            for domain in domains
        },
        "target_tokens": targets,
        "actual_tokens": actual_tokens,
        "record_counts": record_counts,
        "record_sets": {
            split: {
                domain: {
                    "count": len(selected_hashes[split][domain]),
                    "sha256": hashlib.sha256(
                        "\n".join(sorted(selected_hashes[split][domain])).encode("ascii")
                    ).hexdigest(),
                }
                for domain in domains
            }
            for split in _SPLITS
        },
        "prompt_sets": {
            split: {
                "count": len(selected_prompts[split]),
                "sha256": hashlib.sha256(
                    "\n".join(sorted(selected_prompts[split])).encode("ascii")
                ).hexdigest(),
            }
            for split in _SPLITS
        },
        "token_sha256": token_digest.hexdigest(),
        "tokenizer": _tokenizer_metadata(tokenizer, "trace-chat"),
        "chat_template_kwargs": {"add_generation_prompt": False},
        "sources": {
            **dict(source_metadata or {}),
            "files": file_metadata,
            "audit": audit,
        },
        "trace": {
            "expected_generator_model": expected_generator_model,
            "validation_fraction": validation_fraction,
            "boundary_preserved": True,
            "maximum_trace_tokens": int(sequence_length),
            "selected_length": _number_summary(selected_lengths),
            "rendered_minus_declared_tokens": _number_summary(usage_deltas),
            "rejected_overlong": rejected_overlong,
            "partial_records": partial_records,
            "selected_source_dataset_records": {
                split: dict(sorted(source_counts[split].items())) for split in _SPLITS
            },
        },
    }
    return _write_corpus_artifact(output_path, chunks, chunk_splits, chunk_domains, manifest)


def build_hf_trace_corpus(
    tokenizer: Any,
    output: str | Path,
    *,
    repo_id: str,
    revision: str | None,
    sources: Sequence[TraceSource],
    expected_generator_model: str,
    cache_dir: str | Path | None = None,
    train_tokens: int = 1_572_864,
    validation_tokens: int = 262_144,
    sequence_length: int = 2048,
    seed: int = 20260718,
) -> CalibrationCorpus:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("HF trace calibration requires huggingface_hub") from exc

    info = HfApi().dataset_info(repo_id, revision=revision)
    resolved_revision = str(info.sha)
    downloaded = [
        (
            Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=source.filename,
                    repo_type="dataset",
                    revision=resolved_revision,
                    cache_dir=None if cache_dir is None else str(cache_dir),
                )
            ).resolve(),
            source,
        )
        for source in sources
    ]
    return build_trace_corpus_from_jsonl(
        downloaded,
        tokenizer,
        output,
        expected_generator_model=expected_generator_model,
        train_tokens=train_tokens,
        validation_tokens=validation_tokens,
        sequence_length=sequence_length,
        seed=seed,
        source_metadata={
            "repo_id": repo_id,
            "requested_revision": revision,
            "resolved_revision": resolved_revision,
        },
    )


def load_eaddario_records(
    *,
    repo_id: str = "eaddario/imatrix-calibration",
    sources: Sequence[EaddarioSource] = DEFAULT_EADDARIO_SOURCES,
    cache_dir: str | Path | None = None,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Download selected parquet sources and return their ``content`` records."""

    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "eaddario calibration data requires huggingface_hub and pyarrow"
        ) from exc

    records: dict[str, list[str]] = {}
    metadata: dict[str, Any] = {"repo_id": repo_id, "files": {}}
    for source in sources:
        resolved = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=source.filename,
                repo_type="dataset",
                cache_dir=None if cache_dir is None else str(cache_dir),
            )
        ).resolve()
        schema = pq.read_schema(resolved)
        column = (
            "content" if "content" in schema.names else "text" if "text" in schema.names else None
        )
        if column is None:
            raise ValueError(f"{source.filename} has neither a content nor text column")
        table = pq.read_table(resolved, columns=[column])
        parquet_values = table.column(column).to_pylist()
        values = [
            line.strip()
            for value in parquet_values
            for line in str(value).splitlines()
            if line.strip()
        ]
        if not values:
            raise ValueError(f"{source.filename} contains no non-empty records")
        records[source.domain] = values
        metadata["files"][source.domain] = {
            "filename": source.filename,
            "weight": float(source.weight),
            "parquet_rows": len(parquet_values),
            "records": len(values),
            "identity": _file_identity(resolved),
        }
    return records, metadata


def build_eaddario_corpus(
    tokenizer: Any,
    output: str | Path,
    *,
    repo_id: str = "eaddario/imatrix-calibration",
    sources: Sequence[EaddarioSource] = DEFAULT_EADDARIO_SOURCES,
    cache_dir: str | Path | None = None,
    train_tokens: int = 1_572_864,
    validation_tokens: int = 262_144,
    sequence_length: int = 2048,
    seed: int = 20260718,
    render_mode: str = "plain",
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> CalibrationCorpus:
    records, metadata = load_eaddario_records(repo_id=repo_id, sources=sources, cache_dir=cache_dir)
    weights = {source.domain: source.weight for source in sources}
    return build_corpus_from_records(
        records,
        tokenizer,
        output,
        train_tokens=train_tokens,
        validation_tokens=validation_tokens,
        sequence_length=sequence_length,
        domain_weights=weights,
        seed=seed,
        render_mode=render_mode,
        chat_template_kwargs=chat_template_kwargs,
        source_metadata=metadata,
    )


def load_corpus(path: str | Path) -> CalibrationCorpus:
    root = Path(path).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"calibration corpus manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != _FORMAT:
        raise ValueError(f"unsupported calibration corpus format: {manifest.get('format')!r}")
    arrays = {
        name: np.load(root / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in ("tokens", "offsets", "split_ids", "domain_ids")
    }
    tokens = arrays["tokens"]
    offsets = arrays["offsets"]
    split_ids = arrays["split_ids"]
    domain_ids = arrays["domain_ids"]
    if tokens.dtype != np.int32 or offsets.dtype != np.int64:
        raise ValueError("calibration token/offset arrays have invalid dtypes")
    if offsets.ndim != 1 or split_ids.ndim != 1 or domain_ids.ndim != 1:
        raise ValueError("calibration corpus metadata arrays must be one-dimensional")
    if offsets.size != split_ids.size + 1 or split_ids.size != domain_ids.size:
        raise ValueError("calibration corpus arrays have inconsistent lengths")
    if int(offsets[0]) != 0 or int(offsets[-1]) != int(tokens.size):
        raise ValueError("calibration offsets do not cover the token array")
    if np.any(offsets[1:] <= offsets[:-1]):
        raise ValueError("calibration chunks must be non-empty")
    if np.any(split_ids > max(_SPLIT_NAMES)):
        raise ValueError("calibration corpus contains an invalid split id")
    if np.any(domain_ids >= len(manifest["domains"])):
        raise ValueError("calibration corpus contains an invalid domain id")
    corpus = CalibrationCorpus(root, manifest, tokens, offsets, split_ids, domain_ids)
    for split, expected in manifest["actual_tokens"].items():
        if corpus.token_count(split) != int(expected):
            raise ValueError(f"calibration split {split} token count does not match manifest")
    return corpus


__all__ = [
    "CalibrationBatch",
    "CalibrationCorpus",
    "DEFAULT_EADDARIO_SOURCES",
    "EADDARIO_SOURCE_SIZES",
    "EaddarioSource",
    "TraceSource",
    "build_corpus_from_records",
    "build_eaddario_corpus",
    "build_hf_trace_corpus",
    "build_trace_corpus_from_jsonl",
    "eaddario_sources",
    "load_corpus",
    "load_eaddario_records",
]
