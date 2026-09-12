"""Build a small llama-compatible tokenizer cache from a native HF model."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import struct
from pathlib import Path
from typing import Any

from mfq.formats.assets import (
    HF_CHAT_TEMPLATE_ASSET,
    HF_GENERATION_CONFIG_ASSET,
    HF_TOKENIZER_CONFIG_ASSET,
    HF_TOKENIZER_JSON_ASSET,
    MODEL_CONFIG_ASSET,
    TOKENIZER_GGUF_ASSET,
    minicpmo45_resampler_pos_embed_asset,
)
from mfq.formats.io import open_mmap


class HfTokenizerError(RuntimeError):
    pass


DEEPSEEK_V4_CHAT_TEMPLATE = r"""{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] }}{% elif message['role'] == 'user' %}{{ '<｜User｜>' + message['content'] }}{% elif message['role'] == 'assistant' %}{{ '<｜Assistant｜>' + message['content'] + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<｜Assistant｜>' }}{% if enable_thinking %}{{ '<think>' }}{% else %}{{ '</think>' }}{% endif %}{% endif %}"""


DEEPSEEK_V41_CHAT_TEMPLATE = r"""{{ bos_token }}{%- set thinking = enable_thinking | default(true) -%}{%- set effort = reasoning_effort | default('high') -%}{%- if thinking -%}<｜System｜>Reasoning Effort: {%- if effort == 'low' -%}25{%- elif effort == 'xhigh' -%}75{%- elif effort == 'max' -%}100{%- elif effort is number -%}{{ effort }}{%- else -%}50{%- endif -%} (range 1-100, the higher the value, the more thorough the reasoning)\n\n{%- endif -%}{%- for message in messages -%}{%- if message.role == 'system' -%}{%- if not loop.first or not thinking -%}<｜System｜>{%- endif -%}{{ message.content or '' }}{%- elif message.role == 'user' -%}<｜User｜>{%- if message.content is string -%}{{ message.content }}{%- else -%}{%- for block in message.content -%}{%- if block.type == 'text' -%}{{ block.text }}{%- elif block.type in ['image', 'image_url'] -%}<｜deepseek_image｜>{%- endif -%}{%- endfor -%}{%- endif -%}{%- elif message.role == 'tool' -%}<｜User｜><tool_result>{{ message.content or '' }}</tool_result>{%- elif message.role == 'assistant' -%}{%- if message.reasoning_content is defined and message.reasoning_content -%}{{ message.reasoning_content }}</think>{%- endif -%}{{ message.content or '' }}{%- if message.tool_calls is defined and message.tool_calls -%}\n\n<｜DSML｜ calls>\n{%- for call in message.tool_calls -%}<｜DSML｜ invoke name=\"{{ call.function.name }}\">\n<｜DSML｜ parameter name=\"arguments\" string=\"false\">{{ call.function.arguments | tojson }}</｜DSML｜ parameter>\n</｜DSML｜ invoke>{%- endfor -%}\n</｜DSML｜ calls>{%- endif -%}{{ eos_token }}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}<｜Assistant｜>{%- if thinking -%}<think>{%- else -%}</think>{%- endif -%}{%- endif -%}"""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HfTokenizerError(f"cannot read {path.name}") from error
    if not isinstance(value, dict):
        raise HfTokenizerError(f"invalid {path.name}")
    return value


def _token_content(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    return None


def _integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        return next((item for item in value if isinstance(item, int)), None)
    return None


def _tokenizer_pre(model_type: str) -> str:
    if model_type in {"qwen3_5", "qwen3_6", "qwen3_8", "qwen4_exp"}:
        return "qwen35"
    if model_type.startswith("minicpmo"):
        return "qwen2"
    if model_type.startswith("deepseek_v4"):
        return "joyai-llm"
    raise HfTokenizerError(f"unsupported native HF tokenizer family: {model_type}")


def _special_id(
    name: str,
    tokenizer_config: dict[str, Any],
    config: dict[str, Any],
    generation_config: dict[str, Any],
    token_ids: dict[str, int],
) -> int | None:
    content = _token_content(tokenizer_config.get(f"{name}_token"))
    if content is not None and content in token_ids:
        return token_ids[content]
    text_config = config.get("text_config")
    candidates = (
        config.get(f"{name}_token_id"),
        text_config.get(f"{name}_token_id") if isinstance(text_config, dict) else None,
        generation_config.get(f"{name}_token_id"),
    )
    return next((value for item in candidates if (value := _integer(item)) is not None), None)


def _fingerprint_payloads(payloads: tuple[tuple[str, bytes], ...]) -> str:
    digest = hashlib.sha256()
    digest.update(b"mfq-hf-tokenizer-gguf-v4\0")
    for name, payload in payloads:
        digest.update(name.encode("utf-8"))
        digest.update(payload)
    return digest.hexdigest()[:20]


def _fingerprint(root: Path) -> str:
    payloads = tuple(
        (name, path.read_bytes())
        for name in (
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        )
        if (path := root / name).is_file()
    )
    return _fingerprint_payloads(payloads)


def _cache_root(cache_directory: str | Path | None) -> Path:
    root = (
        Path(cache_directory).expanduser().resolve()
        if cache_directory is not None
        else Path(os.environ.get("MFQ_SERVER_TOKENIZER_CACHE_DIR", "~/.cache/mfq/tokenizers"))
        .expanduser()
        .resolve()
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _json_payload(payload: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HfTokenizerError(f"cannot read embedded {name}") from error
    if not isinstance(value, dict):
        raise HfTokenizerError(f"invalid embedded {name}")
    return value


def _write_tokenizer_gguf(
    *,
    output: Path,
    model_name: str,
    tokenizer: dict[str, Any],
    tokenizer_config: dict[str, Any],
    config: dict[str, Any],
    generation_config: dict[str, Any],
) -> Path:
    if output.is_file() and output.stat().st_size > 0:
        return output

    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise HfTokenizerError("HF config.json has no model_type")
    if not isinstance(tokenizer_config.get("chat_template"), (str, list)):
        tokenizer_config = dict(tokenizer_config)
        if model_type.startswith("deepseek_v41"):
            tokenizer_config["chat_template"] = DEEPSEEK_V41_CHAT_TEMPLATE
        elif model_type.startswith("deepseek_v4"):
            tokenizer_config["chat_template"] = DEEPSEEK_V4_CHAT_TEMPLATE

    model = tokenizer.get("model")
    if not isinstance(model, dict) or model.get("type") != "BPE":
        raise HfTokenizerError("only HF BPE tokenizers are supported by the native runtime")
    raw_vocab = model.get("vocab")
    raw_merges = model.get("merges")
    raw_added = tokenizer.get("added_tokens", [])
    if not isinstance(raw_vocab, dict) or not isinstance(raw_merges, list) or not isinstance(raw_added, list):
        raise HfTokenizerError("invalid HF BPE tokenizer.json")

    try:
        from gguf import GGUFWriter, TokenType
    except ModuleNotFoundError as error:
        raise HfTokenizerError("native HF loading requires the lightweight 'gguf' package") from error

    assigned: dict[int, tuple[str, int]] = {}
    token_ids: dict[str, int] = {}
    for token, raw_id in raw_vocab.items():
        if not isinstance(token, str) or not isinstance(raw_id, int) or raw_id < 0:
            raise HfTokenizerError("invalid HF BPE vocabulary entry")
        assigned[raw_id] = (token, int(TokenType.NORMAL))
        token_ids[token] = raw_id
    for entry in raw_added:
        if not isinstance(entry, dict):
            raise HfTokenizerError("invalid HF added token entry")
        token = entry.get("content")
        raw_id = entry.get("id")
        if not isinstance(token, str) or not isinstance(raw_id, int) or raw_id < 0:
            raise HfTokenizerError("invalid HF added token entry")
        control_alias = (
            token.startswith("<|fim_") or
            token in {"<|repo_name|>", "<|file_sep|>", "</s>"}
        )
        token_type = (
            TokenType.CONTROL
            if entry.get("special") is True or control_alias
            else TokenType.USER_DEFINED
        )
        assigned[raw_id] = (token, int(token_type))
        token_ids[token] = raw_id

    text_config = config.get("text_config")
    configured_vocab = _integer(config.get("vocab_size"))
    if configured_vocab is None and isinstance(text_config, dict):
        configured_vocab = _integer(text_config.get("vocab_size"))
    minimum_vocab = max(assigned, default=-1) + 1
    vocabulary_size = max(minimum_vocab, configured_vocab or 0)
    if vocabulary_size <= 0:
        raise HfTokenizerError("HF tokenizer has an empty vocabulary")
    tokens: list[str] = []
    token_types: list[int] = []
    for token_id in range(vocabulary_size):
        token, token_type = assigned.get(
            token_id, (f"[PAD{token_id}]", int(TokenType.UNUSED))
        )
        tokens.append(token)
        token_types.append(token_type)

    merges: list[str] = []
    for merge in raw_merges:
        if isinstance(merge, str):
            merges.append(merge)
        elif (isinstance(merge, list) and len(merge) == 2 and
              all(isinstance(item, str) for item in merge)):
            merges.append(" ".join(merge))
        else:
            raise HfTokenizerError("invalid HF BPE merge entry")

    temporary = output.with_name(
        f".{output.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    )
    temporary.unlink(missing_ok=True)
    try:
        writer = GGUFWriter(temporary, "llama")
        try:
            writer.add_name(model_name)
            writer.add_tokenizer_model("gpt2")
            writer.add_tokenizer_pre(_tokenizer_pre(model_type))
            writer.add_token_list(tokens)
            writer.add_token_types(token_types)
            writer.add_token_merges(merges)
            for name, method_name in (
                ("bos", "add_bos_token_id"),
                ("eos", "add_eos_token_id"),
                ("unk", "add_unk_token_id"),
                ("pad", "add_pad_token_id"),
            ):
                token_id = _special_id(
                    name, tokenizer_config, config, generation_config, token_ids
                )
                if token_id is not None:
                    getattr(writer, method_name)(token_id)
            if isinstance(tokenizer_config.get("add_bos_token"), bool):
                writer.add_add_bos_token(tokenizer_config["add_bos_token"])
            if isinstance(tokenizer_config.get("add_eos_token"), bool):
                writer.add_add_eos_token(tokenizer_config["add_eos_token"])
            chat_template = tokenizer_config.get("chat_template")
            if isinstance(chat_template, (str, list)):
                writer.add_chat_template(chat_template)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
        finally:
            writer.close()
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output


def ensure_hf_tokenizer_gguf(
    model_directory: str | Path,
    cache_directory: str | Path | None = None,
) -> Path:
    """Return a reusable tokenizer-only GGUF for a supported HF checkpoint."""

    root = Path(model_directory).expanduser().resolve()
    bundled = root / "tokenizer.gguf"
    if bundled.is_file():
        return bundled
    if not root.is_dir():
        raise HfTokenizerError(f"HF model directory does not exist: {root}")

    tokenizer = _read_json(root / "tokenizer.json")
    tokenizer_config = _read_json(root / "tokenizer_config.json")
    config = _read_json(root / "config.json")
    generation_path = root / "generation_config.json"
    generation_config = _read_json(generation_path) if generation_path.is_file() else {}
    output = _cache_root(cache_directory) / (
        f"{root.name}-{_fingerprint(root)}.tokenizer.gguf"
    )
    return _write_tokenizer_gguf(
        output=output,
        model_name=root.name,
        tokenizer=tokenizer,
        tokenizer_config=tokenizer_config,
        config=config,
        generation_config=generation_config,
    )


def ensure_mfq_tokenizer_gguf(
    model_file: str | Path,
    cache_directory: str | Path | None = None,
) -> Path:
    """Build a reusable tokenizer GGUF from assets embedded in an MFQ model."""

    path = Path(model_file).expanduser().resolve()
    if not path.is_file():
        raise HfTokenizerError(f"MFQ model does not exist: {path}")
    with open_mmap(path) as store:
        if TOKENIZER_GGUF_ASSET in store.records:
            payload = store.read_blob(TOKENIZER_GGUF_ASSET)
            fingerprint = hashlib.sha256(payload).hexdigest()[:20]
            output = _cache_root(cache_directory) / (
                f"{path.stem}-{fingerprint}.tokenizer.gguf"
            )
            if output.is_file() and output.stat().st_size > 0:
                return output
            temporary = output.with_name(
                f".{output.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
            )
            temporary.unlink(missing_ok=True)
            try:
                temporary.write_bytes(payload)
                os.replace(temporary, output)
            finally:
                temporary.unlink(missing_ok=True)
            return output
        required = (
            MODEL_CONFIG_ASSET,
            HF_TOKENIZER_JSON_ASSET,
            HF_TOKENIZER_CONFIG_ASSET,
        )
        missing = [name for name in required if name not in store.records]
        if missing:
            raise HfTokenizerError(
                "MFQ model has no embedded native tokenizer metadata: "
                + ", ".join(missing)
            )
        payloads = tuple(
            (name, store.read_blob(name))
            for name in (
                MODEL_CONFIG_ASSET,
                HF_TOKENIZER_JSON_ASSET,
                HF_TOKENIZER_CONFIG_ASSET,
                HF_GENERATION_CONFIG_ASSET,
                HF_CHAT_TEMPLATE_ASSET,
            )
            if name in store.records
        )

    values = dict(payloads)
    tokenizer = _json_payload(values[HF_TOKENIZER_JSON_ASSET], "tokenizer.json")
    tokenizer_config = _json_payload(
        values[HF_TOKENIZER_CONFIG_ASSET], "tokenizer_config.json"
    )
    config = _json_payload(values[MODEL_CONFIG_ASSET], "model_config.json")
    generation_config = (
        _json_payload(values[HF_GENERATION_CONFIG_ASSET], "generation_config.json")
        if HF_GENERATION_CONFIG_ASSET in values
        else {}
    )
    if not isinstance(tokenizer_config.get("chat_template"), (str, list)):
        template = values.get(HF_CHAT_TEMPLATE_ASSET)
        if template is not None:
            try:
                tokenizer_config["chat_template"] = template.decode("utf-8")
            except UnicodeDecodeError as error:
                raise HfTokenizerError(
                    "cannot read embedded chat_template.jinja"
                ) from error

    model_name = path.stem
    output = _cache_root(cache_directory) / (
        f"{model_name}-{_fingerprint_payloads(payloads)}.tokenizer.gguf"
    )
    return _write_tokenizer_gguf(
        output=output,
        model_name=model_name,
        tokenizer=tokenizer,
        tokenizer_config=tokenizer_config,
        config=config,
        generation_config=generation_config,
    )


def native_hf_asset_environment(
    model_directory: str | Path,
    cache_directory: str | Path | None = None,
) -> dict[str, str]:
    """Materialize deterministic non-weight constants omitted by HF checkpoints."""

    root = Path(model_directory).expanduser().resolve()
    if not root.is_dir():
        return {}
    config = _read_json(root / "config.json")
    model_type = config.get("model_type")
    if not isinstance(model_type, str):
        return {}
    cache_root = (
        Path(cache_directory).expanduser().resolve()
        if cache_directory is not None
        else Path(os.environ.get("MFQ_SERVER_ASSET_CACHE_DIR", "~/.cache/mfq/assets"))
        .expanduser()
        .resolve()
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    if model_type.startswith("minicpmo"):
        output = cache_root / "minicpmo45-resampler-pos-embed-v1.bf16"
        if not output.is_file():
            asset = minicpmo45_resampler_pos_embed_asset()
            temporary = output.with_name(
                f".{output.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
            )
            try:
                temporary.write_bytes(asset.data)
                os.replace(temporary, output)
            finally:
                temporary.unlink(missing_ok=True)
        return {"MFQ_MINICPMO45_RESAMPLER_POSITION_ASSET": str(output)}
    if not model_type.startswith("deepseek_v41"):
        return {}

    text_config = config.get("text_config")
    if not isinstance(text_config, dict) or not text_config.get("engram_layer_ids"):
        return {}
    tokenizer_path = root / "tokenizer.json"
    fingerprint_hash = hashlib.sha256()
    fingerprint_hash.update(tokenizer_path.read_bytes())
    fingerprint_hash.update(
        json.dumps(
            {
                name: text_config.get(name)
                for name in (
                    "engram_layer_ids",
                    "engram_num_embeddings",
                    "engram_max_ngram_size",
                    "engram_vocab_size",
                    "engram_n_heads",
                    "engram_head_dim",
                    "engram_pad_token_id",
                    "engram_compressed_vocab_size",
                )
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    fingerprint = fingerprint_hash.hexdigest()[:20]
    output = cache_root / f"deepseek-v41-engram-layout-v2-{fingerprint}.bin"
    if not output.is_file():
        try:
            from tokenizers import Regex, Tokenizer, normalizers
        except ModuleNotFoundError as error:
            raise HfTokenizerError(
                "DeepSeek-V4.1 Engram requires the lightweight 'tokenizers' package"
            ) from error
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        normalizer = normalizers.Sequence(
            [
                normalizers.NFKC(),
                normalizers.NFD(),
                normalizers.StripAccents(),
                normalizers.Lowercase(),
                normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
                normalizers.Replace(Regex(r"^ $"), "\ue000"),
                normalizers.Strip(),
                normalizers.Replace("\ue000", " "),
            ]
        )
        vocabulary_size = tokenizer.get_vocab_size(with_added_tokens=True)
        lookup: list[int] = [0] * vocabulary_size
        keys: dict[str, int] = {}
        for token_id in range(vocabulary_size):
            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
            raw = tokenizer.id_to_token(token_id)
            if raw is None:
                raise HfTokenizerError(
                    f"DeepSeek-V4.1 tokenizer has no token for id {token_id}"
                )
            key = raw if "\ufffd" in decoded else normalizer.normalize_str(decoded)
            if not key:
                key = decoded
            lookup[token_id] = keys.setdefault(key, len(keys))
        expected = text_config.get("engram_compressed_vocab_size")
        if isinstance(expected, int) and len(keys) != expected:
            raise HfTokenizerError(
                "DeepSeek-V4.1 Engram compressed vocabulary mismatch: "
                f"expected {expected}, generated {len(keys)}"
            )
        layer_ids = text_config.get("engram_layer_ids")
        table_rows = text_config.get("engram_num_embeddings")
        max_ngram_size = text_config.get("engram_max_ngram_size")
        hash_vocab_size = text_config.get("engram_vocab_size")
        n_heads = text_config.get("engram_n_heads")
        if (
            not isinstance(layer_ids, list)
            or not layer_ids
            or not all(isinstance(value, int) and value >= 0 for value in layer_ids)
            or not isinstance(table_rows, list)
            or len(table_rows) != len(layer_ids)
            or not all(isinstance(value, int) and value > 0 for value in table_rows)
            or not isinstance(max_ngram_size, int)
            or max_ngram_size < 2
            or not isinstance(hash_vocab_size, int)
            or hash_vocab_size <= 0
            or not isinstance(n_heads, int)
            or n_heads <= 0
        ):
            raise HfTokenizerError("DeepSeek-V4.1 Engram layout is incomplete")

        # These constants are part of the trained hash-table layout. Keep their
        # generation beside the tokenizer map so the native worker has no
        # NumPy/SymPy dependency and cannot silently choose a different RNG.
        import numpy as np

        max_long = np.iinfo(np.int64).max
        multiplier_bound = max(1, (int(max_long) // len(keys)) // 2)
        multipliers: list[int] = []
        for layer_id in layer_ids:
            generator = np.random.default_rng(10007 * layer_id)
            values = generator.integers(
                low=0,
                high=multiplier_bound,
                size=max_ngram_size,
                dtype=np.int64,
            )
            multipliers.extend(int(value * np.int64(2) + np.int64(1)) for value in values)

        def is_prime(value: int) -> bool:
            if value < 2:
                return False
            if value % 2 == 0:
                return value == 2
            divisor = 3
            while divisor * divisor <= value:
                if value % divisor == 0:
                    return False
                divisor += 2
            return True

        seen_primes: set[int] = set()
        primes: list[int] = []
        for table_index, _layer_id in enumerate(layer_ids):
            table_primes: list[int] = []
            for _ngram in range(max_ngram_size - 1):
                current = hash_vocab_size - 1
                for _head in range(n_heads):
                    current += 1
                    while not is_prime(current) or current in seen_primes:
                        current += 1
                    seen_primes.add(current)
                    table_primes.append(current)
            if sum(table_primes) != table_rows[table_index]:
                raise HfTokenizerError(
                    "DeepSeek-V4.1 Engram table rows do not match its prime layout"
                )
            primes.extend(table_primes)

        payload = b"".join(
            (
                struct.pack(
                    "<4sIIIIII",
                    b"D41T",
                    2,
                    len(lookup),
                    len(keys),
                    len(layer_ids),
                    max_ngram_size,
                    n_heads,
                ),
                struct.pack(f"<{len(lookup)}I", *lookup),
                struct.pack(f"<{len(layer_ids)}i", *layer_ids),
                struct.pack(f"<{len(multipliers)}q", *multipliers),
                struct.pack(f"<{len(primes)}I", *primes),
            )
        )
        temporary = output.with_name(
            f".{output.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
        )
        try:
            temporary.write_bytes(payload)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    return {"MFQ_DEEPSEEK_V41_ENGRAM_TOKEN_MAP": str(output)}


__all__ = [
    "HfTokenizerError",
    "DEEPSEEK_V4_CHAT_TEMPLATE",
    "DEEPSEEK_V41_CHAT_TEMPLATE",
    "ensure_hf_tokenizer_gguf",
    "ensure_mfq_tokenizer_gguf",
    "native_hf_asset_environment",
]
