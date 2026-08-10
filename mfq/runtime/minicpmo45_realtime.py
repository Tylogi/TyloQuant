from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import json
import os
import tempfile
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.websockets import WebSocket, WebSocketDisconnect

SAMPLE_RATE_IN = 16_000
SAMPLE_RATE_OUT = 24_000
DEFAULT_WEB_ROOT = Path(__file__).resolve().parents[2] / "cpp_runtime" / "web"
DEFAULT_DUPLEX_SYSTEM_PROMPT = "Streaming Omni Conversation."
DEFAULT_DUPLEX_CONFIG: dict[str, Any] = {
    "system_prompt": DEFAULT_DUPLEX_SYSTEM_PROMPT,
    "decode_mode": "sampling",
    "temperature": 0.7,
    "top_k": 100,
    "top_p": 0.8,
    "text_repetition_penalty": 1.05,
    "text_repetition_window_size": 512,
    "length_penalty": 1.0,
    "listen_prob_scale": 1.0,
    "force_listen_count": 0,
}


def _session_system_prompt(
    payload: dict[str, Any],
    effective_config: dict[str, Any],
) -> str:
    value = (
        payload["system_prompt"]
        if "system_prompt" in payload
        else effective_config.get("system_prompt", DEFAULT_DUPLEX_SYSTEM_PROMPT)
    )
    if not isinstance(value, str):
        raise ValueError("system_prompt must be a string")
    return value


def _encode_f32(values: np.ndarray) -> str:
    data = np.ascontiguousarray(values, dtype="<f4")
    return base64.b64encode(data.tobytes()).decode("ascii")


def _decode_f32(value: str, *, maximum_samples: int) -> np.ndarray:
    try:
        raw = base64.b64decode(value, validate=True)
    except Exception as error:
        raise ValueError("audio must be valid base64 float32 PCM") from error
    if len(raw) == 0 or len(raw) % 4 != 0:
        raise ValueError("audio must contain float32 PCM samples")
    if len(raw) // 4 > maximum_samples:
        raise ValueError("audio payload is too large")
    result = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("audio contains a non-finite sample")
    return result


def _ws_url(base_url: str, path: str) -> str:
    parsed = urlsplit(base_url.rstrip("/") + "/" + path.lstrip("/"))
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunsplit((scheme, parsed.netloc, parsed.path, parsed.query, ""))


def _backend_token2wav_steps(
    backend_url: str,
    api_key: str,
    explicit_steps: int | None,
) -> int:
    if explicit_steps is not None:
        if explicit_steps <= 0:
            raise ValueError("--token2wav-steps must be positive")
        return explicit_steps
    import httpx

    headers = (
        {"Authorization": f"Bearer {api_key}"}
        if api_key
        else None
    )
    try:
        response = httpx.get(
            f"{backend_url.rstrip('/')}/health",
            headers=headers,
            timeout=5.0,
            trust_env=False,
        )
        health = response.json() if response.is_success else {}
        defaults = health.get("tts_sampling_defaults", {})
        value = defaults.get("token2wav_steps") if isinstance(defaults, dict) else None
        if isinstance(value, int) and value > 0:
            return value
    except Exception:
        pass
    return 10


class MiniCPMOMel:
    def __init__(self) -> None:
        from transformers.audio_utils import mel_filter_bank

        self.filters = mel_filter_bank(
            num_frequency_bins=201,
            num_mel_filters=80,
            min_frequency=0.0,
            max_frequency=8000.0,
            sampling_rate=SAMPLE_RATE_IN,
            norm="slaney",
            mel_scale="slaney",
        )

    def extract(self, waveform: np.ndarray, *, fixed_floor: bool) -> np.ndarray:
        from transformers.audio_utils import spectrogram, window_function

        features = spectrogram(
            np.asarray(waveform, dtype=np.float32),
            window_function(400, "hann"),
            frame_length=400,
            hop_length=160,
            power=2.0,
            mel_filters=self.filters,
            log_mel="log10",
        )[:, :-1]
        if fixed_floor:
            features = np.maximum(features, -10.0)
        else:
            features = np.maximum(features, float(features.max()) - 8.0)
        return np.ascontiguousarray((features + 4.0) / 4.0, dtype=np.float32)


class ExactStreamingMel:
    def __init__(self, extractor: MiniCPMOMel) -> None:
        self.extractor = extractor
        self.pending = np.empty(0, dtype=np.float32)
        self.history = np.empty(0, dtype=np.float32)
        self.first = True
        self.first_padded = False
        self.last_emitted_frame = 0

    def append(self, audio: np.ndarray) -> list[tuple[np.ndarray, int, int]]:
        self.pending = np.concatenate((self.pending, audio.astype(np.float32, copy=False)))
        outputs: list[tuple[np.ndarray, int, int]] = []
        if self.first and not self.first_padded and self.pending.size >= SAMPLE_RATE_IN:
            first_target_samples = 1035 * SAMPLE_RATE_IN // 1000
            self.pending = np.concatenate(
                (
                    np.zeros(first_target_samples - SAMPLE_RATE_IN, dtype=np.float32),
                    self.pending,
                )
            )
            self.first_padded = True
        while True:
            required = 16_480 if self.first else 16_000
            if self.pending.size < required:
                break
            chunk = self.pending[:required]
            self.pending = self.pending[required:]
            self.history = np.concatenate((self.history, chunk))
            full = self.extractor.extract(
                self.history,
                fixed_floor=self.history.size < 5 * SAMPLE_RATE_IN,
            )
            core_start = self.last_emitted_frame
            core_end = core_start + 100
            prefix = 0 if self.first else 2
            suffix = 2
            start = max(0, core_start - prefix)
            end = min(full.shape[1], core_end + suffix)
            if end <= start:
                raise RuntimeError("streaming Mel extractor emitted no stable frames")
            outputs.append((np.ascontiguousarray(full[:, start:end]), prefix, suffix))
            self.last_emitted_frame = core_end
            self.first = False
        return outputs


def _clone_tensors(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tensors(item) for item in value)
    return copy.deepcopy(value)


class AppleToken2Wav:
    def __init__(self, model_path: Path, *, n_timesteps: int = 10) -> None:
        import onnxruntime
        import s3tokenizer
        import torch
        from hyperpyyaml import load_hyperpyyaml
        from stepaudio2.flashcosyvoice.modules.hifigan import HiFTGenerator
        from stepaudio2.token2wav import _setup_cosyvoice2_alias

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.n_timesteps = n_timesteps
        _setup_cosyvoice2_alias()
        self.audio_tokenizer = s3tokenizer.load_model(
            str(model_path / "speech_tokenizer_v2_25hz.onnx")
        ).to(self.device).eval()
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1
        self.speaker_model = onnxruntime.InferenceSession(
            str(model_path / "campplus.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        with (model_path / "flow.yaml").open("r", encoding="utf-8") as handle:
            self.flow = load_hyperpyyaml(handle)["flow"]
        self.flow.load_state_dict(
            torch.load(model_path / "flow.pt", map_location="cpu", weights_only=True),
            strict=True,
        )
        self.flow.to(self.device).eval()
        self.hift = HiFTGenerator()
        hift_state = {
            key.replace("generator.", ""): value
            for key, value in torch.load(
                model_path / "hift.pt", map_location="cpu", weights_only=True
            ).items()
        }
        self.hift.load_state_dict(hift_state, strict=True)
        self.hift.to(self.device).eval()
        self.mel_cache_len = 8
        self.source_cache_len = self.mel_cache_len * 480
        self.speech_window = torch.from_numpy(
            np.hamming(2 * self.source_cache_len).astype(np.float32)
        ).to(self.device)
        self.cache: tuple[Any, ...] | None = None
        self.stream_cache: dict[str, Any] | None = None
        self.hift_cache: dict[str, Any] | None = None
        self.stream_cache_base: dict[str, Any] | None = None
        self.hift_cache_base: dict[str, Any] | None = None
        self.token_buffer: list[int] = []
        self.prepared_prompt_key: tuple[str, int, int] | None = None

    def _load_audio(self, path: Path) -> tuple[Any, int]:
        import soundfile as sf
        import torch

        values, rate = sf.read(path, dtype="float32", always_2d=True)
        return torch.from_numpy(values.mean(axis=1)), int(rate)

    def _prepare_prompt(self, prompt_path: Path) -> tuple[Any, ...]:
        import s3tokenizer
        import torch
        import torchaudio
        import torchaudio.compliance.kaldi as kaldi
        from stepaudio2.flashcosyvoice.utils.audio import mel_spectrogram

        audio, rate = self._load_audio(prompt_path)
        if rate != SAMPLE_RATE_IN:
            audio = torchaudio.functional.resample(audio, rate, SAMPLE_RATE_IN)
        mel = s3tokenizer.log_mel_spectrogram(audio)
        mel, mel_lengths = s3tokenizer.padding([mel])
        prompt_tokens, prompt_token_lengths = self.audio_tokenizer.quantize(
            mel.to(self.device), mel_lengths.to(self.device)
        )
        speaker_features = kaldi.fbank(
            audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=SAMPLE_RATE_IN
        )
        speaker_features -= speaker_features.mean(dim=0, keepdim=True)
        speaker = self.speaker_model.run(
            None,
            {
                self.speaker_model.get_inputs()[0].name:
                    speaker_features.unsqueeze(0).cpu().numpy()
            },
        )[0]
        speaker_embedding = torch.tensor(speaker, device=self.device)
        waveform = audio.unsqueeze(0)
        if SAMPLE_RATE_IN != SAMPLE_RATE_OUT:
            waveform = torchaudio.functional.resample(
                waveform, SAMPLE_RATE_IN, SAMPLE_RATE_OUT
            )
        prompt_mel = mel_spectrogram(waveform).transpose(1, 2)
        prompt_mel = prompt_mel.to(self.device)
        prompt_mel_lengths = torch.tensor(
            [prompt_mel.shape[1]], dtype=torch.int32, device=self.device
        )
        target = prompt_tokens.shape[1] * self.flow.up_rate
        if prompt_mel.shape[1] < target:
            prompt_mel = torch.nn.functional.pad(
                prompt_mel,
                (0, 0, 0, target - prompt_mel.shape[1]),
                mode="replicate",
            )
        return (
            prompt_tokens,
            prompt_token_lengths,
            speaker_embedding,
            prompt_mel,
            prompt_mel_lengths,
        )

    def prepare(self, prompt_path: Path) -> None:
        import torch

        resolved = prompt_path.resolve()
        stat = resolved.stat()
        prompt_key = (str(resolved), stat.st_size, stat.st_mtime_ns)
        if self.prepared_prompt_key == prompt_key and self.cache is not None:
            self.reset_turn()
            return

        self.cache = self._prepare_prompt(prompt_path)
        prompt_tokens, _, speaker, prompt_mel, _ = self.cache
        silence = torch.full(
            (1, 3), 4218, dtype=prompt_tokens.dtype, device=self.device
        )
        self.stream_cache = self.flow.setup_cache(
            torch.cat((prompt_tokens, silence), dim=1),
            prompt_mel,
            speaker,
            n_timesteps=self.n_timesteps,
        )
        self.hift_cache = {
            "mel": torch.zeros(1, prompt_mel.shape[2], 0, device=self.device),
            "source": torch.zeros(1, 1, 0, device=self.device),
            "speech": torch.zeros(1, 0, device=self.device),
        }
        self.stream_cache_base = _clone_tensors(self.stream_cache)
        self.hift_cache_base = _clone_tensors(self.hift_cache)
        self.prepared_prompt_key = prompt_key
        self.reset_turn()

    def reset_turn(self) -> None:
        if self.stream_cache_base is None or self.hift_cache_base is None:
            return
        self.stream_cache = _clone_tensors(self.stream_cache_base)
        self.hift_cache = _clone_tensors(self.hift_cache_base)
        self.token_buffer = [4218, 4218, 4218]

    def _stream(self, tokens: list[int], *, last_chunk: bool) -> np.ndarray:
        import torch
        from stepaudio2.token2wav import fade_in_out

        if self.cache is None or self.stream_cache is None or self.hift_cache is None:
            raise RuntimeError("Token2wav is not prepared")
        _, _, speaker, prompt_mel, _ = self.cache
        token_tensor = torch.tensor([tokens], dtype=torch.int32, device=self.device)
        with torch.inference_mode(), nullcontext():
            chunk_mel, self.stream_cache = self.flow.inference_chunk(
                token=token_tensor,
                spk=speaker,
                cache=self.stream_cache,
                last_chunk=last_chunk,
                n_timesteps=self.n_timesteps,
            )
            cache_limit = prompt_mel.shape[1] + 100
            if self.stream_cache["estimator_att_cache"].shape[4] > cache_limit:
                value = self.stream_cache["estimator_att_cache"]
                self.stream_cache["estimator_att_cache"] = torch.cat(
                    (value[:, :, :, :, : prompt_mel.shape[1]], value[:, :, :, :, -100:]),
                    dim=4,
                )
            if self.stream_cache["conformer_att_cache"].shape[3] > cache_limit:
                value = self.stream_cache["conformer_att_cache"]
                self.stream_cache["conformer_att_cache"] = torch.cat(
                    (value[:, :, :, : prompt_mel.shape[1], :], value[:, :, :, -100:, :]),
                    dim=3,
                )
            mel = torch.cat((self.hift_cache["mel"], chunk_mel), dim=2)
            speech, source = self.hift(mel, self.hift_cache["source"])
            if self.hift_cache["speech"].shape[-1] > 0:
                speech = fade_in_out(
                    speech, self.hift_cache["speech"], self.speech_window
                )
            self.hift_cache = {
                "mel": mel[..., -self.mel_cache_len :].detach().clone(),
                "source": source[:, :, -self.source_cache_len :].detach().clone(),
                "speech": speech[:, -self.source_cache_len :].detach().clone(),
            }
            if not last_chunk:
                speech = speech[:, : -self.source_cache_len]
        return np.ascontiguousarray(speech.detach().cpu().numpy().reshape(-1), dtype=np.float32)

    def push(
        self,
        codes: list[int],
        *,
        end_of_turn: bool,
        force_flush: bool = False,
    ) -> np.ndarray:
        self.token_buffer.extend(int(code) for code in codes if 0 <= int(code) < 6561)
        pieces: list[np.ndarray] = []
        if force_flush:
            while len(self.token_buffer) >= 8:
                chunk_size = min(28, len(self.token_buffer))
                pieces.append(
                    self._stream(self.token_buffer[:chunk_size], last_chunk=False)
                )
                consumed = min(25, chunk_size - 3)
                del self.token_buffer[:consumed]
        else:
            while len(self.token_buffer) >= 28:
                pieces.append(self._stream(self.token_buffer[:28], last_chunk=False))
                del self.token_buffer[:25]
        if end_of_turn and self.token_buffer:
            pieces.append(self._stream(self.token_buffer, last_chunk=True))
            self.reset_turn()
        if not pieces:
            return np.empty(0, dtype=np.float32)
        waveform = np.concatenate(pieces)
        if not end_of_turn and 0 < waveform.size < SAMPLE_RATE_OUT:
            waveform = np.pad(
                waveform,
                (SAMPLE_RATE_OUT - waveform.size, 0),
                mode="constant",
            )
        return waveform


class RealtimeGateway:
    def __init__(
        self,
        backend_url: str,
        assets: Path,
        *,
        api_key: str = "",
        token2wav_steps: int = 10,
    ) -> None:
        self.backend_url = backend_url
        self.assets = assets
        self.api_key = api_key
        self.extractor = MiniCPMOMel()
        self.renderer = AppleToken2Wav(
            assets / "assets" / "token2wav", n_timesteps=token2wav_steps
        )
        self.session_lock = asyncio.Lock()
        self.default_reference_waveform: np.ndarray | None = None
        self.default_reference_mel: np.ndarray | None = None

    async def backend_runtime_defaults(self) -> dict[str, Any]:
        import httpx

        headers = (
            {"Authorization": f"Bearer {self.api_key}"}
            if self.api_key
            else None
        )
        try:
            async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
                response = await client.get(
                    f"{self.backend_url.rstrip('/')}/health",
                    headers=headers,
                )
            health = response.json() if response.is_success else {}
            value = health.get("duplex_sampling_defaults", {})
            return dict(value) if isinstance(value, dict) else {}
        except Exception:
            return {}

    def reference_waveform(self, payload: dict[str, Any]) -> tuple[np.ndarray, Path | None]:
        voice = payload.get("voice") or {}
        encoded = voice.get("ref_audio_base64") or voice.get("tts_ref_audio_base64")
        if encoded:
            waveform = _decode_f32(encoded, maximum_samples=10 * 60 * SAMPLE_RATE_IN)
            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                path = Path(handle.name)
            sf.write(path, waveform, SAMPLE_RATE_IN)
            return waveform, path
        import soundfile as sf

        path = self.assets / "assets" / "system_ref_audio.wav"
        if self.default_reference_waveform is None:
            values, rate = sf.read(path, dtype="float32", always_2d=True)
            waveform = values.mean(axis=1)
            if rate != SAMPLE_RATE_IN:
                import scipy.signal

                waveform = scipy.signal.resample_poly(
                    waveform, SAMPLE_RATE_IN, rate
                )
            self.default_reference_waveform = np.asarray(
                waveform, dtype=np.float32
            )
        return self.default_reference_waveform, None

    async def forward_step(
        self,
        backend: Any,
        client: Any,
        input_payload: dict[str, Any],
    ) -> None:
        await backend.send(json.dumps({"type": "input.append", "input": input_payload}))
        while True:
            event = json.loads(await backend.recv())
            if event.get("type") == "response.step.done":
                return
            if event.get("kind") == "audio_tokens":
                waveform = await asyncio.to_thread(
                    self.renderer.push,
                    event.get("audio_tokens") or [],
                    end_of_turn=bool(event.get("end_of_turn", False)),
                    force_flush=bool(event.get("force_flush", False)),
                )
                if waveform.size:
                    event = dict(event)
                    event["kind"] = "audio"
                    event.pop("audio_tokens", None)
                    event["audio"] = _encode_f32(waveform)
                    event["sample_rate"] = SAMPLE_RATE_OUT
                else:
                    continue
            await client.send_json(event)

    async def serve(self, client: Any) -> None:
        import websockets

        if self.session_lock.locked():
            await client.send_json(
                {"type": "error", "error": {"message": "the native worker is busy"}}
            )
            await client.close(code=1013)
            return
        async with self.session_lock:
            await client.send_json({"type": "session.queue_done"})
            backend = None
            temporary_reference: Path | None = None
            try:
                first = await client.receive_json()
                if first.get("type") != "session.init":
                    raise ValueError("session.init must be the first message")
                payload = first.get("payload") or {}
                if not isinstance(payload, dict):
                    raise ValueError("session.init payload must be an object")
                config = payload.get("config") or {}
                if not isinstance(config, dict):
                    raise ValueError("session config must be an object")
                effective_config = {
                    **DEFAULT_DUPLEX_CONFIG,
                    **await self.backend_runtime_defaults(),
                    **config,
                }
                system_prompt = _session_system_prompt(
                    payload, effective_config
                )
                force_listen_count = int(effective_config["force_listen_count"])
                if not 0 <= force_listen_count <= 60:
                    raise ValueError("config.force_listen_count must be between 0 and 60")
                reference, temporary_reference = self.reference_waveform(payload)
                if temporary_reference is None:
                    if self.default_reference_mel is None:
                        self.default_reference_mel = self.extractor.extract(
                            reference, fixed_floor=False
                        )
                    reference_mel = self.default_reference_mel
                else:
                    reference_mel = self.extractor.extract(
                        reference, fixed_floor=False
                    )
                prompt_path = temporary_reference or (
                    self.assets / "assets" / "system_ref_audio.wav"
                )
                await asyncio.to_thread(self.renderer.prepare, prompt_path)
                headers = (
                    {"Authorization": f"Bearer {self.api_key}"}
                    if self.api_key
                    else None
                )
                backend = await websockets.connect(
                    _ws_url(self.backend_url, "/backend"),
                    additional_headers=headers,
                    max_size=128 * 1024 * 1024,
                    proxy=None,
                )
                init_payload = {
                    "mode": "full_duplex",
                    "system_prompt": system_prompt,
                    "config": effective_config,
                    "reference_audio_features": _encode_f32(reference_mel),
                    "reference_audio_frames": int(reference_mel.shape[1]),
                }
                await backend.send(json.dumps({"type": "session.init", "payload": init_payload}))
                created = json.loads(await backend.recv())
                if created.get("type") != "session.created":
                    raise RuntimeError(f"backend initialization failed: {created}")
                created["config"] = effective_config
                created["token2wav_steps"] = self.renderer.n_timesteps
                await client.send_json(created)
                mel_stream = ExactStreamingMel(self.extractor)
                audio_chunk_count = 0
                while True:
                    request = await client.receive_json()
                    request_type = request.get("type")
                    if request_type == "session.close":
                        await client.send_json(
                            {"type": "session.closed", "reason": request.get("reason", "user_stop")}
                        )
                        break
                    if request_type != "input.append":
                        raise ValueError(f"unsupported message type: {request_type}")
                    input_payload = request.get("input") or {}
                    has_input = False
                    text_input = input_payload.get("text")
                    if text_input is not None:
                        if not isinstance(text_input, str) or not text_input.strip():
                            raise ValueError("input.text must be a non-empty string")
                        await self.forward_step(
                            backend,
                            client,
                            {
                                "text": text_input,
                                "force_speak": True,
                                "max_new_speak_tokens": int(
                                    input_payload.get("max_new_speak_tokens", 20)
                                ),
                            },
                        )
                        has_input = True
                    if "audio" in input_payload:
                        audio = _decode_f32(
                            input_payload["audio"],
                            maximum_samples=10 * SAMPLE_RATE_IN,
                        )
                        chunks = mel_stream.append(audio)
                        for mel, prefix, suffix in chunks:
                            force_listen = bool(
                                input_payload.get("force_listen", False)
                            ) or audio_chunk_count < force_listen_count
                            audio_chunk_count += 1
                            await self.forward_step(
                                backend,
                                client,
                                {
                                    "audio_features": _encode_f32(mel),
                                    "audio_frames": int(mel.shape[1]),
                                    "audio_prefix_extra_frames": prefix,
                                    "audio_suffix_extra_frames": suffix,
                                    "force_listen": force_listen,
                                    "max_new_speak_tokens": int(
                                        input_payload.get("max_new_speak_tokens", 20)
                                    ),
                                },
                            )
                        has_input = True
                    if not has_input:
                        raise ValueError("input.append requires audio or text")
            finally:
                if backend is not None:
                    await backend.close()
                if temporary_reference is not None:
                    temporary_reference.unlink(missing_ok=True)


def build_app(gateway: RealtimeGateway, web_root: Path | None = None) -> Any:
    import httpx
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from starlette.background import BackgroundTask

    root = (web_root or DEFAULT_WEB_ROOT).resolve()
    if not (root / "index.html").is_file():
        raise FileNotFoundError(f"MFQ WebUI was not found at {root}")
    backend_http = httpx.AsyncClient(timeout=None, trust_env=False)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> Any:
        try:
            yield
        finally:
            await backend_http.aclose()

    app = FastAPI(title="MFQ MiniCPM-o Realtime", lifespan=lifespan)

    excluded_request_headers = {"connection", "content-length", "host", "transfer-encoding"}
    excluded_response_headers = {"connection", "content-length", "transfer-encoding"}

    async def proxy(request: Request, path: str) -> StreamingResponse:
        headers = {
            name: value
            for name, value in request.headers.items()
            if name.lower() not in excluded_request_headers
        }
        if gateway.api_key and "authorization" not in headers:
            headers["authorization"] = f"Bearer {gateway.api_key}"
        upstream_request = backend_http.build_request(
            request.method,
            f"{gateway.backend_url.rstrip('/')}{path}",
            params=request.query_params,
            headers=headers,
            content=await request.body(),
        )
        upstream = await backend_http.send(upstream_request, stream=True)
        response_headers = {
            name: value
            for name, value in upstream.headers.items()
            if name.lower() not in excluded_response_headers
        }
        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers=response_headers,
            background=BackgroundTask(upstream.aclose),
        )

    @app.get("/realtime/capabilities")
    async def realtime_capabilities() -> dict[str, Any]:
        headers = (
            {"Authorization": f"Bearer {gateway.api_key}"}
            if gateway.api_key
            else None
        )
        health: dict[str, Any] = {}
        try:
            response = await backend_http.get(
                f"{gateway.backend_url.rstrip('/')}/health",
                headers=headers,
            )
            health = response.json() if response.is_success else {}
            available = health.get("duplex_available") is True
        except Exception:
            available = False
        tts_defaults = health.get("tts_sampling_defaults", {})
        if not isinstance(tts_defaults, dict):
            tts_defaults = {}
        return {
            "available": available,
            "input": ["audio"],
            "output": ["text", "audio"],
            "input_sample_rate": SAMPLE_RATE_IN,
            "output_sample_rate": SAMPLE_RATE_OUT,
            "defaults": {
                **DEFAULT_DUPLEX_CONFIG,
                **(
                    health.get("duplex_sampling_defaults", {})
                    if isinstance(health.get("duplex_sampling_defaults"), dict)
                    else {}
                ),
                "max_new_speak_tokens_per_chunk": 20,
                "tts_temperature": tts_defaults.get("temperature", 0.8),
                "tts_repetition_penalty": tts_defaults.get(
                    "repetition_penalty", 1.05
                ),
                "token2wav_steps": gateway.renderer.n_timesteps,
                "playback_delay_ms": 200,
            },
        }

    @app.api_route(
        "/v1/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def proxy_v1(request: Request, path: str) -> StreamingResponse:
        return await proxy(request, f"/v1/{path}")

    @app.api_route(
        "/api/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def proxy_api(request: Request, path: str) -> StreamingResponse:
        return await proxy(request, f"/api/{path}")

    @app.get("/health")
    async def health(request: Request) -> StreamingResponse:
        return await proxy(request, "/health")

    @app.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket) -> None:
        if websocket.query_params.get("mode", "audio") != "audio":
            await websocket.close(code=1008, reason="audio mode is required")
            return
        await websocket.accept()
        try:
            await gateway.serve(websocket)
        except WebSocketDisconnect:
            pass
        except Exception as error:
            try:
                await websocket.send_json(
                    {"type": "error", "error": {"message": str(error)}}
                )
                await websocket.close(code=1011)
            except Exception:
                pass

    app.mount("/", StaticFiles(directory=root, html=True), name="web")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MiniCPM-o 4.5 realtime audio gateway for an MFQ native worker"
    )
    parser.add_argument("--backend", default="http://127.0.0.1:8080")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--web-root", type=Path, default=DEFAULT_WEB_ROOT)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--token2wav-steps", type=int)
    arguments = parser.parse_args()
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    gateway = RealtimeGateway(
        arguments.backend,
        arguments.assets,
        api_key=arguments.api_key,
        token2wav_steps=_backend_token2wav_steps(
            arguments.backend,
            arguments.api_key,
            arguments.token2wav_steps,
        ),
    )
    import uvicorn

    uvicorn.run(
        build_app(gateway, arguments.web_root),
        host=arguments.host,
        port=arguments.port,
    )


if __name__ == "__main__":
    main()
