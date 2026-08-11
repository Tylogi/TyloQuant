from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECODE = (ROOT / "cpp_runtime" / "mfq_decode.cpp").read_text(encoding="utf-8")
METAL_DECODE = (ROOT / "cpp_runtime" / "metal" / "mfq_decode_mlx.cpp").read_text(
    encoding="utf-8"
)
METAL_QWEN = (
    ROOT / "cpp_runtime" / "metal" / "mlx_qwen35_causal_lm.cpp"
).read_text(encoding="utf-8")
METAL_DSV4 = (
    ROOT / "cpp_runtime" / "metal" / "mlx_deepseek_v4_causal_lm.cpp"
).read_text(encoding="utf-8")
METAL_MINICPM = (
    ROOT / "cpp_runtime" / "metal" / "mlx_minicpmo45.cpp"
).read_text(encoding="utf-8")
SERVER = (ROOT / "cpp_runtime" / "mfq_server.cpp").read_text(encoding="utf-8")
HEADER = (ROOT / "cpp_runtime" / "mfq_server.h").read_text(encoding="utf-8")


def test_native_session_identifier_reaches_the_cuda_runtime() -> None:
    assert "std::string session_id;" in HEADER
    assert 'body.contains("mfq_session_id")' in SERVER
    assert "valid_mfq_session_id" in SERVER
    assert "const MfqPromptCachePlan & cache_plan" in DECODE
    assert "cache_plan.session_id" in DECODE


def test_full_attention_session_state_copies_only_visible_linear_kv() -> None:
    assert "struct TextSessionState" in DECODE
    assert "supports_text_session_state" in DECODE
    assert "saved.ring" in DECODE
    assert "std::min<int64_t>(cache_pos, saved.capacity)" in DECODE
    assert "restore_text_session_state" in DECODE


def test_deepseek_v4_session_state_preserves_local_and_compressed_caches() -> None:
    assert "TextSessionStateKind::DeepseekV4" in DECODE
    assert "struct Dsv4PoolSessionState" in DECODE
    assert "saved.local_cache = dsv4->local_cache.clone()" in DECODE
    assert "capture_dsv4_pool_session_state(" in DECODE
    assert "restore_dsv4_pool_session_state(" in DECODE
    assert "cache_pos / source.ratio" in DECODE


def test_glm_dsa_session_state_preserves_mla_and_index_caches() -> None:
    assert "TextSessionStateKind::GlmDsa" in DECODE
    assert "struct GlmDsaBlockSessionState" in DECODE
    assert "saved.kv_cache = glm->kv_cache.narrow(" in DECODE
    assert "saved.index_cache = glm->index_cache.narrow(" in DECODE
    assert "glm->shared_state->reset()" in DECODE
    assert 'a == "--check-text-session-state"' in DECODE
    assert '"text_session_state_check dsv4=1 glm_dsa=1\\n"' in DECODE


def test_partial_stable_prefix_is_saved_before_generation_suffix() -> None:
    assert "stable_prefix_tokens < prompt.size()" in DECODE
    assert "store_session_snapshot(stable_prefix_tokens)" in DECODE
    assert "tokens.size() > maximum_prefix_tokens" in DECODE


def test_session_cache_uses_exact_prefixes_and_reports_suffix_prefill() -> None:
    assert "!std::equal(" in DECODE
    assert "tokens.begin(), tokens.end(), prompt.begin()))" in DECODE
    assert "tokens.size() >= prompt.size()" in DECODE
    assert "prompt.size() - reused_tokens" in DECODE
    assert "MFQ_SERVER_MAX_KV_SESSIONS" in DECODE
    assert "MFQ_SERVER_MAX_KV_SNAPSHOTS_PER_SESSION" in DECODE
    assert "MFQ_SERVER_KV_SESSION_BYTES" in DECODE


def test_session_cache_retains_history_and_exposes_lifecycle_controls() -> None:
    assert "std::vector<TextSessionState>> states_" in DECODE
    assert "fork_session(" in DECODE
    assert "close_session(" in DECODE
    assert 'server.Post("/api/runtime/sessions/fork"' in SERVER
    assert 'R"(/api/runtime/sessions/' in SERVER
    assert "MfqSessionControl" in HEADER


def test_metal_runtime_matches_native_session_lifecycle_and_limits() -> None:
    assert "class MlxServerTextSessionCache" in METAL_DECODE
    assert "MFQ_SERVER_MAX_KV_SESSIONS" in METAL_DECODE
    assert "MFQ_SERVER_MAX_KV_SNAPSHOTS_PER_SESSION" in METAL_DECODE
    assert "MFQ_SERVER_KV_SESSION_BYTES" in METAL_DECODE
    assert "fork_session(" in METAL_DECODE
    assert "close_session(" in METAL_DECODE
    assert "MfqSessionControl session_control" in METAL_DECODE
    assert "backend=metal" in METAL_DECODE


def test_all_metal_text_graphs_capture_and_restore_prefix_state() -> None:
    for source in (METAL_QWEN, METAL_DSV4, METAL_MINICPM):
        assert "capture_text_session_state" in source
        assert "restore_text_session_state" in source
        assert "prompt.size() - reused_tokens" in source
