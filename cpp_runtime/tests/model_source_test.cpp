#include "mfq/fp8_sq_blob.h"
#include "mfq/hf_model_source.h"
#include "mfq/hf_safetensors_source.h"
#include "mfq/mfq_model_source.h"
#include "mfq/mapped_tensor_pages.h"
#include "mfq/mapped_prefetch_scope.h"
#include "mfq/tq_rows.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
void write_scalar(std::ofstream& stream, T value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_string(std::ofstream& stream, std::string_view value) {
    write_scalar(stream, static_cast<std::uint32_t>(value.size()));
    stream.write(value.data(), static_cast<std::streamsize>(value.size()));
}

std::string graph_json() {
    return R"json({
        "schema_version": 1,
        "architecture": "qwen3_5",
        "canonical_naming": {
            "namespace": "mfq.tensor",
            "version": 1,
            "component_roots": ["model"]
        },
        "topology": {"text_layers": 1, "vision_layers": 0, "predictor_layers": 0},
        "graph": {
            "kind": "causal_lm",
            "backbone": "qwen3_5",
            "components": [{
                "kind": "text",
                "tensor_root": "model",
                "implementation": "qwen3_5"
            }]
        },
        "capabilities": ["text"]
    })json";
}

void write_mfq(
    const std::filesystem::path& path,
    std::string_view tensor_name = "model.token_embedding.weight",std::int64_t elements=1) {
    const std::string config = R"({"model_type":"qwen3_5"})";
    std::ofstream stream(path, std::ios::binary);
    stream.write("MFQ1", 4);
    write_scalar<std::uint32_t>(stream, 2);
    write_string(stream, "qwen3_5");
    write_scalar<std::uint32_t>(stream, 0);
    write_scalar<std::uint32_t>(stream, 2);
    write_string(stream, tensor_name);
    write_string(stream, "BF16");
    write_scalar<std::uint64_t>(stream, 12+2*elements);
    write_string(stream, "__mfq_asset__/model_config.json");
    write_string(stream, "BLOB");
    write_scalar<std::uint64_t>(stream, config.size());
    write_scalar<std::uint32_t>(stream, 1);
    write_scalar<std::int64_t>(stream, elements);
    std::vector<char> tensor(static_cast<std::size_t>(2*elements));
    for(std::size_t i=0;i<tensor.size();++i)tensor[i]=static_cast<char>(i*17+3);
    tensor[0]=0x12;tensor[1]=0x34;
    stream.write(tensor.data(),static_cast<std::streamsize>(tensor.size()));
    stream.write(config.data(), static_cast<std::streamsize>(config.size()));
}

void write_hf(const std::filesystem::path& root) {
    std::filesystem::create_directories(root / ".mfq-assets" / "hf");
    std::ofstream(root / "config.json") << R"({"model_type":"qwen3_5"})";
    std::ofstream(root / ".mfq-assets" / "model_graph.json") << graph_json();
    std::ofstream(root / ".mfq-assets" / "hf" / "source_tensor_map.json")
        << R"({"schema":"mfq.hf-source-map","version":1,"canonical_to_source":{"model.token_embedding.weight":"raw.embed.weight","model.blocks.0.mlp.experts.0.gate.weight":"raw.e0.weight","model.blocks.0.mlp.experts.0.gate.weight_scale":"raw.e0.scale","model.blocks.0.mlp.experts.1.gate.weight":"raw.e1.weight","model.blocks.0.mlp.experts.1.gate.weight_scale":"raw.e1.scale"}})";
    const std::string header =
        R"({"raw.embed.weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"raw.e0.weight":{"dtype":"I8","shape":[1,16],"data_offsets":[2,18]},"raw.e0.scale":{"dtype":"F8_E8M0","shape":[1,1],"data_offsets":[18,19]},"raw.e1.weight":{"dtype":"I8","shape":[1,16],"data_offsets":[19,35]},"raw.e1.scale":{"dtype":"F8_E8M0","shape":[1,1],"data_offsets":[35,36]}})";
    std::ofstream stream(root / "model.safetensors", std::ios::binary);
    write_scalar<std::uint64_t>(stream, header.size());
    stream.write(header.data(), static_cast<std::streamsize>(header.size()));
    const char tensor[] = {0x12, 0x34};
    stream.write(tensor, sizeof(tensor));
    std::array<char, 34> experts{};
    experts[0] = 0x45;
    experts[16] = 0x7f;
    experts[17] = 0x46;
    experts[33] = 0x7e;
    stream.write(experts.data(), experts.size());
}

void write_hf_mxfp8(const std::filesystem::path& root) {
    std::filesystem::create_directories(root);
    std::ofstream(root / "config.json") << R"({"model_type":"qwen3_5"})";
    const std::string header =
        R"({"block128.weight":{"dtype":"F8_E4M3","shape":[128,128],"data_offsets":[0,16384]},"block128.scale":{"dtype":"F8_E8M0","shape":[1,1],"data_offsets":[16384,16385]},"block32.weight":{"dtype":"F8_E4M3","shape":[33,64],"data_offsets":[16385,18497]},"block32.scale":{"dtype":"F8_E8M0","shape":[2,2],"data_offsets":[18497,18501]},"row32.weight":{"dtype":"F8_E4M3","shape":[3,64],"data_offsets":[18501,18693]},"row32.scale":{"dtype":"F8_E8M0","shape":[3,2],"data_offsets":[18693,18699]}})";
    std::ofstream stream(root / "model.safetensors", std::ios::binary);
    write_scalar<std::uint64_t>(stream, header.size());
    stream.write(header.data(), static_cast<std::streamsize>(header.size()));
    const std::vector<char> payload(18699);
    stream.write(payload.data(), static_cast<std::streamsize>(payload.size()));
}

void write_hf_fp8_128(
    const std::filesystem::path& root,
    bool explicit_source_map = false) {
    std::filesystem::create_directories(root);
    std::ofstream(root / "config.json") << R"({"model_type":"qwen3_5"})";
    if (explicit_source_map) {
        std::filesystem::create_directories(root / ".mfq-assets" / "hf");
        std::ofstream(root / ".mfq-assets" / "hf" / "source_tensor_map.json")
            << R"({"schema":"mfq.hf-source-map","version":1,"canonical_to_source":{"model.block.0.linear_attention.gate.weight":"block.weight","model.block.0.linear_attention.gate.weight_scale":"block.weight_scale_inv"}})";
    }
    const std::string header =
        R"({"block.weight":{"dtype":"F8_E4M3","shape":[128,128],"data_offsets":[0,16384]},"block.weight_scale_inv":{"dtype":"BF16","shape":[1,1],"data_offsets":[16384,16386]}})";
    std::ofstream stream(root / "model.safetensors", std::ios::binary);
    write_scalar<std::uint64_t>(stream, header.size());
    stream.write(header.data(), static_cast<std::streamsize>(header.size()));
    std::array<char, 16384> values{};
    values.front() = 0x12;
    values.back() = 0x34;
    stream.write(values.data(), static_cast<std::streamsize>(values.size()));
    const std::array<char, 2> bf16_one = {
        static_cast<char>(0x80), static_cast<char>(0x3f)};
    stream.write(bf16_one.data(), static_cast<std::streamsize>(bf16_one.size()));
}

void require_bytes(const std::vector<std::byte>& bytes) {
    require(bytes.size() == 14, "unexpected tensor byte count");
    require(std::to_integer<unsigned char>(bytes[12]) == 0x12 &&
                std::to_integer<unsigned char>(bytes[13]) == 0x34,
            "unexpected tensor bytes");
}

void write_shared_fixture(const std::filesystem::path& path, int index,
                          std::string_view name, std::string_view architecture = "test-shared") {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path, std::ios::binary);
    stream.write("MFQ1", 4);
    write_scalar<std::uint32_t>(stream, 2);
    write_string(stream, architecture);
    write_scalar<std::uint32_t>(stream, 4);
    for (const auto& field : std::vector<std::pair<std::string, std::string>>{
            {"split.no", std::to_string(index)}, {"split.count", "3"},
            {"split.tensors.count", "3"}, {"split.records.count", "3"}}) {
        write_string(stream, field.first);
        write_string(stream, field.second);
    }
    write_scalar<std::uint32_t>(stream, 1);
    write_string(stream, name);
    write_string(stream, "BF16");
    write_scalar<std::uint64_t>(stream, 14);
    write_scalar<std::uint32_t>(stream, 1);
    write_scalar<std::int64_t>(stream, 1);
    write_scalar<std::uint16_t>(stream, static_cast<std::uint16_t>(index + 10));
}

void test_shared_engram_shards(const std::filesystem::path& root, const std::string& suffix) {
    const std::string table_name = "model.block.1.associative_memory.embedding.weight";
    const auto shared = root / "Engram" / ("model-EW-Engram-MXFP8-00002-of-00003" + suffix);
    write_shared_fixture(shared, 1, table_name);
    for (const std::string precision : {"Q2.5", "Q3"}) {
        const auto first = root / precision / ("model-EW-" + precision + "-00001-of-00003" + suffix);
        const auto last = root / precision / ("model-EW-" + precision + "-00003-of-00003" + suffix);
        write_shared_fixture(first, 0, "model.token_embedding.weight");
        write_shared_fixture(last, 2, "model.output.weight");
        for (auto mode : {mfq::FileReadMode::Buffered, mfq::FileReadMode::Direct}) {
            mfq::MfqModelSource source(last, mode);
            require(source.source_paths().size() == 3, "shared checkpoint lost shards");
            require(source.source_paths()[1] == std::filesystem::canonical(shared), "shared table was copied or not resolved");
            const auto bytes = source.read(table_name);
            require(bytes.size() == 14 && std::to_integer<int>(bytes[12]) == 11,
                    "shared table read returned incorrect bytes");
        }
    }
    const auto first = root / "Q3" / ("model-EW-Q3-00001-of-00003" + suffix);
    auto rejects = [&] {
        try { mfq::MfqModelSource source(first); return false; }
        catch (const std::runtime_error&) { return true; }
    };
    const auto duplicate = root / "Engram" / ("model-EW-Engram-MXFP4-00002-of-00003" + suffix);
    write_shared_fixture(duplicate, 1, table_name);
    require(rejects(), "ambiguous shared table precision was guessed");
    std::filesystem::remove(duplicate);
    write_shared_fixture(shared, 1, "model.output.other.weight");
    require(rejects(), "shared filename allowed a non-table substitution");
    write_shared_fixture(shared, 1, table_name, "wrong-architecture");
    require(rejects(), "shared table from incompatible architecture was accepted");
    std::filesystem::remove(shared);
    require(rejects(), "missing shared Engram table was accepted");
    const auto other_suffix = suffix == ".mfq" ? ".tq" : ".mfq";
    const auto other = root / "Engram" /
        (std::string("model-EW-Engram-MXFP8-00002-of-00003") + other_suffix);
    write_shared_fixture(other, 1, table_name);
    require(rejects(), "shared shards from different container suffixes were mixed");
    std::filesystem::remove(other);
    // An ordinary complete checkpoint remains loadable beside unusable tables.
    write_shared_fixture(root / "Q3" / ("model-EW-Q3-00002-of-00003" + suffix), 1, table_name);
    write_shared_fixture(shared, 1, "invalid.weight");
    mfq::MfqModelSource legacy(first);
    require(legacy.source_paths()[1].parent_path().filename() == "Q3", "legacy shard was replaced");
}

void test_tq_reserved_descriptor_bits() {
    std::vector<std::uint8_t> payload(64 + 8 + 8 + 32);
    std::memcpy(payload.data(), "RTRELL05", 8);
    mfq::TqLayout::integer(payload.data() + 8, 5, 4);
    mfq::TqLayout::integer(payload.data() + 12, 1, 4);
    mfq::TqLayout::integer(payload.data() + 16, 256, 4);
    mfq::TqLayout::integer(payload.data() + 24, 16, 4);
    mfq::TqLayout::integer(payload.data() + 28, 32, 4);
    mfq::TqRows rows(payload.size(), [&](auto offset, auto* output, auto bytes) {
        std::memcpy(output, payload.data() + offset, bytes);
    });
    require(rows.slice_rows(0, 1) == payload, "valid TQ row changed");
    for (const int bit : {35, 36, 37, 38, 39, 56, 63}) {
        mfq::TqLayout::integer(payload.data() + 64, std::uint64_t(1) << bit, 8);
        bool rejected = false;
        try { rows.slice_rows(0, 1); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "reserved TQ descriptor bit was accepted");
    }
}

} // namespace

int main() {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto root = std::filesystem::temp_directory_path() /
        ("mfq-model-source-test-" + std::to_string(nonce));
    try {
        std::filesystem::create_directories(root);
        test_shared_engram_shards(root / "shared-mfq", ".mfq");
        test_shared_engram_shards(root / "shared-tq", ".tq");
        test_shared_engram_shards(root / "shared-tq-upper", ".TQ");
        test_tq_reserved_descriptor_bits();
        const auto mfq_path = root / "model.mfq";
        const auto legacy_mfq_path = root / "legacy-model.mfq";
        const auto legacy_gguf_path = root / "legacy-gguf-model.mfq";
        const auto hf_path = root / "hf";
        write_mfq(mfq_path);
        write_mfq(root/"mapped-large.mfq","model.token_embedding.weight",131072);
        write_mfq(
            legacy_mfq_path,
            "model.language_model.embed_tokens.weight");
        write_mfq(legacy_gguf_path, "token_embd.weight");
        write_hf(hf_path);
        write_hf_mxfp8(root / "hf-mxfp8");
        write_hf_fp8_128(root / "hf-fp8-128");
        write_hf_fp8_128(root / "hf-fp8-128-explicit", true);

        {
        mfq::MfqModelSource mfq_source(mfq_path);
        mfq::MfqModelSource legacy_mfq_source(legacy_mfq_path);
        mfq::MfqModelSource legacy_gguf_source(legacy_gguf_path);
        mfq::HfModelSource hf_source(hf_path);
        const mfq::ModelSource& generic_hf = hf_source;
        require(!generic_hf.mapped_tensor_reader("model.token_embedding.weight"),
                "unmapped source must advertise an empty mapped reader");
        require(!generic_hf.mapped_tensor_batch_reader("model.token_embedding.weight"),
                "unmapped source must advertise an empty mapped batch reader");
        const auto opened_mfq = mfq::open_model_source(mfq_path);
        const auto opened_hf = mfq::open_model_source(hf_path);

        // The retained reader must outlive the source and preserve canonical
        // legacy-name binding, including strict tensor-relative bounds.
        auto retained = [&] {
            mfq::MfqModelSource temporary(legacy_gguf_path);
            return temporary.tensor_reader("model.token_embedding.weight");
        }();
        std::vector<std::byte> retained_bytes(14);
        retained(0, retained_bytes.data(), retained_bytes.size());
        require_bytes(retained_bytes);
        bool range_rejected = false;
        try { retained(13, retained_bytes.data(), 2); }
        catch (const std::out_of_range&) { range_rejected = true; }
        require(range_rejected, "retained reader accepted out-of-range bytes");
        retained = {}; // Close the independently owned file before cleanup.

        for(auto mode:{mfq::FileReadMode::Buffered,mfq::FileReadMode::Direct}) {
            auto batch=[&] {
                mfq::MfqModelSource temporary(legacy_gguf_path,mode);
                return temporary.tensor_batch_reader("model.token_embedding.weight");
            }();
            require(bool(batch),"retained source is missing its batch reader");
            std::vector<std::byte> actual(14);
            batch({{8,actual.data()+8,6},{0,actual.data(),5},{5,actual.data()+5,3}});
            require_bytes(actual);
            bool rejected=false;
            try{batch({{13,actual.data(),2}});}catch(const std::out_of_range&){rejected=true;}
            require(rejected,"batched reader accepted bytes beyond the canonical tensor");
        }

        {
            auto mapped=[&] {
                mfq::MfqModelSource temporary(legacy_gguf_path);
                return temporary.mapped_tensor_reader("model.token_embedding.weight");
            }();
            std::vector<std::byte> all(14);mapped(0,all.data(),all.size());require_bytes(all);
            for(std::size_t offset=0;offset<=all.size();++offset)for(std::size_t count=0;count<=all.size()-offset;++count) {
                std::vector<std::byte> part(count);mapped(offset,part.data(),count);
                require(std::equal(part.begin(),part.end(),all.begin()+offset),"mapped reader slice differs");
            }
            mapped(14,nullptr,0);
            int rejected=0;
            try {mapped(13,all.data(),2);}catch(const std::out_of_range&){++rejected;}
            try {mapped(UINT64_MAX,all.data(),1);}catch(const std::out_of_range&){++rejected;}
            try {mapped(0,nullptr,1);}catch(const std::invalid_argument&){++rejected;}
            require(rejected==3,"mapped reader boundary guards failed");
            std::vector<std::future<void>> readers;
            for(int worker=0;worker<8;++worker)readers.push_back(std::async(std::launch::async,[mapped] {
                for(int i=0;i<64;++i){std::vector<std::byte> bytes(14);mapped(0,bytes.data(),bytes.size());require_bytes(bytes);}
            }));
            for(auto& worker:readers)worker.get();
        }
        {
            mfq::MfqModelSource large(root/"mapped-large.mfq");
            auto ordinary=large.tensor_reader("model.token_embedding.weight");
            const mfq::ModelSource& generic=large;
            auto mapped=generic.mapped_tensor_reader("model.token_embedding.weight");
            for(std::size_t offset:{0u,4093u,65521u,65536u,131071u,262143u}) {
                const auto count=std::min<std::size_t>(67,262156-offset);
                std::vector<std::byte> expected(count),actual(count);
                ordinary(offset,expected.data(),count);mapped(offset,actual.data(),count);
                require(actual==expected,"mapped tensor crossing page/granularity boundary differs");
            }
            require(mfq::release_mapped_tensor_pages()>0,"mapped source trim did not retain any live reader");
            std::vector<std::byte> expected(8197),actual(8197);
            ordinary(65521,expected.data(),expected.size());mapped(65521,actual.data(),actual.size());
            require(actual==expected,"mapped reader changed after releasing clean source pages");
        }

        {
            auto batch=[&] {
                mfq::MfqModelSource temporary(legacy_gguf_path);
                return temporary.mapped_tensor_batch_reader("model.token_embedding.weight");
            }();
            require(bool(batch),"retained mapped batch reader missing");
            std::vector<std::byte> actual(14);
            {
                mfq::MappedPrefetchScope prefetch(true);
                batch({{8,actual.data()+8,6},{0,actual.data(),5},{5,actual.data()+5,3},{14,nullptr,0}});
            }
            require_bytes(actual);batch({});
            for(const auto bad:std::vector<mfq::ReadSpan>{{13,actual.data(),2},{UINT64_MAX,actual.data(),0},{0,nullptr,1}}) {
                std::fill(actual.begin(),actual.end(),std::byte{0x7f});
                bool rejected=false;
                try{batch({{0,actual.data(),1},bad});}catch(const std::exception&){rejected=true;}
                require(rejected && std::all_of(actual.begin(),actual.end(),[](auto x){return x==std::byte{0x7f};}),
                        "mapped batch must reject all bad ranges before writing");
            }
            std::vector<std::future<void>> readers;
            for(int worker=0;worker<8;++worker)readers.push_back(std::async(std::launch::async,[batch] {
                mfq::MappedPrefetchScope transient(false,true);
                for(int i=0;i<32;++i) {
                    std::vector<std::byte> bytes(14);
                    batch({{7,bytes.data()+7,7},{0,bytes.data(),7}});require_bytes(bytes);
                }
            }));
            for(auto& worker:readers)worker.get();
            auto large_batch=[&] {
                mfq::MfqModelSource temporary(root/"mapped-large.mfq");
                return temporary.mapped_tensor_batch_reader("model.token_embedding.weight");
            }();
            mfq::MfqModelSource large(root/"mapped-large.mfq");
            const auto expected=large.read("model.token_embedding.weight");
            constexpr size_t count=4357,width=67;
            std::vector<std::byte> received(count*width);
            std::vector<mfq::ReadSpan> spans;
            for(size_t i=0;i<count;++i)spans.push_back({(i*4093)%(expected.size()-width+1),received.data()+i*width,width});
            {
                mfq::MappedPrefetchScope transient(true,true);
                large_batch(spans);
            }
            require(mfq::release_mapped_tensor_pages()>0,"mapped batch trim lost its independent owner");
            std::fill(received.begin(),received.end(),std::byte{0});large_batch(spans);
            for(const auto& span:spans)require(std::equal(span.destination,span.destination+span.size,expected.begin()+span.offset),
                    "mapped batch across pages/prefetch chunks differs");
            auto sparse=std::vector<mfq::ReadSpan>(spans.begin(),spans.begin()+64);
            std::reverse(sparse.begin(),sparse.end());
            sparse.push_back(sparse.front());
            sparse.push_back({0,nullptr,0});
            for(int repeat=0;repeat<2;++repeat) {
                std::fill(received.begin(),received.end(),std::byte{0});
                mfq::MappedPrefetchScope transient(false,true);
                large_batch(sparse);
                for(const auto& span:sparse)if(span.size)
                    require(std::equal(span.destination,span.destination+span.size,expected.begin()+span.offset),
                            "sparse mapped page release changed reread data");
            }
        }

        {
            mfq::MfqModelSource direct(legacy_gguf_path,mfq::FileReadMode::Direct);
            require(direct.supports_parallel_tensor_reads(), "direct source refused independent reads");
            mfq::MfqModelSource buffered(legacy_gguf_path,mfq::FileReadMode::Buffered);
            require(!buffered.supports_parallel_tensor_reads(), "seek-based source advertised parallel reads");
            require_bytes(direct.read("model.token_embedding.weight"));
            require(direct.model_config_json()==legacy_gguf_source.model_config_json(),"direct assets differ");
            const auto stats=direct.file_read_stats();
            require(stats.mode==mfq::FileReadMode::Direct && stats.files==1 && stats.calls>0 &&
                    stats.staging_bytes==0 && stats.errors==0,"direct model source accounting differs");
            auto direct_retained=[&] {
                mfq::MfqModelSource temporary(legacy_gguf_path,mfq::FileReadMode::Direct);
                return temporary.tensor_reader("model.token_embedding.weight");
            }();
            direct_retained(0,retained_bytes.data(),retained_bytes.size());
            require_bytes(retained_bytes);
            range_rejected=false;
            try { direct_retained(13,retained_bytes.data(),2); }
            catch (const std::out_of_range&) { range_rejected=true; }
            require(range_rejected,"direct retained reader accepted out-of-range bytes");
        }

        require(mfq_source.tensors().size() == 1,
                "MFQ assets leaked into tensor enumeration");
        require(
            legacy_mfq_source.find_tensor("model.token_embedding.weight") !=
                    nullptr &&
                legacy_mfq_source.find_tensor(
                    "model.language_model.embed_tokens.weight") == nullptr,
            "legacy MFQ tensor name was not canonicalized");
        require_bytes(
            legacy_mfq_source.read("model.token_embedding.weight"));
        const auto& gguf_compatibility =
            legacy_gguf_source.legacy_tensor_compatibility();
        require(
            gguf_compatibility.canonical_to_stored.at(
                "model.token_embedding.weight") == "token_embd.weight" &&
                gguf_compatibility.layout.norm_weight_offset == 0.0 &&
                !gguf_compatibility.layout.linear_attention_a_is_log &&
                gguf_compatibility.layout.qwen_gdn_gguf_layout,
            "legacy MFQ tensor layout was not preserved");
        require_bytes(
            legacy_gguf_source.read("model.token_embedding.weight"));

        for (const mfq::ModelSource* source :
             std::vector<const mfq::ModelSource*>{
                 opened_mfq.get(), opened_hf.get()}) {
            const auto* tensor = source->find_tensor(
                "model.token_embedding.weight");
            require(tensor != nullptr, "canonical tensor is missing");
            require(tensor->dtype == "BF16" && tensor->nbytes == 14,
                    "canonical tensor metadata differs by source format");
            require_bytes(source->read("model.token_embedding.weight"));
            std::byte ignored{};
            source->read_range_into(
                "model.token_embedding.weight", 2, &ignored, 0);
            require(source->resolved_model_graph().backbone == "qwen3_5",
                    "model graph differs by source format");
            require(source->model_config_json() ==
                        R"({"model_type":"qwen3_5"})",
                    "model config differs by source format");
        }
        require(!mfq_source.find_tensor("__mfq_asset__/model_graph.json"),
                "MFQ graph asset is exposed as a tensor");
        require(!mfq_source.has_asset("__mfq_asset__/model_graph.json"),
                "legacy MFQ unexpectedly has a stored graph");
        require(mfq_source.has_asset("__mfq_asset__/model_config.json"),
                "MFQ config asset is missing");
        require(hf_source.has_asset("__mfq_asset__/model_graph.json"),
                "HF graph asset is missing");
        require(hf_source.find_tensor("model.token_embedding.weight")->shape ==
                    std::optional<std::vector<std::int64_t>>({1}),
                "HF tensor shape was lost");
        const auto dense_cross = hf_source.read(
            "model.token_embedding.weight", 11, 3);
        require(dense_cross.size() == 3 &&
                    std::to_integer<unsigned char>(dense_cross[1]) == 0x12 &&
                    std::to_integer<unsigned char>(dense_cross[2]) == 0x34,
                "dense segmented range read is invalid");
        const auto* mx = hf_source.find_tensor(
            "model.blocks.0.mlp.experts.0.gate.weight");
        require(mx != nullptr && mx->dtype == "MXFP4" && mx->nbytes == 73,
                "native MX view was not synthesized");
        const auto mx_bytes = hf_source.read(mx->name);
        require(mx_bytes.size() == 73 &&
                    std::to_integer<char>(mx_bytes[0]) == 'M' &&
                    std::to_integer<char>(mx_bytes[3]) == '1' &&
                    std::to_integer<unsigned char>(mx_bytes[56]) == 0x45 &&
                    std::to_integer<unsigned char>(mx_bytes.back()) == 0x7f,
                "native MX payload is invalid");
        const auto mx_cross = hf_source.read(mx->name, 55, 2);
        require(mx_cross.size() == 2 &&
                    std::to_integer<unsigned char>(mx_cross[1]) == 0x45,
                "native MX segmented range read is invalid");
        const auto* mfe = hf_source.find_tensor(
            "model.blocks.0.mlp.experts.gate.weight");
        require(mfe != nullptr && mfe->dtype == "MFE" && mfe->nbytes == 232,
                "virtual MFE view was not synthesized");
        const auto mfe_bytes = hf_source.read(mfe->name);
        require(mfe_bytes.size() == 232 &&
                    std::to_integer<char>(mfe_bytes[0]) == 'M' &&
                    std::to_integer<char>(mfe_bytes[3]) == '1',
                "virtual MFE payload is invalid");

        mfq::HfModelSource fp8_128_source(root / "hf-fp8-128");
        const auto* fp8_128 = fp8_128_source.find_tensor("block.weight");
        require(fp8_128 != nullptr && fp8_128->dtype == "FP8-128SQ" &&
                    fp8_128->stored_dtype == "FP8-128SQ" &&
                    fp8_128->nbytes == 16734,
                "native block FP8 was not exposed as lossless FP8-128SQ");
        require(fp8_128_source.find_tensor("block.weight_scale_inv") == nullptr,
                "consumed block FP8 scale leaked into tensor enumeration");
        const auto fp8_128_bytes = fp8_128_source.read(fp8_128->name);
        const auto fp8_128_layout = mfq::fp8sq::parse(
            fp8_128->dtype,
            reinterpret_cast<const std::uint8_t*>(fp8_128_bytes.data()),
            fp8_128_bytes.size());
        const auto fp8_128_rows = mfq::fp8sq::row_metadata(
            reinterpret_cast<const std::uint8_t*>(fp8_128_bytes.data()),
            fp8_128_layout);
        require(fp8_128_layout.outputs == 128 &&
                    fp8_128_layout.width == 128 &&
                    fp8_128_layout.block_rows == 128 &&
                    fp8_128_layout.block_columns == 128 &&
                    fp8_128_layout.scale_kind == mfq::fp8sq::ScaleKind::Bf16 &&
                    std::all_of(
                        fp8_128_rows.q.begin(), fp8_128_rows.q.end(),
                        [](std::uint8_t q) { return q == 8; }) &&
                    std::to_integer<unsigned char>(
                        fp8_128_bytes[fp8_128_layout.symbols]) == 0x12 &&
                    std::to_integer<unsigned char>(
                        fp8_128_bytes[fp8_128_layout.scales - 1]) == 0x34 &&
                    std::to_integer<unsigned char>(
                        fp8_128_bytes[fp8_128_layout.scales]) == 0x80 &&
                    std::to_integer<unsigned char>(
                        fp8_128_bytes[fp8_128_layout.scales + 1]) == 0x3f,
                "lossless FP8-128SQ payload differs from HF source bytes");

        mfq::HfModelSource fp8_128_explicit_source(
            root / "hf-fp8-128-explicit");
        const auto* fp8_128_explicit = fp8_128_explicit_source.find_tensor(
            "model.block.0.linear_attention.gate.weight");
        require(fp8_128_explicit != nullptr &&
                    fp8_128_explicit->dtype == "FP8-128SQ" &&
                    fp8_128_explicit->stored_dtype == "FP8-128SQ" &&
                    fp8_128_explicit->nbytes == 16734,
                "explicit HF source map did not bind block FP8 scale");
        require(fp8_128_explicit_source.find_tensor(
                    "model.block.0.linear_attention.gate.weight_scale") == nullptr,
                "explicit block FP8 scale leaked into tensor enumeration");

        mfq::HfModelSource mxfp8_source(root / "hf-mxfp8");
        for (const auto& [name, nbytes] :
             std::array<std::pair<const char*, std::uint64_t>, 3>{{
                 {"block128.weight", 16441},
                 {"block32.weight", 2172},
                 {"row32.weight", 254},
             }}) {
            const auto* tensor = mxfp8_source.find_tensor(name);
            require(tensor != nullptr && tensor->dtype == "MXFP8" &&
                        tensor->nbytes == nbytes,
                    "valid native MXFP8 geometry was rejected");
        }

        bool rejected = false;
        try {
            static_cast<void>(hf_source.read(
                "model.token_embedding.weight", 13, 2));
        } catch (const std::out_of_range&) {
            rejected = true;
        }
        require(rejected, "out-of-range tensor read was accepted");
        } // Close all source file handles before Windows fixture cleanup.

        std::filesystem::remove_all(root);
        std::cout << "model source parity passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove_all(root);
        std::cerr << "model source parity failed: " << error.what() << '\n';
        return 1;
    }
}
