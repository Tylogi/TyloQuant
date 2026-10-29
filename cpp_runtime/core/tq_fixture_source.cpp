#include "mfq/tq_fixture_source.h"
#include "mfq/mfq_model_source.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <numeric>
#include <mutex>
#include <limits>
#include <unordered_set>

namespace mfq {
namespace {
using json=nlohmann::json;
constexpr std::uint64_t gamma=0x9e3779b97f4a7c15ULL;
std::uint64_t mix(std::uint64_t value) {
    value=(value^(value>>30))*0xbf58476d1ce4e5b9ULL;
    value=(value^(value>>27))*0x94d049bb133111ebULL;
    return value^(value>>31);
}
void little(std::byte* output,std::uint64_t value,std::size_t bytes) {
    for(std::size_t i=0;i<bytes;++i)output[i]=std::byte(value>>(8*i));
}
struct Segment {
    struct Piece {std::shared_ptr<FileRangeReader> file;std::uint64_t offset,size;};
    std::vector<Piece> pieces;
    std::unordered_map<std::string,std::array<int,8>> histograms;
    std::uint64_t rows=0,width=0,gs=0,bit_sum=0,seed=0,anchor=0,size=0;
    std::uint64_t groups=0,scale_offset=0,packed_offset=0;
    std::array<std::byte,64> header{};
    // Bounded synthetic metadata only. Encoded weight bytes still come from
    // the source files; one-row reads must not reshuffle 4096 row precisions.
    mutable std::mutex block_mutex;
    mutable std::uint64_t cached_block=std::numeric_limits<std::uint64_t>::max();
    mutable std::shared_ptr<const std::vector<std::uint64_t>> cached_descriptors;
    Segment(const json& spec,const std::vector<std::shared_ptr<FileRangeReader>>& files) {
        rows=spec.at("rows");width=spec.at("width");gs=spec.at("gs");
        bit_sum=spec.at("bit_sum");seed=spec.at("seed");anchor=spec.at("anchor_bits");size=spec.at("nbytes");
        if(!rows || rows>UINT32_MAX || !width || width>UINT32_MAX || width%256 ||
           gs<16 || gs>(1u<<20) || (gs&(gs-1)) || bit_sum<rows || bit_sum>8*rows ||
           width*bit_sum/8>UINT32_MAX || anchor>0x7bff)
            throw std::invalid_argument("invalid synthetic TQ segment geometry");
        groups=(width+gs-1)/gs;scale_offset=64+rows*8;
        packed_offset=scale_offset+(rows*groups+1)/2;
        if(size!=packed_offset+width*bit_sum/8)throw std::invalid_argument("synthetic TQ segment size differs");
        std::memcpy(header.data(),"RTRELL05",8);
        const std::array<std::uint64_t,6> fields{5,rows,width,0,gs,width*bit_sum/8};
        for(std::size_t i=0;i<fields.size();++i)little(header.data()+8+i*4,fields[i],4);
        for(const auto& item:spec.at("histograms").items()) {
            const auto counts=item.value().get<std::array<int,8>>();
            const auto colon=item.key().find(':');
            if(colon==std::string::npos)throw std::invalid_argument("invalid TQ histogram key");
            std::uint64_t count=0,bits=0;
            for(int i=0;i<8;++i) {
                if(counts[i]<0)throw std::invalid_argument("negative TQ histogram count");
                count+=counts[i];bits+=std::uint64_t(counts[i])*(i+1);
            }
            if(count!=std::stoull(item.key().substr(0,colon)) || count>4096 ||
               bits!=std::stoull(item.key().substr(colon+1)))
                throw std::invalid_argument("TQ histogram budget differs");
            histograms.emplace(item.key(),counts);
        }
        std::uint64_t packed_size=0;
        for(const auto& p:spec.at("packed")) {
            const auto index=p.at("file").get<std::size_t>();
            if(index>=files.size())throw std::invalid_argument("TQ backing file index out of bounds");
            pieces.push_back({files[index],p.at("offset"),p.at("nbytes")});
            packed_size+=pieces.back().size;
        }
        if(packed_size!=width*bit_sum/8)throw std::invalid_argument("TQ backing stream size differs");
    }
    std::vector<std::uint8_t> block_bits(std::uint64_t block) const {
        const auto start=block*4096,count=std::min<std::uint64_t>(4096,rows-start);
        const auto budget=bit_sum*(start+count)/rows-bit_sum*start/rows;
        const auto& counts=histograms.at(std::to_string(count)+":"+std::to_string(budget));
        std::vector<std::uint8_t> result;result.reserve(count);
        for(int i=0;i<8;++i)result.insert(result.end(),counts[i],std::uint8_t(i+1));
        std::uint64_t state=seed^block;
        for(std::size_t i=result.size()-1;i>0;--i) {
            state+=gamma;std::swap(result[i],result[mix(state)%(i+1)]);
        }
        return result;
    }
    std::shared_ptr<const std::vector<std::uint64_t>> block_descriptors(std::uint64_t block)const {
        std::lock_guard<std::mutex> guard(block_mutex);
        if(cached_descriptors && cached_block==block)return cached_descriptors;
        const auto bits=block_bits(block);
        auto records=std::make_shared<std::vector<std::uint64_t>>();records->reserve(bits.size());
        auto prefix=bit_sum*(block*4096)/rows;
        for(const auto precision:bits) {
            records->push_back(prefix*width/32 | (std::uint64_t(precision-1)<<32) | (anchor<<40));
            prefix+=precision;
        }
        cached_block=block;cached_descriptors=records;return records;
    }
    void read(std::uint64_t offset,std::byte* output,std::size_t bytes) const {
        if(offset>size || bytes>size-offset || (bytes && !output))throw std::out_of_range("synthetic TQ read is out of bounds");
        const auto end=offset+bytes;
        auto section=[&](std::uint64_t begin,std::uint64_t finish,auto emit) {
            const auto first=std::max(offset,begin),last=std::min(end,finish);
            if(first<last)emit(first-begin,last-begin,output+(first-offset));
        };
        section(0,64,[&](auto first,auto last,auto* out){std::memcpy(out,header.data()+first,last-first);});
        section(64,scale_offset,[&](auto first,auto last,auto* out) {
            const auto row_begin=first/8,row_end=(last+7)/8;
            for(auto block=row_begin/4096;block<(row_end+4095)/4096;++block) {
                const auto records=block_descriptors(block);
                const auto begin=std::max(row_begin,block*4096)-block*4096;
                const auto end=std::min(row_end,block*4096+records->size())-block*4096;
                for(auto i=begin;i<end;++i) {
                    const auto row=block*4096+i;
                    if(row>=row_begin && row<row_end) {
                        const auto value=(*records)[i];
                        std::array<std::byte,8> encoded{};little(encoded.data(),value,8);
                        const auto a=std::max(first,row*8),b=std::min(last,(row+1)*8);
                        std::memcpy(out+a-first,encoded.data()+a-row*8,b-a);
                    }
                }
            }
        });
        section(scale_offset,packed_offset,[&](auto first,auto last,auto* out) {
            for(auto i=first;i<last;++i) {
                auto value=std::uint8_t(mix(seed+(i+1)*gamma));
                if((rows*groups&1) && i+1==packed_offset-scale_offset)value&=15;
                out[i-first]=std::byte(value);
            }
        });
        section(packed_offset,size,[&](auto first,auto last,auto* out) {
            std::uint64_t start=0;
            for(const auto& p:pieces) {
                const auto a=std::max(first,start),b=std::min(last,start+p.size);
                if(a<b)p.file->read(p.offset+a-start,out+a-first,b-a);
                start+=p.size;
            }
        });
    }
};

class Fixture final:public ModelSource {
    std::shared_ptr<const ModelSource> base_;
    std::vector<TensorMetadata> tensors_;
    std::unordered_map<std::string,std::size_t> index_;
    std::unordered_map<std::string,TensorReader> readers_;
    std::unordered_map<std::string,std::string> metadata_;
public:
    explicit Fixture(const std::filesystem::path& path) {
        std::ifstream input(path);json spec;input>>spec;
        if(spec.at("format")!="mfq-tq-fixture-v1" || spec.at("synthetic")!=true)
            throw std::invalid_argument("expected an explicit synthetic TQ fixture");
        base_=std::make_shared<MfqModelSource>(std::filesystem::u8path(spec.at("reference").get<std::string>()));
        tensors_=base_->tensors();metadata_=base_->metadata();metadata_["tq.synthetic"]="true";
        // Synthetic conversion can replace a legacy combined Gate/Up bank
        // with two ordinary tensors, while retaining all unchanged assets.
        const auto removed=spec.value("remove_tensors",std::vector<std::string>{});
        const std::unordered_set<std::string> removed_names(removed.begin(),removed.end());
        tensors_.erase(std::remove_if(tensors_.begin(),tensors_.end(),[&](const auto& t) {
            return removed_names.count(t.name)!=0;
        }),tensors_.end());
        std::unordered_set<std::string> names;
        for(const auto& t:tensors_)names.insert(t.name);
        for(const auto& item:spec.at("tensors").items()) {
            if(removed_names.count(item.key()))throw std::invalid_argument("removed TQ tensor was also defined");
            if(names.insert(item.key()).second) {
                TensorMetadata tensor;tensor.name=item.key();tensors_.push_back(std::move(tensor));
            }
        }
        std::vector<std::shared_ptr<FileRangeReader>> files;
        for(const auto& file:spec.at("source_files"))
            files.push_back(std::make_shared<FileRangeReader>(std::filesystem::u8path(file.get<std::string>()),model_file_read_mode()));
        json segmented=json::object();
        for(auto& tensor:tensors_) {
            const auto item=spec.at("tensors").find(tensor.name);
            if(item==spec.at("tensors").end())continue;
            tensor.dtype=tensor.stored_dtype="TQ";
            tensor.shape=item->at("shape").get<std::vector<std::int64_t>>();
            tensor.nbytes=item->at("nbytes");
            std::vector<std::shared_ptr<Segment>> segments;
            for(const auto& segment:item->at("segments"))segments.push_back(std::make_shared<Segment>(segment,files));
            auto reader=[segments,total=tensor.nbytes](std::uint64_t offset,std::byte* out,std::size_t bytes) {
                if(offset>total || bytes>total-offset)throw std::out_of_range("synthetic TQ tensor read is out of bounds");
                std::uint64_t start=0,end=offset+bytes;
                for(const auto& segment:segments) {
                    const auto a=std::max(offset,start),b=std::min(end,start+segment->size);
                    if(a<b)segment->read(a-start,out+a-offset,b-a);
                    start+=segment->size;
                }
            };
            readers_.emplace(tensor.name,std::move(reader));
            if(segments.size()>1) {
                auto names=json::array();
                for(std::size_t i=0;i<segments.size();++i) {
                    auto name=tensor.name+".__tq_rows__."+std::to_string(i);
                    names.push_back(name);
                    auto segment=segments[i];
                    readers_.emplace(name,[segment](auto offset,auto* out,auto bytes){segment->read(offset,out,bytes);});
                }
                segmented[tensor.name]=std::move(names);
            }
        }
        // Append segment metadata after the loop: vector growth must not
        // invalidate the references used to replace the original records.
        for(const auto& item:segmented.items()) {
            const auto& tensor=spec.at("tensors").at(item.key());
            const auto& segments=tensor.at("segments");
            for(std::size_t i=0;i<segments.size();++i) {
                TensorMetadata metadata;
                metadata.name=item.value()[i].get<std::string>();metadata.dtype=metadata.stored_dtype="TQ";
                metadata.shape=std::vector<std::int64_t>{segments[i].at("rows").get<std::int64_t>(),
                    tensor.at("shape").back().get<std::int64_t>()};
                metadata.nbytes=segments[i].at("nbytes");tensors_.push_back(std::move(metadata));
            }
        }
        metadata_["tq.segments"]=segmented.dump();
        for(std::size_t i=0;i<tensors_.size();++i)index_.emplace(tensors_[i].name,i);
    }
    const std::vector<std::filesystem::path>& source_paths()const noexcept override{return base_->source_paths();}
    std::string_view architecture()const noexcept override{return base_->architecture();}
    const std::unordered_map<std::string,std::string>& metadata()const noexcept override{return metadata_;}
    const std::vector<TensorMetadata>& tensors()const noexcept override{return tensors_;}
    const TensorMetadata* find_tensor(std::string_view name)const noexcept override {
        const auto item=index_.find(std::string(name));return item==index_.end()?nullptr:&tensors_[item->second];
    }
    TensorReader tensor_reader(std::string_view name)const override {
        const auto item=readers_.find(std::string(name));return item==readers_.end()?base_->tensor_reader(name):item->second;
    }
    void read_range_into(std::string_view name,std::uint64_t offset,std::byte* out,std::size_t size)const override {
        tensor_reader(name)(offset,out,size);
    }
    bool supports_parallel_tensor_reads()const noexcept override{return true;}
    const std::vector<std::string>& assets()const noexcept override{return base_->assets();}
    bool has_asset(std::string_view name)const noexcept override{return base_->has_asset(name);}
    std::vector<std::byte> read_asset(std::string_view name)const override{return base_->read_asset(name);}
    std::optional<ModelGraph> model_graph()const override{return base_->model_graph();}
};
}
std::shared_ptr<const ModelSource> open_tq_fixture_source(const std::filesystem::path& path) {
    return std::make_shared<Fixture>(path);
}
} // namespace mfq
