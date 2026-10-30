#pragma once
#include "mfq/model_source.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>

namespace mfq {
struct TqRowSelection {
    std::vector<std::int64_t> selected;
    std::vector<std::uint32_t> inverse;
};

// Preserve first-occurrence order without allocating a hash node per row.
inline TqRowSelection select_tq_rows(const std::vector<std::int64_t>& ids,
        std::int64_t rows_per_shard,std::size_t shards) {
    if(rows_per_shard<1 || !shards || ids.size()>std::uint64_t(INT32_MAX))
        throw std::invalid_argument("invalid TQ row selection geometry");
    TqRowSelection result;result.selected.reserve(ids.size());result.inverse.reserve(ids.size());
    std::size_t capacity=1;while(capacity<ids.size()*2)capacity*=2;
    constexpr auto empty=std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> slots(capacity,empty);
    for(const auto id:ids) {
        if(id<0 || std::uint64_t(id/rows_per_shard)>=shards)
            throw std::out_of_range("TQ stage shard ID is out of bounds");
        auto key=std::uint64_t(id);key^=key>>33;key*=0xff51afd7ed558ccdULL;key^=key>>33;
        auto at=std::size_t(key)&(capacity-1);
        while(slots[at]!=empty && result.selected[slots[at]]!=id)at=(at+1)&(capacity-1);
        if(slots[at]==empty){slots[at]=std::uint32_t(result.selected.size());result.selected.push_back(id);}
        result.inverse.push_back(slots[at]);
    }
    return result;
}

struct TqLayout {
    std::uint32_t rows=0,width=0,gs=0,packed_bytes=0;
    bool columns=false,compact=false;
    std::size_t groups=0,scales=0,packed=0,scale_bytes=0,bytes=0;
    static std::uint64_t integer(const std::uint8_t* p,int bytes) {
        std::uint64_t value=0;
        for(int i=0;i<bytes;++i)value|=std::uint64_t(p[i])<<(8*i);
        return value;
    }
    static void integer(std::uint8_t* p,std::uint64_t value,int bytes) {
        for(int i=0;i<bytes;++i)p[i]=std::uint8_t(value>>(8*i));
    }
    std::uint32_t offset(std::uint64_t descriptor)const noexcept {
        return std::uint32_t(descriptor)&(compact?0x7ffffffeU:0xffffffffU);
    }
    static std::uint32_t tail_seed(std::uint64_t d)noexcept {
        return ((d>>35)&31)|(((d>>56)&255)<<5)|((d&1)<<13)|(((d>>31)&1)<<14);
    }
    bool valid_descriptor(std::uint64_t d)const noexcept {
        const auto bits=((d>>32)&7)+1;
        if(((d>>40)&0xffff)>0x7bff)return false;
        if(!compact)return !(d&~0x00ffff07ffffffffULL);
        return !(std::uint64_t(width)*bits/32&1) && tail_seed(d)<(1U<<(16-bits));
    }
    std::uint64_t rebase(std::uint64_t d,std::uint64_t words)const {
        const std::uint64_t mask=compact?0x7ffffffeULL:0xffffffffULL;
        if(words&~mask)throw std::overflow_error("TQ row word offset exceeds format capacity");
        return (d&~mask)|words;
    }
    TqLayout()=default;
    TqLayout(const std::uint8_t* p,std::size_t size) {
        if(size<64)throw std::invalid_argument("truncated TQ header");
        compact=!std::memcmp(p,"RTRELL06",8) && integer(p+8,4)==6;
        if((!compact && (std::memcmp(p,"RTRELL05",8) || integer(p+8,4)!=5)) || integer(p+20,4)>1)
            throw std::invalid_argument("invalid TQ header");
        rows=std::uint32_t(integer(p+12,4));width=std::uint32_t(integer(p+16,4));
        columns=integer(p+20,4)!=0;gs=std::uint32_t(integer(p+24,4));
        packed_bytes=std::uint32_t(integer(p+28,4));
        if(!rows || !width || width%32 || (compact?(!columns || width%256==0):width%256!=0) ||
           (!columns && (gs<16 || gs>(1u<<20) || (gs&(gs-1)))))
            throw std::invalid_argument("invalid TQ row geometry");
        for(int i=32;i<64;++i)if(p[i])throw std::invalid_argument("nonzero TQ reserved coefficients");
        groups=columns?0:(std::size_t(width)+gs-1)/gs;
        scales=64+std::size_t(rows)*8;
        scale_bytes=columns?std::size_t(width)*2:(std::size_t(rows)*groups+1)/2;
        packed=scales+scale_bytes;bytes=packed+packed_bytes;
        if(size!=bytes)throw std::invalid_argument("TQ tensor byte count differs");
    }
};

// Retained bounded reads; no expanded weights and no expert/cache ownership.
class TqRows {
    ModelSource::TensorReader read_;
    TqLayout layout_;
    std::array<std::uint8_t,64> header_{};
public:
    TqRows(std::size_t bytes,ModelSource::TensorReader read):read_(std::move(read)) {
        if(bytes<64)throw std::invalid_argument("truncated TQ header");
        read_(0,reinterpret_cast<std::byte*>(header_.data()),header_.size());
        layout_=TqLayout(header_.data(),bytes);
    }
    const TqLayout& layout()const noexcept{return layout_;}
    // Write one ordinary TQ row into caller-owned staging storage.
    // This avoids temporary descriptor/payload vectors for sparse CPU tables.
    std::size_t slice_row_into(std::uint64_t row,std::uint8_t* target,std::size_t capacity)const {
        const auto& m=layout_;
        if(row>=m.rows || !target)throw std::out_of_range("invalid TQ table row");
        std::array<std::uint8_t,8> record{};
        read_(64+row*8,reinterpret_cast<std::byte*>(record.data()),record.size());
        const auto descriptor=TqLayout::integer(record.data(),8);
        const auto first=m.offset(descriptor);
        const auto bits=((descriptor>>32)&7)+1;
        const auto packed_bytes=std::size_t(m.width)*bits/8;
        if(!m.valid_descriptor(descriptor) ||
           std::uint64_t(first)*4+packed_bytes>m.packed_bytes ||
           (!row && first) || (row+1==m.rows && std::uint64_t(first)*4+packed_bytes!=m.packed_bytes))
            throw std::invalid_argument("invalid TQ row descriptor or payload range");
        const auto scale_bytes=m.columns?m.scale_bytes:(m.groups+1)/2;
        const auto bytes=72+scale_bytes+packed_bytes;
        if(capacity<bytes)throw std::overflow_error("TQ row staging capacity is too small");
        std::memcpy(target,header_.data(),64);
        TqLayout::integer(target+12,1,4);TqLayout::integer(target+28,packed_bytes,4);
        TqLayout::integer(target+64,m.rebase(descriptor,0),8);
        auto* scales=target+72;
        if(m.columns)read_(m.scales,reinterpret_cast<std::byte*>(scales),scale_bytes);
        else if(m.groups==1) {
            std::uint8_t selector=0;
            read_(m.scales+row/2,reinterpret_cast<std::byte*>(&selector),1);
            scales[0]=(selector>>((row&1)*4))&15;
        } else {
            const auto first_selector=std::size_t(row)*m.groups;
            const auto start=first_selector/2,end=(first_selector+m.groups+1)/2;
            std::vector<std::uint8_t> selectors(end-start);
            read_(m.scales+start,reinterpret_cast<std::byte*>(selectors.data()),selectors.size());
            std::memset(scales,0,scale_bytes);
            for(std::size_t i=0;i<m.groups;++i) {
                const auto at=i+(first_selector&1);
                scales[i/2]|=((selectors[at/2]>>((at&1)*4))&15)<<((i&1)*4);
            }
        }
        read_(m.packed+std::uint64_t(first)*4,reinterpret_cast<std::byte*>(scales+scale_bytes),packed_bytes);
        return bytes;
    }
    std::vector<std::uint8_t> slice_rows(std::uint64_t begin,std::uint64_t end)const {
        const auto& m=layout_;
        if(begin>=end || end>m.rows)throw std::out_of_range("invalid TQ row range");
        const auto count=std::size_t(end-begin);
        std::vector<std::uint8_t> descriptors(count*8);
        read_(64+begin*8,reinterpret_cast<std::byte*>(descriptors.data()),descriptors.size());
        const auto first=m.offset(TqLayout::integer(descriptors.data(),8));
        auto expected=first;
        for(std::size_t row=0;row<count;++row) {
            auto* record=descriptors.data()+row*8;
            const auto descriptor=TqLayout::integer(record,8);
            const auto bits=((descriptor>>32)&7)+1;
            if(!m.valid_descriptor(descriptor) || m.offset(descriptor)!=expected)
                throw std::invalid_argument("invalid TQ row descriptor");
            TqLayout::integer(record,m.rebase(descriptor,expected-first),8);
            expected+=std::uint64_t(m.width)*bits/32;
        }
        if(first*4>m.packed_bytes || expected*4>m.packed_bytes ||
           (begin==0 && first!=0) || (end==m.rows && expected*4!=m.packed_bytes))
            throw std::invalid_argument("TQ row payload range differs");
        const auto packed_bytes=std::size_t(expected-first)*4;
        const auto scale_bytes=m.columns?m.scale_bytes:(count*m.groups+1)/2;
        std::vector<std::uint8_t> result(64+count*8+scale_bytes+packed_bytes);
        std::memcpy(result.data(),header_.data(),64);
        TqLayout::integer(result.data()+12,count,4);
        TqLayout::integer(result.data()+28,packed_bytes,4);
        std::memcpy(result.data()+64,descriptors.data(),descriptors.size());
        auto* scales=result.data()+64+descriptors.size();
        if(m.columns)read_(m.scales,reinterpret_cast<std::byte*>(scales),scale_bytes);
        else {
            const auto first_selector=std::size_t(begin)*m.groups,selectors=count*m.groups;
            const auto start_byte=first_selector/2,end_byte=(first_selector+selectors+1)/2;
            std::vector<std::uint8_t> source(end_byte-start_byte);
            read_(m.scales+start_byte,reinterpret_cast<std::byte*>(source.data()),source.size());
            for(std::size_t i=0;i<selectors;++i) {
                const auto index=i+(first_selector&1);
                scales[i/2]|=((source[index/2]>>((index&1)*4))&15)<<((i&1)*4);
            }
        }
        read_(m.packed+first*4,reinterpret_cast<std::byte*>(scales+scale_bytes),packed_bytes);
        return result;
    }
    std::vector<std::uint8_t> slice_columns(std::uint64_t begin,std::uint64_t end)const {
        const auto& m=layout_;
        if(begin>=end || end>m.width || begin%256 || end%256)
            throw std::out_of_range("TQ column slices require complete encoded rings");
        const auto width=std::uint32_t(end-begin);
        const auto gs=m.columns?m.gs:std::min(m.gs,256u);
        const auto groups=m.columns?0:(std::size_t(width)+gs-1)/gs;
        const auto scale_bytes=m.columns?std::size_t(width)*2:(std::size_t(m.rows)*groups+1)/2;
        std::vector<std::uint8_t> descriptors(std::size_t(m.rows)*8),scales(m.scale_bytes);
        read_(64,reinterpret_cast<std::byte*>(descriptors.data()),descriptors.size());
        read_(m.scales,reinterpret_cast<std::byte*>(scales.data()),scales.size());
        std::uint64_t expected=0,packed_bytes=0;
        for(std::size_t row=0;row<m.rows;++row) {
            const auto descriptor=TqLayout::integer(descriptors.data()+row*8,8);
            const auto bits=((descriptor>>32)&7)+1;
            if((descriptor & ~0x00ffff07ffffffffULL) || ((descriptor>>40)&0xffff)>0x7bff ||
               std::uint32_t(descriptor)!=expected)
                throw std::invalid_argument("invalid TQ row descriptor");
            expected+=std::uint64_t(m.width)*bits/32;
            packed_bytes+=std::uint64_t(width)*bits/8;
        }
        if(expected*4!=m.packed_bytes)throw std::invalid_argument("TQ row payload range differs");
        std::vector<std::uint8_t> result(64+descriptors.size()+scale_bytes+packed_bytes);
        std::memcpy(result.data(),header_.data(),64);
        TqLayout::integer(result.data()+16,width,4);
        TqLayout::integer(result.data()+24,gs,4);
        TqLayout::integer(result.data()+28,packed_bytes,4);
        auto* target_scales=result.data()+64+descriptors.size();
        auto* target_words=target_scales+scale_bytes;
        if(m.columns)std::memcpy(target_scales,scales.data()+begin*2,scale_bytes);
        std::size_t cursor=0;
        for(std::size_t row=0;row<m.rows;++row) {
            const auto descriptor=TqLayout::integer(descriptors.data()+row*8,8);
            const auto bits=((descriptor>>32)&7)+1;
            TqLayout::integer(result.data()+64+row*8,(descriptor & 0xffffffff00000000ULL)|(cursor/4),8);
            if(!m.columns)for(std::size_t group=0;group<groups;++group) {
                const auto source=row*m.groups+(begin+group*gs)/m.gs;
                const auto target=row*groups+group;
                target_scales[target/2]|=((scales[source/2]>>((source&1)*4))&15)<<((target&1)*4);
            }
            const auto bytes=std::size_t(width)*bits/8;
            read_(m.packed+std::uint32_t(descriptor)*4+begin*bits/8,
                reinterpret_cast<std::byte*>(target_words+cursor),bytes);
            cursor+=bytes;
        }
        return result;
    }
};

inline std::vector<std::shared_ptr<TqRows>> tq_row_segments(
        std::size_t bytes, ModelSource::TensorReader read, std::uint64_t logical_width) {
    if (bytes < 64) throw std::invalid_argument("truncated TQ tensor header");
    std::array<std::uint8_t, 64> header{};
    read(0, reinterpret_cast<std::byte*>(header.data()), header.size());
    if (std::memcmp(header.data(), "RTQEXP01", 8)) {
        auto rows = std::make_shared<TqRows>(bytes, std::move(read));
        if (!logical_width || logical_width > rows->layout().width)
            throw std::invalid_argument("TQ logical width differs");
        return {std::move(rows)};
    }
    const auto count = TqLayout::integer(header.data() + 12, 4);
    const auto outputs = TqLayout::integer(header.data() + 16, 4);
    const auto width = TqLayout::integer(header.data() + 20, 4);
    if (TqLayout::integer(header.data() + 8, 4) != 1 || !count || !outputs || !width
            || width != logical_width || count > (bytes - 64) / 16
            || count > 1048576 || TqLayout::integer(header.data() + 24, 4) != 16
            || TqLayout::integer(header.data() + 28, 4)
            || std::any_of(header.begin() + 32, header.end(), [](auto b) { return b != 0; }))
        throw std::invalid_argument("invalid TQ expert container header");
    std::vector<std::uint8_t> directory(std::size_t(count) * 16);
    read(64, reinterpret_cast<std::byte*>(directory.data()), directory.size());
    std::vector<std::shared_ptr<TqRows>> result;
    result.reserve(count);
    std::uint64_t end = 64 + directory.size();
    for (std::size_t index = 0; index < count; ++index) {
        const auto offset = TqLayout::integer(directory.data() + index * 16, 8);
        const auto size = TqLayout::integer(directory.data() + index * 16 + 8, 8);
        if (offset != end || size < 64 || offset > bytes || size > bytes - offset)
            throw std::invalid_argument("invalid TQ expert container span");
        auto child_read = [read, offset, size](std::uint64_t begin, std::byte* destination, std::size_t length) {
            if (begin > size || length > size - begin)
                throw std::out_of_range("TQ expert read exceeds its payload");
            read(offset + begin, destination, length);
        };
        auto child = std::make_shared<TqRows>(size, std::move(child_read));
        if (child->layout().rows != outputs || logical_width > child->layout().width
                || (index && child->layout().width != result.front()->layout().width))
            throw std::invalid_argument("TQ expert container dimensions differ");
        result.push_back(std::move(child));
        end = offset + size;
    }
    if (end != bytes) throw std::invalid_argument("TQ expert container length differs");
    return result;
}

inline std::vector<std::int64_t> tq_tensor_shape(std::size_t bytes,
        const ModelSource::TensorReader& read, std::vector<std::int64_t> logical = {}) {
    if (bytes < 64) throw std::invalid_argument("truncated TQ tensor header");
    std::array<std::uint8_t,64> header{};
    read(0, reinterpret_cast<std::byte*>(header.data()), header.size());
    const bool expert_container = !std::memcmp(header.data(), "RTQEXP01", 8);
    if (logical.empty()) {
        if (expert_container) logical = {
            std::int64_t(TqLayout::integer(header.data()+12,4)),
            std::int64_t(TqLayout::integer(header.data()+16,4)),
            std::int64_t(TqLayout::integer(header.data()+20,4))};
        else logical = {std::int64_t(TqLayout::integer(header.data()+12,4)),
            std::int64_t(TqLayout::integer(header.data()+16,4))};
    }
    if ((logical.size()!=2 && logical.size()!=3) ||
        std::any_of(logical.begin(),logical.end(),[](auto dim){return dim<=0 || dim>INT32_MAX;}))
        throw std::invalid_argument("invalid TQ logical shape");
    auto segments=tq_row_segments(bytes,read,std::uint64_t(logical.back()));
    const auto rows=std::uint64_t(logical[logical.size()-2]);
    const auto experts=logical.size()==3?std::uint64_t(logical.front()):1;
    if ((expert_container && (segments.size()!=experts || segments.front()->layout().rows!=rows)) ||
        (!expert_container && segments.front()->layout().rows!=experts*rows))
        throw std::invalid_argument("TQ logical rows differ");
    return logical;
}
} // namespace mfq
