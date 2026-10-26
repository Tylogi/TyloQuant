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
    bool columns=false;
    std::size_t groups=0,scales=0,packed=0,scale_bytes=0,bytes=0;
    static std::uint64_t integer(const std::uint8_t* p,int bytes) {
        std::uint64_t value=0;
        for(int i=0;i<bytes;++i)value|=std::uint64_t(p[i])<<(8*i);
        return value;
    }
    static void integer(std::uint8_t* p,std::uint64_t value,int bytes) {
        for(int i=0;i<bytes;++i)p[i]=std::uint8_t(value>>(8*i));
    }
    TqLayout()=default;
    TqLayout(const std::uint8_t* p,std::size_t size) {
        if(size<64 || std::memcmp(p,"RTRELL05",8) || integer(p+8,4)!=5 || integer(p+20,4)>1)
            throw std::invalid_argument("invalid RTRELL05 header");
        rows=std::uint32_t(integer(p+12,4));width=std::uint32_t(integer(p+16,4));
        columns=integer(p+20,4)!=0;gs=std::uint32_t(integer(p+24,4));
        packed_bytes=std::uint32_t(integer(p+28,4));
        if(!rows || !width || width%256 ||
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
    // Write one ordinary RTRELL05 row into caller-owned staging storage.
    // This avoids temporary descriptor/payload vectors for sparse CPU tables.
    std::size_t slice_row_into(std::uint64_t row,std::uint8_t* target,std::size_t capacity)const {
        const auto& m=layout_;
        if(row>=m.rows || !target)throw std::out_of_range("invalid TQ table row");
        std::array<std::uint8_t,8> record{};
        read_(64+row*8,reinterpret_cast<std::byte*>(record.data()),record.size());
        const auto descriptor=TqLayout::integer(record.data(),8);
        const auto first=std::uint32_t(descriptor);
        const auto bits=((descriptor>>32)&7)+1;
        const auto packed_bytes=std::size_t(m.width)*bits/8;
        if((descriptor & ~0x00ffff07ffffffffULL) || ((descriptor>>40)&0xffff)>0x7bff ||
           std::uint64_t(first)*4+packed_bytes>m.packed_bytes ||
           (!row && first) || (row+1==m.rows && std::uint64_t(first)*4+packed_bytes!=m.packed_bytes))
            throw std::invalid_argument("invalid TQ row descriptor or payload range");
        const auto scale_bytes=m.columns?m.scale_bytes:(m.groups+1)/2;
        const auto bytes=72+scale_bytes+packed_bytes;
        if(capacity<bytes)throw std::overflow_error("TQ row staging capacity is too small");
        std::memcpy(target,header_.data(),64);
        TqLayout::integer(target+12,1,4);TqLayout::integer(target+28,packed_bytes,4);
        TqLayout::integer(target+64,descriptor&0xffffffff00000000ULL,8);
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
        const auto first=TqLayout::integer(descriptors.data(),4);
        auto expected=first;
        for(std::size_t row=0;row<count;++row) {
            auto* record=descriptors.data()+row*8;
            const auto descriptor=TqLayout::integer(record,8);
            const auto bits=((descriptor>>32)&7)+1;
            const auto anchor=(descriptor>>40)&0xffff;
            if((descriptor & ~0x00ffff07ffffffffULL) || anchor>0x7bff ||
               TqLayout::integer(record,4)!=expected)
                throw std::invalid_argument("invalid TQ row descriptor");
            TqLayout::integer(record,expected-first,4);
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
};
} // namespace mfq
