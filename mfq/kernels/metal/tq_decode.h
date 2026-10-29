inline half4 tq_book(uint4 state) {
    uint4 raw = state * 0x83dcd12du;
    uint4 pairs = (raw & 0x00ff00ffu) + ((raw >> 8) & 0x00ff00ffu);
    uint4 sums = (pairs & 65535u) + (pairs >> 16);
    return fma(as_type<half4>(ushort4(sums + 0x6400u)),
        half4(as_type<half>(ushort(0x1eeeu))),
        half4(as_type<half>(ushort(0xc931u))));
}

template <bool POOLED, typename Descriptors>
inline device const uint* tq_words(device const uint* words, Descriptors descriptors, uint rows, uint expert) {
    if constexpr (POOLED) return words + descriptors[rows + expert];
    else return words;
}

template <typename ScalePtr>
inline float tq_group_scale(ScalePtr scales, device const half* relative, uint group, float anchor) {
    uint code = (uint(scales[group >> 1]) >> ((group & 1u) * 4u)) & 15u;
    return anchor * float(relative[code]);
}

inline float tq_group_scale(device const float* scales, device const half*, uint group, float) {
    return scales[group];
}

inline float tq_group_scale(constant const float* scales, device const half*, uint group, float) {
    return scales[group];
}

template <typename S>
inline float4 tq_column_scale4(S scales, uint at) {
    return float4(scales[at], scales[at + 1u], scales[at + 2u], scales[at + 3u]);
}

template <typename S>
inline float4 tq_column_scale4(device const S* scales, uint at) {
    return float4(*(device const vec<S, 4>*)(scales + at));
}

template <typename S>
inline float4 tq_column_scale4(constant const S* scales, uint at) {
    return float4(*(constant const vec<S, 4>*)(scales + at));
}

template <uint COLUMNS, uint WIDTH, uint GS, typename ScalePtr>
inline half4 tq_decode4(device const uint* words, ulong descriptor,
        ScalePtr scales, device const half* relative, uint row, uint col) {
    uint bits = (uint(descriptor >> 32) & 7u) + 1u;
    uint ring_words = 8u * bits;
    uint pos = (252u - (col & 255u)) * bits;
    uint first = pos >> 5;
    device const uint* ring = words + uint(descriptor) + (col >> 8) * ring_words;
    uint a = ring[first];
    uint b = ring[first + 1u == ring_words ? 0u : first + 1u];
    uint c = ring[first + 2u >= ring_words ? first + 2u - ring_words : first + 2u];
    ulong low = ulong(a) | (ulong(b) << 32), high = ulong(b) | (ulong(c) << 32);
    uint phase = pos & 31u, start = phase + 2u * bits;
    uint pair0 = uint((start >= 32u ? high : low) >> (start & 31u));
    uint pair1 = uint(low >> phase);
    uint4 state = uint4(pair0 >> bits, pair0, pair1 >> bits, pair1) & 65535u;
    float anchor = float(as_type<half>(ushort(descriptor >> 40)));
    float4 factor;
    if constexpr (COLUMNS) {
        uint at = col + (COLUMNS == 2u ? row / GS * WIDTH : 0u);
        factor = tq_column_scale4(scales, at) * anchor;
    }
    else {
        uint group = row * ((WIDTH + GS - 1u) / GS) + col / GS;
        factor = float4(tq_group_scale(scales, relative, group, anchor));
    }
    return half4(float4(tq_book(state)) * factor);
}

template <uint K>
inline ulong tq_row_offset(uint row, uint col) {
    return ulong(row) * K + col;
}

template <uint K, typename T>
inline void tq_store4(device T* output, uint row, uint col, half4 value) {
    ulong offset = tq_row_offset<K>(row, col);
    if constexpr (K % 4u == 0u)
        *(device vec<T, 4>*)(output + offset) = vec<T, 4>(value);
    else for (uint j = 0u; j < 4u; ++j)
        if (col + j < K) output[offset + j] = T(value[j]);
}

struct TqInput8 {
    float4 first;
    float4 second;
};

template <uint K, typename T>
inline float4 tq_input4(device const T* x, uint col, uint part) {
    uint at = col + part * 4u;
    if constexpr (K % 8u == 0u) return float4(*(device const vec<T, 4>*)(x + at));
    float4 value = 0.f;
    if constexpr (K % 4u == 0u) {
        if (at + 4u <= K) return float4(*(device const vec<T, 4>*)(x + at));
    }
    for (uint j = 0; j < 4u; ++j) if (at + j < K) value[j] = float(x[at + j]);
    return value;
}

template <uint K>
inline float4 tq_input4(TqInput8 input, uint, uint part) {
    return part == 0u ? input.first : input.second;
}

template <typename T> struct TqRmsInput {
    device const T* values;
    float inverse;
};

template <uint K, typename T>
inline float4 tq_input4(TqRmsInput<T> input, uint col, uint part) {
    return tq_input4<K>(input.values, col, part) * input.inverse;
}

struct TqWindow {
    uint first, second, third, ring_words;
    uint3 shifts;
    uint bits;
    bool2 high;
};

inline TqWindow tq_window(uint bits, uint col) {
    uint words = 8u * bits;
    uint pos = (248u - (col & 255u)) * bits, first = pos >> 5, phase = pos & 31u;
    uint start0 = phase + 5u * bits, start1 = phase + 2u * bits;
    return TqWindow{first, first + 1u == words ? 0u : first + 1u,
        first + 2u >= words ? first + 2u - words : first + 2u, words,
        uint3(start0 & 31u, start1 & 31u, phase), bits, bool2(start0 >= 32u, start1 >= 32u)};
}

struct TqDecoded8 {
    half4 first, second;
};

template <uint COLUMNS, uint WIDTH, uint GS, typename ScalePtr>
inline TqDecoded8 tq_decode8(device const uint* words, ulong descriptor,
        ScalePtr scales, device const half* relative, uint row, uint col) {
    uint bits = (uint(descriptor >> 32) & 7u) + 1u;
    float anchor = float(as_type<half>(ushort(descriptor >> 40)));
    TqWindow plan = tq_window(bits, col);
    device const uint* ring = words + uint(descriptor) + (col >> 8u) * plan.ring_words;
    uint a = ring[plan.first], b = ring[plan.second], c = ring[plan.third];
    ulong low = ulong(a) | (ulong(b) << 32), high = ulong(b) | (ulong(c) << 32);
    uint first = uint((plan.high.x ? high : low) >> plan.shifts.x);
    uint second = uint((plan.high.y ? high : low) >> plan.shifts.y);
    uint third = uint(low >> plan.shifts.z);
    uint4 states0 = uint4(first >> (2u * bits), first >> bits, first, second >> (2u * bits)) & 65535u;
    uint4 states1 = uint4(second >> bits, second, third >> bits, third) & 65535u;
    float4 scale0, scale1;
    if constexpr (COLUMNS) {
        uint at = col + (COLUMNS == 2u ? row / GS * WIDTH : 0u);
        scale0 = anchor * tq_column_scale4(scales, at);
        scale1 = anchor * tq_column_scale4(scales, at + 4u);
    } else {
        uint group = row * ((WIDTH + GS - 1u) / GS) + col / GS;
        scale0 = scale1 = float4(tq_group_scale(scales, relative, group, anchor));
    }
    return {half4(float4(tq_book(states0)) * scale0), half4(float4(tq_book(states1)) * scale1)};
}

template <uint LANES>
inline TqWindow tq_projection_window(TqWindow plan, uint bits, uint col) {
    if constexpr (LANES == 32) return plan;
    else if constexpr (LANES == 16) {
        bool second_half = (col & 128u) != 0u;
        plan.first -= second_half ? plan.ring_words / 2u : 0u;
        plan.second = second_half ? plan.first + 1u : plan.second;
        plan.third = second_half ? plan.first + 2u : plan.third;
        return plan;
    } else return tq_window(bits, col);
}

template <uint COLUMNS, uint WIDTH, uint K, uint GS, typename ScalePtr, typename Input>
inline float4 tq_accum8(device const uint* ring,
        ScalePtr scales, device const half* relative, uint row, uint col,
        TqWindow plan, float anchor, Input input, float4 acc) {
    uint a = ring[plan.first], b = ring[plan.second], c = ring[plan.third];
    ulong low_window = ulong(a) | (ulong(b) << 32);
    ulong high_window = ulong(b) | (ulong(c) << 32);
    float scale = anchor;
    if constexpr (!COLUMNS) {
        uint group = row * ((WIDTH + GS - 1u) / GS) + col / GS;
        scale = tq_group_scale(scales, relative, group, anchor);
    }
    uint first = uint((plan.high.x ? high_window : low_window) >> plan.shifts.x);
    uint second = uint((plan.high.y ? high_window : low_window) >> plan.shifts.y);
    uint third = uint(low_window >> plan.shifts.z);
    uint4 states0 = uint4(first >> (2u * plan.bits), first >> plan.bits, first, second >> (2u * plan.bits)) & 65535u;
    uint4 states1 = uint4(second >> plan.bits, second, third >> plan.bits, third) & 65535u;
    #pragma unroll
    for (uint part = 0u; part < 2u; ++part) {
        uint4 state = part == 0u ? states0 : states1;
        uint at = col + part * 4u;
        float4 factor = scale;
        if constexpr (COLUMNS) {
            uint sc = at + (COLUMNS == 2u ? row / GS * WIDTH : 0u);
            factor = anchor * tq_column_scale4(scales, sc);
        }
        float4 weight = float4(half4(float4(tq_book(state)) * factor));
        acc = fma(weight, tq_input4<K>(input, col, part), acc);
    }
    return acc;
}

template <typename T, uint LANES, uint COLUMNS, uint WIDTH, uint K, uint GS, typename ScalePtr, typename Input>
inline float tq_dot(device const uint* words, ulong descriptor,
        ScalePtr scales, device const half* relative, uint row,
        Input x, uint lane, uint begin = 0u, uint end = K) {
    if (begin >= end) return 0.f;
    float4 acc = 0.f;
    uint bits = (uint(descriptor >> 32) & 7u) + 1u;
    float anchor = float(as_type<half>(ushort(descriptor >> 40)));
    TqWindow plan = tq_window(bits, lane * 8u);
    device const uint* row_words = words + uint(descriptor);
    device const uint* cursor = row_words + (begin >> 8u) * plan.ring_words;
    if constexpr (LANES == 16) {
        TqWindow second = tq_projection_window<16>(plan, bits, 128u);
        for (uint col = begin + lane * 8u; col < end; col += 256u) {
            acc = tq_accum8<COLUMNS, WIDTH, K, GS>(
                cursor, scales, relative, row, col, plan, anchor, x, acc);
            if constexpr (K % 256u == 0u)
                acc = tq_accum8<COLUMNS, WIDTH, K, GS>(
                    cursor, scales, relative, row, col + 128u, second, anchor, x, acc);
            else if (col + 128u < end)
                acc = tq_accum8<COLUMNS, WIDTH, K, GS>(
                    cursor, scales, relative, row, col + 128u, second, anchor, x, acc);
            cursor += plan.ring_words;
        }
    } else {
        for (uint col = begin + lane * 8u; col < end; col += LANES * 8u) {
            acc = tq_accum8<COLUMNS, WIDTH, K, GS>(
                LANES == 32 ? cursor : row_words + (col >> 8) * plan.ring_words, scales, relative, row, col,
                tq_projection_window<LANES>(plan, bits, col), anchor, x, acc);
            if constexpr (LANES == 32) cursor += plan.ring_words;
        }
    }
    return acc.x + acc.y + acc.z + acc.w;
}

template <typename T, uint LANES, uint CG, uint CU, uint WG, uint WU, uint K,
          uint GG, uint GU, typename SG, typename SU>
inline float2 tq_dot_pair(device const uint* words_g, ulong desc_g, SG scales_g,
        device const uint* words_u, ulong desc_u, SU scales_u, device const half* relative,
        uint row_g, uint row_u, device const T* x, uint lane) {
    if constexpr (LANES == 32 && CG == CU && WG == WU && GG == GU) {
        bool upper = lane >= 16u;
        float value = tq_dot<T, 16, CG, WG, K, GG>(upper ? words_u : words_g,
            upper ? desc_u : desc_g, upper ? scales_u : scales_g, relative,
            upper ? row_u : row_g, x, lane & 15u);
        return upper ? float2(0.f, value) : float2(value, 0.f);
    }
    float4 gate = 0.f, up = 0.f;
    uint bg = (uint(desc_g >> 32) & 7u) + 1u, bu = (uint(desc_u >> 32) & 7u) + 1u;
    float ag = float(as_type<half>(ushort(desc_g >> 40)));
    float au = float(as_type<half>(ushort(desc_u >> 40)));
    TqWindow plan_g = tq_window(bg, lane * 8u), plan_u = tq_window(bu, lane * 8u);
    device const uint* row_g_words = words_g + uint(desc_g);
    device const uint* row_u_words = words_u + uint(desc_u);
    device const uint* cursor_g = row_g_words;
    device const uint* cursor_u = row_u_words;
    for (uint col = lane * 8u; col < K; col += LANES * 8u) {
        TqInput8 input{tq_input4<K>(x, col, 0u), tq_input4<K>(x, col, 1u)};
        gate = tq_accum8<CG, WG, K, GG>(
            LANES == 32 ? cursor_g : row_g_words + (col >> 8) * plan_g.ring_words, scales_g, relative, row_g, col,
            tq_projection_window<LANES>(plan_g, bg, col), ag, input, gate);
        up = tq_accum8<CU, WU, K, GU>(
            LANES == 32 ? cursor_u : row_u_words + (col >> 8) * plan_u.ring_words, scales_u, relative, row_u, col,
            tq_projection_window<LANES>(plan_u, bu, col), au, input, up);
        if constexpr (LANES == 32) {
            cursor_g += plan_g.ring_words;
            cursor_u += plan_u.ring_words;
        }
    }
    return float2(gate.x + gate.y + gate.z + gate.w, up.x + up.y + up.z + up.w);
}

template <typename T, uint COLUMNS, uint WIDTH, uint K, uint GS, uint BM, uint LANES, bool COMPACT = false, typename ScalePtr, typename Sources>
inline void tq_dot_batch_accumulate(device const uint* words, ulong descriptor, ScalePtr scales,
        device const half* relative, uint row, device const T* x, Sources sources, uint lane,
        thread float4 (&acc)[BM],
        uint begin = 0u, uint end = K) {
    #pragma unroll
    for (uint m = 0u; m < BM; ++m) acc[m] = 0.f;
    for (uint col = begin + lane * 8u; col < end; col += LANES * 8u) {
        TqDecoded8 decoded;
        if constexpr (K >= 1024u) decoded = tq_decode8<COLUMNS, WIDTH, GS>(words, descriptor, scales, relative, row, col);
        else decoded = {tq_decode4<COLUMNS, WIDTH, GS>(words, descriptor, scales, relative, row, col),
            tq_decode4<COLUMNS, WIDTH, GS>(words, descriptor, scales, relative, row, col + 4u)};
        float4 first = float4(decoded.first), second = float4(decoded.second);
        #pragma unroll
        for (uint m = 0u; m < BM; ++m) {
            if (sources[m] >= 0) {
                device const T* input = x + uint(sources[m]) * K;
                if constexpr (COMPACT) {
                    float4 a = tq_input4<K>(input, col, 0u), b = tq_input4<K>(input, col, 1u);
                    acc[m].xy = fma(first.xy, a.xy, acc[m].xy);
                    acc[m].xy = fma(first.zw, a.zw, acc[m].xy);
                    acc[m].xy = fma(second.xy, b.xy, acc[m].xy);
                    acc[m].xy = fma(second.zw, b.zw, acc[m].xy);
                } else {
                    acc[m] = fma(first, tq_input4<K>(input, col, 0u), acc[m]);
                    acc[m] = fma(second, tq_input4<K>(input, col, 1u), acc[m]);
                }
            }
        }
    }
}

template <typename T, uint COLUMNS, uint WIDTH, uint K, uint GS, uint BM, uint LANES = 16,
          bool COMPACT = false, typename ScalePtr>
inline float4 tq_dot_batch(device const uint* words, ulong descriptor, ScalePtr scales,
        device const half* relative, uint row, device const T* x, int4 sources, uint lane,
        uint begin = 0u, uint end = K) {
    static_assert(BM <= 4, "TQ vector batch supports at most four rows");
    float4 acc[BM];
    tq_dot_batch_accumulate<T, COLUMNS, WIDTH, K, GS, BM, LANES, COMPACT>(
        words, descriptor, scales, relative, row, x, sources, lane, acc, begin, end);
    float4 result = 0.f;
    #pragma unroll
    for (uint m = 0u; m < BM; ++m) result[m] = acc[m].x + acc[m].y + acc[m].z + acc[m].w;
    return result;
}

template <typename T, uint CG, uint CU, uint WG, uint WU, uint K, uint GG, uint GU,
          uint BM, typename SG, typename SU, typename Sources>
inline void tq_dot_pair_batch_accumulate(device const uint* words_g, ulong desc_g, SG scales_g,
        device const uint* words_u, ulong desc_u, SU scales_u, device const half* relative,
        uint row_g, uint row_u, device const T* x, Sources sources, uint lane, bool upper,
        thread float4 (&acc)[BM]) {
    if constexpr (CG == CU && WG == WU && GG == GU && metal::is_same_v<SG, SU>) {
        tq_dot_batch_accumulate<T, CG, WG, K, GG, BM, 32>(upper ? words_u : words_g,
            upper ? desc_u : desc_g, upper ? scales_u : scales_g, relative,
            upper ? row_u : row_g, x, sources, lane, acc);
    } else if (upper) {
        tq_dot_batch_accumulate<T, CU, WU, K, GU, BM, 32>(words_u, desc_u, scales_u,
            relative, row_u, x, sources, lane, acc);
    } else {
        tq_dot_batch_accumulate<T, CG, WG, K, GG, BM, 32>(words_g, desc_g, scales_g,
            relative, row_g, x, sources, lane, acc);
    }
}
