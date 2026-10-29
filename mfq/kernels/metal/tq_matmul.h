template <typename T, uint OUT, uint TILE_N, typename Order>
inline void tq_zero_rows(device T* y, Order order, uint begin, uint count,
        uint n0, uint local, uint threads) {
    for (uint index = local; index < count * TILE_N; index += threads) {
        uint col = n0 + index % TILE_N;
        if (col < OUT) y[uint(order[begin + index / TILE_N]) * OUT + col] = T(0);
    }
}

template <uint OUT, typename T>
inline void tq_store_projection4(device T* y, uint row, uint col, float4 value) {
    if constexpr (OUT % 4u == 0u) {
        if (col < OUT) *(device vec<T, 4>*)(y + tq_row_offset<OUT>(row, col)) = vec<T, 4>(value);
    } else {
        #pragma unroll
        for (uint j = 0u; j < 4u; ++j)
            if (col + j < OUT) y[tq_row_offset<OUT>(row, col + j)] = T(value[j]);
    }
}

template <typename T, uint K, uint OUT, uint BM, typename Decode, typename InputRow, typename Store>
inline void tq_nax_matmul_apply(device const T* x, uint rows, uint n0, uint m0,
        Decode decode, InputRow input_row, Store store, uint begin = 0u, uint end = K, uint n_end = OUT) {
    short2 coord = mlx::steel::BaseNAXFrag::get_coord();
    bool upper = (__metal_get_thread_index_in_simdgroup(ushort()) & 1u) != 0u;
    uint offsets[BM / 16][2];
    bool valid[BM / 16][2];
    #pragma unroll
    for (uint fm = 0u; fm < BM / 16u; ++fm) {
        #pragma unroll
        for (uint r = 0u; r < 2u; ++r) {
            uint m = m0 + fm * 16u + uint(coord.y) + r * 8u;
            valid[fm][r] = m < rows;
            offsets[fm][r] = m < rows ? input_row(m) * K : 0u;
        }
    }
    mlx::steel::NAXTile<float, BM / 16, 2> accum;
    accum.clear();
    for (uint kk = begin; kk < end; kk += 16u) {
        mlx::steel::NAXTile<half, BM / 16, 1> a;
        mlx::steel::NAXTile<half, 2, 1> b;
        #pragma unroll
        for (uint fk = 0u; fk < 1u; ++fk) {
            uint col = kk + fk * 16u + uint(coord.x);
            #pragma unroll
            for (uint fm = 0u; fm < BM / 16u; ++fm) {
                #pragma unroll
                for (uint r = 0u; r < 2u; ++r) {
                    half4 value = 0.h;
                    if (valid[fm][r] && col < K) {
                        device const T* source = x + offsets[fm][r];
                        value = half4(tq_input4<K>(source, col, 0u));
                    }
                    #pragma unroll
                    for (uint j = 0u; j < 4u; ++j) a.frag_at(fm, fk)[r * 4u + j] = value[j];
                }
            }
            #pragma unroll
            for (uint fn = 0u; fn < 2u; ++fn) {
                uint n = n0 + fn * 16u + uint(coord.y) + (upper ? 8u : 0u);
                uint aligned = kk + fk * 16u + (uint(coord.x) & ~7u);
                TqDecoded8 value{half4(0.h), half4(0.h)};
                if (n < n_end && aligned < K) value = decode(n, aligned);
                half4 other_first = simd_shuffle_xor(value.first, 1u);
                half4 other_second = simd_shuffle_xor(value.second, 1u);
                half4 row0 = upper ? other_second : value.first;
                half4 row1 = upper ? value.second : other_first;
                #pragma unroll
                for (uint j = 0u; j < 4u; ++j) {
                    b.frag_at(fn, fk)[j] = row0[j];
                    b.frag_at(fn, fk)[4u + j] = row1[j];
                }
            }
        }
        mlx::steel::tile_matmad_nax(accum, a, metal::bool_constant<false>{}, b, metal::bool_constant<true>{});
    }
    #pragma unroll
    for (uint fm = 0u; fm < BM / 16u; ++fm) {
        #pragma unroll
        for (uint fn = 0u; fn < 2u; ++fn) {
            #pragma unroll
            for (uint r = 0u; r < 2u; ++r) {
                uint m = m0 + fm * 16u + uint(coord.y) + r * 8u;
                uint n = n0 + fn * 16u + uint(coord.x);
                if (m < rows && n < n_end) store(m, n,
                    float4(accum.frag_at(fm, fn)[r * 4u], accum.frag_at(fm, fn)[r * 4u + 1u],
                        accum.frag_at(fm, fn)[r * 4u + 2u], accum.frag_at(fm, fn)[r * 4u + 3u]));
            }
        }
    }
}

template <typename T, uint K, uint OUT, uint BM, typename O, typename Decode, typename InputRow, typename OutputRow>
inline void tq_nax_matmul(device const T* x, device O* y, uint rows, uint n0, uint m0,
        Decode decode, InputRow input_row, OutputRow output_row, uint begin = 0u, uint end = K, uint n_end = OUT) {
    auto store = [&](uint row, uint col, float4 value) {
        if (col + 3u < n_end) tq_store_projection4<OUT>(y, output_row(row), col, value);
        else {
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j)
                if (col + j < n_end) y[tq_row_offset<OUT>(output_row(row), col + j)] = O(value[j]);
        }
    };
    tq_nax_matmul_apply<T, K, OUT, BM>(x, rows, n0, m0, decode, input_row, store, begin, end, n_end);
}

template <typename T, uint K, uint OUT, typename Decode, typename InputRow, typename OutputRow>
inline void tq_simd_matmul(device const T* x, device T* y, uint rows, uint m0, uint n0,
        uint local, Decode decode, InputRow input_row, OutputRow output_row,
        threadgroup half* shared_a, threadgroup half* shared_b) {
    constexpr uint BM = 32, BN = 32, BK = 32, THREADS = 128;
    uint lane = local % 32u, group = local / 32u;
    uint fr = ((lane / 4u) & 4u) + ((lane / 2u) & 3u);
    uint fc = ((lane / 4u) & 2u) * 2u + (lane & 1u) * 2u;
    uint sm = group / 2u, sn = group % 2u;
    metal::simdgroup_matrix<float, 8, 8> accum[2][2];
    #pragma unroll
    for (uint fm = 0u; fm < 2u; ++fm) {
        #pragma unroll
        for (uint fn = 0u; fn < 2u; ++fn) {
            accum[fm][fn].thread_elements()[0] = 0.f;
            accum[fm][fn].thread_elements()[1] = 0.f;
        }
    }
    for (uint kk = 0u; kk < K; kk += BK) {
        for (uint index = local; index < BM * BK / 4u; index += THREADS) {
            uint row = m0 + index / (BK / 4u), col = kk + (index % (BK / 4u)) * 4u;
            half4 value = 0.h;
            if (row < rows && col < K) value = half4(tq_input4<K>(x + input_row(row) * K, col, 0u));
            *(threadgroup half4*)(shared_a + index * 4u) = value;
        }
        for (uint index = local; index < BN * BK / 4u; index += THREADS) {
            uint row = n0 + index / (BK / 4u), col = kk + (index % (BK / 4u)) * 4u;
            half4 value = row < OUT && col < K ? decode(row, col) : half4(0.h);
            *(threadgroup half4*)(shared_b + index * 4u) = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (m0 + sm * 16u < rows) {
        for (uint k = 0u; k < BK; k += 8u) {
            metal::simdgroup_matrix<half, 8, 8> a[2], b[2];
            #pragma unroll
            for (uint f = 0u; f < 2u; ++f) {
                #pragma unroll
                for (uint j = 0u; j < 2u; ++j) {
                    a[f].thread_elements()[j] = shared_a[(sm * 16u + f * 8u + fr) * BK + k + fc + j];
                    b[f].thread_elements()[j] = shared_b[(sn * 16u + f * 8u + fc + j) * BK + k + fr];
                }
            }
            #pragma unroll
            for (uint fm = 0u; fm < 2u; ++fm) {
                #pragma unroll
                for (uint fn = 0u; fn < 2u; ++fn)
                    simdgroup_multiply_accumulate(accum[fm][fn], a[fm], b[fn], accum[fm][fn]);
            }
        }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for (uint fm = 0u; fm < 2u; ++fm) {
        uint m = m0 + sm * 16u + fm * 8u + fr;
        if (m < rows) {
            uint destination = output_row(m) * OUT;
            #pragma unroll
            for (uint fn = 0u; fn < 2u; ++fn) {
                uint n = n0 + sn * 16u + fn * 8u + fc;
                #pragma unroll
                for (uint j = 0u; j < 2u; ++j)
                    if (n + j < OUT) y[destination + n + j] = T(accum[fm][fn].thread_elements()[j]);
            }
        }
    }
}

template <typename T, uint K, uint OUT, uint BM, uint TM, bool DIRECT_A = false,
        typename Decode, typename InputRow, typename OutputRow>
inline void tq_nax_shared_matmul(device const T* x, device T* y, uint rows, uint m0, uint n0,
        uint local, Decode decode, InputRow input_row, OutputRow output_row,
        threadgroup half* shared_a, threadgroup half* shared_b) {
    constexpr uint BN = 64, BK = DIRECT_A ? 64 : 32, THREADS = BM * 2, LOAD_M = TM == 16 ? BM / 2 : BM;
    short2 coord = mlx::steel::BaseNAXFrag::get_coord();
    uint sm = (local / 32u) / 2u, sn = (local / 32u) % 2u;
    uint offsets[TM / 16][2];
    bool valid[TM / 16][2];
    if constexpr (DIRECT_A) {
        #pragma unroll
        for (uint f = 0u; f < TM / 16u; ++f) {
            #pragma unroll
            for (uint r = 0u; r < 2u; ++r) {
                uint row = m0 + sm * TM + f * 16u + uint(coord.y) + r * 8u;
                valid[f][r] = row < rows;
                offsets[f][r] = row < rows ? input_row(row) * K : 0u;
            }
        }
    }
    mlx::steel::NAXTile<float, TM / 16, 2> accum;
    accum.clear();
    for (uint kk = 0u; kk < K; kk += BK) {
        if constexpr (!DIRECT_A) for (uint index = local; index < LOAD_M * BK / 4u; index += THREADS) {
            uint row = m0 + index / (BK / 4u), col = kk + (index % (BK / 4u)) * 4u;
            half4 value = 0.h;
            if (row < rows && col < K) value = half4(tq_input4<K>(x + input_row(row) * K, col, 0u));
            *(threadgroup half4*)(shared_a + index * 4u) = value;
        }
        constexpr uint VALUES = DIRECT_A ? 32u : THREADS == 128u ? 16u : 8u;
        for (uint index = local; index < BN * BK / VALUES; index += THREADS) {
            uint row = n0 + index / (BK / VALUES), col = kk + (index % (BK / VALUES)) * VALUES;
            #pragma unroll
            for (uint part = 0u; part < VALUES / 8u; ++part) {
                TqDecoded8 value{half4(0.h), half4(0.h)};
                if (row < OUT && col + part * 8u < K) value = decode(row, col + part * 8u);
                uint offset = index * VALUES + part * 8u;
                if constexpr (DIRECT_A) offset = index / (BK / VALUES) * BK
                    + (((index % (BK / VALUES)) * VALUES + part * 8u)
                        ^ ((index / (BK / VALUES) & 7u) * 8u));
                *(threadgroup half4*)(shared_b + offset) = value.first;
                *(threadgroup half4*)(shared_b + offset + 4u) = value.second;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (m0 + sm * TM < rows) {
        for (uint k = 0u; k < min(uint(BK), uint(K) - kk); k += 32u) {
        mlx::steel::NAXTile<half, TM / 16, 2> a;
        mlx::steel::NAXTile<half, 2, 2> b;
        #pragma unroll
        for (uint fk = 0u; fk < 2u; ++fk) {
            #pragma unroll
            for (uint f = 0u; f < TM / 16u; ++f) {
                #pragma unroll
                for (uint r = 0u; r < 2u; ++r) {
                    uint ar = sm * TM + f * 16u + uint(coord.y) + r * 8u;
                    uint col = k + fk * 16u + uint(coord.x);
                    half4 av = 0.h;
                    if constexpr (DIRECT_A) {
                        if (valid[f][r] && kk + col < K)
                            av = half4(tq_input4<K>(x + offsets[f][r], kk + col, 0u));
                    } else av = *(threadgroup half4*)(shared_a + ar * BK + col);
                    #pragma unroll
                    for (uint j = 0u; j < 4u; ++j) a.frag_at(f, fk)[r * 4u + j] = av[j];
                }
            }
            #pragma unroll
            for (uint f = 0u; f < 2u; ++f) {
                #pragma unroll
                for (uint r = 0u; r < 2u; ++r) {
                    uint br = sn * 32u + f * 16u + uint(coord.y) + r * 8u;
                    uint col = k + fk * 16u + uint(coord.x);
                    if constexpr (DIRECT_A) col ^= (br & 7u) * 8u;
                    half4 bv = *(threadgroup half4*)(shared_b + br * BK + col);
                    #pragma unroll
                    for (uint j = 0u; j < 4u; ++j) b.frag_at(f, fk)[r * 4u + j] = bv[j];
                }
            }
        }
        mlx::steel::tile_matmad_nax(accum, a, metal::bool_constant<false>{}, b, metal::bool_constant<true>{});
        }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for (uint fm = 0u; fm < TM / 16u; ++fm) {
        #pragma unroll
        for (uint fn = 0u; fn < 2u; ++fn) {
            #pragma unroll
            for (uint r = 0u; r < 2u; ++r) {
                uint m = m0 + sm * TM + fm * 16u + uint(coord.y) + r * 8u;
                uint n = n0 + sn * 32u + fn * 16u + uint(coord.x);
                if (m < rows) {
                    tq_store_projection4<OUT>(y, output_row(m), n,
                        float4(accum.frag_at(fm, fn)[r * 4u], accum.frag_at(fm, fn)[r * 4u + 1u],
                            accum.frag_at(fm, fn)[r * 4u + 2u], accum.frag_at(fm, fn)[r * 4u + 3u]));
                }
            }
        }
    }
}
