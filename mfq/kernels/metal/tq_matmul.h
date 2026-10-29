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

template <typename T, uint K, uint OUT, uint BM, uint TM, bool DIRECT_A = false, bool CACHE_INPUT_ROWS = false,
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
    constexpr uint INPUT_LOADS = (LOAD_M * BK / 4u + THREADS - 1u) / THREADS;
    uint input_rows[INPUT_LOADS];
    if constexpr (CACHE_INPUT_ROWS && !DIRECT_A) {
        #pragma unroll
        for (uint entry = 0u; entry < INPUT_LOADS; ++entry) {
            uint row = m0 + (local + entry * THREADS) / (BK / 4u);
            input_rows[entry] = row < rows ? input_row(row) : 0xffffffffu;
        }
    }
    mlx::steel::NAXTile<float, TM / 16, 2> accum;
    accum.clear();
    for (uint kk = 0u; kk < K; kk += BK) {
        if constexpr (!DIRECT_A) for (uint index = local; index < LOAD_M * BK / 4u; index += THREADS) {
            uint row = m0 + index / (BK / 4u), col = kk + (index % (BK / 4u)) * 4u;
            half4 value = 0.h;
            if constexpr (CACHE_INPUT_ROWS) {
                uint input = input_rows[(index - local) / THREADS];
                if (input != 0xffffffffu && col < K) value = half4(tq_input4<K>(x + input * K, col, 0u));
            } else if (row < rows && col < K) value = half4(tq_input4<K>(x + input_row(row) * K, col, 0u));
            *(threadgroup half4*)(shared_a + index * 4u) = value;
        }
        constexpr uint VALUES = BN * BK / THREADS;
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

template <typename T, uint K, uint OUT, typename DecodeG, typename DecodeU, typename InputRow, typename OutputRow>
inline void tq_simd_shared_pair(device const T* x, device T* y, uint rows, uint m0, uint n0,
        uint local, DecodeG decode_g, DecodeU decode_u, InputRow input_row, OutputRow output_row,
        float limit, bool gelu, threadgroup half* shared_a, threadgroup half* shared_g, threadgroup half* shared_u) {
    constexpr uint BM = 32, BN = 32, BK = 32, THREADS = 256;
    uint lane = local % 32u, group = (local / 32u) % 4u, projection = local / 128u;
    threadgroup half* shared_b = projection ? shared_u : shared_g;
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
        for (uint index = local % 128u; index < BN * BK / 4u; index += 128u) {
            uint row = n0 + index / (BK / 4u), col = kk + (index % (BK / 4u)) * 4u;
            half4 value = row < OUT && col < K
                ? (projection ? decode_u(row, col) : decode_g(row, col)) : half4(0.h);
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
            #pragma unroll
            for (uint fn = 0u; fn < 2u; ++fn) {
                uint n = n0 + sn * 16u + fn * 8u + fc;
                #pragma unroll
                for (uint j = 0u; j < 2u; ++j)
                    if (n + j < OUT) shared_b[(m - m0) * BN + n + j - n0] = half(accum[fm][fn].thread_elements()[j]);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = local; index < BM * BN; index += THREADS) {
        uint m = m0 + index / BN, n = n0 + index % BN;
        if (m < rows && n < OUT) {
            float gate = float(shared_g[index]), up = float(shared_u[index]);
            if (limit > 0.f) { gate = min(gate, limit); up = clamp(up, -limit, limit); }
            gate = gelu ? .5f * gate * (1.f + tanh(.7978845608028654f * gate * (1.f + .044715f * gate * gate)))
                : gate / (1.f + exp(-gate));
            y[output_row(m) * OUT + n] = T(gate * up);
        }
    }
}

template <typename T, uint K, uint OUT, uint BM, uint TM, typename DecodeG, typename DecodeU,
        typename InputRow, typename OutputRow>
inline void tq_nax_shared_pair(device const T* x, device T* y, uint rows, uint m0, uint n0,
        uint local, DecodeG decode_g, DecodeU decode_u, InputRow input_row, OutputRow output_row,
        float limit, bool gelu, threadgroup half* shared_a, threadgroup half* shared_g, threadgroup half* shared_u) {
    constexpr uint BK = 32, BN = 64, THREADS = BM * 2, LOAD_M = TM == 16 ? BM / 2 : BM;
    short2 coord = mlx::steel::BaseNAXFrag::get_coord();
    uint sm = (local / 32u) / 2u, sn = (local / 32u) % 2u;
    constexpr uint INPUT_LOADS = (LOAD_M * BK / 4u + THREADS - 1u) / THREADS;
    uint input_rows[INPUT_LOADS];
    #pragma unroll
    for (uint i = 0u; i < INPUT_LOADS; ++i) {
        uint row = m0 + (local + i * THREADS) / (BK / 4u);
        input_rows[i] = row < rows ? input_row(row) : 0xffffffffu;
    }
    mlx::steel::NAXTile<float, TM / 16, 2> gate, up;
    gate.clear(); up.clear();
    for (uint kk = 0u; kk < K; kk += BK) {
        for (uint i = local; i < LOAD_M * BK / 4u; i += THREADS) {
            uint source = input_rows[(i - local) / THREADS], col = kk + (i % (BK / 4u)) * 4u;
            half4 value = 0.h;
            if (source != 0xffffffffu && col < K) value = half4(tq_input4<K>(x + source * K, col, 0u));
            *(threadgroup half4*)(shared_a + i * 4u) = value;
        }
        constexpr uint VALUES = BN * BK / THREADS;
        for (uint i = local; i < BN * BK / VALUES; i += THREADS) {
            uint row = n0 + i / (BK / VALUES), col = kk + (i % (BK / VALUES)) * VALUES;
            #pragma unroll
            for (uint part = 0u; part < VALUES / 8u; ++part) {
                TqDecoded8 g{half4(0.h), half4(0.h)}, u = g;
                if (row < OUT && col + part * 8u < K) {
                    g = decode_g(row, col + part * 8u); u = decode_u(row, col + part * 8u);
                }
                uint offset = i * VALUES + part * 8u;
                *(threadgroup half4*)(shared_g + offset) = g.first;
                *(threadgroup half4*)(shared_g + offset + 4u) = g.second;
                *(threadgroup half4*)(shared_u + offset) = u.first;
                *(threadgroup half4*)(shared_u + offset + 4u) = u.second;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (m0 + sm * TM < rows) {
            mlx::steel::NAXTile<half, TM / 16, 2> a;
            mlx::steel::NAXTile<half, 2, 2> bg, bu;
            #pragma unroll
            for (uint fk = 0u; fk < 2u; ++fk) {
                #pragma unroll
                for (uint f = 0u; f < TM / 16u; ++f) {
                    #pragma unroll
                    for (uint r = 0u; r < 2u; ++r) {
                        uint ar = sm * TM + f * 16u + uint(coord.y) + r * 8u, col = fk * 16u + uint(coord.x);
                        half4 value = *(threadgroup half4*)(shared_a + ar * BK + col);
                        #pragma unroll
                        for (uint j = 0u; j < 4u; ++j) a.frag_at(f, fk)[r * 4u + j] = value[j];
                    }
                }
                #pragma unroll
                for (uint f = 0u; f < 2u; ++f) {
                    #pragma unroll
                    for (uint r = 0u; r < 2u; ++r) {
                        uint br = sn * 32u + f * 16u + uint(coord.y) + r * 8u, col = fk * 16u + uint(coord.x);
                        half4 g = *(threadgroup half4*)(shared_g + br * BK + col);
                        half4 u = *(threadgroup half4*)(shared_u + br * BK + col);
                        #pragma unroll
                        for (uint j = 0u; j < 4u; ++j) {
                            bg.frag_at(f, fk)[r * 4u + j] = g[j]; bu.frag_at(f, fk)[r * 4u + j] = u[j];
                        }
                    }
                }
            }
            mlx::steel::tile_matmad_nax(gate, a, metal::bool_constant<false>{}, bg, metal::bool_constant<true>{});
            mlx::steel::tile_matmad_nax(up, a, metal::bool_constant<false>{}, bu, metal::bool_constant<true>{});
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
                    float4 g, u;
                    #pragma unroll
                    for (uint j = 0u; j < 4u; ++j) {
                        g[j] = float(T(gate.frag_at(fm, fn)[r * 4u + j]));
                        u[j] = float(T(up.frag_at(fm, fn)[r * 4u + j]));
                    }
                    if (limit > 0.f) { g = min(g, float4(limit)); u = clamp(u, float4(-limit), float4(limit)); }
                    g = gelu ? .5f * g * (1.f + tanh(.7978845608028654f * g * (1.f + .044715f * g * g)))
                        : g / (1.f + exp(-g));
                    tq_store_projection4<OUT>(y, output_row(m), n, g * u);
                }
            }
        }
    }
}
