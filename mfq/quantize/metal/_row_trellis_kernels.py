# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Metal kernels for the experimental row Trellis codec."""

HEADER = r"""
inline half exl3_decode_state(uint state, uint codebook) {
    uint raw = state * 0x83dcd12du;
    uint sum = (raw & 255u) + ((raw >> 8) & 255u)
        + ((raw >> 16) & 255u) + (raw >> 24);
    half value = as_type<half>(ushort(sum + 0x6400u));
    return metal::fma(value, as_type<half>(ushort(0x1eeeu)),
                      as_type<half>(ushort(0xc931u)));
}

inline float fast_value(float base, float weight, float step,
                        float inverse, uint refine_bits) {
    if (refine_bits == 0u) return base;
    float maximum = float((1u << refine_bits) - 1u);
    float code = clamp(rint((weight - base) * inverse + maximum * .5f), 0.f, maximum);
    return metal::fma(code - maximum * .5f, step, base);
}

inline float hybrid_value(uint state, float weight, float step, float inverse,
                          uint refine_bits, device const half* codebook) {
    return fast_value(float(codebook[state]), weight, step, inverse, refine_bits);
}

inline uint read_bits(device const uint* words, uint start, uint bits) {
    uint shift = start & 31u;
    uint value = words[start >> 5] << shift;
    if (shift + bits > 32u) value |= words[(start >> 5) + 1u] >> (32u - shift);
    return value >> (32u - bits);
}
"""
SEARCH = r"""
uint lane = thread_position_in_threadgroup.x;
uint slot = threadgroup_position_in_grid.x;

threadgroup float tile_values[TILE_LEN];
threadgroup float reduction_costs[THREADS];
threadgroup ushort reduction_edges[THREADS];
threadgroup ushort shared_edge;
threadgroup float local_costs[LOCAL_COST_COUNT];

device float* slot_costs = temp_costs;
if (!LOCAL_COSTS) slot_costs += slot * 2 * EDGES;
device uint* slot_edges = temp_edges + slot * TILE_LEN * TRACE_WORDS;

for (uint tile = slot; tile < NUM_TILES; tile += ACTIVE_GROUPS) {
    for (uint i = lane; i < TILE_LEN; i += THREADS) { tile_values[i] = float(input_tiles[tile * TILE_LEN + i]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First pass starts halfway around the cyclic tile and finds
    // the state immediately before logical position zero.
    for (uint step = 0; step < TILE_LEN; ++step) {
        uint ri = (step + (TILE_LEN/2)) & (TILE_LEN-1u);
        uint write_bank = 1u - (step & 1u);
        uint read_bank = step & 1u;
        float weight = tile_values[ri]; float objective = objectives[((tile / OBJECTIVE_CANDIDATES) % OBJECTIVE_GROUPS) * TILE_LEN + ri];

        for (uint out_edge = lane; out_edge < EDGES; out_edge += THREADS) {
            float parent[MAX_SYMBOLS];
            half value[MAX_SYMBOLS];
            #pragma unroll
            for(uint s=0u;s<MAX_SYMBOLS;++s) {
                uint candidate=(s<<STATE_SHIFT)|out_edge;
                uint previous=candidate>>BITS;
                parent[s]=step==0u?0.f:(LOCAL_COSTS?local_costs[read_bank*EDGES+previous]:slot_costs[read_bank*EDGES+previous]);
                value[s]=codebook[candidate];
            }

            uint state = out_edge;
            uint in_edge = state >> BITS;
            float delta = (fast_value(float(value[0u]),weight,config[0],config[1],REFINE_BITS) - weight) * objective;
            float best_cost;
            if (step == 0) {
                best_cost = delta * delta;
            } else {
                float previous_cost = parent[0u];
                best_cost = metal::fma(delta, delta, previous_cost);
            }
            ushort best_edge = ushort(in_edge);

            for (uint symbol = 1; symbol < MAX_SYMBOLS; ++symbol) {
                state = (symbol << STATE_SHIFT) | out_edge;
                in_edge = state >> BITS;
                delta = (fast_value(float(value[symbol]),weight,config[0],config[1],REFINE_BITS) - weight) * objective;
                float cost;
                if (step == 0) {
                    cost = delta * delta;
                } else {
                    float previous_cost = parent[symbol];
                    cost = metal::fma(delta, delta, previous_cost);
                }
                if (cost < best_cost) {
                    best_cost = cost;
                    best_edge = ushort(in_edge);
                }
            }

            if (LOCAL_COSTS) {
                local_costs[write_bank * EDGES + out_edge] = best_cost;
            } else {
                slot_costs[write_bank * EDGES + out_edge] = best_cost;
            }
            uint selected=uint(best_edge)>>(STATE_SHIFT-BITS);
            uint packed=selected<<((out_edge%PER_WORD)*PACK_BITS);
            for(uint offset=1u;offset<PER_WORD;offset<<=1u)
                packed|=simd_shuffle_xor(packed,offset);
            if(out_edge%PER_WORD==0u) slot_edges[ri*TRACE_WORDS+out_edge/PER_WORD]=packed;
        }
        if (LOCAL_COSTS) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
    }

    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    float local_cost = as_type<half>(ushort(0x7c00u));
    ushort local_edge = ushort(0xffffu);
    for (uint edge = lane; edge < EDGES; edge += THREADS) {
        float cost = LOCAL_COSTS ? local_costs[edge] : slot_costs[edge];
        if (cost < local_cost || (cost == local_cost && edge < local_edge)) {
            local_cost = cost;
            local_edge = ushort(edge);
        }
    }
    reduction_costs[lane] = local_cost;
    reduction_edges[lane] = local_edge;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = THREADS >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            float other_cost = reduction_costs[lane + offset];
            ushort other_edge = reduction_edges[lane + offset];
            float own_cost = reduction_costs[lane];
            ushort own_edge = reduction_edges[lane];
            if (other_cost < own_cost || (other_cost == own_cost && other_edge < own_edge)) {
                reduction_costs[lane] = other_cost;
                reduction_edges[lane] = other_edge;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        ushort edge = reduction_edges[0];
        for (int step = (TILE_LEN-1); step >= 0; --step) {
            uint ri = (uint(step) + (TILE_LEN/2)) & (TILE_LEN-1u);
            uint selected=(slot_edges[ri*TRACE_WORDS+uint(edge)/PER_WORD]>>((uint(edge)%PER_WORD)*PACK_BITS))&((1u<<BITS)-1u);
            edge=ushort((uint(edge)>>BITS)|(selected<<(STATE_SHIFT-BITS)));
            if (ri == 0) break;
        }
        shared_edge = edge;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Second pass fixes the incoming edge at position zero and
    // backtracks from that same edge to enforce tail biting.
    for (uint step = 0; step < TILE_LEN; ++step) {
        uint write_bank = 1u - (step & 1u);
        uint read_bank = step & 1u;
        float weight = tile_values[step]; float objective = objectives[((tile / OBJECTIVE_CANDIDATES) % OBJECTIVE_GROUPS) * TILE_LEN + step];

        for (uint out_edge = lane; out_edge < EDGES; out_edge += THREADS) {
            float parent[MAX_SYMBOLS];
            half value[MAX_SYMBOLS];
            #pragma unroll
            for(uint s=0u;s<MAX_SYMBOLS;++s) {
                uint candidate=(s<<STATE_SHIFT)|out_edge;
                uint previous=candidate>>BITS;
                parent[s]=step==0u?0.f:(LOCAL_COSTS?local_costs[read_bank*EDGES+previous]:slot_costs[read_bank*EDGES+previous]);
                value[s]=codebook[candidate];
            }

            uint state = out_edge;
            uint in_edge = state >> BITS;
            float delta = (fast_value(float(value[0u]),weight,config[0],config[1],REFINE_BITS) - weight) * objective;
            float best_cost;
            if (step == 0) {
                best_cost = in_edge == uint(shared_edge)
                    ? delta * delta
                    : as_type<half>(ushort(0x7c00u));
            } else {
                float previous_cost = parent[0u];
                best_cost = metal::fma(delta, delta, previous_cost);
            }
            ushort best_edge = ushort(in_edge);

            for (uint symbol = 1; symbol < MAX_SYMBOLS; ++symbol) {
                state = (symbol << STATE_SHIFT) | out_edge;
                in_edge = state >> BITS;
                delta = (fast_value(float(value[symbol]),weight,config[0],config[1],REFINE_BITS) - weight) * objective;
                float cost;
                if (step == 0) {
                    cost = in_edge == uint(shared_edge)
                        ? delta * delta
                        : as_type<half>(ushort(0x7c00u));
                } else {
                    float previous_cost = parent[symbol];
                    cost = metal::fma(delta, delta, previous_cost);
                }
                if (cost < best_cost) {
                    best_cost = cost;
                    best_edge = ushort(in_edge);
                }
            }

            if (LOCAL_COSTS) {
                local_costs[write_bank * EDGES + out_edge] = best_cost;
            } else {
                slot_costs[write_bank * EDGES + out_edge] = best_cost;
            }
            uint selected=uint(best_edge)>>(STATE_SHIFT-BITS);
            uint packed=selected<<((out_edge%PER_WORD)*PACK_BITS);
            for(uint offset=1u;offset<PER_WORD;offset<<=1u)
                packed|=simd_shuffle_xor(packed,offset);
            if(out_edge%PER_WORD==0u) slot_edges[step*TRACE_WORDS+out_edge/PER_WORD]=packed;
        }
        if (LOCAL_COSTS) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
    }

    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    if (lane == 0) {
        ushort edge = shared_edge;
        for (int step = (TILE_LEN-1); step >= 0; --step) {
            uint selected=(slot_edges[uint(step)*TRACE_WORDS+uint(edge)/PER_WORD]>>((uint(edge)%PER_WORD)*PACK_BITS))&((1u<<BITS)-1u);
            ushort previous=ushort((uint(edge)>>BITS)|(selected<<(STATE_SHIFT-BITS)));
            ushort encoded = ushort((uint(previous) << BITS) | uint(edge));
            edge = previous;
            indices[tile * TILE_LEN + uint(step)] = as_type<short>(encoded);
            quantized[tile * TILE_LEN + uint(step)] = float(
                hybrid_value(uint(encoded),tile_values[uint(step)],config[0],config[1],REFINE_BITS,codebook)
            );
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
"""

ANCHOR = r"""
uint i=thread_position_in_grid.x;
if(i>=COUNT) return;
uint raw=uint(as_type<ushort>(half(input[i])));
uint code=min((raw+3u+((raw>>3)&1u))>>3,3967u);
output[i]=as_type<half>(ushort(code<<3));
"""

PACK = r"""
uint index=thread_position_in_grid.x;
if(index>=WORDS) return;
uint group=index/(TILE_LEN*Q/32u);
uint word=index%(TILE_LEN*Q/32u);
bool refinement=word>=TILE_LEN*BASE_BITS/32u;
uint bits=refinement?REFINE_BITS:BASE_BITS;
if(refinement) word-=TILE_LEN*BASE_BITS/32u;
uint value=0u;
for(uint j=0;j<32u;++j) {
    uint stream_bit=word*32u+j;
    uint element=stream_bit/bits;
    uint shift=bits-1u-stream_bit%bits;
    uint state=uint(as_type<ushort>(states[group*TILE_LEN+element]));
    uint code=state&((1u<<BASE_BITS)-1u);
    if(refinement) {
        float base=float(exl3_decode_state(state,2));
        float maximum=float((1u<<REFINE_BITS)-1u);
        float r=clamp(rint((normalized[group*TILE_LEN+element]-base)*config[1]+maximum*.5f),0.f,maximum);
        code=uint(r);
    }
    value|=((code>>shift)&1u)<<(31u-j);
}
packed[index]=value;
"""

DECODE = r"""
uint i=thread_position_in_grid.x;
if(i>=COUNT) return;
uint row=i/COLUMNS;
uint col=i%COLUMNS;
uint descriptor=uint(rows[row*3u])|(uint(rows[row*3u+1u])<<8)|(uint(rows[row*3u+2u])<<16);
uint q=((descriptor>>12)&7u)+1u;
uint prefix=(descriptor>>15)&255u;
ulong offset=blocks[row/32u]+ulong(prefix*(COLUMNS/32u));
uint base_bits=min(q,3u);
uint refine=q-base_bits;
float anchor=float(as_type<half>(ushort((descriptor&4095u)<<3)));
uint group=col/GROUP_SIZE;
uint lane=col%GROUP_SIZE;
uint gain_index=row*(COLUMNS/GROUP_SIZE)+group;
uint gain=(payload[gain_index/8u]>>((gain_index%8u)*4u))&15u;
float scale=anchor*float(lut[gain]);
device const uint* base=payload+offset+group*(GROUP_SIZE*q/32u);
uint bit=((lane+1u)*base_bits+GROUP_SIZE*base_bits-STATE_BITS)%(GROUP_SIZE*base_bits);
uint shift=bit&31u;
uint word=bit>>5;
uint value=base[word]<<shift;
if(shift+STATE_BITS>32u) value|=base[(word+1u)%(GROUP_SIZE*base_bits/32u)]>>(32u-shift);
uint state=value>>(32u-STATE_BITS);
float decoded=float(exl3_decode_state(state,2));
if(refine>0u) {
    device const uint* residual=base+GROUP_SIZE*base_bits/32u;
    uint code=read_bits(residual,lane*refine,refine);
    float midpoint=float((1u<<refine)-1u)*.5f;
    decoded=metal::fma(float(code)-midpoint,steps[q-1u],decoded);
}
output[i]=decoded*scale*float(columns_scale[SCALE_COLUMNS?col:0u]);
"""

PROJECT = r"""
uint row=threadgroup_position_in_grid.x;
uint lane=thread_position_in_threadgroup.x;
uint descriptor=uint(descriptors[row*3u])|(uint(descriptors[row*3u+1u])<<8)|(uint(descriptors[row*3u+2u])<<16);
uint q=((descriptor>>12)&7u)+1u;
uint prefix=(descriptor>>15)&255u;
ulong offset=blocks[row/32u]+ulong(prefix*(COLUMNS/32u));
float anchor=float(as_type<half>(ushort((descriptor&4095u)<<3)));
uint base_bits=min(q,3u);
uint refine=q-base_bits;
float accum[M];
for(uint m=0;m<M;++m) accum[m]=0.f;
for(uint col=lane;col<COLUMNS;col+=THREADS) {
    uint group=col/GROUP_SIZE;
    uint position=col%GROUP_SIZE;
    uint gain_index=row*(COLUMNS/GROUP_SIZE)+group;
    uint gain=(payload[gain_index/8u]>>((gain_index%8u)*4u))&15u;
    float scale=anchor*float(lut[gain]);
    device const uint* base=payload+offset+group*(GROUP_SIZE*q/32u);
    uint bit=((position+1u)*base_bits+GROUP_SIZE*base_bits-STATE_BITS)%(GROUP_SIZE*base_bits);
    uint shift=bit&31u;
    uint word=bit>>5;
    uint value=base[word]<<shift;
    if(shift+STATE_BITS>32u) value|=base[(word+1u)%(GROUP_SIZE*base_bits/32u)]>>(32u-shift);
    uint state=value>>(32u-STATE_BITS);
    float decoded=float(exl3_decode_state(state,2));
    if(refine>0u) {
        device const uint* residual=base+GROUP_SIZE*base_bits/32u;
        uint code=read_bits(residual,position*refine,refine);
        decoded=metal::fma(float(code)-float((1u<<refine)-1u)*.5f,steps[q-1u],decoded);
    }
    float weight=decoded*scale*float(columns_scale[SCALE_COLUMNS?col:0u]);
    for(uint m=0;m<M;++m) accum[m]=metal::fma(weight,input[m*COLUMNS+col],accum[m]);
}
threadgroup float partial[M*(THREADS/32)];
for(uint m=0;m<M;++m) {
    float sum=simd_sum(accum[m]);
    if((lane&31u)==0u) partial[m*(THREADS/32)+(lane>>5)]=sum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if(lane<32u) {
    for(uint m=0;m<M;++m) {
        float sum=simd_sum(lane<(THREADS/32)?partial[m*(THREADS/32)+lane]:0.f);
        if(lane==0u) output[m*ROWS+row]=sum;
    }
}
"""
