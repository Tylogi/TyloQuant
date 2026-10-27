// Included inside the NoPE/latent MoE test's anonymous namespace.

float dense_tolerance(tb::ScalarType dtype) {
    return dtype == tb::kFloat32 ? 1e-6f : (dtype == tb::kFloat16 ? 3e-3f : 3e-2f);
}

void require_dense_close(const Tensor& actual, const Tensor& expected, float tolerance,
                         const std::string& context) {
    require(actual.sizes().vec() == expected.sizes().vec() &&
                actual.scalar_type() == expected.scalar_type(), context + " shape/dtype changed");
    const auto values = host_values(actual);
    const auto reference = host_values(expected);
    for (std::size_t i = 0; i < values.size(); ++i)
        require(std::isfinite(values[i]) && std::abs(values[i] - reference[i]) <= tolerance,
                context + " differs at " + std::to_string(i) + ": " +
                    std::to_string(values[i]) + " / " + std::to_string(reference[i]));
}

void check_dense_grouped() {
    constexpr int experts = 32, routes = 22, n = 19, k = 37;
    for (const auto dtype : {tb::kFloat32, tb::kFloat16, tb::kBFloat16}) {
        auto bank = upload(pattern(experts * n * k, 0.08f, 3.0f), {experts, n, k}, dtype);
        for (const int rows : {1, 8, 64, 512, 4096}) {
            for (const bool repeated : {false, true}) {
                std::vector<float> values(rows * routes);
                for (int pair = 0; pair < rows * routes; ++pair)
                    values[pair] = repeated ? pair % 2 : (pair * 13) % experts;
                auto ids = upload(values, {rows, routes}, tb::kFloat32).to(tb::kInt32);
                auto route = build_moe_route_plan(ids, experts);
                auto flat = upload(pattern(rows * k, 0.15f, 7.0f), {rows, k}, tb::kFloat32);
                auto routed = upload(pattern(rows * routes * k, 0.15f, 11.0f),
                                     {rows, routes, k}, tb::kFloat16);
                for (const auto& input : {flat, routed}) {
                    auto expected = dense_routed_projection_gather(bank, input, ids);
                    auto actual = dense_routed_projection(bank, input, route);
                    require_dense_close(actual, expected, dense_tolerance(dtype),
                        "grouped dense rows=" + std::to_string(rows) + " repeated=" + std::to_string(repeated));
                }
            }
        }
        // Every pair has a distinct expert; no route has another row to reuse.
        auto ids = tb::arange(32, tb::TensorOptions().dtype(tb::kInt32).device(tb::kCUDA)).reshape({16, 2});
        auto input = upload(pattern(16 * k, 0.15f, 9.0f), {16, k}, dtype);
        require_dense_close(dense_routed_projection(bank, input, ids),
                            dense_routed_projection_gather(bank, input, ids), dense_tolerance(dtype),
                            "fully dispersed dense routes");
    }
    // Unsupported source precision keeps the bounded legacy path.
    auto bank = upload(pattern(4 * 7 * 5, 0.1f, 3.0f), {4, 7, 5}, tb::kFloat64);
    auto input = upload(pattern(3 * 5, 0.1f, 4.0f), {3, 5}, tb::kFloat32);
    auto ids = upload({0, 1, 1, 2, 2, 3}, {3, 2}, tb::kFloat32).to(tb::kInt64);
    require_dense_close(dense_routed_projection(bank, input, ids, 1),
                        dense_routed_projection_gather(bank, input, ids, 1), 1e-6f, "F64 fallback");
    // Above the map's expert limit, retain gather even for supported dtypes.
    auto many = upload(pattern(4097 * 7 * 5, 0.1f, 3.0f), {4097, 7, 5}, tb::kFloat32);
    require_dense_close(dense_routed_projection(many, input, ids, 1),
                        dense_routed_projection_gather(many, input, ids, 1), 1e-6f, "expert-count fallback");
}

Tensor dense_reference_activation(Tensor gate, Tensor up, mfq::models::MlpActivation kind, double limit) {
    if (kind == mfq::models::MlpActivation::relu_squared) return relu_squared_cuda(up.contiguous());
    if (kind == mfq::models::MlpActivation::gelu_gated) return gelu_mul_cuda(gate.contiguous(), up.contiguous());
    if (limit > 0.0)
        return (tb::silu(tb::clamp_max(gate.to(tb::kFloat32), limit)) *
                tb::clamp(up.to(tb::kFloat32), -limit, limit)).to(tb::kFloat16).contiguous();
    return (up * tb::silu(gate)).contiguous();
}

void check_dense_gated_layouts() {
    using Kind = mfq::models::MlpActivation;
    for (const auto dtype : {tb::kFloat32, tb::kFloat16, tb::kBFloat16}) {
        MoeGeometry g;
        g.hidden = g.latent;
        MoeWeights weights;
        make_moe_tensors(g, dtype, weights);
        for (const int rows : {1, 64}) {
            auto input = upload(pattern(rows * g.hidden, 0.15f, 5.0f), {1, rows, g.hidden}, tb::kFloat16);
            auto flat = input.reshape({rows, g.hidden});
            for (const auto kind : {Kind::relu_squared, Kind::silu_gated, Kind::gelu_gated}) {
                for (const double limit : {0.0, 0.05}) {
                    auto ffn = make_moe_ffn(g, dtype, weights, false);
                    ffn->activation = kind;
                    ffn->moe_ungated_experts = kind == Kind::relu_squared;
                    ffn->swiglu_limit = limit;
                    auto gate = upload(pattern(g.experts * g.expert_width * g.latent, 0.2f, 29.0f),
                                       {g.experts, g.expert_width, g.latent}, dtype);
                    if (!ffn->moe_ungated_experts) ffn->moe_dense_gate = gate;
                    auto selected = moe_topk_cuda(tb::matmul(flat.to(tb::kFloat32), ffn->moe_router.transpose(0, 1)),
                        g.top_k, true, false, true, false, ffn->moe_router_bias, 1e-20, 5.0);
                    auto up = dense_routed_projection_gather(ffn->moe_dense_up, flat, selected[0]);
                    Tensor gate_projection;
                    if (!ffn->moe_ungated_experts)
                        gate_projection = dense_routed_projection_gather(gate, flat, selected[0]);
                    if (kind == Kind::gelu_gated && dtype == tb::kBFloat16) {
                        // Keep the existing activation's dtype restriction.
                        auto rejects_bf16_gelu = [&](auto&& run) {
                            bool rejected = false;
                            try {
                                (void)run();
                            } catch (const std::runtime_error& error) {
                                rejected = std::string(error.what()).find("gelu_mul: dtype") != std::string::npos;
                            }
                            require(rejected, "BF16 GEGLU changed its existing dtype restriction");
                        };
                        rejects_bf16_gelu([&] { return dense_reference_activation(gate_projection, up, kind, limit); });
                        CudaExecutionContext execution;
                        rejects_bf16_gelu([&] { return ffn->forward(execution, input); });
                        continue;
                    }
                    auto hidden = dense_reference_activation(gate_projection, up, kind, limit);
                    auto down = dense_routed_projection_gather(ffn->moe_dense_down, hidden, selected[0]);
                    CudaExecutionContext execution;
                    auto shared = ffn->shared->forward(execution, flat).reshape({rows, g.hidden}).to(tb::kFloat16);
                    auto reduced = moe_weighted_reduce_cuda(down, selected[1]);
                    // The FFN returns the activation dtype whatever the bank stores.
                    auto expected = (reduced + shared.to(reduced.scalar_type())).to(input.scalar_type());
                    const auto tolerance = std::max(dense_tolerance(dtype), dense_tolerance(input.scalar_type()));
                    require_dense_close(ffn->forward(execution, input), expected, tolerance,
                                        "dense FFN activation and split banks");
                    if (!ffn->moe_ungated_experts) {
                        ffn->moe_dense_gate_up = tb::cat({gate, ffn->moe_dense_up}, 1).contiguous();
                        ffn->moe_dense_gate = Tensor();
                        ffn->moe_dense_up = Tensor();
                        require_dense_close(ffn->forward(execution, input), expected, tolerance,
                                            "dense FFN combined Gate/Up bank");
                    }
                }
            }
        }
    }
}

std::size_t dense_plan_bytes(const MoeRoutePlan& route) {
    std::set<const void*> allocations;
    std::size_t bytes = 0;
    for (const auto& tensor : {route.ids_dst, route.expert_bounds, route.tile_bounds, route.tile_experts,
                              route.mma_tile_bounds, route.mma_tile_experts, route.wide_tile_bounds,
                              route.wide_tile_experts, route.counts, route.cursors}) {
        if (tensor.defined() && tensor.numel() && allocations.insert(tensor.data_ptr()).second)
            bytes += tensor.numel() * tensor.element_size();
    }
    return bytes;
}

template <typename F>
float dense_time_ms(F&& projection) {
    auto warmup = projection();
    MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
    cudaEvent_t begin, end;
    MFQ_CUDA_CHECK(cudaEventCreate(&begin));
    MFQ_CUDA_CHECK(cudaEventCreate(&end));
    MFQ_CUDA_CHECK(cudaEventRecord(begin, mfq_current_cuda_stream()));
    for (int i = 0; i < 3; ++i) {
        auto result = projection();
    }
    MFQ_CUDA_CHECK(cudaEventRecord(end, mfq_current_cuda_stream()));
    MFQ_CUDA_CHECK(cudaEventSynchronize(end));
    float elapsed = 0.0f;
    MFQ_CUDA_CHECK(cudaEventElapsedTime(&elapsed, begin, end));
    MFQ_CUDA_CHECK(cudaEventDestroy(begin));
    MFQ_CUDA_CHECK(cudaEventDestroy(end));
    return elapsed / 3;
}

// CUDA events include allocation/launch gaps. Plan construction is reported
// separately because FFN reuses the same plan for all its dense projections.
// Workspace bounds count the plan, gathered banks, int64 indices and retained
// chunk outputs, excluding the common input/bank and the final output tensor.
void benchmark_dense_routed() {
    constexpr int experts = 64, n = 256, k = 256;
    std::cout << "dtype,tokens,routes,old_A_ms,new_ms,old_A2_ms,plan_ms,old_gather_bytes,new_gather_bytes,"
                 "old_workspace_bound_bytes,new_workspace_bytes\n";
    for (const auto dtype : {tb::kBFloat16, tb::kFloat16, tb::kFloat32}) {
        auto bank = upload(pattern(experts * n * k, 0.08f, 3.0f), {experts, n, k}, dtype);
        for (const int rows : {1, 8, 64, 512, 4096}) {
            for (const int routes : {8, 22}) {
                std::vector<float> values(rows * routes);
                for (int pair = 0; pair < rows * routes; ++pair) values[pair] = (pair * 13) % experts;
                auto ids = upload(values, {rows, routes}, tb::kFloat32).to(tb::kInt32);
                auto input = upload(pattern(rows * k, 0.15f, 7.0f), {rows, k}, dtype);
                auto route = build_moe_route_plan(ids, experts);
                const float plan_ms = dense_time_ms([&] {
                    return build_moe_route_plan(ids, experts).ids_dst;
                });
                const auto old = [&] { return dense_routed_projection_gather(bank, input, ids); };
                const auto grouped = [&] { return dense_routed_projection(bank, input, route); };
                const float before = dense_time_ms(old);
                const float after = dense_time_ms(grouped);
                const float before2 = dense_time_ms(old);
                const auto row_bytes = std::size_t(routes) * n * k * bank.element_size();
                const auto chunk = std::min<std::size_t>(rows,
                    std::max<std::size_t>(1, kDenseRoutedGatherBudgetBytes / row_bytes));
                const auto plan_bytes = dense_plan_bytes(route);
                const auto retained = rows > int(chunk) ? std::size_t(rows) * routes * n * bank.element_size() : 0;
                const auto old_workspace = plan_bytes + chunk * row_bytes +
                    chunk * routes * sizeof(int64_t) + retained;
                const char* name = dtype == tb::kFloat32 ? "F32" : (dtype == tb::kFloat16 ? "F16" : "BF16");
                std::cout << name << ',' << rows << ',' << routes << ',' << std::fixed << std::setprecision(4)
                          << before << ',' << after << ',' << before2 << ',' << plan_ms << ','
                          << rows * row_bytes << ",0," << old_workspace << ',' << plan_bytes << '\n';
                std::cout.flush();
            }
        }
    }
}

// Run each case in a fresh process: the native tensor allocator's high-water
// mark is cumulative. Extra peak includes the output and route plan, and
// excludes the pre-existing bank/input/IDs and CUDA-library allocations.
void measure_dense_workspace(const std::string& dtype_name, int rows, int routes, bool grouped) {
    require(rows > 0 && routes > 0, "workspace measurement needs positive tokens/routes");
    require(dtype_name == "BF16" || dtype_name == "F16" || dtype_name == "F32", "unknown bank dtype");
    const auto dtype = dtype_name == "BF16" ? tb::kBFloat16 :
        (dtype_name == "F16" ? tb::kFloat16 : tb::kFloat32);
    constexpr int experts = 64, n = 256, k = 256;
    auto bank = upload(pattern(experts * n * k, 0.08f, 3.0f), {experts, n, k}, dtype);
    std::vector<float> values(rows * routes);
    for (int pair = 0; pair < rows * routes; ++pair) values[pair] = (pair * 13) % experts;
    auto ids = upload(values, {rows, routes}, tb::kFloat32).to(tb::kInt32);
    auto input = upload(pattern(rows * k, 0.15f, 7.0f), {rows, k}, dtype);
    MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
    const auto context = mfq::cuda::default_context(bank.get_device());
    const auto before = context->memory_stats();
    require(before.peak == before.allocated, "workspace measurement requires a fresh allocator peak");
    auto route = build_moe_route_plan(ids, experts);
    auto output = grouped ? dense_routed_projection(bank, input, route) :
        dense_routed_projection_gather(bank, input, ids);
    MFQ_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
    const auto extra_peak = context->memory_stats().peak - before.allocated;
    const auto output_bytes = output.numel() * output.element_size();
    std::cout << dtype_name << ',' << rows << ',' << routes << ',' << (grouped ? "grouped" : "gather")
              << ',' << extra_peak << ',' << output_bytes << '\n';
}
