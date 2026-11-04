static void check_index_bounds() {
    using namespace mfq::cuda;
    for (auto dtype : {kInt32, kInt64}) for (int64_t bad : {-1, 3}) {
        for (int operation = 1; operation <= 7; ++operation) {
            auto source = ones({3, 4}, TensorOptions{}.device(kCUDA));
            auto destination = zeros_like(source);
            auto indices = tensor<int64_t>({bad}).to(kCUDA).to(dtype);
            bool rejected = false;
            try {
                Tensor result;
                switch (operation) {
                case 1: result = source.index_select(0, indices); break; // vectorized row path
                case 2: result = source.transpose(0, 1).index_select(1, indices); break;
                case 3: result = source.gather(0, indices.reshape({1, 1})); break;
                case 4: destination.scatter_(0, indices.reshape({1, 1}), source.narrow(0, 0, 1).narrow(1, 0, 1)); break;
                case 5: destination.index_copy_(0, indices, source.narrow(0, 0, 1)); break;
                case 6: destination.index_fill_(0, indices, 99); break;
                case 7: destination.scatter_add_(0, indices.reshape({1, 1}),
                    source.narrow(0, 0, 1).narrow(1, 0, 1)); break;
                }
                if (result.defined()) (void)result.cpu();
                else (void)source.cpu();
            } catch (const std::out_of_range& error) {
                rejected = std::string(error.what()).find("out of range") != std::string::npos;
            }
            require(rejected, "invalid tensor index did not report a recoverable bounds error");
            require(host_values(destination) == std::vector<float>(12, 0.f), "invalid index wrote destination");
            require(host_values(source.index_select(0, tensor<int64_t>({2}).to(kCUDA))) ==
                std::vector<float>(4, 1.f), "bounds failure poisoned the CUDA context");
        }
    }
    auto wide = ones({3, 4}, TensorOptions{}.device(kCUDA));
    bool overflow_rejected = false;
    try {
        auto index = tensor<int64_t>({std::numeric_limits<int64_t>::max()}).to(kCUDA);
        (void)wide.index_select(0, index).cpu();
    } catch (const std::out_of_range&) { overflow_rejected = true; }
    require(overflow_rejected, "index addressing overflowed before the bounds check");
    // Values change at replay time, after host shape validation/capture.
    auto source = ones({3, 4}, TensorOptions{}.device(kCUDA));
    auto indices = tensor<int64_t>({0}).to(kCUDA);
    Graph graph;
    graph.prepare_memory();
    auto result = source.index_select(0, indices);
    (void)source.cpu();
    result = {};
    graph.capture_begin();
    result = source.index_select(0, indices);
    graph.capture_end();
    indices.fill_(3);
    graph.replay();
    bool rejected = false;
    try { (void)result.cpu(); } catch (const std::out_of_range&) { rejected = true; }
    require(rejected, "graph replay bypassed index bounds validation");
    indices.fill_(2);
    graph.replay();
    require(host_values(result) == std::vector<float>(4, 1.f), "graph did not recover after an invalid index");
    std::cout << "index_bounds negative/upper FP32 int32/int64 "
                 "vectorized/strided/gather/scatter/copy/fill/graph PASS\n";
}
