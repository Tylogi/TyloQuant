static void check_kv_bounds() {
    namespace tb = mfq_tensor_backend;
    const auto options = tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    auto cache = HotKvCache::allocate(1, 8, 2, 4, 4, options);
    auto cold = tb::zeros({1, 1, 6, 4}, options);
    bool rejected = false;
    try { cache.set_cold_storage(cold, cold.narrow(2, 0, 5)); }
    catch (const std::exception& error) { rejected = std::string(error.what()).find("capacity") != std::string::npos; }
    check(rejected, "exclusive KV accepted mismatched physical tier capacities");
    cache.set_cold_storage(cold, cold.clone());
    auto keys = tb::ones({1, 1, 1, 4}, options);
    cache.free_cold = cache.free_cold.narrow(0, 0, 5);
    rejected = false;
    try { cache.append(keys, keys, tb::zeros({1}, options.dtype(tb::kInt64))); }
    catch (const std::exception& error) { rejected = std::string(error.what()).find("capacity") != std::string::npos; }
    check(rejected, "exclusive KV launched with an undersized allocation table");
    cache.set_cold_storage(cold, cold.clone());
    for (int64_t position : {-1, 8}) {
        rejected = false;
        try {
            cache.append(keys, keys, tb::full({1}, position, options.dtype(tb::kInt64)));
            (void)cache.mapping.cpu();
        } catch (const std::out_of_range& error) {
            rejected = std::string(error.what()).find("exclusive KV") != std::string::npos;
        }
        check(rejected, "exclusive KV did not reject an invalid logical position");
        auto map = cache.mapping.cpu();
        for (int i = 0; i < 8; ++i) check(map.data_ptr<int32_t>()[i] == -1, "invalid append changed mapping");
    }
    cache.append(keys, keys, tb::zeros({1}, options.dtype(tb::kInt64)));
    auto stored = cache.gather(1).first.cpu();
    check(stored.data_ptr<float>()[0] == 1, "exclusive KV did not recover after a bounds failure");
    std::cout << "exclusive_kv_bounds host_capacity=1 position_bounds=1 recovery=1 PASS\n";
}
