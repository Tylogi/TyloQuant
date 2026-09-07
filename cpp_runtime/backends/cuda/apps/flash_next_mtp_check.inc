static int run_flash_next_mtp_check(Model& model,CudaFlashNextMtp& mtp) {
    namespace tb=mfq_tensor_backend;
    MFQ_RUNTIME_CHECK(model.c.is_flash_next() && model.c.vocab_size>=8 && model.c.max_position_embeddings>=24,
        "Flash-Next MTP diagnostic requires vocab>=8 and context>=24");
    const auto width=mtp.hidden_norm.numel();
    auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64);
    auto ids=tb::tensor(std::vector<int64_t>{1,2,3,4,5,6,7},options).reshape({1,7});
    auto previous=(tb::arange(7*width,options).remainder(31).to(tb::kFloat32)*.01-.15).reshape({1,7,width});
    const auto json_tensor=[](const tb::Tensor& value) {
        auto host=value.to(tb::kFloat32).contiguous().cpu();
        return nlohmann::json{{"shape",host.sizes().vec()},
            {"data",std::vector<float>(host.data_ptr<float>(),host.data_ptr<float>()+host.numel())}};
    };
    nlohmann::json result;result["architecture"]=model.c.model_graph.backbone;
    result["previous"]=json_tensor(previous);
    for (int64_t i=0;i<int64_t(mtp.lengths.size());++i) {
        nlohmann::json row;
        mtp.reset();auto full=mtp.evaluate(model,previous,ids,i);
        row["full"]=json_tensor(full.first);row["multi"]=json_tensor(full.second);
        row["logits"]=json_tensor(model.logits_from_hidden(full.first));
        const auto saved=mtp.lengths;
        row["uncached"]=json_tensor(mtp.evaluate(model,previous,ids,i,false).first);
        MFQ_RUNTIME_CHECK(saved==mtp.lengths,"uncached MTP changed layer positions");
        mtp.reset();std::vector<tb::Tensor> pieces;
        for (auto [begin,count]:std::vector<std::pair<int64_t,int64_t>>{{0,2},{2,1},{3,4}})
            pieces.push_back(mtp.evaluate(model,previous.narrow(1,begin,count),ids.narrow(1,begin,count),i).first);
        row["chunked"]=json_tensor(tb::cat(pieces,1));
        mtp.reset();row["reset"]=json_tensor(mtp.evaluate(model,previous,ids,i).first);
        // Changing batch resets every layer, not just the selected depth.
        row["batch"]=json_tensor(mtp.evaluate(model,previous.repeat({2,1,1}),ids.repeat({2,1}),i).first);
        row["batch_reset"]=json_tensor(mtp.evaluate(model,previous,ids,i).first);
        auto pos=ids.reshape({7})+2;
        if (model.c.is_qwen4()) pos=tb::stack({pos+10,pos,pos+2,pos+4},0);
        row["axis"]=json_tensor(mtp.evaluate(model,previous,ids,i,false,pos).first);
        mtp.reset();
        for (int64_t other=0;other<int64_t(mtp.lengths.size());++other)
            if (other!=i) (void)mtp.evaluate(model,previous,ids,other);
        row["independent"]=json_tensor(mtp.evaluate(model,previous,ids,i).first);
        result["layers"].push_back(std::move(row));
    }
    model.reset(1);tb::Tensor raw;
    auto normalized=model.hidden_forward(ids,mfq_nullopt,mfq_nullopt,nullptr,mfq_nullopt,&raw);
    result["target_raw"]=json_tensor(raw);result["target_normalized"]=json_tensor(normalized);
    // Exercise the production server generation path, target rollback and
    // callback early-stop. The two architectures retain distinct head equations.
    MfqSamplingParams sampling;sampling.max_tokens=12;sampling.temperature=0;sampling.top_k=1;
    std::vector<int64_t> prompt{1,2,3,4,5,6,7},expected,generated;
    model.reset(1);auto current=ids;
    for (int i=0;i<sampling.max_tokens;++i) {
        auto next=model.next_token(current);expected.push_back(next.item<int64_t>());current=next.reshape({1,1});
    }
    const auto count=generate_qwen35_mtp_tokens(model,mtp,prompt,sampling,
        [&](int64_t token) {generated.push_back(token);return true;},{});
    MFQ_RUNTIME_CHECK(count==sampling.max_tokens && generated==expected && mtp.last_cycles>0,
        "Flash-Next MTP greedy tokens disagree with incremental target");
    result["greedy"]=generated;result["cycles"]=mtp.last_cycles;
    result["accepted"]=mtp.last_accepted;result["rejected"]=mtp.last_rejected;
    generated.clear();
    const auto stopped=generate_qwen35_mtp_tokens(model,mtp,prompt,sampling,
        [&](int64_t token) {generated.push_back(token);return generated.size()<3;},{});
    MFQ_RUNTIME_CHECK(stopped==3 && generated==std::vector<int64_t>(expected.begin(),expected.begin()+3),
        "Flash-Next MTP callback emitted extra or incorrect tokens");
    sampling.max_tokens=8;sampling.temperature=.8;sampling.top_k=16;sampling.top_p=.95;
    sampling.presence_penalty=.2;sampling.frequency_penalty=.1;sampling.repetition_penalty=1.05;sampling.seed=20260907;
    MFQ_RUNTIME_CHECK(generate_qwen35_mtp_tokens(model,mtp,prompt,sampling,[](int64_t){return true;},{})==8,
        "Flash-Next stochastic MTP failed to generate requested tokens");
    std::cout<<"flash_next_mtp_check "<<result.dump()<<'\n';return 0;
}
