#pragma once
#include <filesystem>
#include <chrono>

Tensor qsa_pool_reference(const Tensor& raw,const Tensor& weight,const Tensor& positions,
        const RotaryEmbedding& rotary,int64_t tokens,int64_t pool) {
    const auto groups=tokens/pool;
    auto mean=raw.narrow(1,0,groups*pool).reshape({raw.size(0),groups,pool,raw.size(2)})
        .to(tb::kFloat32).mean(2).to(raw.scalar_type());
    auto starts=tb::arange(groups,positions.options().dtype(tb::kInt64))*pool;
    return rotary.forward(attention_ops::rms_norm(mean,weight+1,1e-6).unsqueeze(1),
        positions.index_select(-1,starts)).squeeze(1);
}

void qsa_incremental_pool_case() {
    int cases=0;
    int64_t processed=0,recomputed=0,device_rows=0;
    const bool fused_store=runtime_options::qsa_pool_store_fused();
    {
        CudaExecutionContext execution;
        constexpr int maximum=64,hidden=32,heads=4,kv=2,width=32,iw=8,ih=2,pool=4,budget=16;
        auto rotary=std::make_shared<RotaryEmbedding>(8,maximum,10000.);
        auto norm=data({width},1,0),inorm=data({iw},1,0);
        QsaWeights weights{linear(hidden,heads*2*width,1),linear(hidden,kv*width,2),linear(hidden,kv*width,3),
            linear(heads*width,hidden,4),linear(hidden,(ih+1)*iw,5),norm,norm,inorm,inorm};
        QsaConfig cfg{heads,kv,width,ih,iw,pool,budget,maximum,1e-6};
        auto memory=std::make_shared<KvMemoryManager>(KvMemoryOptions{});
        Qsa actual(weights,cfg,rotary,true,true,{},memory),oracle(weights,cfg,rotary);
        memory->finalize();
        actual.activate_memory();
        const auto* pooled_address=actual.pooled_cache().data_ptr();
        auto inputs=data({1,maximum,hidden},17,.4f);
        auto positions=tb::arange(maximum,inputs.options().dtype(tb::kInt64));
        // The first short prefill enters graph execution with no position history.
        const void* positions_address=nullptr;
        Tensor occupied;
        for(int reuse=0;reuse<2;++reuse) {
            actual.prepare_graph({},maximum);
            // Retained graphs capture the pooled position buffer by address, so
            // a later request must reuse it even when the allocator could not.
            const auto pooled_positions=actual.resident_cache_tensors().front();
            if(!reuse)positions_address=pooled_positions.data_ptr();
            else if(pooled_positions.data_ptr()!=positions_address)
                throw std::runtime_error("managed QSA pooled positions moved between requests");
            for(int tokens=0;tokens<24;++tokens) {
                auto x=inputs.narrow(1,tokens,1),pos=positions.narrow(0,tokens,1);
                Tensor output;
                DecodeWindow window(mfq_current_cuda_stream());
                window.capture([&] {output=actual.forward_graph(execution,x,pos,pos);},[]{},[&] {output={};});
                window.run();
                actual.advance_graph(1);
                equal(output,oracle.forward(execution,x,pos,positions.narrow(0,0,tokens+1),true),
                    "managed QSA empty-prefix graph output");
                if(actual.pooled_groups()!=(tokens+1)/pool || actual.pooled_cache().data_ptr()!=pooled_address)
                    throw std::runtime_error("managed QSA pooling lost its watermark or preallocated storage");
                if(tokens+1>=pool && tokens+1>budget)
                    equal(actual.pooled_cache().narrow(1,0,(tokens+1)/pool),
                        oracle.pooled_cache().narrow(1,0,(tokens+1)/pool),"managed QSA graph pooled storage");
                ++cases;
            }
            actual.reset();
            oracle.reset();
            // Hold a block of the same size so a freed buffer cannot return here.
            occupied=tb::empty({maximum},inputs.options().dtype(tb::kInt64));
        }
    }
    for(int width:{8,64,256,512})for(auto dtype:{tb::kFloat16,tb::kFloat32})
    for(int axes:{1,3})for(int batch:{1,2}) {
        constexpr int maximum=128,pool=4;
        auto raw=data({batch,maximum,width+8},13,.4f).to(dtype).narrow(2,0,width);
        auto weight=data({width},19,.1f).to(tb::kFloat32);
        std::vector<int64_t> p(axes*batch*maximum);
        for(int a=0;a<axes;++a)for(int b=0;b<batch;++b)for(int t=0;t<maximum;++t)
            p[(a*batch+b)*maximum+t]=(t*t+3*t+17*a+5*b)%maximum;
        auto positions=tb::tensor(p).reshape({axes,batch,maximum}).to(tb::kCUDA);
        RotaryEmbedding rotary(8,maximum,10000.,axes==3?std::vector<int64_t>{2,1,1}:
            std::vector<int64_t>{},axes==3,true);
        PooledCache cache;
        auto counter=tb::zeros({1},positions.options());
        if(width<=256 && fused_store)cache.count_group_rows(counter);
        int64_t tokens=0;
        for(int chunk:{3,1,2,7,16,5,31}) {
            tokens+=chunk;
            auto result=cache.update(rotary,raw.narrow(1,0,tokens),weight,
                positions.narrow(-1,0,tokens),tokens,pool,1e-6);
            if(tokens>=pool)equal(result,qsa_pool_reference(raw,weight,positions,rotary,tokens,pool),
                "incremental pooled prefix");
            if(cache.completed()!=tokens/pool || cache.processed_groups()!=tokens/pool)
                throw std::runtime_error("incremental pooling processed old or incomplete groups");
            recomputed+=tokens/pool;
            ++cases;
        }
        auto snapshot=cache.capture(tokens,pool);
        if(width<=256 && fused_store) {
            const auto actual=counter.item<int64_t>();
            if(actual!=batch*(tokens/pool))throw std::runtime_error("kernel processed extra group rows");
            device_rows+=actual;
        }
        cache.truncate(13,pool);
        const auto before=cache.processed_groups();
        // Change the discarded suffix so stale cached groups cannot pass.
        auto replacement=raw.clone();
        replacement.narrow(1,13,maximum-13).copy_(raw.narrow(1,13,maximum-13)*-.7);
        auto result=cache.update(rotary,replacement,weight,positions,49,pool,1e-6);
        equal(result,qsa_pool_reference(replacement,weight,positions,rotary,49,pool),"pool truncate replay");
        if(cache.processed_groups()-before!=49/pool-13/pool)
            throw std::runtime_error("pool truncate watermark differs");
        cache.reset();
        cache.restore(snapshot,tokens,pool);
        result=cache.update(rotary,raw,weight,positions,73,pool,1e-6);
        equal(result,qsa_pool_reference(raw,weight,positions,rotary,73,pool),"pool session restore");
        if(cache.processed_groups()!=73/pool-tokens/pool)
            throw std::runtime_error("restored pool recomputed a saved prefix");
        processed+=cache.processed_groups();
        cache.reset();
        result=cache.update(rotary,replacement,weight,positions,21,pool,1e-6);
        equal(result,qsa_pool_reference(replacement,weight,positions,rotary,21,pool),"pool cancel and reuse");
        cases+=3;
    }
    // Captured graphs replay with changing, non-affine three-axis positions.
    for(int rows:{1,2,6}) {
        constexpr int maximum=128,pool=4,width=64;
        auto raw=data({1,maximum,width},13,.4f),weight=data({width},19,.1f).to(tb::kFloat32);
        std::vector<int64_t> p(3*maximum);
        for(int a=0;a<3;++a)for(int t=0;t<maximum;++t)p[a*maximum+t]=(t*t+11*a)%maximum;
        auto positions=tb::tensor(p).reshape({3,1,maximum}).to(tb::kCUDA);
        RotaryEmbedding rotary(8,maximum,10000.,{2,1,1},true,true);
        PooledCache cache;
        auto counter=tb::zeros({1},positions.options());
        cache.count_group_rows(counter);
        int64_t tokens=7;
        cache.reserve(raw,maximum/pool);
        cache.update(rotary,raw,weight,positions,tokens,pool,1e-6);
        cache.prepare_graph(positions.narrow(-1,0,tokens),maximum);
        auto cp=tb::zeros({rows},positions.options()),pos=tb::zeros({3,1,rows},positions.options());
        const auto upload=[&] {
            cp.copy_(tb::arange(tokens,tokens+rows,positions.options()));
            pos.copy_(positions.narrow(-1,tokens,rows));
        };
        upload();
        DecodeWindow window(mfq_current_cuda_stream());
        window.capture([&] {cache.update_graph(rotary,raw,weight,pos,cp,pool,1e-6);},[]{},[]{});
        counter.zero_();
        for(int replay=0;replay<8;++replay) {
            upload();
            window.run();
            tokens+=rows;
            cache.advance_graph(tokens,pool);
            equal(cache.storage().narrow(1,0,tokens/pool),
                qsa_pool_reference(raw,weight,positions,rotary,tokens,pool),"graph completed pooling MRoPE");
            if(cache.processed_groups()!=tokens/pool)throw std::runtime_error("graph pool group count differs");
            ++cases;
        }
        const auto actual=counter.item<int64_t>();
        if(fused_store && actual!=tokens/pool-7/pool)
            throw std::runtime_error("graph kernel pooled an incomplete or repeated group");
        device_rows+=actual;
        cache.truncate(13,pool);
        tokens=13;
        upload();
        window.run();
        tokens+=rows;
        cache.advance_graph(tokens,pool);
        equal(cache.storage().narrow(1,0,tokens/pool),qsa_pool_reference(raw,weight,positions,rotary,tokens,pool),
            "graph pool truncate and replay");
        ++cases;
    }
    {
        CudaExecutionContext execution;
        constexpr int maximum=128,hidden=32,heads=4,kv=2,width=32,iw=8,ih=2,pool=4,budget=16;
        auto rotary=std::make_shared<RotaryEmbedding>(8,maximum,10000.,std::vector<int64_t>{2,1,1},true,true);
        auto norm=data({width},1,0),weight=data({iw},1,0);
        QsaWeights weights{linear(hidden,heads*2*width,1),linear(hidden,kv*width,2),linear(hidden,kv*width,3),
            linear(heads*width,hidden,4),linear(hidden,(ih+1)*iw,5),norm,norm,weight,weight};
        QsaConfig cfg{heads,kv,width,ih,iw,pool,budget,maximum,1e-6};
        Qsa actual(weights,cfg,rotary),oracle(weights,cfg,rotary);
        oracle.set_incremental_pooling(false);
        auto inputs=data({1,maximum,hidden},17,.4f);
        std::vector<int64_t> p(3*maximum);
        for(int a=0;a<3;++a)for(int t=0;t<maximum;++t)p[a*maximum+t]=(t*t+11*a)%maximum;
        auto positions=tb::tensor(p).reshape({3,1,maximum}).to(tb::kCUDA);
        int64_t tokens=0;
        const auto eager=[&](int count) {
            auto pos=positions.narrow(-1,tokens,count),full=positions.narrow(-1,0,tokens+count);
            auto x=inputs.narrow(1,tokens,count);
            equal(actual.forward(execution,x,pos,full,true),oracle.forward(execution,x,pos,full,true),
                "QSA incremental eager MRoPE");
            tokens+=count;
            ++cases;
        };
        for(int count:{3,1,7,17,5})eager(count);
        if(actual.pooled_processed_groups()!=tokens/pool)
            throw std::runtime_error("QSA eager adapter repeated old groups");
        Qwen4BlockSessionState state;
        std::size_t bytes=0;
        actual.capture_session(state,tokens,bytes);
        actual.reset();
        actual.restore_session(state,tokens);
        if(actual.pooled_groups()!=state.pooled.completed)throw std::runtime_error("QSA session lost watermark");
        eager(4);
        if(actual.pooled_processed_groups()!=1)throw std::runtime_error("QSA restored pooling repeated prefix");
        actual.prepare_graph(positions.narrow(-1,0,tokens),maximum);
        for(int count:{1,3,6}) {
            auto pos=positions.narrow(-1,tokens,count);
            auto cp=tb::arange(tokens,tokens+count,positions.options());
            auto x=inputs.narrow(1,tokens,count);
            Tensor output;
            DecodeWindow window(mfq_current_cuda_stream());
            window.capture([&] {
                output=actual.forward_graph(execution,x,pos,cp);
            },[]{},[&] {
                output={};
            });
            window.run();
            actual.advance_graph(count);
            equal(output,oracle.forward(execution,x,pos,positions.narrow(-1,0,tokens+count),true),
                "QSA prefill to graph MRoPE");
            tokens+=count;
            equal(actual.pooled_cache().narrow(1,0,tokens/pool),oracle.pooled_cache().narrow(1,0,tokens/pool),
                "QSA graph pooled cache");
            ++cases;
        }
        actual.leave_graph();
        actual.truncate(13);
        oracle.truncate(13);
        tokens=13;
        eager(8);
        actual.reset();
        oracle.reset();
        tokens=0;
        eager(21);
        actual.capture_session(state,tokens,bytes);
        TextSessionState saved;
        saved.cache_pos=tokens;
        saved.tokens.resize(tokens,1);
        saved.qwen4_positions=positions.narrow(-1,0,tokens);
        saved.payload=std::vector<Qwen4BlockSessionState>{state};
        std::vector<CudaPagedPayload> blocks;
        for(std::size_t i=0;i<(saved.tokens.size()+7)/8;++i)
            blocks.push_back(encode_cuda_qwen4_paged_block(saved,8,i));
        auto decoded=decode_cuda_qwen4_paged_session(blocks,saved.tokens,8);
        auto restored=std::get<std::vector<Qwen4BlockSessionState>>(decoded.payload).front();
        equal(restored.pooled.values,state.pooled.values,"paged pooled cache snapshot",0,0);
        if(restored.pooled.completed!=state.pooled.completed)throw std::runtime_error("paged watermark differs");
        actual.reset();
        actual.restore_session(restored,tokens);
        eager(4);
        actual.capture_session(state,tokens,bytes);
        // Encode the old schema independently: it has no pooled value or watermark fields.
        CudaPagedWriter legacy;
        constexpr std::array<uint8_t,8> magic{'M','F','Q','C','U','D','2',0};
        legacy.raw(magic.data(),magic.size());
        legacy.scalar<uint32_t>(1);
        legacy.scalar<uint32_t>(0);
        legacy.scalar<uint32_t>(uint32_t(tokens));
        legacy.scalar<uint32_t>(1);
        legacy.scalar<uint32_t>(1);
        for(const auto* tensor:{&state.keys,&state.values,&state.index}) {
            legacy.scalar<uint32_t>(1);
            legacy.tensor(*tensor);
        }
        for(int field=0;field<3;++field)legacy.scalar<uint32_t>(0);
        legacy.scalar<int64_t>(0);
        legacy.scalar<uint32_t>(0);
        legacy.scalar<uint32_t>(1);
        legacy.tensor(positions.narrow(-1,0,tokens));
        legacy.scalar<uint32_t>(0);
        auto old=decode_cuda_qwen4_paged_session({std::move(legacy).finish()},std::vector<int64_t>(tokens,1),maximum);
        restored=std::get<std::vector<Qwen4BlockSessionState>>(old.payload).front();
        actual.reset();
        actual.restore_session(restored,tokens);
        if(actual.pooled_groups()!=0)throw std::runtime_error("uncached session restored a stale watermark");
        eager(4);
    }
    std::cout<<"QSA incremental pool cases="<<cases<<" device_group_rows="<<device_rows
        <<" restored_new_groups="<<processed
        <<" full_recompute_groups="<<recomputed<<" boundary/MRoPE/truncate/session/graph/cancel/reuse PASS\n";
}

void qsa_incremental_pool_benchmark() {
    constexpr int maximum=16384,pool=32,iw=128,hidden=64,width=64,heads=4,kv=2,ih=2,budget=256;
    const auto stream=mfq_current_cuda_stream();
    auto raw=data({1,maximum,iw},13,.4f),weight=data({iw},19,.1f).to(tb::kFloat32);
    auto positions=tb::arange(maximum,raw.options().dtype(tb::kInt64));
    auto rotary=std::make_shared<RotaryEmbedding>(64,maximum,10000.);
    auto inputs=data({1,maximum,hidden},17,.4f),norm=data({width},1,0);
    QsaWeights weights{linear(hidden,heads*2*width,1),linear(hidden,kv*width,2),linear(hidden,kv*width,3),
        linear(heads*width,hidden,4),linear(hidden,(ih+1)*iw,5),norm,norm,weight,weight};
    QsaConfig cfg{heads,kv,width,ih,iw,pool,budget,maximum,1e-6};
    const std::string old=std::getenv("MFQ_QSA_POOL_WRITE_FUSED")?std::getenv("MFQ_QSA_POOL_WRITE_FUSED"):"";
    const auto fusion=[](const char* value) {
#ifdef _WIN32
        _putenv_s("MFQ_QSA_POOL_WRITE_FUSED",value);
#else
        if(*value)setenv("MFQ_QSA_POOL_WRITE_FUSED",value,1);
        else unsetenv("MFQ_QSA_POOL_WRITE_FUSED");
#endif
    };
    const auto measure=[&](auto&& run) {
        cudaEvent_t begin,end;
        MFQ_CUDA_CHECK(cudaEventCreate(&begin));
        MFQ_CUDA_CHECK(cudaEventCreate(&end));
        MFQ_CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto start=std::chrono::steady_clock::now();
        MFQ_CUDA_CHECK(cudaEventRecord(begin,stream));
        run();
        MFQ_CUDA_CHECK(cudaEventRecord(end,stream));
        MFQ_CUDA_CHECK(cudaEventSynchronize(end));
        const auto wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        float ms=0;
        MFQ_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));
        MFQ_CUDA_CHECK(cudaEventDestroy(begin));
        MFQ_CUDA_CHECK(cudaEventDestroy(end));
        return std::pair<float,double>{ms,wall};
    };
    for(int chunk:{512,2048})for(int arm:{0,1,0}) {
        const bool incremental=arm==1;
        fusion(incremental?"1":"0");
        PooledCache cache;
        cache.set_incremental(incremental);
        // Warm the same geometry and parameter preparation before timing.
        cache.update(*rotary,raw,weight,positions,maximum,pool,1e-6);
        cache.reset();
        Tensor result;
        const auto pooling=measure([&] {
            for(int repeat=0;repeat<10;++repeat) {
                cache.reset();
                for(int end=chunk;end<=maximum;end+=chunk)
                    result=cache.update(*rotary,raw,weight,positions,end,pool,1e-6);
            }
        });
        equal(result,qsa_pool_reference(raw,weight,positions,*rotary,maximum,pool),"pool benchmark prefix");
        CudaExecutionContext execution;
        Qsa model(weights,cfg,rotary);
        model.set_incremental_pooling(incremental);
        model.forward(execution,inputs.narrow(1,0,chunk),positions.narrow(0,0,chunk),
            positions.narrow(0,0,chunk),true);
        model.reset();
        const auto prefill=measure([&] {
            for(int start=0;start<maximum;start+=chunk)
                result=model.forward(execution,inputs.narrow(1,start,chunk),positions.narrow(0,start,chunk),
                    positions.narrow(0,0,start+chunk),true);
        });
        std::cout<<"QSA pooling benchmark tokens="<<maximum<<" chunk="<<chunk<<" incremental="<<incremental
            <<" pooling_ms="<<pooling.first/10<<" pooling_wall_ms="<<pooling.second/10
            <<" groups="<<cache.processed_groups()<<" prefill_ms="<<prefill.first
            <<" prefill_wall_ms="<<prefill.second<<" dtype=fp16 pool=32 index_width=128 heads=4 kv_heads=2\n";
    }
    fusion(old.c_str());
}


void qsa_scalar_history_chain_case() {
    int cases=0,scalar_replays=0,chain_replays=0;
    for(int mode:{0,1,2})for(int width:{32,256}) {
        CudaExecutionContext execution;
        constexpr int hidden=32,heads=4,kv=2,iw=8,ih=2,pool=4,maximum=128,budget=16;
        auto rotary=std::make_shared<RotaryEmbedding>(8,maximum,10000.);
        auto norm=data({width},1,0),inorm=data({iw},1,0);
        QsaWeights weights{linear(hidden,heads*2*width,1),linear(hidden,kv*width,2),linear(hidden,kv*width,3),
            linear(heads*width,hidden,4),linear(hidden,(ih+1)*iw,5),norm,norm,inorm,inorm};
        QsaConfig cfg{heads,kv,width,ih,iw,pool,budget,maximum,1e-6};
        std::shared_ptr<KvMemoryManager> memory;
        if(mode) {
            KvMemoryOptions policy;policy.placement=mode==1?"auto":"auto_ssd";policy.budget_policy=true;
            policy.gpu_budget=2*(maximum+maximum/pool)*iw*2+8*kv*width*4;
            policy.ram_limit_set=true;policy.ram_limit=mode==1?2*maximum*kv*width*4:256;
            policy.directory=(std::filesystem::current_path()/"qsa-scalar-history-chain-test").string();
            memory=std::make_shared<KvMemoryManager>(policy);
        }
        Qsa actual(weights,cfg,rotary,true,true,{},memory),oracle(weights,cfg,rotary,true,true,{},memory);
        if(memory)memory->finalize();
        auto inputs=data({1,maximum,hidden},17,.4f);
        auto history=tb::arange(6,inputs.options().dtype(tb::kInt64));
        actual.forward(execution,inputs.narrow(1,0,6),history,history,true);
        oracle.forward(execution,inputs.narrow(1,0,6),history,history,true);
        actual.prepare_graph(history,maximum);oracle.prepare_graph(history,maximum);
        std::map<int,std::unique_ptr<Cell>> chains;
        Cell scalar(1,hidden),next_actual(1,hidden),next_oracle(1,hidden);
        const auto compare_state=[&] {
            if(!actual.supports_session())return;
            Qwen4BlockSessionState a,b;std::size_t bytes=0;
            actual.capture_session(a,actual.position(),bytes);oracle.capture_session(b,oracle.position(),bytes);
            equal(a.keys,b.keys,"scalar-chain KV keys",0,0);
            equal(a.values,b.values,"scalar-chain KV values",0,0);
            equal(a.index,b.index,"scalar-chain indexer",0,0);
        };
        const auto run=[&](int tokens) {
            const auto start=actual.position();
            auto positions=tb::arange(start,start+tokens,history.options());
            auto& cell=chains[tokens];if(!cell)cell=std::make_unique<Cell>(tokens,hidden);
            cell->x.copy_(inputs.narrow(1,start,tokens));cell->pos.copy_(positions);
            if(!cell->window.valid())cell->window.capture([&] {
                for(int row=0;row<tokens;++row)
                    actual.forward_graph(execution,cell->x.narrow(1,row,1),cell->pos.narrow(0,row,1),
                        cell->pos.narrow(0,row,1),{},true,true);
            },[]{},[]{});
            cell->window.run();actual.advance_graph(tokens);++chain_replays;
            for(int row=0;row<tokens;++row) {
                scalar.x.copy_(inputs.narrow(1,start+row,1));scalar.pos.copy_(positions.narrow(0,row,1));
                if(!scalar.window.valid())scalar.window.capture([&] {
                    oracle.forward_graph(execution,scalar.x,scalar.pos,scalar.pos,{},true,true);
                },[]{},[]{});
                scalar.window.run();oracle.advance_graph(1);++scalar_replays;
            }
            history=tb::cat({history,positions},-1);compare_state();
            const auto next=actual.position();
            auto next_positions=tb::arange(next,next+1,history.options());
            for(auto* next_cell:{&next_actual,&next_oracle}) {
                next_cell->x.copy_(inputs.narrow(1,next,1));next_cell->pos.copy_(next_positions);
            }
            if(!next_actual.window.valid())next_actual.window.capture([&] {
                next_actual.output=actual.forward_graph(execution,next_actual.x,next_actual.pos,next_actual.pos);
            },[]{},[&]{next_actual.output={};});
            if(!next_oracle.window.valid())next_oracle.window.capture([&] {
                next_oracle.output=oracle.forward_graph(execution,next_oracle.x,next_oracle.pos,next_oracle.pos);
            },[]{},[&]{next_oracle.output={};});
            next_actual.window.run();actual.advance_graph(1);
            next_oracle.window.run();oracle.advance_graph(1);
            equal(next_actual.output,next_oracle.output,"scalar-chain next-token readout",0,0);
            history=tb::cat({history,next_positions},-1);compare_state();++cases;
        };
        for(int tokens:{2,3,4,6,8,2,3})run(tokens);
        actual.truncate(13);oracle.truncate(13);history=history.narrow(0,0,13);
        run(4);run(2);
    }
    std::cout<<"QSA scalar history chain exact KV/index/readout, repeated graph, rollback, GPU/RAM/SSD PASS cases="
        <<cases<<" scalar_replays="<<scalar_replays<<" chain_replays="<<chain_replays<<'\n';
}

void qsa_history_prime_case() {
    int cases=0;
    for(int mode:{0,1,2}) {
        CudaExecutionContext execution;
        constexpr int hidden=32,heads=4,kv=2,width=32,iw=8,ih=2,pool=4,maximum=512,budget=64;
        auto rotary=std::make_shared<RotaryEmbedding>(8,maximum,10000.);
        auto norm=data({width},1,0),inorm=data({iw},1,0);
        QsaWeights weights{linear(hidden,heads*2*width,1),linear(hidden,kv*width,2),linear(hidden,kv*width,3),
            linear(heads*width,hidden,4),linear(hidden,(ih+1)*iw,5),norm,norm,inorm,inorm};
        QsaConfig cfg{heads,kv,width,ih,iw,pool,budget,maximum,1e-6};
        std::shared_ptr<KvMemoryManager> memory;
        if(mode) {
            KvMemoryOptions policy;policy.placement=mode==1?"auto":"auto_ssd";policy.budget_policy=true;
            policy.gpu_budget=(maximum+maximum/pool)*iw*2+(mode==1?maximum:4)*kv*width*4;
            policy.ram_limit_set=true;policy.ram_limit=mode==1?maximum*kv*width*4:256;
            policy.directory=(std::filesystem::current_path()/"qsa-history-prime-test").string();
            memory=std::make_shared<KvMemoryManager>(policy);
        }
        Qsa actual(weights,cfg,rotary,true,true,{},memory),oracle(weights,cfg,rotary);
        if(memory)memory->finalize();
        auto inputs=data({1,maximum,hidden},17,.4f);
        int64_t offset=0;
        for(int tokens:{17,128,257}) {
            auto positions=tb::arange(offset,offset+tokens,inputs.options().dtype(tb::kInt64));
            auto full=tb::arange(offset+tokens,positions.options());
            auto x=inputs.narrow(1,offset,tokens);
            oracle.forward(execution,x,positions,full,true);
            if(actual.forward(execution,x,positions,full,true,nullptr,{},true,true).defined())
                throw std::runtime_error("history priming computed an unused attention output");
            offset+=tokens;
            if(actual.position()!=offset)throw std::runtime_error("history priming did not advance KV");
            if(actual.supports_session()) {
                Qwen4BlockSessionState a,b;std::size_t bytes=0;
                actual.capture_session(a,offset,bytes);oracle.capture_session(b,offset,bytes);
                equal(a.keys,b.keys,"history priming keys",0,0);
                equal(a.values,b.values,"history priming values",0,0);
                equal(a.index,b.index,"history priming index",0,0);
            }
            auto next=tb::arange(offset,offset+1,positions.options());
            actual.prepare_graph(full,maximum);
            {
                auto next_x=inputs.narrow(1,offset,1);
                DecodeWindow window(mfq_current_cuda_stream());
                window.capture([&] {actual.forward_graph(execution,next_x,next,next,{},true,true);},[]{},[]{});
                window.run();actual.advance_graph(1);
            }
            oracle.forward(execution,inputs.narrow(1,offset,1),next,tb::arange(offset+1,positions.options()),true);
            actual.leave_graph();++offset;
            next=tb::arange(offset,offset+1,positions.options());
            full=tb::arange(offset+1,positions.options());x=inputs.narrow(1,offset,1);
            equal(actual.forward(execution,x,next,full,true),oracle.forward(execution,x,next,full,true),
                "history priming next-token readout");
            ++offset;++cases;
        }
        actual.truncate(65);oracle.truncate(65);
        auto positions=tb::arange(65,73,inputs.options().dtype(tb::kInt64));
        auto full=tb::arange(73,positions.options()),x=inputs.narrow(1,65,8);
        equal(actual.forward(execution,x,positions,full,true),oracle.forward(execution,x,positions,full,true),
            "history priming rollback and continuation");++cases;
    }
    std::cout<<"QSA cache-only priming cases="<<cases<<" exact KV/index and continuation oracle PASS\n";
}

void qsa_prefill_fusion_case() {
    struct Setting {
        std::string old;
        Setting(){if(const auto* p=std::getenv("MFQ_QSA_PREFILL_FUSED"))old=p;}
        static void set(const char* p) {
#ifdef _WIN32
            _putenv_s("MFQ_QSA_PREFILL_FUSED",p);
#else
            if(*p)setenv("MFQ_QSA_PREFILL_FUSED",p,1);else unsetenv("MFQ_QSA_PREFILL_FUSED");
#endif
        }
        ~Setting(){set(old.c_str());}
    } setting;
    int cases=0,grouped_calls=0;
    for(int width:{8,256})for(bool grouped:{false,true}) {
        CudaExecutionContext execution;
        constexpr int hidden=32,kv=2,iw=8,ih=2,pool=4,maximum=2048,budget=512;
        const int heads=width==256?24:4;
        auto rotary=std::make_shared<RotaryEmbedding>(std::min(width,8),maximum,10000.);
        auto norm=data({width},1,0),inorm=data({iw},1,0);
        QsaWeights weights{linear(hidden,heads*2*width,1),linear(hidden,kv*width,2),linear(hidden,kv*width,3),
            linear(heads*width,hidden,4),linear(hidden,(ih+1)*iw,5),norm,norm,inorm,inorm};
        if(grouped)weights.input_projection=[q=weights.query,k=weights.key,v=weights.value,i=weights.index_query_key,&grouped_calls]
                (CudaExecutionContext& context,const Tensor& x) {
            ++grouped_calls;return std::vector<Tensor>{q(context,x),k(context,x),v(context,x),i(context,x)};
        };
        QsaConfig cfg{heads,kv,width,ih,iw,pool,budget,maximum,1e-6};
        Qsa actual(weights,cfg,rotary),oracle(weights,cfg,rotary);
        auto inputs=data({1,maximum,hidden},17,.4f);
        int64_t offset=0;
        for(int tokens:{128,257,511,512,1}) {
            auto positions=tb::arange(offset,offset+tokens,inputs.options().dtype(tb::kInt64));
            auto history=tb::arange(offset+tokens,positions.options());
            auto x=inputs.narrow(1,offset,tokens);
            Setting::set("0");auto expected=oracle.forward(execution,x,positions,history,true);
            Setting::set("1");auto result=actual.forward(execution,x,positions,history,true);
            equal(result,expected,"QSA prefill fused projection/normalization/RoPE");
            ++cases;offset+=tokens;
        }
        actual.truncate(385);oracle.truncate(385);
        auto positions=tb::arange(385,642,inputs.options().dtype(tb::kInt64));
        auto history=tb::arange(642,positions.options());auto x=inputs.narrow(1,385,257);
        Setting::set("0");auto expected=oracle.forward(execution,x,positions,history,true);
        Setting::set("1");auto result=actual.forward(execution,x,positions,history,true);
        equal(result,expected,"QSA prefill cache truncation and replay");++cases;
    }
    if(grouped_calls!=10)throw std::runtime_error("QSA prefill grouped projections were not used for every large chunk");
    std::cout<<"QSA prefill fusion cases="<<cases<<" grouped_calls="<<grouped_calls
             <<" dense/sparse prefixes, changing chunk sizes, decode transition, cache truncation PASS\n";
}
