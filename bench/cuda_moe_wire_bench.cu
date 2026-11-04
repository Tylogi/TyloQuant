#include "runtime/moe_pipeline.h"
#include "packed_bench_utils.h"
#include <cuda_runtime.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>

namespace {
using namespace mfq::bench;
void check(cudaError_t error){if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));}
struct Buffer {
    uint8_t* data=nullptr;std::size_t bytes;
    explicit Buffer(std::size_t n):bytes(n){check(cudaMalloc(reinterpret_cast<void**>(&data),n));}
    ~Buffer(){if(data)cudaFree(data);}
};
struct Graph {
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    ~Graph(){if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);}
};
// Retained MFQ reference from before warp-shared descriptor lookup.
__global__ void reference_scatter(const uint8_t* wire,const uint32_t* cancelled) {
    if(*cancelled)return;
    const auto* h=reinterpret_cast<const uint64_t*>(wire);const auto count=h[0];if(!count)return;
    const auto* d=h+4;const auto stride=uint64_t(blockDim.x)*gridDim.x*16;
    for(auto offset=h[1]+(uint64_t(blockIdx.x)*blockDim.x+threadIdx.x)*16;offset<h[2];offset+=stride) {
        uint64_t low=0,high=count;
        while(low<high){const auto mid=(low+high)/2;if(d[mid*3+1]<=offset)low=mid+1;else high=mid;}
        if(!low)continue;
        const auto* field=d+(low-1)*3;const auto within=offset-field[1];if(within>=field[2])continue;
        auto* output=reinterpret_cast<uint8_t*>(field[0])+within;
        if((reinterpret_cast<uint64_t>(output)&15)==0 && field[2]-within>=16)
            *reinterpret_cast<uint4*>(output)=*reinterpret_cast<const uint4*>(wire+offset);
        else for(uint64_t byte=0;byte<16 && within+byte<field[2];++byte)output[byte]=wire[offset+byte];
    }
}
struct Layout {int layer;std::vector<std::size_t> sizes;};
struct HostBuffer {
    uint8_t* data=nullptr;void* alias=nullptr;std::size_t bytes;
    explicit HostBuffer(std::size_t n):bytes(n) {
        check(cudaHostAlloc(reinterpret_cast<void**>(&data),n,cudaHostAllocMapped));
        const auto status=cudaHostGetDevicePointer(&alias,data,0);
        if(status!=cudaSuccess){cudaFreeHost(data);data=nullptr;check(status);}
    }
    ~HostBuffer(){if(data)cudaFreeHost(data);}
};
std::vector<Layout> read_layouts(const char* path) {
    std::ifstream file(path);if(!file)throw std::runtime_error("missing wire field layout");
    std::vector<Layout> result;std::string line;
    while(std::getline(file,line)) {
        std::istringstream input(line);Layout row{};std::size_t count;
        if(!(input>>row.layer>>count) || !count)throw std::runtime_error("invalid wire layout header");
        row.sizes.resize(count);for(auto& size:row.sizes)if(!(input>>size) || !size)throw std::runtime_error("invalid field size");
        result.push_back(std::move(row));
    }
    return result;
}
void bench(const Layout& layout,int alignment,const cudaDeviceProp& device) {
    const auto count=layout.sizes.size();const auto header=(32+24*count+15)&~std::size_t(15);
    std::size_t wire_bytes=header,output_bytes=0,payload=0;
    std::vector<std::size_t> source_offsets,destination_offsets;
    for(std::size_t i=0;i<count;++i) {
        wire_bytes=(wire_bytes+15)&~std::size_t(15);source_offsets.push_back(wire_bytes);wire_bytes+=layout.sizes[i];
        output_bytes=(output_bytes+15)&~std::size_t(15);destination_offsets.push_back(output_bytes+alignment);
        output_bytes+=alignment+layout.sizes[i]+32;payload+=layout.sizes[i];
    }
    wire_bytes=(wire_bytes+15)&~std::size_t(15);output_bytes=(output_bytes+15)&~std::size_t(15);
    // Rotating fields exceed four queried L2 capacities, including no model
    // tensors. The complete retained batch is timed inside captured graphs.
    const int banks=std::max(1,int((std::size_t(device.l2CacheSize)*4+payload-1)/payload));
    Buffer wires(wire_bytes*banks),outputs(output_bytes*banks),cancelled(4);
    std::vector<uint8_t> host(wires.bytes),expected(outputs.bytes,0xa7),actual(outputs.bytes);
    const auto make_host=[&](int step,bool empty) {
        std::fill(expected.begin(),expected.end(),0xa7);
        for(int bank=0;bank<banks;++bank) {
            auto* wire=host.data()+wire_bytes*bank;auto* words=reinterpret_cast<uint64_t*>(wire);
            words[0]=empty?0:count;words[1]=header;words[2]=wire_bytes;words[3]=0;
            for(std::size_t f=0;f<count;++f) {
                words[4+3*f]=reinterpret_cast<uint64_t>(outputs.data+output_bytes*bank+destination_offsets[f]);
                words[5+3*f]=source_offsets[f];words[6+3*f]=layout.sizes[f];
                for(std::size_t i=0;i<layout.sizes[f];++i) {
                    const auto value=uint8_t(i*29+f*71+bank*17+step*13);
                    wire[source_offsets[f]+i]=value;
                    if(!empty)expected[output_bytes*bank+destination_offsets[f]+i]=value;
                }
            }
        }
    };
    int old_blocks=0,old_threads=0,blocks=0,threads=0;
    check(cudaOccupancyMaxPotentialBlockSize(&old_blocks,&old_threads,reference_scatter,0,0));
    check(mfq::cuda::moe_wire_launch_geometry(&blocks,&threads));
    std::array<Graph,2> graphs;
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    for(int mode=0;mode<2;++mode) {
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        for(int bank=0;bank<banks;++bank) {
            const auto* wire=wires.data+wire_bytes*bank;
            if(mode)mfq::cuda::moe_scatter_wire(wire,reinterpret_cast<uint32_t*>(cancelled.data),blocks,threads,stream);
            else reference_scatter<<<old_blocks,old_threads,0,stream>>>(wire,reinterpret_cast<uint32_t*>(cancelled.data));
        }
        check(cudaStreamEndCapture(stream,&graphs[mode].graph));
        check(cudaGraphInstantiate(&graphs[mode].executable,graphs[mode].graph,0));
    }
    int exact=0;
    for(int step=0;step<4;++step)for(int mode=0;mode<2;++mode) {
        const bool empty=step==2,abort=step==3;make_host(step,empty);
        if(abort)std::fill(expected.begin(),expected.end(),0xa7);
        check(cudaMemcpyAsync(wires.data,host.data(),host.size(),cudaMemcpyHostToDevice,stream));
        check(cudaMemsetAsync(outputs.data,0xa7,outputs.bytes,stream));check(cudaMemsetAsync(cancelled.data,abort?1:0,4,stream));
        check(cudaGraphLaunch(graphs[mode].executable,stream));check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(actual.data(),outputs.data,outputs.bytes,cudaMemcpyDeviceToHost));
        if(actual!=expected) {
            const auto mismatch=std::mismatch(actual.begin(),actual.end(),expected.begin()).first-actual.begin();
            throw std::runtime_error("wire copy differs: mode="+std::to_string(mode)+" step="+std::to_string(step)+
                " layer="+std::to_string(layout.layer)+" alignment="+std::to_string(alignment)+" byte="+std::to_string(mismatch));
        }
        ++exact;
    }
    make_host(0,false);check(cudaMemcpyAsync(wires.data,host.data(),host.size(),cudaMemcpyHostToDevice,stream));check(cudaMemsetAsync(cancelled.data,0,4,stream));
    cudaEvent_t begin,end;check(cudaEventCreate(&begin));check(cudaEventCreate(&end));
    const auto measure=[&](int mode,int repetitions) {
        check(cudaEventRecord(begin,stream));for(int i=0;i<repetitions;++i)check(cudaGraphLaunch(graphs[mode].executable,stream));
        check(cudaEventRecord(end,stream));check(cudaEventSynchronize(end));float ms;check(cudaEventElapsedTime(&ms,begin,end));
        return double(ms)*1000/(repetitions*banks);
    };
    for(int mode=0;mode<2;++mode)measure(mode,1);
    int repetitions=1;while(measure(0,repetitions)*repetitions*banks<20000)repetitions*=2;
    std::array<std::vector<double>,2> samples;
    for(int sample=0;sample<7;++sample)for(int order=0;order<2;++order) {
        const int mode=(sample+order)%2;samples[mode].push_back(measure(mode,repetitions));
    }
    check(cudaEventDestroy(begin));check(cudaEventDestroy(end));
    check(cudaStreamDestroy(stream));
    std::cout<<std::setprecision(10)<<"{\"layer\":"<<layout.layer<<",\"fields\":"<<count<<",\"payload_bytes\":"<<payload
        <<",\"destination_offset_mod16\":"<<alignment<<",\"banks\":"<<banks<<",\"exact_cases\":"<<exact
        <<",\"whole_model_loads\":0";timing(samples[0],samples[1]);std::cout<<"}"<<std::endl;
}
void mapped_bench(const Layout& layout,int alignment,const cudaDeviceProp& device) {
    const auto count=layout.sizes.size(),header=(32+24*count+15)&~std::size_t(15);
    const auto table_bytes=32+count*sizeof(mfq::cuda::MoeMappedCopyDescriptor);
    std::size_t span=0,output_span=0,payload=0;
    std::vector<std::size_t> offsets,destinations;
    for(auto size:layout.sizes) {
        span=(span+15)&~std::size_t(15);offsets.push_back(span);span+=size;
        output_span=(output_span+15)&~std::size_t(15);destinations.push_back(output_span+alignment);
        output_span+=alignment+size+32;payload+=size;
    }
    span=(span+15)&~std::size_t(15);output_span=(output_span+15)&~std::size_t(15);
    const int banks=std::max(1,int((std::size_t(device.l2CacheSize)*4+payload-1)/payload));
    HostBuffer input(span*banks),headers(header*banks),tables(table_bytes*banks);
    Buffer wire((header+span)*banks),output(output_span*banks),abort(4);
    std::vector<uint8_t> expected(output.bytes),actual(output.bytes);
    int blocks=0,threads=0,copy_blocks=0,copy_threads=0;
    check(mfq::cuda::moe_wire_launch_geometry(&blocks,&threads));
    check(mfq::cuda::moe_mapped_copy_launch_geometry(&copy_blocks,&copy_threads));
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    Graph graph;
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    for(int bank=0;bank<banks;++bank)mfq::cuda::moe_copy_mapped(
        static_cast<uint8_t*>(tables.alias)+table_bytes*bank,reinterpret_cast<uint32_t*>(abort.data),copy_blocks,copy_threads,stream);
    check(cudaStreamEndCapture(stream,&graph.graph));
    check(cudaGraphInstantiate(&graph.executable,graph.graph,0));
    const auto fill=[&](int step,bool empty) {
        std::fill(expected.begin(),expected.end(),0xa7);
        for(int bank=0;bank<banks;++bank) {
            auto* old=reinterpret_cast<uint64_t*>(headers.data+header*bank);
            auto* table=reinterpret_cast<uint64_t*>(tables.data+table_bytes*bank);
            std::memset(old,0,header);table[0]=old[0]=empty?0:count;table[1]=span;table[2]=table[3]=0;
            old[1]=header;old[2]=header+span;
            auto* fields=reinterpret_cast<mfq::cuda::MoeMappedCopyDescriptor*>(table+4);
            for(std::size_t f=0;f<count;++f) {
                auto* destination=output.data+output_span*bank+destinations[f];
                old[4+3*f]=reinterpret_cast<uint64_t>(destination);old[5+3*f]=header+offsets[f];old[6+3*f]=layout.sizes[f];
                fields[f]={reinterpret_cast<uint64_t>(destination),
                    reinterpret_cast<uint64_t>(static_cast<uint8_t*>(input.alias)+span*bank+offsets[f]),offsets[f],layout.sizes[f]};
                for(std::size_t i=0;i<layout.sizes[f];++i) {
                    const auto value=uint8_t(i*29+f*71+bank*17+step*13);
                    input.data[span*bank+offsets[f]+i]=value;
                    if(!empty)expected[output_span*bank+destinations[f]+i]=value;
                }
            }
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
    };
    const auto submit=[&](int mode) {
        if(mode) {check(cudaGraphLaunch(graph.executable,stream));return;}
        for(int bank=0;bank<banks;++bank) {
            auto* target=wire.data+(header+span)*bank;
            check(cudaMemcpyAsync(target,headers.data+header*bank,header,cudaMemcpyHostToDevice,stream));
            // This legal packed layout lets the existing interval builder
            // combine every field into one DMA. Compare with that best case.
            if(reinterpret_cast<uint64_t*>(headers.data+header*bank)[0])
                check(cudaMemcpyAsync(target+header,input.data+span*bank,span,cudaMemcpyHostToDevice,stream));
            mfq::cuda::moe_scatter_wire(target,reinterpret_cast<uint32_t*>(abort.data),blocks,threads,stream);
        }
    };
    int exact=0;
    for(int step=0;step<4;++step)for(int mode=0;mode<2;++mode) {
        fill(step,step==2);if(step==3)std::fill(expected.begin(),expected.end(),0xa7);
        check(cudaMemsetAsync(output.data,0xa7,output.bytes,stream));check(cudaMemsetAsync(abort.data,step==3?1:0,4,stream));
        submit(mode);check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(actual.data(),output.data,output.bytes,cudaMemcpyDeviceToHost));
        if(actual!=expected)throw std::runtime_error("mapped copy changed fields or guards: layer="+
            std::to_string(layout.layer)+" alignment="+std::to_string(alignment)+" mode="+std::to_string(mode));
        ++exact;
    }
    fill(0,false);check(cudaMemsetAsync(abort.data,0,4,stream));check(cudaStreamSynchronize(stream));
    const auto measure=[&](int mode,int repeats) {
        const auto start=std::chrono::steady_clock::now();
        for(int i=0;i<repeats;++i)submit(mode);
        check(cudaStreamSynchronize(stream));
        return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/(repeats*banks);
    };
    for(int mode=0;mode<2;++mode)measure(mode,1);
    int repeats=1;while(measure(0,repeats)*repeats*banks<20000)repeats*=2;
    std::array<std::vector<double>,2> samples;
    for(int sample=0;sample<7;++sample)for(int order=0;order<2;++order) {
        const int mode=(sample+order)%2;samples[mode].push_back(measure(mode,repeats));
    }
    check(cudaStreamDestroy(stream));
    std::cout<<std::setprecision(10)<<"{\"scope\":\"mapped_RAM_copy\",\"layer\":"<<layout.layer<<",\"fields\":"<<count
        <<",\"payload_bytes\":"<<payload<<",\"destination_offset_mod16\":"<<alignment<<",\"banks\":"<<banks
        <<",\"exact_cases\":"<<exact<<",\"whole_model_loads\":0,\"host_submission_included\":true";
    timing(samples[0],samples[1]);std::cout<<"}"<<std::endl;
}
}
int main(int argc,char** argv)try {
    if(argc<2 || argc>3 || (argc==3 && std::string(argv[2])!="--mapped"))
        throw std::invalid_argument("expected retained wire layout file and optional --mapped");
    cudaDeviceProp device;check(cudaGetDeviceProperties(&device,0));
    for(const auto& row:read_layouts(argv[1]))for(int alignment:{0,8,4,1}) {
        if(argc==3)mapped_bench(row,alignment,device);else bench(row,alignment,device);
    }
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
