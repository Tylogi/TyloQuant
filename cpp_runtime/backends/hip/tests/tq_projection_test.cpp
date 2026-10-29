#include "tq.h"
#include "mfq/tq_rows.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>

namespace tb=mfq_tensor_backend;
int main(int argc,char** argv) {
    try {
        if(argc<2)throw std::invalid_argument("expected independent CPU fixture paths");
        if(hipSetDevice(0)!=hipSuccess)return 77;
        for(int file=1;file<argc;++file) {
            std::ifstream in(argv[file],std::ios::binary);
            std::uint32_t bytes=0,columns=0;
            in.read(reinterpret_cast<char*>(&bytes),4);
            in.read(reinterpret_cast<char*>(&columns),4);
            std::vector<std::uint8_t> blob(bytes);
            in.read(reinterpret_cast<char*>(blob.data()),bytes);
            const mfq::TqLayout layout(blob.data(),blob.size());
            std::vector<float> reference(std::size_t(layout.rows)*columns);
            in.read(reinterpret_cast<char*>(reference.data()),reference.size()*4);
            if(!in || in.peek()!=EOF)throw std::runtime_error("invalid native fixture");
            auto op=to_device_tq(blob,columns,0);
            auto decoded=op.materialize(tb::kFloat32).cpu().contiguous();
            const auto* actual=decoded.data_ptr<float>();
            for(std::size_t i=0;i<reference.size();++i)
                if(actual[i]!=reference[i])throw std::runtime_error("native TQ decode differs from NumPy oracle");
            std::vector<float> basis(std::size_t(columns)*columns,0);
            for(std::size_t c=0;c<columns;++c)basis[c*columns+c]=1;
            for(auto type:{tb::kFloat16,tb::kFloat32}) {
                auto eye=tb::from_blob(basis.data(),{std::int64_t(columns),std::int64_t(columns)},
                    tb::TensorOptions().dtype(tb::kFloat32)).clone().to(type).to(tb::Device(tb::kCUDA,0));
                for(std::size_t c=0;c<columns;++c) {
                    auto column=op.forward(eye.narrow(0,c,1)).cpu().to(tb::kFloat32).contiguous();
                    for(std::size_t r=0;r<layout.rows;++r)
                        if(column.data_ptr<float>()[r]!=reference[r*columns+c])
                            throw std::runtime_error("single-token packed math changed a reconstructed weight");
                }
            }
            for(auto type:{tb::kFloat16,tb::kBFloat16,tb::kFloat32}) {
                for(int tokens:{1,2,17,128}) {
                    std::vector<float> values(std::size_t(tokens)*columns);
                    for(std::size_t i=0;i<values.size();++i)values[i]=std::sin(float(i)*.37f)*.2f;
                    auto source=tb::from_blob(values.data(),{tokens,std::int64_t(columns)},
                        tb::TensorOptions().dtype(tb::kFloat32)).clone().to(type);
                    auto rounded=source.to(tb::kFloat32).contiguous();
                    auto x=source.to(tb::Device(tb::kCUDA,0));
                    auto raw=op.forward(x);
                    auto y=raw.cpu().to(tb::kFloat32).contiguous();
                    const auto* input=rounded.data_ptr<float>();
                    const auto* output=y.data_ptr<float>();
                    if(tokens==1) {
                        std::vector<float> skip(layout.rows);
                        for(std::size_t r=0;r<layout.rows;++r)skip[r]=std::cos(float(r)*.19f)*.03f;
                        auto residual=tb::from_blob(skip.data(),{1,std::int64_t(layout.rows)},
                            tb::TensorOptions().dtype(tb::kFloat32)).clone().to(type).to(tb::Device(tb::kCUDA,0));
                        auto fused=tq_forward_residual(op,x,residual).cpu().to(tb::kFloat32).contiguous();
                        auto expected=(raw+residual).cpu().to(tb::kFloat32).contiguous();
                        for(std::size_t r=0;r<layout.rows;++r)
                            if(fused.data_ptr<float>()[r]!=expected.data_ptr<float>()[r])
                                throw std::runtime_error("native fused TQ residual differs from separate addition");
                    }
                    double error=0,scale=0;
                    for(int t=0;t<tokens;++t)for(std::size_t r=0;r<layout.rows;++r) {
                        double expected=0;
                        for(std::size_t c=0;c<columns;++c)expected+=double(input[t*columns+c])*reference[r*columns+c];
                        error+=std::pow(double(output[t*layout.rows+r])-expected,2);scale+=expected*expected;
                    }
                    const double limit=type==tb::kFloat16?.001:type==tb::kBFloat16?.004:3e-6;
                    if(std::sqrt(error/scale)>limit)throw std::runtime_error("native TQ forward differs from FP64 oracle");
                }
            }
            auto invalid=blob;invalid[64+4]|=0x80;
            bool rejected=false;
            try{(void)to_device_tq(invalid,columns,0);}catch(const std::invalid_argument&){rejected=true;}
            if(!rejected)throw std::runtime_error("invalid TQ descriptor accepted");
            std::cout<<argv[file]<<": exact decode, 12 forwards, fused residual and invalid descriptor passed\n";
        }
        return 0;
    }catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n';return 1;
    }
}
