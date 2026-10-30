#include "prefill_budget.h"

#include <cassert>
#include <limits>

int main() {
    using namespace mfq::engine;
    constexpr std::size_t row=327680, fixed=std::size_t(2)<<30;
    const auto memory=[&](int tokens) {return fixed+std::size_t(tokens)*row;};
    assert(select_prefill_chunk(131072,8192,memory(8192),fixed,row)==8192);
    assert(select_prefill_chunk(4096,8192,memory(8192),fixed,row)==4096);
    assert(select_prefill_chunk(8192,2048,memory(8192),fixed,row)==2048);
    assert(select_prefill_chunk(8192,8192,memory(4100),fixed,row)==4096);
    assert(select_prefill_chunk(8192,8192,memory(256)-1,fixed,row)==0);
    assert(select_prefill_chunk(8192,8192,fixed-1,fixed,row)==0);
    assert(select_prefill_chunk(17,8192,memory(17),fixed,row)==17);
    assert(select_prefill_chunk(17,8192,memory(16),fixed,row)==0);
    assert(select_prefill_chunk(8192,999999,memory(16384),fixed,row)==8192);
    // More transfer storage must reduce the compute tile. A later request can
    // grow back to 8192; the policy never latches an earlier low-memory result.
    assert(select_prefill_chunk(8192,8192,memory(8192),fixed+4096*row,row)==4096);
    assert(select_prefill_chunk(8192,8192,memory(8192),fixed,row)==8192);
    const auto maximum=std::numeric_limits<std::size_t>::max();
    assert(prefill_budget_add(maximum-100,101)==maximum);
    assert(prefill_budget_multiply(maximum,2)==maximum);
    assert((PrefillMoeGeometry{8,4096,128}.activation_bytes_per_token()==8*(4096+32*4)));
    assert((PrefillMoeGeometry{2,65,32}.activation_bytes_per_token()==2*(3*32+3*4)));
    assert(pressure_prefill_chunk(8192,8192,128*row,0,row)==128);
    assert(pressure_prefill_chunk(8192,8192,0,fixed,row)==1);
    assert(select_prefill_chunk(8192,8192,maximum,maximum,row)==0);
    for(int rows=0;rows<10000;++rows) {
        const auto result=select_prefill_chunk(8192,8192,memory(rows),fixed,row);
        assert(result>=0 && result<=8192 && result<=rows);
        assert(result==0 || result%256==0);
    }
}
