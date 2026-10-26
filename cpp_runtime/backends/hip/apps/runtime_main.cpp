#include "mfq/fatal_diagnostics.h"

namespace mfq::cuda::commands {
int run_runtime(int argc, char** argv);
}

int main(int argc, char** argv) {
    mfq::install_fatal_diagnostics();
    return mfq::cuda::commands::run_runtime(argc, argv);
}
