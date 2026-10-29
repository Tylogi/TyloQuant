#pragma once
#include "mfq/model_source.h"

namespace mfq {
// Explicit synthetic benchmark input. No normal model load creates this source.
std::shared_ptr<const ModelSource> open_tq_fixture_source(const std::filesystem::path&);
}
