#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*mfq_avfoundation_frame_callback)(
    void * context,
    int target_index,
    const uint8_t * y_plane,
    size_t y_stride,
    const uint8_t * uv_plane,
    size_t uv_stride,
    int width,
    int height,
    double presentation_seconds);

int mfq_avfoundation_sample_video(
    const char * path,
    const int64_t * target_pts,
    int target_count,
    int32_t time_base_numerator,
    int32_t time_base_denominator,
    int parallelism,
    mfq_avfoundation_frame_callback callback,
    void * context,
    char * error_message,
    size_t error_capacity);

#ifdef __cplusplus
}
#endif
