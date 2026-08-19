#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>

#include "mfq_avfoundation_video.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <future>
#include <string>
#include <vector>

namespace {

int fail(
        NSString * message,
        char * error_message,
        size_t error_capacity) {
    if (error_message != nullptr && error_capacity > 0) {
        std::snprintf(
            error_message,
            error_capacity,
            "%s",
            message.UTF8String ?: "AVFoundation video processing failed");
    }
    return -1;
}

NSString * reader_error(AVAssetReader * reader, NSString * fallback) {
    return reader.error.localizedDescription ?: fallback;
}

struct DecodedFrame {
    std::vector<uint8_t> nv12;
    int width = 0;
    int height = 0;
    double presentation_seconds = 0.0;
    std::string error;
};

DecodedFrame decode_frame(
        AVURLAsset * asset,
        AVAssetTrack * track,
        NSDictionary * output_settings,
        CMTime target) {
    @autoreleasepool {
        DecodedFrame result;
        NSError * error = nil;
        AVAssetReader * reader =
            [[AVAssetReader alloc] initWithAsset:asset error:&error];
        if (reader == nil) {
            result.error = (error.localizedDescription ?: @"unable to seek video").UTF8String;
            return result;
        }
        reader.timeRange = CMTimeRangeMake(
            target, CMTimeMakeWithSeconds(0.25, 600));
        AVAssetReaderTrackOutput * output =
            [[AVAssetReaderTrackOutput alloc] initWithTrack:track
                                             outputSettings:output_settings];
        output.alwaysCopiesSampleData = NO;
        if (![reader canAddOutput:output]) {
            result.error = "unable to configure hardware video decoding";
            return result;
        }
        [reader addOutput:output];
        if (![reader startReading]) {
            result.error = reader_error(reader, @"unable to decode video frame").UTF8String;
            return result;
        }
        CMSampleBufferRef sample = [output copyNextSampleBuffer];
        if (sample == nullptr) {
            result.error = reader_error(reader, @"unable to decode video frame").UTF8String;
            return result;
        }
        CVPixelBufferRef pixels = CMSampleBufferGetImageBuffer(sample);
        if (pixels == nullptr || CVPixelBufferGetPlaneCount(pixels) < 2) {
            CFRelease(sample);
            result.error = "AVFoundation returned an unsupported pixel format";
            return result;
        }
        if (CVPixelBufferLockBaseAddress(
                pixels, kCVPixelBufferLock_ReadOnly) != kCVReturnSuccess) {
            CFRelease(sample);
            result.error = "unable to access decoded video pixels";
            return result;
        }
        result.width = static_cast<int>(CVPixelBufferGetWidth(pixels));
        result.height = static_cast<int>(CVPixelBufferGetHeight(pixels));
        result.presentation_seconds = CMTimeGetSeconds(
            CMSampleBufferGetPresentationTimeStamp(sample));
        const auto * y_source = static_cast<const uint8_t *>(
            CVPixelBufferGetBaseAddressOfPlane(pixels, 0));
        const auto * uv_source = static_cast<const uint8_t *>(
            CVPixelBufferGetBaseAddressOfPlane(pixels, 1));
        const size_t y_stride = CVPixelBufferGetBytesPerRowOfPlane(pixels, 0);
        const size_t uv_stride = CVPixelBufferGetBytesPerRowOfPlane(pixels, 1);
        const size_t y_size =
            static_cast<size_t>(result.width) * static_cast<size_t>(result.height);
        result.nv12.resize(y_size + y_size / 2);
        for (int row = 0; row < result.height; ++row) {
            std::memcpy(
                result.nv12.data() + static_cast<size_t>(row) * result.width,
                y_source + static_cast<size_t>(row) * y_stride,
                result.width);
        }
        for (int row = 0; row < result.height / 2; ++row) {
            std::memcpy(
                result.nv12.data() + y_size + static_cast<size_t>(row) * result.width,
                uv_source + static_cast<size_t>(row) * uv_stride,
                result.width);
        }
        CVPixelBufferUnlockBaseAddress(pixels, kCVPixelBufferLock_ReadOnly);
        CFRelease(sample);
        return result;
    }
}

}  // namespace

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
        size_t error_capacity) {
    @autoreleasepool {
        if (path == nullptr || target_pts == nullptr || callback == nullptr ||
            target_count <= 0 || time_base_numerator <= 0 ||
            time_base_denominator <= 0 || parallelism <= 0) {
            return fail(@"invalid AVFoundation video sampling arguments",
                        error_message, error_capacity);
        }

        NSString * file_path = [NSString stringWithUTF8String:path];
        if (file_path == nil) {
            return fail(@"video path is not valid UTF-8",
                        error_message, error_capacity);
        }
        AVURLAsset * asset = [AVURLAsset URLAssetWithURL:
            [NSURL fileURLWithPath:file_path] options:nil];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        AVAssetTrack * track =
            [[asset tracksWithMediaType:AVMediaTypeVideo] firstObject];
#pragma clang diagnostic pop
        if (track == nil) {
            return fail(@"video contains no video stream",
                        error_message, error_capacity);
        }

        NSDictionary * output_settings = @{
            (NSString *) kCVPixelBufferPixelFormatTypeKey:
                @(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange),
            (NSString *) kCVPixelBufferMetalCompatibilityKey: @YES,
        };
        int delivered = 0;
        while (delivered < target_count) {
            const int batch = std::min(parallelism, target_count - delivered);
            std::vector<std::future<DecodedFrame>> futures;
            futures.reserve(batch);
            for (int offset = 0; offset < batch; ++offset) {
                const CMTime target = CMTimeMake(
                    target_pts[delivered + offset] * time_base_numerator,
                    time_base_denominator);
                futures.emplace_back(std::async(
                    std::launch::async,
                    [asset, track, output_settings, target, callback, context,
                     target_index = delivered + offset]() {
                        DecodedFrame frame = decode_frame(
                            asset, track, output_settings, target);
                        if (frame.error.empty()) {
                            const int callback_result = callback(
                                context,
                                target_index,
                                frame.nv12.data(),
                                frame.width,
                                frame.nv12.data() +
                                    static_cast<size_t>(frame.width) * frame.height,
                                frame.width,
                                frame.width,
                                frame.height,
                                frame.presentation_seconds);
                            if (callback_result != 0) {
                                frame.error = "video frame callback failed";
                            }
                        }
                        return frame;
                    }));
            }
            for (auto & future : futures) {
                DecodedFrame frame = future.get();
                if (!frame.error.empty()) {
                    return fail(
                        [NSString stringWithUTF8String:frame.error.c_str()],
                        error_message,
                        error_capacity);
                }
            }
            delivered += batch;
        }
        return delivered;
    }
}
