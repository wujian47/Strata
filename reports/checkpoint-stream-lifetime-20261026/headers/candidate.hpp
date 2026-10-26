#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

namespace strata::core::conversation_detail {

// Each prefill chunk on a later split device may run on a short-lived worker.
// Own the private handles until that worker exits, rather than leaking them with
// the destroyed thread-local vector. Failed runtime calls during shutdown are
// best effort, including restoring the previously selected device.
struct CopyResources {
    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t> events;
    ~CopyResources() {
        int previous = 0;
        if (cudaGetDevice(&previous) != cudaSuccess) return;
        int selected = previous;
        const size_t count = streams.size() > events.size() ? streams.size() : events.size();
        for (size_t i = 0; i < count; ++i) {
            const auto stream = i < streams.size() ? streams[i] : nullptr;
            const auto event = i < events.size() ? events[i] : nullptr;
            if (!stream && !event) continue;
            if (selected != (int) i) {
                if (cudaSetDevice((int) i) != cudaSuccess) continue;
                selected = (int) i;
            }
            if (event) (void) cudaEventDestroy(event);
            if (stream) (void) cudaStreamDestroy(stream);
        }
        if (selected != previous) (void) cudaSetDevice(previous);
    }
};

inline CopyResources& copy_resources() {
    thread_local CopyResources resources;
    return resources;
}

/// Conversation snapshot / checkpoint copies go through a private non-blocking stream of the current device
/// instead of the legacy (null) stream.  A plain cudaMemcpy is ordered against every blocking stream of the
/// device, and while another thread captures a graph on one (a drafter or a stage worker), HIP refuses it with
/// "operation would make the legacy stream depend on a capturing blocking stream".  The caller's contract is
/// unchanged: the device was synchronized before the copy (the copy itself is complete on return).
inline cudaError_t copy_nonblocking(void* dst, const void* src, size_t bytes) {
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return e;
    auto& streams = copy_resources().streams;
    if (dev < 0) dev = 0;
    if ((size_t) dev >= streams.size()) streams.resize((size_t) dev + 1, nullptr);
    if (!streams[(size_t) dev]) {
        e = cudaStreamCreateWithFlags(&streams[(size_t) dev], cudaStreamNonBlocking);
        if (e != cudaSuccess) { streams[(size_t) dev] = nullptr; return e; }
    }
    cudaStream_t s = streams[(size_t) dev];
    e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, s);
    if (e != cudaSuccess) return e;
    return cudaStreamSynchronize(s);
}

} // namespace strata::core::conversation_detail
