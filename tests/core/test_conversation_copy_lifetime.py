#!/usr/bin/env python3
"""CPU regression for checkpoint copy handle lifetime; no CUDA toolkit or GPU needed.

The fake runtime checks handle ownership and API ordering, not device execution.
Run with STRATA_COPY_HEADER=/path/to/original.hpp to exercise an unfixed header.
"""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
HEADER = Path(os.environ.get("STRATA_COPY_HEADER", ROOT / "src/core/conversation_copy.hpp")).resolve()

CACHE = r"""
#pragma once
#include <cstddef>
#include <vector>
namespace strata::core {
struct ConversationCheckpoint {
    std::vector<int> ids, imgs, gdn, ple, tails, dead, block_pos;
    std::size_t used = 0;
    bool pinned = false;
    std::vector<ConversationCheckpoint> stage_parts;
};
}
"""

PROGRESS = r"""
#pragma once
#include <atomic>
namespace strata::core {
inline std::atomic<int> progress_calls{0};
inline void progress_beat() { ++progress_calls; }
}
"""

CUDA = r"""
#pragma once
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>
struct FakeStream { int device; };
struct FakeEvent { int device; bool recorded = false; };
using cudaStream_t = FakeStream*;
using cudaEvent_t = FakeEvent*;
using cudaError_t = int;
inline constexpr int cudaSuccess = 0, cudaErrorMemoryAllocation = 2, cudaErrorInvalidValue = 11;
inline constexpr unsigned cudaStreamNonBlocking = 1, cudaEventDisableTiming = 2;
inline constexpr int cudaMemcpyDefault = 4;
namespace fake {
inline std::mutex mutex;
inline std::map<cudaStream_t, int> streams;
inline std::map<cudaEvent_t, int> events;
inline int creates = 0, destroys = 0, event_creates = 0, event_destroys = 0;
inline int synchronizes = 0, waits = 0, records = 0, bad_calls = 0, create_attempts = 0;
inline std::vector<int> exit_devices;
inline std::vector<std::string> operations;
inline thread_local int device = 0, fail_stream = 0, fail_event = 0, blocked_device = -1;
inline thread_local bool fail_get = false;
struct ExitProbe {
    ~ExitProbe() { std::lock_guard<std::mutex> lock(mutex); exit_devices.push_back(device); }
};
inline void observe_exit() { thread_local ExitProbe probe; (void) probe; }
inline int bad() { ++bad_calls; return cudaErrorInvalidValue; }
}
inline cudaError_t cudaGetDevice(int* device) {
    if (fake::fail_get) { fake::fail_get = false; return cudaErrorInvalidValue; }
    *device = fake::device; return cudaSuccess;
}
inline cudaError_t cudaSetDevice(int device) {
    if (device == fake::blocked_device) return cudaErrorInvalidValue;
    fake::device = device; return cudaSuccess;
}
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned flags) {
    std::lock_guard<std::mutex> lock(fake::mutex); ++fake::create_attempts;
    if (flags != cudaStreamNonBlocking) return fake::bad();
    if (fake::fail_stream) { --fake::fail_stream; *stream = nullptr; return cudaErrorMemoryAllocation; }
    *stream = new FakeStream{fake::device}; fake::streams[*stream] = fake::device; ++fake::creates;
    return cudaSuccess;
}
inline cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(fake::mutex);
    auto it = fake::streams.find(stream);
    if (it == fake::streams.end() || it->second != fake::device) return fake::bad();
    fake::streams.erase(it); delete stream; ++fake::destroys; return cudaSuccess;
}
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t bytes, int kind, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(fake::mutex);
    auto it = fake::streams.find(stream);
    if (it == fake::streams.end() || it->second != fake::device || kind != cudaMemcpyDefault) return fake::bad();
    std::memcpy(dst, src, bytes); fake::operations.push_back("copy"); return cudaSuccess;
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(fake::mutex);
    auto it = fake::streams.find(stream);
    if (it == fake::streams.end() || it->second != fake::device) return fake::bad();
    ++fake::synchronizes; fake::operations.push_back("sync"); return cudaSuccess;
}
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned flags) {
    std::lock_guard<std::mutex> lock(fake::mutex);
    if (flags != cudaEventDisableTiming) return fake::bad();
    if (fake::fail_event) { --fake::fail_event; *event = nullptr; return cudaErrorMemoryAllocation; }
    *event = new FakeEvent{fake::device}; fake::events[*event] = fake::device; ++fake::event_creates;
    return cudaSuccess;
}
inline cudaError_t cudaEventDestroy(cudaEvent_t event) {
    std::lock_guard<std::mutex> lock(fake::mutex);
    auto it = fake::events.find(event);
    if (it == fake::events.end() || it->second != fake::device) return fake::bad();
    fake::events.erase(it); delete event; ++fake::event_destroys; return cudaSuccess;
}
inline cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(fake::mutex);
    auto e = fake::events.find(event); auto s = fake::streams.find(stream);
    if (e == fake::events.end() || s == fake::streams.end() || e->second != fake::device || s->second != fake::device) return fake::bad();
    event->recorded = true; ++fake::records; fake::operations.push_back("record"); return cudaSuccess;
}
inline cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags) {
    std::lock_guard<std::mutex> lock(fake::mutex);
    auto e = fake::events.find(event);
    if (stream != nullptr || e == fake::events.end() || e->second != fake::device || !event->recorded || flags != 0) return fake::bad();
    ++fake::waits; fake::operations.push_back("wait"); return cudaSuccess;
}
"""

PROGRAM = r"""
#include "conversation_copy.hpp"
#include <atomic>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
namespace detail = strata::core::conversation_detail;

static void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

static void clean() {
    check(fake::streams.empty(), "private streams remain after worker exit");
    check(fake::events.empty(), "private events remain after worker exit");
    check(fake::bad_calls == 0, "wrong-device handle use or changed copy flags");
}

static void thread_workers() {
    std::atomic<int> failures{0};
    for (int i = 0; i < 128; ++i) {
        std::thread worker([&] {
            fake::observe_exit(); int input = 42, output = 0;
            if (detail::copy_nonblocking(&output, &input, sizeof(input)) != cudaSuccess || output != 42) ++failures;
        });
        worker.join();
    }
    check(failures == 0, "worker copy failed");
    check(fake::creates == 128 && fake::destroys == 128, "short-lived worker handles were not reclaimed");
    check(fake::synchronizes == 128 && strata::core::progress_calls == 128, "synchronous copy/progress behavior changed");
    clean();
}

static void queued_ordering() {
    int result = -1;
    std::thread worker([&] {
        fake::observe_exit(); int source = 17, destination = 0;
        result = detail::copy_nonblocking(&destination, &source, sizeof(source), true);
        if (result == cudaSuccess) result = detail::order_after_queued();
        if (destination != source) result = cudaErrorInvalidValue;
    });
    worker.join();
    check(result == cudaSuccess, "queued copy/order failed");
    check(fake::operations == std::vector<std::string>{"copy", "record", "wait"}, "queued copy no longer orders the legacy stream after its event");
    check(fake::synchronizes == 0 && strata::core::progress_calls == 0, "queued path introduced a host wait");
    check(fake::event_creates == 1 && fake::event_destroys == 1, "queued-copy event was not reclaimed");
    clean();
}

static void multiple_devices() {
    int failures = 0;
    std::thread worker([&] {
        fake::observe_exit();
        for (int device : {0, 1, 0, 1}) {
            cudaSetDevice(device); int source = device + 9, destination = 0;
            if (detail::copy_nonblocking(&destination, &source, sizeof(source), true) != cudaSuccess ||
                detail::order_after_queued() != cudaSuccess || destination != source) ++failures;
        }
        cudaSetDevice(2);
    });
    worker.join();
    check(failures == 0, "multi-device copy failed");
    check(fake::creates == 2 && fake::event_creates == 2, "handles were not reused by thread and device");
    check(fake::exit_devices == std::vector<int>{2}, "cleanup did not restore the previous current device");
    clean();
}

static void stream_create_retry() {
    int result = -1;
    std::thread worker([&] {
        fake::observe_exit(); fake::fail_stream = 1; cudaStream_t stream = nullptr;
        if (detail::private_stream(stream) != cudaErrorMemoryAllocation) return;
        result = detail::private_stream(stream);
    });
    worker.join();
    check(result == cudaSuccess && fake::create_attempts == 2 && fake::creates == 1, "stream creation could not retry after allocation failure");
    clean();
}

static void event_create_retry() {
    int result = -1;
    std::thread worker([&] {
        fake::observe_exit(); fake::fail_event = 1;
        if (detail::order_after_queued() != cudaErrorMemoryAllocation) return;
        result = detail::order_after_queued();
    });
    worker.join();
    check(result == cudaSuccess && fake::event_creates == 1 && fake::waits == 1, "event creation could not retry after allocation failure");
    clean();
}

static void get_device_failure() {
    int result = -1;
    std::thread worker([&] {
        fake::observe_exit(); fake::fail_get = true; cudaStream_t stream = nullptr;
        if (detail::private_stream(stream) != cudaErrorInvalidValue || fake::creates != 0) return;
        result = detail::private_stream(stream);
    });
    worker.join();
    check(result == cudaSuccess && fake::creates == 1, "get-device failure did not propagate or retry");
    clean();
}

static void destructor_get_device_failure() {
    int result = -1;
    std::thread worker([&] {
        fake::observe_exit(); cudaSetDevice(1); cudaStream_t stream = nullptr;
        result = detail::private_stream(stream);
        fake::fail_get = true;
    });
    worker.join();
    check(result == cudaSuccess, "cleanup test setup failed");
    check(fake::destroys == 1, "cleanup skipped owned handles after get-device failure");
    clean();
}

static void destructor_set_device_failure() {
    int result = -1;
    std::thread worker([&] {
        fake::observe_exit(); cudaStream_t stream = nullptr;
        cudaSetDevice(0); result = detail::private_stream(stream);
        cudaSetDevice(1); if (result == cudaSuccess) result = detail::private_stream(stream);
        cudaSetDevice(2); fake::blocked_device = 1;
    });
    worker.join();
    check(result == cudaSuccess, "cleanup test setup failed");
    check(fake::bad_calls == 0, "cleanup destroyed a handle on the wrong device after set-device failure");
    check(fake::streams.size() == 1 && fake::streams.begin()->second == 1, "cleanup did not preserve the handle whose device could not be selected");
    check(fake::exit_devices == std::vector<int>{2}, "best-effort cleanup failed to restore the previous device");
    // The fake makes the failed device available again, so the test can release the intentionally skipped handle.
    auto stream = fake::streams.begin()->first; cudaSetDevice(1); cudaStreamDestroy(stream);
    clean();
}

int main(int argc, char** argv) {
    try {
        check(argc == 2, "choose a scenario");
        const std::string name = argv[1];
        if (name == "thread-workers") thread_workers();
        else if (name == "queued-ordering") queued_ordering();
        else if (name == "multiple-devices") multiple_devices();
        else if (name == "stream-create-retry") stream_create_retry();
        else if (name == "event-create-retry") event_create_retry();
        else if (name == "get-device-failure") get_device_failure();
        else if (name == "destructor-get-device-failure") destructor_get_device_failure();
        else if (name == "destructor-set-device-failure") destructor_set_device_failure();
        else throw std::runtime_error("unknown scenario");
        std::cout << "PASS " << name << '\n'; return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
"""


class ConversationCopyLifetime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        if not compiler or not shutil.which(compiler[0]):
            raise RuntimeError("A C++17 compiler is required (set CXX); this regression must not silently skip.")
        cls.directory = tempfile.TemporaryDirectory(prefix="strata-copy-lifetime-")
        cls.addClassCleanup(cls.directory.cleanup)
        work = Path(cls.directory.name)
        includes = work / "include"
        (includes / "strata/core").mkdir(parents=True)
        (includes / "cuda_runtime.h").write_text(CUDA, encoding="utf-8")
        (includes / "strata/core/conversation_cache.hpp").write_text(CACHE, encoding="utf-8")
        (includes / "strata/core/progress.hpp").write_text(PROGRESS, encoding="utf-8")
        # Copy only the selected production header. Its neighbors must not shadow the explicit fake dependencies.
        (includes / "conversation_copy.hpp").write_bytes(HEADER.read_bytes())
        source = work / "test.cpp"
        source.write_text(PROGRAM, encoding="utf-8")
        cls.executable = work / ("copy-lifetime.exe" if os.name == "nt" else "copy-lifetime")
        subprocess.run([*compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pthread",
                        "-I", str(includes), str(source), "-o", str(cls.executable)], check=True, timeout=60)

    def scenario(self, name):
        completed = subprocess.run([str(self.executable), name], text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, timeout=15)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_short_lived_workers_release_handles(self):
        self.scenario("thread-workers")

    def test_queued_copy_keeps_event_order_without_host_wait(self):
        self.scenario("queued-ordering")

    def test_multi_device_cleanup_restores_previous_device(self):
        self.scenario("multiple-devices")

    def test_stream_creation_can_retry(self):
        self.scenario("stream-create-retry")

    def test_event_creation_can_retry(self):
        self.scenario("event-create-retry")

    def test_get_device_failure_is_returned_and_retryable(self):
        self.scenario("get-device-failure")

    def test_cleanup_still_releases_handles_if_get_device_fails(self):
        self.scenario("destructor-get-device-failure")

    def test_failed_device_selection_never_destroys_other_device_handles(self):
        self.scenario("destructor-set-device-failure")


if __name__ == "__main__":
    unittest.main()
