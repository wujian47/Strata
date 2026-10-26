// Standalone CUDA1-pressure diagnostic; run only with the model stopped. Build against real CUDA and an explicitly selected
// checkpoint-copy header; never add checkpoint_stream_lifetime/ to the include
// path (that directory contains a fake cuda_runtime.h).
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace stream_probe {
struct State {
    std::mutex mutex;
    std::map<cudaStream_t, int> streams;
    std::map<cudaEvent_t, int> events;
    std::atomic<unsigned> created{0}, destroyed{0}, copies{0}, events_created{0}, events_destroyed{0};
    std::atomic<unsigned> worker_entries{0}, worker_exits{0}, device_restore_errors{0};
    std::atomic<unsigned> nonblocking_creates{0}, timing_disabled_events{0}, api_errors{0};
    std::string first_error_stage;
    int first_error_code = 0;
    std::string first_error_text;
} state;
constexpr uint64_t pressure_target_free_bytes = uint64_t{256} << 20;
uint64_t pressure_arena_bytes = 0;

bool checked(cudaError_t code, const char* stage) {
    if (code == cudaSuccess) return true;
    ++state.api_errors;
    std::lock_guard<std::mutex> lock(state.mutex);
    if (state.first_error_stage.empty()) {
        state.first_error_stage = stage;
        state.first_error_code = static_cast<int>(code);
        state.first_error_text = cudaGetErrorString(code);
    }
    return false;
}

cudaError_t create_stream(cudaStream_t* stream, unsigned flags) {
    const auto result = ::cudaStreamCreateWithFlags(stream, flags);
    if (!checked(result, "cudaStreamCreateWithFlags")) return result;
    int device = -1;
    checked(::cudaGetDevice(&device), "create_stream.cudaGetDevice");
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.streams[*stream] = device;
    }
    ++state.created;
    if (flags == cudaStreamNonBlocking) ++state.nonblocking_creates;
    return result;
}

cudaError_t destroy_stream(cudaStream_t stream) {
    const auto result = ::cudaStreamDestroy(stream);
    if (!checked(result, "cudaStreamDestroy")) return result;
    std::lock_guard<std::mutex> lock(state.mutex);
    state.streams.erase(stream);
    ++state.destroyed;
    return result;
}

cudaError_t create_event(cudaEvent_t* event, unsigned flags) {
    const auto result = ::cudaEventCreateWithFlags(event, flags);
    if (!checked(result, "cudaEventCreateWithFlags")) return result;
    int device = -1;
    checked(::cudaGetDevice(&device), "create_event.cudaGetDevice");
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.events[*event] = device;
    }
    ++state.events_created;
    if (flags == cudaEventDisableTiming) ++state.timing_disabled_events;
    return result;
}

cudaError_t destroy_event(cudaEvent_t event) {
    const auto result = ::cudaEventDestroy(event);
    if (!checked(result, "cudaEventDestroy")) return result;
    std::lock_guard<std::mutex> lock(state.mutex);
    state.events.erase(event);
    ++state.events_destroyed;
    return result;
}

cudaError_t copy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
    const auto result = ::cudaMemcpyAsync(dst, src, bytes, kind, stream);
    if (checked(result, "cudaMemcpyAsync")) ++state.copies;
    return result;
}
} // namespace stream_probe

// The real runtime was already included above. Intercept only calls originating
// in the selected header; delegates above always invoke real libcudart.
#define cudaStreamCreateWithFlags stream_probe::create_stream
#define cudaStreamDestroy stream_probe::destroy_stream
#define cudaEventCreateWithFlags stream_probe::create_event
#define cudaEventDestroy stream_probe::destroy_event
#define cudaMemcpyAsync stream_probe::copy
#include HEADER_TO_TEST
#undef cudaStreamCreateWithFlags
#undef cudaStreamDestroy
#undef cudaEventCreateWithFlags
#undef cudaEventDestroy
#undef cudaMemcpyAsync

namespace stream_probe {
struct WorkerLifetime {
    int expected_device = 1;
    WorkerLifetime() { ++state.worker_entries; }
    ~WorkerLifetime() {
        int device = -1;
        if (!checked(cudaGetDevice(&device), "worker_exit.cudaGetDevice") || device != expected_device)
            ++state.device_restore_errors;
        ++state.worker_exits;
    }
};

std::string escaped(const std::string& value) {
    std::string output;
    for (unsigned char ch : value) {
        if (ch == '\\' || ch == '"') output += '\\';
        if (ch >= 32) output += static_cast<char>(ch);
    }
    return output;
}

struct MemorySample {
    const char* phase = "progress";
    unsigned after_workers = 0;
    int device = -1;
    uint64_t free_bytes = 0, total_bytes = 0;
    int error_code = 0;
};

std::vector<MemorySample> memory;

bool sample_memory(unsigned after_workers, bool require_headroom = false, const char* phase = "progress") {
    bool okay = true;
    for (int device = 0; device < 2; ++device) {
        MemorySample sample;
        sample.phase = phase;
        sample.after_workers = after_workers;
        sample.device = device;
        auto error = cudaSetDevice(device);
        bool step_okay = checked(error, "memory.cudaSetDevice");
        if (step_okay) {
            error = cudaDeviceSynchronize();
            step_okay = checked(error, "memory.cudaDeviceSynchronize");
        }
        if (step_okay) {
            size_t free_bytes = 0, total_bytes = 0;
            error = cudaMemGetInfo(&free_bytes, &total_bytes);
            sample.free_bytes = free_bytes;
            sample.total_bytes = total_bytes;
            checked(error, "memory.cudaMemGetInfo");
        }
        sample.error_code = static_cast<int>(error);
        if (error != cudaSuccess || (require_headroom && sample.free_bytes < (uint64_t{1} << 30))) okay = false;
        memory.push_back(sample);
    }
    return okay;
}

void report(const char* status, unsigned requested, unsigned completed, unsigned value_errors,
            bool multi_device, double elapsed_s, int runtime_version, int driver_version) {
    std::cout << "{\"kind\":\"real_cuda_checkpoint_stream_pressure\",\"status\":\"" << status
              << "\",\"pressure_device\":1,\"target_free_bytes\":" << pressure_target_free_bytes
              << ",\"pressure_arena_bytes\":" << pressure_arena_bytes
              << ",\"requested_workers\":" << requested << ",\"completed_workers\":" << completed
              << ",\"multi_device\":" << (multi_device ? "true" : "false")
              << ",\"worker_entries\":" << state.worker_entries << ",\"worker_exits\":" << state.worker_exits
              << ",\"created\":" << state.created << ",\"destroyed\":" << state.destroyed
              << ",\"live\":" << state.streams.size() << ",\"copies\":" << state.copies
              << ",\"events_created\":" << state.events_created << ",\"events_destroyed\":" << state.events_destroyed
              << ",\"events_live\":" << state.events.size() << ",\"nonblocking_creates\":" << state.nonblocking_creates
              << ",\"timing_disabled_events\":" << state.timing_disabled_events
              << ",\"device_restore_errors\":" << state.device_restore_errors
              << ",\"value_errors\":" << value_errors << ",\"api_errors\":" << state.api_errors
              << ",\"first_api_error\":";
    if (state.first_error_stage.empty()) std::cout << "null";
    else std::cout << "{\"stage\":\"" << escaped(state.first_error_stage) << "\",\"code\":" << state.first_error_code
                   << ",\"text\":\"" << escaped(state.first_error_text) << "\"}";
    std::cout << ",\"runtime_version\":" << runtime_version << ",\"driver_version\":" << driver_version
              << ",\"elapsed_s\":" << elapsed_s << ",\"memory_samples\":[";
    for (size_t i = 0; i < memory.size(); ++i) {
        if (i) std::cout << ',';
        const auto& sample = memory[i];
        std::cout << "{\"phase\":\"" << sample.phase << "\",\"after_workers\":" << sample.after_workers << ",\"device\":" << sample.device
                  << ",\"free_bytes\":" << sample.free_bytes << ",\"total_bytes\":" << sample.total_bytes
                  << ",\"error_code\":" << sample.error_code << '}';
    }
    std::cout << "]}\n";
}
} // namespace stream_probe

int main(int argc, char** argv) {
    using namespace stream_probe;
    unsigned requested = 4096;
    bool multi_device = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--threads" && i + 1 < argc) {
            const std::string number = argv[++i];
            if (number.empty() || number.find_first_not_of("0123456789") != std::string::npos) return 2;
            const unsigned long parsed = std::strtoul(number.c_str(), nullptr, 10);
            if (parsed < 1 || parsed > 4096) return 2;
            requested = static_cast<unsigned>(parsed);
        } else return 2;
    }
    const auto started = std::chrono::steady_clock::now();
    int runtime_version = 0, driver_version = 0, device_count = 0;
    checked(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
    checked(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
    checked(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (state.api_errors || device_count < 2) {
        report("refused_device_unavailable", requested, 0, 0, multi_device, 0, runtime_version, driver_version);
        return 2;
    }
    if (!sample_memory(0, true, "preflight")) {
        report("refused_free_memory", requested, 0, 0, multi_device, 0, runtime_version, driver_version);
        return 2;
    }
    int* buffers[2] = {nullptr, nullptr};
    bool allocation_okay = true;
    unsigned completed = 0, value_errors = 0;
    for (int device = 0; device < 2; ++device) {
        if (!checked(cudaSetDevice(device), "warm.cudaSetDevice") ||
            !checked(cudaMalloc(reinterpret_cast<void**>(&buffers[device]), 2 * sizeof(int)), "warm.cudaMalloc")) {
            allocation_okay = false;
            break;
        }
        int source = 42, destination = 0;
        if (!checked(cudaMemcpy(buffers[device], &source, sizeof(source), cudaMemcpyHostToDevice), "warm.host_to_device") ||
            !checked(cudaMemcpy(&destination, buffers[device], sizeof(destination), cudaMemcpyDeviceToHost), "warm.device_to_host") ||
            !checked(cudaDeviceSynchronize(), "warm.cudaDeviceSynchronize") || destination != source) {
            if (destination != source) ++value_errors;
            allocation_okay = false;
            break;
        }
    }
    const bool baseline_okay = allocation_okay && sample_memory(0, true, "warm");
    const bool refused_headroom = allocation_okay && !baseline_okay && !state.api_errors;
    void* pressure_arena = nullptr;
    bool pressure_okay = false;
    if (baseline_okay) {
        size_t free_bytes = 0, total_bytes = 0;
        if (checked(cudaSetDevice(1), "pressure.cudaSetDevice") &&
            checked(cudaMemGetInfo(&free_bytes, &total_bytes), "pressure.cudaMemGetInfo") &&
            free_bytes > pressure_target_free_bytes) {
            const size_t reserve_bytes = free_bytes - pressure_target_free_bytes;
            if (checked(cudaMalloc(&pressure_arena, reserve_bytes), "pressure.cudaMalloc")) {
                pressure_arena_bytes = reserve_bytes;
                if (sample_memory(0, false, "pressure")) {
                    const auto available = memory.back().free_bytes; // CUDA1 is sampled last.
                    pressure_okay = available >= (uint64_t{192} << 20) && available <= (uint64_t{320} << 20);
                }
            }
        }
    }
    const bool refused_pressure = baseline_okay && !pressure_okay;
    if (pressure_okay) {
        for (unsigned i = 0; i < requested; ++i) {
            const auto errors_before = state.api_errors.load();
            auto task = std::async(std::launch::async, [&]() -> bool {
                // Construct first so this observer is destroyed after header TLS.
                thread_local WorkerLifetime lifetime;
                lifetime.expected_device = multi_device ? 0 : 1;
                bool good = true;
                const int first = multi_device ? 0 : 1;
                for (int device = first; device < 2; ++device) {
                    if (!checked(cudaSetDevice(device), "worker.cudaSetDevice")) { good = false; break; }
                    int source = 42, destination = 0;
                    if (!checked(strata::core::conversation_detail::copy_nonblocking(
                            buffers[device], &source, sizeof(source)), "worker.header_host_to_device") ||
                        !checked(strata::core::conversation_detail::copy_nonblocking(
                            &destination, buffers[device], sizeof(destination)), "worker.header_device_to_host")) {
                        good = false;
                        break;
                    }
                    if (destination != source) { ++value_errors; good = false; break; }
                }
                if (!checked(cudaSetDevice(lifetime.expected_device), "worker.final_cudaSetDevice")) good = false;
                return good;
            });
            const bool good = task.get();
            if (good) ++completed;
            if (!good || state.api_errors != errors_before) break;
            if ((i + 1) % 32 == 0 && !sample_memory(i + 1)) break;
        }
    }
    sample_memory(completed, false, "final");
    for (int device = 0; device < 2; ++device) {
        if (buffers[device] && checked(cudaSetDevice(device), "cleanup.cudaSetDevice"))
            checked(cudaFree(buffers[device]), "cleanup.cudaFree");
    }
    if (pressure_arena && checked(cudaSetDevice(1), "pressure_cleanup.cudaSetDevice"))
        checked(cudaFree(pressure_arena), "pressure_cleanup.cudaFree");
    sample_memory(completed, false, "after_pressure_free");
    const auto expected_streams = requested * (multi_device ? 2 : 1);
    const bool okay = pressure_okay && completed == requested && !value_errors && !state.api_errors &&
        state.worker_entries == requested && state.worker_exits == requested &&
        !state.device_restore_errors && state.created == expected_streams && state.destroyed == expected_streams &&
        state.nonblocking_creates == expected_streams && state.copies == expected_streams * 2 &&
        state.streams.empty() && state.events.empty();
    const bool stream_create_oom = pressure_okay && state.first_error_stage == "cudaStreamCreateWithFlags" &&
        state.first_error_code == static_cast<int>(cudaErrorMemoryAllocation);
    const char* status = okay ? "passed" : refused_headroom ? "refused_free_memory" :
        refused_pressure ? "refused_pressure_setup" : stream_create_oom ? "stream_create_oom" :
        state.api_errors ? "cuda_error" : value_errors ? "value_mismatch" :
        !state.streams.empty() || !state.events.empty() ? "leaked_handles" : "counter_mismatch";
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    report(status, requested, completed, value_errors, multi_device, elapsed, runtime_version, driver_version);
    return okay ? 0 : refused_headroom || refused_pressure ? 2 : 1;
}
