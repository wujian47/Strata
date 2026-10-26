# Independent checkpoint-copy stream lifetime reproduction

Supplemental evidence for Strata PR #3625. **This is not a test of that PR's
exact head.** It compares the v0.1.43 checkpoint-copy header with an independently
prepared v0.1.43 lifetime-management backport. No claim is made about the PR's
queued-copy/event path, current main, HIP or SYCL.

## Observed result

Measured on 2026-10-26 on Ubuntu with two RTX 3080 20 GB cards, NVIDIA driver
580.178.04, GCC 13.3.0 and nvcc 13.0.88. Both recorded runtime and driver API
versions are 13000. No model ran during these probes; an unrelated display
context remained on CUDA0.

Each fresh process warms eight explicit buffer bytes per card, then reserves a
bulk arena on CUDA1 only. The free-memory target is 256 MiB; both runs actually
started the worker loop with **254.5625 MiB** free on CUDA1. Workers are sequential
`std::async(std::launch::async)` jobs. Each successful worker copies the known
integer 42 to CUDA1 and back through the selected header.

| Counter | Original header | Independent backport |
| --- | ---: | ---: |
| Requested workers | 4096 | 4096 |
| Completed workers | 1592 | 4096 |
| Successful stream creates | 1592 | 4096 |
| Stream destroys | 0 | 4096 |
| Live tracked streams | 1592 | 0 |
| Value errors | 0 | 0 |
| First API failure | `cudaStreamCreateWithFlags`, code 2, `out of memory` | none |
| CUDA1 free memory at end of worker loop | 6.5625 MiB | 254.5625 MiB |
| Process exit status | 1 | 0 |

The original's 1593rd worker fails during stream creation. Its `api_errors=2`
counts the same returned failure at the instrumented runtime call and again at
the worker's header-call boundary; it does not establish two separate faults.
The pressure arena is freed before each process exits. Process exit releases
any remaining original-header handles and its CUDA context.

The original failure is deliberately preserved in `results/pressure-original.json`.
This demonstrates a stream-allocation OOM mechanism in this helper under limited
VRAM. It is not proof that every historical engine exit has the same cause.
These are diagnostic counters, not inference-speed benchmark results.

## Source provenance

`headers/original.hpp` is the exact export of
`v0.1.43:src/core/conversation_copy.hpp` (tag commit
`16374b3e6dfe452a163c3a961ae5faa3b4aec84b`). The candidate adds a thread-local owner
that releases private handles on their owning device at worker exit and attempts
to restore the previous current device. It retains nonblocking-copy behavior.
The old header has no queued-copy/event helper; this workload creates no events.

| File | SHA256 |
| --- | --- |
| `real_checkpoint_stream_pressure.cu` | `8ea096df42b4a63aad86423c1df96bf2b54aea8b007f7f391f3e4d829910f466` |
| `headers/original.hpp` | `f8fd8abe761294f493f5d5f5b898ff76f92115297a3e5a0b40e28d9d54c8ac91` |
| `headers/candidate.hpp` | `71315c4789e253b381846d3f091366d6469303dedc618dad1057719f8b067393` |

The probe and both headers are copied byte-for-byte from the tested sources.
The two result wrappers retain their original timestamp, exit status and stderr;
only the absolute executable path in `command` was replaced with `./pressure-*`.
The entire nested `result` object, including every memory sample, is unchanged.
No hostnames, private addresses, usernames, process IDs, service/configuration
files, model prompts, model answers or credentials are included.

## Reproduce in an isolated directory

Use a Linux host with two CUDA devices and a working CUDA toolchain. Stop model
services first and inspect GPU processes. The probe's initial 1 GiB-per-card
free-memory gate is not an exclusive-use check. It intentionally consumes most
of CUDA1's free VRAM for a short diagnostic; do not run it beside other work.
It performs no GPU reset and does not fill the bulk arena.

From this directory, with the CUDA compiler available on PATH:

```sh
nvcc -std=c++17 -O2 -Xcompiler -pthread \
  '-DHEADER_TO_TEST="headers/original.hpp"' real_checkpoint_stream_pressure.cu \
  -o pressure-original
nvcc -std=c++17 -O2 -Xcompiler -pthread \
  '-DHEADER_TO_TEST="headers/candidate.hpp"' real_checkpoint_stream_pressure.cu \
  -o pressure-candidate
```

Use fresh processes for the two arms, with an external time limit. Preserve the
original's nonzero result instead of treating it as an unexpected harness pass:

```sh
if timeout 120s ./pressure-original > original-local.json; then
  original_status=0
else
  original_status=$?
fi
if timeout 120s ./pressure-candidate > candidate-local.json; then
  candidate_status=0
else
  candidate_status=$?
fi
printf 'original=%s candidate=%s\n' "$original_status" "$candidate_status"
```

Exit 0 means every requested worker and value/lifetime check passed. Exit 1
means an API, value or lifetime failure; inspect `status` and `first_api_error`.
Exit 2 refuses unavailable devices or pressure setup. A timeout/partial JSON is
not a valid completed measurement. Exact failure counts can vary with hardware,
driver and available memory; the recorded 1592 threshold is not a universal
expectation. No fake CUDA include directory belongs on these build commands.

## Limits

This packet contains only this controlled v0.1.43 diagnostic and its measured
failure and success. It does not certify PR #3625's exact code, current-main
queued-copy ordering, HIP/SYCL behavior, model correctness, near-full-context
capacity or multi-day serving stability. Separate model tests are deliberately
outside this narrowly scoped packet.
