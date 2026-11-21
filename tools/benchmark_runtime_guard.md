# Guards for controlled benchmark runs

`benchmark_runtime_guard.py` supplies small checks for a controller that already
owns its experiment. It does not launch or stop a service, read credentials, or
send requests. No model or machine-specific settings are included.

Capture a `ProcessIdentity` when launching the service. Before **and after**
reading its health and metrics, call `owns_listener(identity, port)`. A false
result means no listener exists yet; an exception refuses a changed or foreign
owner. Check both the live counters and status with `require_idle` before the
first stop, and only then mark the controller as having changed the service.
An idle check does not reserve the service or replace a maintenance window.

Use `require_config_path` on the actual process arguments when confirming a
restore. Matching configuration contents in an experiment directory do not
establish restoration of the canonical path. Confirm the accepted engine and
its identity separately.

`require_exact_geometry` rejects allocation drift, including changes below two
percent. It does not establish that the same experts are selected: pin and
compare the profile hash as well. `require_answer_evidence` requires each
expected request exactly once and nonempty answer content, but does not score
answer quality or establish full answers. Preserve finish reasons and raw usage
in the benchmark report.

`verdict` requires both explicit execution completion and an independently
passed correctness gate. Any inconclusive gate blocks performance
interpretation, even when data collection finished. These checks do not by
themselves validate an engine or establish a speedup.

Run the CPU regression tests from the repository root:

```sh
python -m unittest discover -s tools -p test_benchmark_runtime_guard.py -v
```

The tests use temporary procfs fixtures and need no GPU or running service.
The production `/proc` checks are Linux-specific; the evidence and report
checks can also run elsewhere.
