"""Small fail-closed checks for Linux benchmark controllers.

These helpers never start or stop a process and never send an HTTP request.
Check listener ownership before and after reading a benchmark service's state.
They do not reserve the service: experiments still need a maintenance window.
"""

from dataclasses import dataclass
import math
import os
from pathlib import Path


LIVE_COUNTERS = ("running", "queued", "waiting", "admitting", "outside_slots")
GEOMETRY_FIELDS = (
    "expert_slots", "expert_slots_primary", "expert_cache_mib", "expert_cache_primary_mib"
)


class GuardError(ValueError):
    """A fixed diagnostic code, without configuration or response contents."""


def _require(condition, code):
    if not condition:
        raise GuardError(code)


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    start_ticks: int


def process_identity(pid, proc_root=Path("/proc")):
    """Return PID/starttime, or None for a missing or zombie process.

    Permission errors and malformed stat files refuse the check rather than
    treating an unobservable process as exited.
    """
    _require(type(pid) is int and pid > 0, "invalid_pid")
    try:
        raw = (Path(proc_root) / str(pid) / "stat").read_text()
    except FileNotFoundError:
        return None
    try:
        fields = raw.rsplit(")", 1)[1].split()
        start = int(fields[19])
        state = fields[0]
    except (IndexError, ValueError) as error:
        raise GuardError("process_stat_invalid") from error
    _require(start >= 0, "process_stat_invalid")
    return None if state == "Z" else ProcessIdentity(pid, start)


def process_is_same(expected, proc_root=Path("/proc")):
    """A reused PID is not the process captured by an earlier controller step."""
    _require(isinstance(expected, ProcessIdentity), "process_identity_required")
    return process_identity(expected.pid, proc_root) == expected


def listening_socket_inodes(port=8080, proc_root=Path("/proc")):
    """Find every TCP LISTEN socket on a port, across IPv4 and IPv6."""
    _require(type(port) is int and 0 < port < 65536, "invalid_port")
    found = set()
    tables_read = 0
    for name in ("tcp", "tcp6"):
        try:
            lines = (Path(proc_root) / "net" / name).read_text().splitlines()[1:]
        except FileNotFoundError:
            continue
        tables_read += 1
        for line in lines:
            try:
                row = line.split()
                if row[3] == "0A" and int(row[1].rsplit(":", 1)[1], 16) == port:
                    found.add(row[9])
            except (IndexError, ValueError) as error:
                raise GuardError("tcp_table_invalid") from error
    _require(tables_read > 0, "tcp_table_unavailable")
    return found


def owns_listener(expected, port=8080, proc_root=Path("/proc")):
    """Return False while no listener exists; refuse foreign or changed owners.

    Every listener on the port must belong to this exact process identity.
    Callers must not consume service state when this returns False.
    """
    _require(process_is_same(expected, proc_root), "process_identity_changed")
    listeners = listening_socket_inodes(port, proc_root)
    if not listeners:
        return False
    owned = set()
    try:
        descriptors = list((Path(proc_root) / str(expected.pid) / "fd").iterdir())
    except FileNotFoundError as error:
        raise GuardError("process_identity_changed") from error
    for descriptor in descriptors:
        try:
            target = os.readlink(descriptor)
        except FileNotFoundError:
            continue  # A descriptor may close while the directory is scanned.
        if target.startswith("socket:[") and target.endswith("]"):
            owned.add(target[8:-1])
    _require(process_is_same(expected, proc_root), "process_identity_changed")
    _require(listening_socket_inodes(port, proc_root) == listeners, "listener_changed")
    _require(listeners.issubset(owned), "endpoint_owner_mismatch")
    return True


def require_idle(metrics, status):
    """Require explicit zero counters and zero in-flight requests."""
    live = metrics.get("live", {})
    _require(isinstance(live, dict), "live_counters_unavailable")
    _require(all(type(live.get(key)) is int and live[key] == 0 for key in LIVE_COUNTERS),
             "service_not_idle")
    activity = status.get("activity", {})
    _require(isinstance(activity, dict), "activity_unavailable")
    count = activity.get("in_flight")
    _require(type(count) is int and count == 0, "service_in_flight")


def require_config_path(argv, expected):
    """Require exactly one explicit --config with the canonical absolute path.

    Deliberately compare the argument, not file contents or resolved symlinks:
    an identical copy in an experiment directory is still a different path.
    """
    expected = os.fspath(expected)
    _require(Path(expected).is_absolute(), "canonical_config_must_be_absolute")
    paths = []
    for index, arg in enumerate(argv):
        if arg == "--config":
            _require(index + 1 < len(argv) and not argv[index + 1].startswith("--"),
                     "config_argument_missing")
            paths.append(argv[index + 1])
        elif arg.startswith("--config="):
            paths.append(arg.split("=", 1)[1])
    _require(len(paths) == 1 and paths[0] == expected, "config_path_mismatch")
    return expected


def require_exact_geometry(actual, reference):
    """Reject even small allocation drift; verify profile identity separately."""
    for key in GEOMETRY_FIELDS:
        left, right = actual.get(key), reference.get(key)
        valid = (type(left) in (int, float) and type(right) in (int, float)
                 and math.isfinite(left) and math.isfinite(right)
                 and left > 0 and right > 0 and left == right)
        _require(valid, "cache_geometry_mismatch")


def require_answer_evidence(report, expected_ids):
    """Require each expected request once, successful and with answer bytes.

    Answer bytes establish answer emission, not completeness or answer quality.
    A fixed-length throughput workload may legitimately end with reason=length.
    """
    expected_ids = tuple(expected_ids)
    _require(bool(expected_ids) and all(isinstance(name, str) and name for name in expected_ids)
             and len(set(expected_ids)) == len(expected_ids), "expected_request_ids_invalid")
    rows = report.get("requests")
    _require(isinstance(rows, list) and len(rows) == len(expected_ids)
             and all(isinstance(row, dict) for row in rows), "request_set_invalid")
    ids = [row.get("id") for row in rows]
    _require(all(isinstance(name, str) for name in ids) and set(ids) == set(expected_ids),
             "request_set_invalid")
    _require(all(row.get("ok") is True for row in rows), "request_failed")
    counts = [row.get("content_utf8_bytes") for row in rows]
    _require(all(type(count) is int and count > 0 for count in counts), "answer_content_missing")
    return {"request_count": len(rows), "min_content_utf8_bytes": min(counts)}


def verdict(*, execution_complete=False, correctness_passed=False, inconclusive_gates=()):
    """Keep execution completion separate from permission to interpret speed.

    The caller must establish every required gate before passing True; an empty
    list of inconclusive gates alone is never evidence of correctness.
    """
    _require(type(execution_complete) is bool and type(correctness_passed) is bool,
             "verdict_requires_booleans")
    allowed = execution_complete and correctness_passed and not inconclusive_gates
    status = "incomplete" if not execution_complete else ("complete" if allowed else "complete_inconclusive")
    return {"status": status,
            "correctness_status": "passed" if allowed else "not_established",
            "performance_interpretation_allowed": allowed}
