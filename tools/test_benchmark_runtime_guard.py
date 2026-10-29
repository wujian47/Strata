"""CPU regression tests; Linux procfs is represented by temporary fixtures."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import benchmark_runtime_guard as guard


class ProcFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "net").mkdir()
        self.stat()
        (self.root / "7" / "fd").mkdir()
        (self.root / "7" / "fd" / "3").touch()
        self.table()

    def stat(self, state="S", start=100):
        folder = self.root / "7"
        folder.mkdir(exist_ok=True)
        (folder / "stat").write_text("7 (worker with ) in name) " + state + " 1 " + "0 " * 17 + str(start))

    def table(self, port=8080, inode="10", state="0A", name="tcp"):
        row = f"0: 0100007F:{port:04X} 00000000:0000 {state} 0:0 00:0 0 1000 0 {inode}\n"
        (self.root / "net" / name).write_text("header\n" + row)

    def identity(self):
        return guard.ProcessIdentity(7, 100)

    def test_comm_parentheses_do_not_shift_starttime(self):
        self.assertEqual(guard.process_identity(7, self.root), self.identity())

    def test_missing_and_zombie_processes_are_not_live(self):
        self.assertIsNone(guard.process_identity(8, self.root))
        self.stat(state="Z")
        self.assertIsNone(guard.process_identity(7, self.root))

    def test_pid_reuse_is_not_an_owned_process(self):
        self.stat(start=101)
        self.assertFalse(guard.process_is_same(self.identity(), self.root))
        with self.assertRaisesRegex(guard.GuardError, "identity_changed"):
            guard.owns_listener(self.identity(), proc_root=self.root)

    def test_permission_errors_do_not_mean_process_exited(self):
        with patch.object(Path, "read_text", side_effect=PermissionError):
            with self.assertRaises(PermissionError):
                guard.process_identity(7, self.root)

    def test_malformed_stat_refuses(self):
        (self.root / "7" / "stat").write_text("7 (broken) S")
        with self.assertRaisesRegex(guard.GuardError, "stat_invalid"):
            guard.process_identity(7, self.root)

    def test_exact_listener_owner_accepts(self):
        with patch.object(guard.os, "readlink", return_value="socket:[10]"):
            self.assertTrue(guard.owns_listener(self.identity(), proc_root=self.root))

    def test_healthy_foreign_listener_refuses(self):
        with patch.object(guard.os, "readlink", return_value="socket:[20]"):
            with self.assertRaisesRegex(guard.GuardError, "owner_mismatch"):
                guard.owns_listener(self.identity(), proc_root=self.root)

    def test_foreign_ipv6_listener_also_refuses(self):
        self.table(name="tcp6", inode="20")
        with patch.object(guard.os, "readlink", return_value="socket:[10]"):
            with self.assertRaisesRegex(guard.GuardError, "owner_mismatch"):
                guard.owns_listener(self.identity(), proc_root=self.root)

    def test_no_listener_waits_instead_of_consuming_metrics(self):
        self.table(state="01")
        self.assertFalse(guard.owns_listener(self.identity(), proc_root=self.root))

    def test_custom_port_is_honored(self):
        self.table(port=8123)
        self.assertFalse(guard.owns_listener(self.identity(), port=8080, proc_root=self.root))
        with patch.object(guard.os, "readlink", return_value="socket:[10]"):
            self.assertTrue(guard.owns_listener(self.identity(), port=8123, proc_root=self.root))

    def test_identity_change_during_fd_scan_refuses(self):
        def reuse(_):
            self.stat(start=101)
            return "socket:[10]"
        with patch.object(guard.os, "readlink", side_effect=reuse):
            with self.assertRaisesRegex(guard.GuardError, "identity_changed"):
                guard.owns_listener(self.identity(), proc_root=self.root)

    def test_listener_change_during_fd_scan_refuses(self):
        with patch.object(guard, "listening_socket_inodes", side_effect=[{"10"}, {"20"}]), \
             patch.object(guard.os, "readlink", return_value="socket:[10]"):
            with self.assertRaisesRegex(guard.GuardError, "listener_changed"):
                guard.owns_listener(self.identity(), proc_root=self.root)

    def test_missing_tcp_tables_refuse(self):
        (self.root / "net" / "tcp").unlink()
        with self.assertRaisesRegex(guard.GuardError, "table_unavailable"):
            guard.listening_socket_inodes(proc_root=self.root)


class EvidenceGuards(unittest.TestCase):
    def idle(self):
        return {"live": dict.fromkeys(guard.LIVE_COUNTERS, 0)}, {"activity": {"in_flight": 0}}

    def test_all_explicit_zero_counters_accept(self):
        guard.require_idle(*self.idle())

    def test_busy_or_unknown_counters_refuse(self):
        for value in (1, None, False, "0"):
            with self.subTest(value=value):
                metrics, status = self.idle()
                metrics["live"]["admitting"] = value
                with self.assertRaisesRegex(guard.GuardError, "not_idle"):
                    guard.require_idle(metrics, status)
        metrics, status = self.idle()
        del metrics["live"]["waiting"]
        with self.assertRaises(guard.GuardError):
            guard.require_idle(metrics, status)

    def test_inflight_is_required_even_when_live_counters_are_zero(self):
        metrics, _ = self.idle()
        for status in ({}, {"activity": {"in_flight": 1}}, {"activity": {"in_flight": False}}):
            with self.subTest(status=status), self.assertRaises(guard.GuardError):
                guard.require_idle(metrics, status)

    def test_restore_checks_path_not_identical_file_contents(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original, copy_path = root / "production.json", root / "experiment.json"
            original.write_text("{}"); copy_path.write_text(original.read_text())
            self.assertEqual(guard.require_config_path(["python", "--config", str(original)], original), str(original))
            with self.assertRaisesRegex(guard.GuardError, "path_mismatch"):
                guard.require_config_path(["python", "--config", str(copy_path)], original)

    def test_duplicate_config_arguments_refuse(self):
        path = str(Path(tempfile.gettempdir()) / "production.json")
        with self.assertRaisesRegex(guard.GuardError, "path_mismatch"):
            guard.require_config_path(["--config", path, "--config=" + path], path)
        self.assertEqual(guard.require_config_path(["--config=" + path], path), path)

    def test_missing_config_value_refuses(self):
        path = str(Path(tempfile.gettempdir()) / "production.json")
        for argv in (["--config"], ["--config", "--port", "8080"]):
            with self.subTest(argv=argv), self.assertRaises(guard.GuardError):
                guard.require_config_path(argv, path)

    def geometry(self):
        return dict(expert_slots=13600, expert_slots_primary=8000,
                    expert_cache_mib=22344, expert_cache_primary_mib=11950)

    def test_exact_geometry_accepts_and_small_drift_refuses(self):
        reference = self.geometry()
        guard.require_exact_geometry(dict(reference), reference)
        for key in guard.GEOMETRY_FIELDS:
            changed = dict(reference); changed[key] += 1
            with self.subTest(key=key), self.assertRaisesRegex(guard.GuardError, "geometry_mismatch"):
                guard.require_exact_geometry(changed, reference)

    def test_missing_or_invalid_geometry_refuses(self):
        for value in (None, False, 0, float("nan"), float("inf")):
            changed = self.geometry(); changed["expert_slots"] = value
            with self.subTest(value=value), self.assertRaises(guard.GuardError):
                guard.require_exact_geometry(changed, self.geometry())

    def report(self):
        return {"requests": [{"id": name, "ok": True, "content_utf8_bytes": 10, "finish_reason": "length"}
                             for name in ("decode_a", "decode_b", "join_a", "join_b")]}

    def ids(self):
        return ("decode_a", "decode_b", "join_a", "join_b")

    def test_fixed_length_answer_emission_accepts_without_quality_claim(self):
        self.assertEqual(guard.require_answer_evidence(self.report(), self.ids()),
                         {"request_count": 4, "min_content_utf8_bytes": 10})

    def test_duplicate_ids_cannot_hide_missing_request(self):
        report = self.report(); report["requests"][3]["id"] = "decode_a"
        with self.assertRaisesRegex(guard.GuardError, "request_set_invalid"):
            guard.require_answer_evidence(report, self.ids())

    def test_reasoning_only_failed_or_missing_requests_refuse(self):
        for key, value in (("content_utf8_bytes", 0), ("content_utf8_bytes", True), ("ok", False)):
            report = self.report(); report["requests"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(guard.GuardError):
                guard.require_answer_evidence(report, self.ids())
        report = self.report(); report["requests"].pop()
        with self.assertRaises(guard.GuardError):
            guard.require_answer_evidence(report, self.ids())

    def test_empty_expected_workload_is_not_success(self):
        with self.assertRaises(guard.GuardError):
            guard.require_answer_evidence({"requests": []}, [])

    def test_inconclusive_completion_cannot_authorize_speed_claim(self):
        result = guard.verdict(execution_complete=True, correctness_passed=True,
                               inconclusive_gates=["output_equivalence"])
        self.assertEqual(result["status"], "complete_inconclusive")
        self.assertFalse(result["performance_interpretation_allowed"])

    def test_empty_inconclusive_list_alone_is_not_a_pass(self):
        self.assertFalse(guard.verdict(execution_complete=True)["performance_interpretation_allowed"])
        self.assertFalse(guard.verdict(correctness_passed=True)["performance_interpretation_allowed"])
        self.assertEqual(guard.verdict()["status"], "incomplete")

    def test_only_completed_and_validated_run_allows_interpretation(self):
        result = guard.verdict(execution_complete=True, correctness_passed=True)
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["performance_interpretation_allowed"])


if __name__ == "__main__":
    unittest.main()
