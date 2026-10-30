"""Run the real piped-read piece selector and loop bounds with a recording reader.

Only the expensive read and decoder state are substituted. The selector and loop
prefix come from generate.cpp; chunk arithmetic comes from batch_read.hpp.
Run from any directory, optionally with --source-dir PATH and --cxx COMPILER.
"""
import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[2]
CXX = "g++"


def harness(source):
    text = (source / "src/program/generate.cpp").read_text(encoding="utf-8")
    selector = re.search(r"const bool pr_on = piped && pr_piece > 0;.*?auto piece_now = .*?\n\s*};", text, re.S)
    if not selector:
        raise ValueError("unsupported piped-read piece selector")
    prefix = re.search(r"int64_t piece = piece_now\(\);.*?for \(int64_t q = a0; q < b0;\) \{.*?if \(!(?:sp\.run\(ids\.data\(\) \+ q, r - q, q, e\)|run\(q, r\))\) return false;", text[selector.end():], re.S)
    if not prefix:
        raise ValueError("unsupported piped-read loop prefix")
    return r'''
#include "strata/program/batch_read.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
using Clock = std::chrono::steady_clock;
struct Reader {
    bool run(const int64_t*, int64_t count, int64_t offset, std::string&) {
        std::cout << offset << ":" << offset + count << "\n";
        return true;
    }
};
bool exercise(char** args) {
    const bool piped = std::atoi(args[1]);
    const int64_t pr_piece = std::atoll(args[2]);
    struct { int batch; } o{std::atoi(args[3])};
    std::vector<int> stages(std::atoi(args[4]));
    const int active_reads = std::atoi(args[5]);
    bool active = active_reads > 0;
    auto batch_on = [&]() { return active; };
    const int64_t C = std::atoll(args[6]), yield_chunks = std::atoll(args[7]);
    const int64_t a0 = std::atoll(args[8]), b0 = std::atoll(args[9]);
    std::vector<int64_t> ids(static_cast<size_t>(b0));
    Reader sp;
    std::string e;
    auto run = [&](int64_t a, int64_t b) { return sp.run(ids.data() + a, b - a, a, e); };
    int steps = 0;
''' + selector.group() + "\n" + prefix.group() + r'''
        q = r;
        if (++steps > 1000) return false;  // fail a zero-progress loop instead of hanging
        active = steps < active_reads;
    }
    return true;
}
int main(int argc, char** argv) {
    return argc == 10 && exercise(argv) ? 0 : 1;
}
'''


class PieceBehavior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which(CXX):
            raise unittest.SkipTest("C++ compiler unavailable: " + CXX)
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cpp = Path(cls.temp.name) / "piece_test.cpp"
        cpp.write_text(harness(SOURCE), encoding="utf-8")
        cls.binary = Path(cls.temp.name) / "piece_test"
        subprocess.run([CXX, "-std=c++17", "-O0", "-Wall", "-Wextra", "-I", str(SOURCE / "include"), str(cpp), "-o", str(cls.binary)], check=True, capture_output=True, text=True, timeout=60)

    def read(self, *, piped=1, piece=4096, batch=4, split=1, active=100, chunk=8192, yields=4, start=100, end=25100):
        result = subprocess.run([str(self.binary), *map(str, (piped, piece, batch, split, active, chunk, yields, start, end))], check=True, capture_output=True, text=True, timeout=10)
        return [tuple(map(int, line.split(":"))) for line in result.stdout.splitlines()]

    def test_idle_reads_whole_interval(self):
        self.assertEqual(self.read(active=0), [(100, 25100)])

    def test_last_decoder_ends_then_whole_remaining_interval(self):
        self.assertEqual(self.read(active=1), [(100, 8292), (8292, 25100)])

    def test_active_rounds_4096_up_to_one_8192_chunk(self):
        self.assertEqual(self.read(), [(100, 8292), (8292, 16484), (16484, 24676), (24676, 25100)])

    def test_active_rounded_down_and_final_tail(self):
        self.assertEqual(self.read(piece=20000), [(100, 16484), (16484, 25100)])

    def test_feature_off_piped_remains_whole(self):
        self.assertEqual(self.read(piece=0), [(100, 25100)])

    def test_feature_off_nonpiped_keeps_original_fixed_piece_after_decoder_ends(self):
        self.assertEqual(self.read(piped=0, active=1), [(100, 8292), (8292, 16484), (16484, 24676), (24676, 25100)])

    def test_feature_off_idle_layer_split_keeps_yield_chunks(self):
        self.assertEqual(self.read(piped=0, active=0, end=70100), [(100, 32868), (32868, 65636), (65636, 70100)])

    def test_feature_off_batch_disabled_remains_whole(self):
        self.assertEqual(self.read(piped=0, batch=0), [(100, 25100)])

    def test_active_piece_cap_and_short_remaining_interval(self):
        self.assertEqual(self.read(piece=999999, chunk=1, end=140), [(100, 132), (132, 140)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--cxx", default=CXX)
    args, remaining = parser.parse_known_args()
    SOURCE, CXX = args.source_dir.resolve(), args.cxx
    unittest.main(argv=[sys.argv[0], *remaining])
