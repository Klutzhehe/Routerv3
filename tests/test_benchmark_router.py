"""Unit test for scripts.benchmark_router against fake_bridge."""

from pathlib import Path
from tests import fake_bridge

fake_bridge.install()

from scripts.benchmark_router import run_benchmark_suite, _classify_net


def test_classify_net_helper():
    assert _classify_net("net_0") == "plain"
    assert _classify_net("diffpair_0_P") == "diffpair"
    assert _classify_net("lengthgrp_0_1") == "lengthgrp"


def test_benchmark_suite_runs_against_fake_bridge(tmp_path):
    board_file = tmp_path / "test_board.kicad_pcb"
    board_file.write_text("fake board content")

    json_file = tmp_path / "bench_summary.json"
    render_dir = tmp_path / "renders"

    results, summary = run_benchmark_suite(
        [str(board_file)],
        checkpoint_path=None,
        enable_ripup=True,
        render_dir=str(render_dir),
        json_out=str(json_file),
    )

    assert len(results) == 1
    assert summary.total_boards == 1
    assert json_file.exists()
