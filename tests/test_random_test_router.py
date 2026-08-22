"""Unit test for scripts.random_test_router."""

from pathlib import Path
from tests import fake_bridge

fake_bridge.install()

from scripts.random_test_router import run_random_test


def test_random_test_helper_runs(tmp_path, monkeypatch):
    # Mock subprocess.run for board generation to just write a fake file
    def mock_run(cmd, check=True):
        p = Path(cmd[2])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("fake board")
    monkeypatch.setattr("subprocess.run", mock_run)

    res = run_random_test(
        checkpoint_path=None,
        seed=42,
        num_nets=2,
        num_diff_pairs=0,
        num_length_groups=0,
        output_dir=str(tmp_path),
        show_plot=False,
    )

    assert res["seed"] == 42
    assert res["completed_nets"] == 2
    assert Path(res["render_png_path"]).exists()

