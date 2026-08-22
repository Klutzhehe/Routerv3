"""Unit test for scripts.generate_curriculum_boards."""

from pathlib import Path
from tests import fake_bridge

fake_bridge.install()

from scripts.generate_curriculum_boards import generate_curriculum_dataset


def test_generate_curriculum_dataset(tmp_path, monkeypatch):
    def mock_run(cmd, check=True):
        p = Path(cmd[2])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("fake board")
    monkeypatch.setattr("subprocess.run", mock_run)

    generate_curriculum_dataset(str(tmp_path), num_boards_per_stage=3)

    stage1 = tmp_path / "stage1_basics"
    stage2 = tmp_path / "stage2_corridors"
    stage3 = tmp_path / "stage3_production"

    assert len(list(stage1.glob("*.kicad_pcb"))) == 3
    assert len(list(stage2.glob("*.kicad_pcb"))) == 3
    assert len(list(stage3.glob("*.kicad_pcb"))) == 3
