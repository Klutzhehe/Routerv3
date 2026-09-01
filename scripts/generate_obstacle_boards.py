"""Generates single-net PCB boards with intermediate obstacle components blocking the direct path."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

def create_board_with_obstacles(
    output_path: Path,
    num_obstacles: int = 4,
    board_width_mm: float = 50.0,
    board_height_mm: float = 50.0,
    seed: int = 42,
):
    random.seed(seed)
    # Start pad on left side, Target pad on right side
    start_x = 10.0
    start_y = 25.0
    target_x = 40.0
    target_y = 25.0

    # Obstacle pads placed in the middle corridor (x between 15 and 35)
    obstacle_pads = []
    for i in range(num_obstacles):
        ox = random.uniform(18.0, 32.0)
        oy = random.uniform(20.0, 30.0)
        obstacle_pads.append((ox, oy))

    lines = [
        '(kicad_pcb (version 20240108) (generator pcbworld)',
        '  (general (thickness 1.6))',
        '  (paper "A4")',
        '  (layers',
        '    (0 "F.Cu" signal)',
        '    (31 "B.Cu" signal)',
        '    (44 "Edge.Cuts" user "Edge.Cuts")',
        '  )',
        '  (setup',
        '    (pad_to_mask_clearance 0.05)',
        '    (pcbplotparams',
        '      (layerselection 0x00010_00000001)',
        '      (plotframeref false)',
        '      (viasonmask false)',
        '      (mode 1)',
        '      (usegerberextensions false)',
        '      (usegerberattributes true)',
        '      (usegerberadvancedattributes true)',
        '      (creategerberjobfile true)',
        '      (dashed_line_dash_ratio 12.000000)',
        '      (dashed_line_gap_ratio 3.000000)',
        '      (svgprecision 4)',
        '      (plotfilter 0)',
        '      (units 1)',
        '    )',
        '  )',
        '  (net 0 "")',
        '  (net 1 "net_target")',
    ]

    for i in range(len(obstacle_pads)):
        lines.append(f'  (net {i+2} "net_obs_{i+1}")')

    # Board outline
    lines.extend([
        f'  (gr_line (start 0 0) (end {board_width_mm} 0) (layer "Edge.Cuts") (stroke (width 0.1) (type default)))',
        f'  (gr_line (start {board_width_mm} 0) (end {board_width_mm} {board_height_mm}) (layer "Edge.Cuts") (stroke (width 0.1) (type default)))',
        f'  (gr_line (start {board_width_mm} {board_height_mm}) (end 0 {board_height_mm}) (layer "Edge.Cuts") (stroke (width 0.1) (type default)))',
        f'  (gr_line (start 0 {board_height_mm}) (end 0 0) (layer "Edge.Cuts") (stroke (width 0.1) (type default)))',
    ])

    # Target net pads (J1 pin 1 -> J2 pin 1)
    lines.extend([
        f'  (footprint "Connector_PinHeader_2.54mm:PinHeader_1x01_P2.54mm_Vertical" (layer "F.Cu")',
        f'    (at {start_x} {start_y})',
        '    (property "Reference" "J1" (at 0 -2.54 0) (layer "F.SilkS"))',
        '    (pad "1" thru_hole circle (at 0 0) (size 1.7 1.7) (drill 1.0) (layers *.Cu *.Mask) (net 1 "net_target"))',
        '  )',
        f'  (footprint "Connector_PinHeader_2.54mm:PinHeader_1x01_P2.54mm_Vertical" (layer "F.Cu")',
        f'    (at {target_x} {target_y})',
        '    (property "Reference" "J2" (at 0 -2.54 0) (layer "F.SilkS"))',
        '    (pad "1" thru_hole circle (at 0 0) (size 1.7 1.7) (drill 1.0) (layers *.Cu *.Mask) (net 1 "net_target"))',
        '  )',
    ])

    # Obstacle pads blocking the middle
    for idx, (ox, oy) in enumerate(obstacle_pads):
        net_id = idx + 2
        net_name = f"net_obs_{idx+1}"
        ref = f"OBS{idx+1}"
        lines.extend([
            f'  (footprint "Connector_PinHeader_2.54mm:PinHeader_1x01_P2.54mm_Vertical" (layer "F.Cu")',
            f'    (at {ox:.2f} {oy:.2f})',
            f'    (property "Reference" "{ref}" (at 0 -2.54 0) (layer "F.SilkS"))',
            f'    (pad "1" thru_hole rect (at 0 0) (size 2.5 2.5) (drill 1.2) (layers *.Cu *.Mask) (net {net_id} "{net_name}"))',
            '  )',
        ])

    lines.append(')')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate obstacle training boards")
    parser.add_argument("--output-dir", type=Path, default=Path("/content/boards/obstacle_stage"))
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.count):
        out_file = args.output_dir / f"obs_board_{i:03d}.kicad_pcb"
        create_board_with_obstacles(out_file, num_obstacles=random.randint(2, 6), seed=1000+i)
    print(f"Generated {args.count} obstacle boards in {args.output_dir}")
