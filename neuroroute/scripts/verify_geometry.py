"""Exactness checks for the lattice geometry. CPU only, no GPU, no KiCad.

Every routing decision the policy makes is filtered through `check_moves` and
`raycast`. If either is wrong, training optimises against a fiction and the
first real DRC run is where you find out -- expensively. So each is checked
against an independently written brute-force reference that walks cells in
plain Python, the same discipline `scripts/verify_spatial_encoder.py` used to
get its 12,000/12,000 result.

Run:  python -m neuroroute.scripts.verify_geometry
"""

from __future__ import annotations

import itertools
import sys

import numpy as np
import torch

from neuroroute.world import geometry as G
from neuroroute.world.spec import (
    DIRECTION_VECTORS,
    NUM_DIRECTIONS,
    OCC_FREE,
    STEP_LENGTHS,
    BoardSpec,
    DesignRules,
)

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------------------


def ref_move_cells(y, x, d, n, radius):
    """Brute-force reference: cells a move touches, as a set."""
    dy, dx = DIRECTION_VECTORS[d]
    diagonal = dy != 0 and dx != 0
    centres = []
    for k in range(1, n + 1):
        centres.append((k * dy, k * dx))
        if diagonal:
            centres.append((k * dy, (k - 1) * dx))
            centres.append(((k - 1) * dy, k * dx))
    out = set()
    for cy, cx in centres:
        for oy in range(-radius, radius + 1):
            for ox in range(-radius, radius + 1):
                out.add((y + cy + oy, x + cx + ox))
    return out


def test_pitch_and_radii():
    print("\n[1] design-rule arithmetic")
    r = DesignRules()
    check("pitch = min_width + min_clearance", abs(r.pitch_mm - 0.4) < 1e-9, f"{r.pitch_mm} mm")
    check("min-width trace occupies 1 cell", r.width_radius_cells(0) == 0)
    check("width radii are monotone", list(r.width_radii()) == sorted(r.width_radii()))
    check("via radii are monotone", list(r.via_radii()) == sorted(r.via_radii()))
    # A 0.6 mm via plus 0.2 mm clearance spans 0.8 mm = 2 cells -> radius 1.
    check("0.6mm via -> radius 1", r.via_radius_cells(0) == 1, f"r={r.via_radius_cells(0)}")

    spec = BoardSpec(height_cells=64, width_cells=64)
    xs = np.array([0, 7, 63])
    mx, my = spec.cell_to_mm(xs, xs)
    bx, by = spec.mm_to_cell(mx, my)
    check("cell -> mm -> cell round-trips", bool(np.array_equal(bx, xs) and np.array_equal(by, xs)))


def test_move_cells_match_reference():
    print("\n[2] move footprint vs brute force")
    rules = DesignRules()
    tables = G.build_tables(rules, "cpu")
    mismatches = 0
    total = 0
    for d, si, wc in itertools.product(range(NUM_DIRECTIONS), range(len(STEP_LENGTHS)), range(rules.num_width_classes)):
        y, x = 40, 40
        cy, cx, valid = G._move_cells(
            tables,
            torch.tensor([y]),
            torch.tensor([x]),
            torch.tensor([d]),
            torch.tensor([si]),
            torch.tensor([wc]),
        )
        got = {(int(a), int(b)) for a, b in zip(cy[valid].tolist(), cx[valid].tolist())}
        want = ref_move_cells(y, x, d, STEP_LENGTHS[si], rules.width_radius_cells(wc))
        total += 1
        if got != want:
            mismatches += 1
    check("all (dir, step, width) footprints exact", mismatches == 0, f"{total - mismatches}/{total}")


def test_check_moves_vs_bruteforce():
    print("\n[3] check_moves vs brute force, random occupancy")
    rng = np.random.default_rng(7)
    rules = DesignRules()
    tables = G.build_tables(rules, "cpu")
    B, L, H, W = 3, 4, 48, 48

    mismatch = 0
    trials = 0
    for _ in range(30):
        occ = torch.from_numpy(
            (rng.random((B, L, H, W)) < 0.08).astype(np.int16) * rng.integers(1, 6, (B, L, H, W)).astype(np.int16)
        )
        M = 64
        b = torch.from_numpy(rng.integers(0, B, M))
        lay = torch.from_numpy(rng.integers(0, L, M))
        y = torch.from_numpy(rng.integers(10, H - 10, M))
        x = torch.from_numpy(rng.integers(10, W - 10, M))
        d = torch.from_numpy(rng.integers(0, NUM_DIRECTIONS, M))
        s = torch.from_numpy(rng.integers(0, len(STEP_LENGTHS), M))
        wc = torch.from_numpy(rng.integers(0, rules.num_width_classes, M))
        nid = torch.from_numpy(rng.integers(0, 5, M))

        ok, free = G.check_moves(occ, b, lay, y, x, d, s, wc, nid, tables)

        occ_np = occ.numpy()
        for i in range(M):
            radius = rules.width_radius_cells(int(wc[i]))
            n = STEP_LENGTHS[int(s[i])]
            cells = ref_move_cells(int(y[i]), int(x[i]), int(d[i]), n, radius)
            good = True
            for (cyi, cxi) in cells:
                if not (0 <= cyi < H and 0 <= cxi < W):
                    good = False
                    break
                v = occ_np[int(b[i]), int(lay[i]), cyi, cxi]
                if v != OCC_FREE and v != int(nid[i]) + 1:
                    good = False
                    break
            trials += 1
            if good != bool(ok[i]):
                mismatch += 1
    check("legality matches brute force", mismatch == 0, f"{trials - mismatch}/{trials}")


def test_free_units_monotone():
    print("\n[4] free_units consistency")
    rng = np.random.default_rng(11)
    rules = DesignRules()
    tables = G.build_tables(rules, "cpu")
    L, H, W = 2, 40, 40
    occ = torch.from_numpy((rng.random((1, L, H, W)) < 0.1).astype(np.int16))
    M = 128
    b = torch.zeros(M, dtype=torch.long)
    lay = torch.from_numpy(rng.integers(0, L, M))
    y = torch.from_numpy(rng.integers(10, H - 10, M))
    x = torch.from_numpy(rng.integers(10, W - 10, M))
    d = torch.from_numpy(rng.integers(0, NUM_DIRECTIONS, M))
    nid = torch.zeros(M, dtype=torch.long)
    wc = torch.zeros(M, dtype=torch.long)

    bad = 0
    per_class = []
    for si in range(len(STEP_LENGTHS)):
        s = torch.full((M,), si, dtype=torch.long)
        ok, free = G.check_moves(occ, b, lay, y, x, d, s, wc, nid, tables)
        per_class.append((ok, free))
        # `ok` must be exactly "the whole step length was clear".
        if not torch.equal(ok, free >= STEP_LENGTHS[si]):
            bad += 1
    check("ok == (free_units >= step length), every class", bad == 0)

    # free_units is a property of the ray, so it must agree across step classes
    # wherever the shorter class did not truncate the reading.
    longest = per_class[-1][1]
    agree = all(
        bool(((per_class[i][1] == longest) | (per_class[i][1] == STEP_LENGTHS[i])).all())
        for i in range(len(STEP_LENGTHS))
    )
    check("free_units agrees across step classes", agree)


def test_raycast_and_step_safety():
    print("\n[5] raycast + per-(direction, step) safety")
    rng = np.random.default_rng(3)
    rules = DesignRules()
    tables = G.build_tables(rules, "cpu")
    L, H, W = 3, 40, 40
    occ = torch.from_numpy((rng.random((2, L, H, W)) < 0.12).astype(np.int16) * 3)

    M = 96
    b = torch.from_numpy(rng.integers(0, 2, M))
    lay = torch.from_numpy(rng.integers(0, L, M))
    y = torch.from_numpy(rng.integers(12, H - 12, M))
    x = torch.from_numpy(rng.integers(12, W - 12, M))
    nid = torch.full((M,), 2, dtype=torch.long)
    wc = torch.zeros(M, dtype=torch.long)

    free = G.raycast(occ, b, lay, y, x, nid, tables, wc)
    safe = G.step_safety(free)

    # Cross-check: every (direction, step) safety bit must equal check_moves.
    bad = 0
    for d in range(NUM_DIRECTIONS):
        for si, n in enumerate(STEP_LENGTHS):
            ok, _ = G.check_moves(
                occ, b, lay, y, x,
                torch.full((M,), d, dtype=torch.long),
                torch.full((M,), si, dtype=torch.long),
                wc, nid, tables,
            )
            if not torch.equal(ok, safe[:, d, si]):
                bad += 1
    check("step_safety == check_moves for all (dir, step)", bad == 0, f"{NUM_DIRECTIONS * len(STEP_LENGTHS) - bad} combos exact")
    check("raycast range is [0, max step]", bool(((free >= 0) & (free <= max(STEP_LENGTHS))).all()))


def test_stamp_and_erase():
    print("\n[6] stamp / erase round-trip")
    rules = DesignRules()
    tables = G.build_tables(rules, "cpu")
    occ = torch.zeros(1, 2, 32, 32, dtype=torch.int16)

    b = torch.tensor([0]); lay = torch.tensor([1])
    y = torch.tensor([16]); x = torch.tensor([8])
    d = torch.tensor([1]); s = torch.tensor([2]); wc = torch.tensor([1]); nid = torch.tensor([4])

    G.stamp_moves(occ, b, lay, y, x, d, s, wc, nid, tables)
    want = ref_move_cells(16, 8, 1, STEP_LENGTHS[2], rules.width_radius_cells(1))
    got = {(int(a), int(c)) for a, c in zip(*torch.nonzero(occ[0, 1], as_tuple=True))}
    check("stamped cells match reference footprint", got == want, f"{len(got)} cells")
    check("stamp wrote net_id+1", bool((occ[occ != 0] == 5).all()))
    check("other layer untouched", bool((occ[0, 0] == 0).all()))

    # A second net must not be able to overwrite the first.
    G.stamp_moves(occ, b, lay, y, x, d, s, wc, torch.tensor([9]), tables)
    check("stamp never clobbers another net", bool((occ[occ != 0] == 5).all()))

    # Erase clears only the owner's copper.
    occ2 = torch.zeros(1, 1, 24, 24, dtype=torch.int16)
    args = (occ2, torch.tensor([0]), torch.tensor([0]),
            torch.tensor([4]), torch.tensor([4]), torch.tensor([4]), torch.tensor([18]),
            torch.tensor([0]))
    G.stamp_segments(*args, torch.tensor([2]), tables)
    n_before = int((occ2 != 0).sum())
    G.stamp_segments(*args, torch.tensor([7]), tables, erase=True)
    check("erase ignores another net's copper", int((occ2 != 0).sum()) == n_before)
    G.stamp_segments(*args, torch.tensor([2]), tables, erase=True)
    check("erase clears the owner's copper", int((occ2 != 0).sum()) == 0)


def test_geodesic():
    print("\n[7] geodesic field")
    L, H, W = 2, 32, 32
    blocked = torch.zeros(1, L, H, W, dtype=torch.bool)
    fld = G.geodesic_field(
        blocked,
        torch.tensor([0]), torch.tensor([16]), torch.tensor([16]),
        iterations=64, downsample=1, via_cost=4.0,
    )
    check("target cell has zero cost", abs(float(fld[0, 0, 16, 16])) < 1e-6)
    check("field is finite on an empty board", bool(torch.isfinite(fld).all()))
    # 8-connected relaxation gives Chebyshev distance in-plane.
    check("in-plane cost is Chebyshev", abs(float(fld[0, 0, 16, 26]) - 10.0) < 1e-4, f"{float(fld[0,0,16,26]):.3f}")
    check("cross-layer cost adds via_cost", abs(float(fld[0, 1, 16, 16]) - 4.0) < 1e-4, f"{float(fld[0,1,16,16]):.3f}")

    # A wall the field must route around, not through.
    blocked = torch.zeros(1, 1, 32, 32, dtype=torch.bool)
    blocked[0, 0, :28, 16] = True
    fld = G.geodesic_field(
        blocked, torch.tensor([0]), torch.tensor([4]), torch.tensor([24]),
        iterations=128, downsample=1,
    )
    detour = float(fld[0, 0, 4, 8])
    check("field detours around a wall", detour > 16.0, f"cost {detour:.1f} vs straight-line 16")

    # Descent direction must point somewhere that reduces the field.
    tables = G.build_tables(DesignRules(), "cpu")
    pos_y = torch.tensor([4]); pos_x = torch.tensor([8]); lay = torch.tensor([0])
    d = G.descent_direction(fld, lay, pos_y, pos_x, tables)
    dy, dx = DIRECTION_VECTORS[int(d)]
    nxt = float(fld[0, 0, 4 + dy, 8 + dx])
    check("descent direction reduces cost", nxt < detour, f"{detour:.1f} -> {nxt:.1f}")


def test_via_span():
    print("\n[8] via occupancy across layers")
    rules = DesignRules()
    tables = G.build_tables(rules, "cpu")
    occ = torch.zeros(1, 6, 24, 24, dtype=torch.int16)
    G.stamp_via(
        occ, torch.tensor([0]), torch.tensor([0]), torch.tensor([5]),
        torch.tensor([12]), torch.tensor([12]), torch.tensor([0]), torch.tensor([3]), tables,
    )
    per_layer = [(occ[0, i] != 0).sum().item() for i in range(6)]
    check("through via marks every layer", all(c > 0 for c in per_layer), f"{per_layer}")
    check("through via is identical on each layer", len(set(per_layer)) == 1)

    occ2 = torch.zeros(1, 6, 24, 24, dtype=torch.int16)
    G.stamp_via(
        occ2, torch.tensor([0]), torch.tensor([2]), torch.tensor([4]),
        torch.tensor([12]), torch.tensor([12]), torch.tensor([1]), torch.tensor([3]), tables,
    )
    spanned = [i for i in range(6) if (occ2[0, i] != 0).any()]
    check("buried via marks only its span", spanned == [2, 3, 4], f"{spanned}")


def main() -> int:
    torch.manual_seed(0)
    print("=" * 68)
    print("NeuroRoute lattice geometry verification")
    print("=" * 68)
    test_pitch_and_radii()
    test_move_cells_match_reference()
    test_check_moves_vs_bruteforce()
    test_free_units_monotone()
    test_raycast_and_step_safety()
    test_stamp_and_erase()
    test_geodesic()
    test_via_span()

    print("\n" + "=" * 68)
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
        return 1
    print("All geometry checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
