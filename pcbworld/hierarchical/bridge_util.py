"""Small shared helpers for talking to `pcbworld_pns_bridge`.

Both of these exist because the same two mistakes were made independently in
every router in this package, and each of them turns a routing failure into
one that cannot be diagnosed from the report.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple


def pad_candidate(bridge: Any, x: int, y: int, layer: int = 0, slop_radius: int = 500_000) -> int:
    """The item id of the PAD at (x, y), not whatever the hit-test found first.

    `query_hover_items()` returns hits in the router's own hit-test order --
    not sorted by kind, and not by distance. On a board with committed copper
    an unrelated track routinely passes within the slop radius of a pad, so
    `candidates[0].id` hands `fix()` the id of that track instead of the pad
    the net is supposed to land on. `fix()` then refuses for a reason that has
    nothing to do with the route, and the failure is indistinguishable from a
    real collision in any aggregate count.

    That bug cost a Colab round in measure_waypoint_fidelity.py and is fixed
    in LineRouteEnv; every router in pcbworld/hierarchical/ still carried the
    `candidates[0]` pattern until this helper existed.
    """
    if not hasattr(bridge, "query_hover_items"):
        return -1
    candidates = bridge.query_hover_items(x, y, layer, slop_radius)
    if not candidates:
        return -1
    pads = [c for c in candidates if getattr(c, "kind", "") == "pad"]
    return (pads[0] if pads else candidates[0]).id


def push_path(
    bridge: Any, waypoints: Sequence[Tuple[int, int]], target: Tuple[int, int]
) -> Optional[dict]:
    """Push a corridor, reporting the FIRST waypoint the head collided at.

    Under `RM_MARK_OBSTACLES` push() is a validator: it marks the collision
    and keeps going, and `fix()` then refuses the whole route. Pushing the
    corridor blind and reading only fix()'s boolean means every failure
    reports "fix() failed" no matter what happened -- a plan that was wrong at
    its first waypoint and a plan that was fine until the last one are the
    same line in the report.

    Returns None when the head stayed clean, else a dict naming the leg and
    the obstacle, so a failed net says which part of its corridor was
    unroutable.
    """
    first_contact: Optional[dict] = None
    path = list(waypoints) + [tuple(target)]

    for index, (wx, wy) in enumerate(path):
        bridge.push(int(wx), int(wy))
        if first_contact is not None:
            continue
        if not hasattr(bridge, "head_collides") or not bridge.head_collides():
            continue

        first_contact = {
            "waypoint_index": index,
            "waypoint": (int(wx), int(wy)),
            "of": len(path),
        }
        # WHAT it hit, when the bridge can say. Another net's copper means the
        # corridor is genuinely wrong; the route's own target pad means the id
        # or the start was, which is a different fix entirely.
        if hasattr(bridge, "get_head_obstacle"):
            try:
                obstacle = bridge.get_head_obstacle()
            except Exception:  # a diagnostic must never take the run down
                obstacle = None
            if obstacle is not None and getattr(obstacle, "found", False):
                first_contact["obstacle_net"] = str(getattr(obstacle, "net", ""))
                first_contact["obstacle_kind"] = str(getattr(obstacle, "kind", ""))

    return first_contact


def describe_contact(contact: Optional[dict]) -> str:
    """One-line summary of push_path()'s result, for RouteResult.error_message."""
    if not contact:
        return ""
    where = f"waypoint {contact['waypoint_index'] + 1}/{contact['of']} at {contact['waypoint']}"
    net = contact.get("obstacle_net")
    if net:
        return f"head collided at {where} against {contact.get('obstacle_kind', 'item')} on net {net!r}"
    return f"head collided at {where}"
