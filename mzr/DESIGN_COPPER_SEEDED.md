# Copper-seeded fields: the architecture multi-pin actually needs

Status: **proposal**, 2026-09-01. Supersedes the MST leg decomposition shipped
in `9f4b5ba`, which works but is a workaround.

## The one error everything traces back to

Each frontier's geodesic field targets a **static pad** -- the opposite end of
its leg -- and is computed once at `load()`. That single choice causes, in
order of how long it took to notice:

1. **Double-routing.** Frontier A walks toward B's pad while B walks toward A's
   pad. Neither field contains the other's copper, so on a symmetric board they
   mirror-route around opposite sides, pass without meeting, and both pad-snap.
   Measured on the gate-clearing stage-0 policy: **20 of 32 nets routed twice,
   1.94x the straight-line copper**, with `completion` reading 1.000 because the
   net *is* connected -- twice. The policy is behaving correctly; the field is
   telling it to.
2. **Multi-pin needs a pre-computed tree.** Because a frontier can only target a
   pad, a k-pin net has to be cut into k-1 two-pin legs *before routing*, by an
   MST over pin coordinates. That tree cannot see keepouts or congestion, and it
   is up to 1.5x longer than the rectilinear Steiner optimum.
3. **Legs cannot share copper.** A leg terminates on its assigned pad, never on
   copper the same net already laid two cells away -- which is exactly what a
   real router does, and why real multi-pin routing is short.
4. **Memory.** `fr_geo`, a coarse 3-D field **per frontier**, is the dominant
   memory term in the system. A k-pin net costs `2(k-1)` of them.

Four separate reward attempts failed to fix (1): leg-gap shaping (copper
1.94x -> 2.13x, lost the gate), terminal wirelength at 12x weight (-> 1.79x,
lost the gate), tip-distance shaping, and a hard per-frontier length budget
(copper -> 1.15x but completion collapsed without a gradient). They are all
patches over the field being wrong.

## The reframe

**A net is not a set of two-pin connections. It is a connected component that
grows until it contains every one of its pins.**

Route by growing copper until all pins share a component. Then:

* **One field per net**, not per frontier: a multi-source distance transform
  seeded from **every cell the net currently owns** -- its pads and all copper
  laid so far.
* **A frontier descends its own net's field**, so it is always heading for the
  nearest copper of its net, whichever pin or trace that belongs to.
* **A merge is "you touched your own net's copper."** No proximity test, no
  clear-segment test, no same-layer test -- the three-part `_try_meet` heuristic
  disappears.
* **A net is done when one flood fill from any pin reaches all of them.**

## What falls out for free

| | today | copper-seeded |
|---|---|---|
| multi-pin | MST into k-1 legs, fixed up front | native; pins join a component |
| junction points | only at pins | **Steiner junctions emerge** -- a frontier stops at the nearest *copper*, which is usually not a pin |
| double-routing | mirror-route, both finish | impossible: A's copper enters B's field the step it is laid |
| meet detection | proximity + clear segment + same layer | touching own-net copper |
| fields per k-pin net | `2(k-1)` | **1**, and frontiers retire as pins join |
| topology | frozen before routing | emerges from obstacles and congestion |

Stage 3 (8 nets): 32 fields -> 8. A 20-pin power net: 38 -> 1.

## The cost, stated plainly

The field is currently **static** -- computed once, never refreshed. That is the
property `DESIGN.md` chose it for, and this proposal gives it up. Two things
make that affordable:

* **Distance-to-copper only ever decreases** as copper is laid, so the field is
  updated by relaxing outward from newly-written cells, not recomputed. This is
  the standard incremental multi-source distance transform.
* Even a periodic full refresh maintains **one field per net** where today it
  maintains `2(k-1)`. At k=2 that is 1 vs 2; the accounting favours the new
  scheme at every pin count.

Refresh cadence becomes a knob (`neuroroute` had `--geodesic-refresh`); a stale
field is a performance issue, not a correctness one, because legality is
enforced by the engine and not by the field.

## Frontier lifecycle

* Seed **one frontier per pin** (k, not `2(k-1)`).
* Designate pin 0's copper the net's initial component; every other frontier
  grows toward the field.
* When a frontier's copper touches the component, **its pin has joined**:
  retire that frontier and fold its copper into the source set.
* Net done when no unjoined pins remain. Frontier count *shrinks* during an
  episode instead of staying fixed.

This also removes the idle-frontier pathology (1/32 nets had a frontier that
never moved): every live frontier has an unjoined pin behind it by definition.

## What breaks, and the migration order

1. `world/geometry.py` needs a multi-source seeded relaxation with an
   incremental entry point. The existing `geodesic_field` is single-target.
2. `fr_geo` moves from `(B, F, L, h, w)` to `(B, N, L, h, w)` -- the memory win,
   and the largest mechanical change.
3. `_try_snap` / `_try_meet` collapse into one "touched own copper" test.
4. `leg_valid` / `leg_done` become per-pin joined flags; `net_done` becomes a
   flood fill.
5. Differential pairs stay as they are -- a pair is genuinely two conductors
   that must run parallel, not one component. `KIND_DIFF_PAIR` keeps the current
   two-leg treatment; it is the one case the reframe does not cover.
6. The reward's `progress` term keeps working unchanged: it is still the drop in
   this frontier's field value. `tip_progress`, `leg_progress` and
   `leg_budget_frac` all become **dead code** and should be deleted, not left as
   flags -- they exist only to patch the bug this removes.

Order: geometry kernel first (testable alone), then the field store, then the
merge test, then delete the patches. `verify_world.py` is the gate at each step;
its flood-fill connectivity checks already encode the new definition of "done".

## The prediction this makes

If the analysis is right, copper/ideal should fall to ~1.0-1.1x **without any
reward term aimed at it**, because a frontier physically cannot walk past copper
it is being drawn toward. If it does not, the diagnosis in this document is
wrong and the reward patches were treating something else.

## Prior art -- this is standard routing practice, not a new idea

The confidence in this proposal comes from it being the established treatment of
multi-terminal nets in production routers, not from novelty.

* **McMurchie & Ebeling, "PathFinder: A Negotiation-Based Performance-Driven
  Router for FPGAs", FPGA'95** -- the negotiated-congestion foundation this
  repo's price model already follows.
  <https://www.cecs.uci.edu/~papers/compendium94-03/papers/1995/fpga95/pdffiles/6a.pdf>
* **VPR / VTR's Adaptive Incremental Router** is the confirmation of the
  mechanism proposed here: it routes a multi-sink net by *"only inserting
  portions of the routing tree of such nets into the priority queue when routing
  remaining connections"* -- that is, later sinks are searched for from the
  net's **already-laid routing**, not from its source pin. "Distance to the
  net's live copper" is precisely this, expressed as a field instead of a
  priority queue.
  <https://dl.acm.org/doi/fullHtml/10.1145/3406959>
* **Hwang's rectilinear Steiner ratio = 3/2** -- a rectilinear minimum spanning
  tree can be up to 1.5x the length of the rectilinear Steiner minimum tree.
  This is a proven bound, and it is the length the MST decomposition in
  `9f4b5ba` gives away *before routing starts*. Seeding from copper recovers it
  without computing a Steiner tree, because a frontier stops at the nearest
  copper -- which is generally not a pin, i.e. a Steiner point.
  <https://en.wikipedia.org/wiki/Rectilinear_Steiner_tree>
* **Lee-style maze routing on multi-terminal nets** expands the wavefront from
  the whole existing net rather than a single cell, for the same reason.
  <https://sites.math.unt.edu/~sgao/pub/paper15.pdf>

**What is NOT precedented**: doing this inside a *learned* simultaneous-frontier
policy. The RL-for-PCB literature -- XRoute (arXiv 2305.13823), the escape-routing
DQN work, DeepPCB -- routes pin-to-pin. So the routing technique is proven and
the RL combination is not; the risk lives in the combination and in refresh cost,
not in whether copper-seeded search is the right formulation.

## Calibration, stated before running anything

* correct architecture for multi-pin -- **high** (reference implementation exists)
* removes double-routing -- **high** (direct mechanism: A's copper is in B's
  field the step it is laid)
* affordable once the field stops being static -- **medium**. This is the real
  risk and the one to measure first.

## Measured: the kernel, and what the refresh actually costs

`geometry.geodesic_field_multi` is built and verified (step 1 of the migration).
Three properties, all exact:

* **single-source parity** with the existing `geodesic_field`: max |diff| = 0.0,
  identical `inf` pattern. It is a strict generalisation; nothing regresses.
* **monotone decrease**: adding a source only ever shortens distances. This is
  what makes an incremental refresh *valid*.
* **incremental == full recompute**: relaxing from the previous field after
  adding sources gives max |diff| = 0.0 against a rebuild from `inf`.

Cost per episode at stage-3 scale (8 nets, 4 layers, 64x64, 48 macro-steps),
CPU, coarse ds=4:

| | fields held | per-episode field time |
|---|---|---|
| today (static, one build) | 256 per-frontier | 160 ms |
| new, refresh every macro-step | 64 per-net | 1866 ms (**11.7x**) |
| new, refresh every 4 | 64 | 498 ms (3.1x) |
| new, refresh every 8 | 64 | 270 ms (1.7x) |
| new, refresh every 16 | 64 | 156 ms (0.98x) |

**Refreshing every step is not affordable** -- that must be said plainly, since
the earlier draft of this document implied the incremental path made the cost
disappear. It does not; it makes it tunable. Memory falls 4x regardless (256
fields -> 64), and a cadence of 8-16 buys the whole architecture for roughly
today's field budget.

A stale field is a *shaping* inaccuracy, never a legality one -- the engine
validates every move against live occupancy -- so cadence is a quality/speed
knob, exactly as `--geodesic-refresh` was in `neuroroute`. Start at 8.
