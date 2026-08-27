"""PathFinder-style congestion price over the lattice.

`mzr/DESIGN.md` section 3. The price is what turns "all nets grow at once" from
a traffic jam into a negotiation: cells that several nets want become
expensive, so nets learn to spread out rather than race for the same channel.

    price(c) = (1 + h(c)) * (1 + p(c))

      p(c)  present congestion   -- contention for c right now, decays fast
      h(c)  historical congestion -- accumulated pressure, decays slowly

From McMurchie & Ebeling, FPGA '95. Its predecessor (Nair 1987) assigned
*infinite* cost to over-capacity resources; PathFinder's contribution was the
**gradual** penalty, and that gradualness is the whole reason this can be an
observation channel a policy learns to read rather than a mask that forbids
actions. A hard constraint would teach the policy nothing about *how much* a
cell is wanted.

Why history matters at all, given present congestion already exists: present
cost alone oscillates. Two nets both back off a contested cell, both find it
free next step, both retry, forever. History breaks the tie -- a cell that has
been hot for many steps stays expensive even when momentarily free, so nets
route around persistently-contended channels instead of thrashing through them.

**Where the contention signal comes from.** The engine's arbitration
(`geometry.resolve_claims`) already decides which cells more than one net
claimed in the same macro-step, because it has to. So congestion here is
*measured contention*, not an estimate from overlapping straight-line demand
-- and it costs one extra scatter over machinery the engine needs anyway.
"""

from __future__ import annotations

import torch

from mzr.world.spec import PriceRules


def contention(
    occ_shape: tuple[int, int, int, int],
    flats: list[torch.Tensor],
    valids: list[torch.Tensor],
    net_ids: list[torch.Tensor],
    *,
    min_buf: torch.Tensor,
    max_buf: torch.Tensor,
    count_buf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Which cells more than one *net* claimed this macro-step.

    Arbitration is by net, not by frontier, for the same reason
    `resolve_claims` is: the two frontiers of one net grow toward each other
    and the two legs of a differential pair travel a cell apart, so both
    routinely claim overlapping cells. That is not contention. Two *different*
    nets claiming one cell is.

    Detected by tracking the min and max claiming net id per cell: they differ
    exactly when at least two distinct nets claimed it. That is a boolean, not
    a count -- which is what PathFinder's ``h`` actually needs ("was this
    over-subscribed"), and it is exact, where a distinct-net *count* would need
    a sort per cell.

    ``count`` is returned alongside as total claiming frontiers per cell. It
    over-counts a net whose own frontiers overlap, so it is a pressure
    diagnostic rather than an over-subscription measure. Do not feed it to
    ``h`` -- that is what ``contested`` is for.

    The three ``*_buf`` arguments are flat (B*L*H*W,) scratch tensors owned by
    the caller, reused every step so this allocates nothing in the hot loop.

    Returns
    -------
    contested : (B, L, H, W) bool
    count : (B, L, H, W) int32
    """
    if not flats:
        zeros_b = torch.zeros(occ_shape, dtype=torch.bool, device=min_buf.device)
        zeros_c = torch.zeros(occ_shape, dtype=torch.int32, device=min_buf.device)
        return zeros_b, zeros_c

    big = torch.iinfo(torch.int32).max
    min_buf.fill_(big)
    max_buf.fill_(-big)
    count_buf.zero_()

    for flat, valid, nid in zip(flats, valids, net_ids):
        ids = nid.view(-1, 1).expand_as(flat).to(torch.int32)
        sel_flat = flat[valid]
        sel_ids = ids[valid]
        min_buf.scatter_reduce_(0, sel_flat, sel_ids, reduce="amin", include_self=True)
        max_buf.scatter_reduce_(0, sel_flat, sel_ids, reduce="amax", include_self=True)
        count_buf.scatter_add_(0, sel_flat, torch.ones_like(sel_ids))

    touched = min_buf != big
    contested = touched & (min_buf != max_buf)
    return contested.view(occ_shape), count_buf.view(occ_shape).clone()


class CongestionPrice:
    """Present and historical congestion over a batch of boards.

    Shapes match the occupancy grid exactly, ``(B, L, H, W)``, so the price
    drops straight into the field encoder as extra channels with no resampling.
    """

    def __init__(
        self,
        shape: tuple[int, int, int, int],
        rules: PriceRules,
        device: torch.device | str = "cpu",
    ) -> None:
        self.rules = rules
        self.shape = shape
        device = torch.device(device)
        self.present = torch.zeros(shape, dtype=torch.float32, device=device)
        self.history = torch.zeros(shape, dtype=torch.float32, device=device)

        n = int(shape[0] * shape[1] * shape[2] * shape[3])
        self._min_buf = torch.empty(n, dtype=torch.int32, device=device)
        self._max_buf = torch.empty(n, dtype=torch.int32, device=device)
        self._count_buf = torch.empty(n, dtype=torch.int32, device=device)

    @property
    def device(self) -> torch.device:
        return self.present.device

    def reset(self, mask: torch.Tensor | None = None) -> None:
        """Clear price. `mask` is (B,) bool to reset only some boards."""
        if mask is None:
            self.present.zero_()
            self.history.zero_()
            return
        sel = mask.view(-1, 1, 1, 1)
        self.present = torch.where(sel, torch.zeros_like(self.present), self.present)
        self.history = torch.where(sel, torch.zeros_like(self.history), self.history)

    def decay(self) -> None:
        """Age both fields by one macro-step.

        Separate from `update` so a caller can advance time without inventing
        claims -- the engine always has claims to report, but tests and any
        future "idle step" path do not, and faking an empty claim list to get
        decay would be a trap.
        """
        r = self.rules
        self.present.mul_(r.present_decay)
        self.history.mul_(r.history_decay)

    def update(
        self,
        flats: list[torch.Tensor],
        valids: list[torch.Tensor],
        net_ids: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decay, then accumulate this macro-step's measured contention.

        Decay happens *before* accumulation so a cell contested this very step
        is not immediately discounted -- otherwise the freshest signal is the
        weakest, which is backwards.

        Returns the ``(contested, count)`` fields, which the engine also uses
        for the reward's congestion term.
        """
        r = self.rules
        self.decay()

        contested, count = contention(
            self.shape,
            flats,
            valids,
            net_ids,
            min_buf=self._min_buf,
            max_buf=self._max_buf,
            count_buf=self._count_buf,
        )

        hot = contested.to(self.present.dtype)
        self.present.add_(hot * r.present_rate).clamp_(0.0, r.max_present)
        self.history.add_(hot * r.history_rate).clamp_(0.0, r.max_history)
        return contested, count

    def field(self) -> torch.Tensor:
        """The PathFinder price, ``(1 + h) * (1 + p)``, as (B, L, H, W).

        Minimum 1.0 on an uncontested cell, so it multiplies a base cost
        without changing it where nothing is contested.
        """
        return (1.0 + self.history) * (1.0 + self.present)

    def observation(self) -> torch.Tensor:
        """Price as two encoder channels, ``(B, 2, L, H, W)``.

        Present and history are kept **separate** rather than handed over as
        the single product from `field()`. They mean different things -- "who
        wants this now" versus "this has been fought over for a while" -- and
        collapsing them costs the policy the ability to tell a transient
        crossing from a genuinely oversubscribed channel. Each is squashed to
        roughly [0, 1] so neither dominates the encoder's input scale.
        """
        p = self.present / self.rules.max_present
        h = self.history / self.rules.max_history
        return torch.stack([p, h], dim=1)

    def total_oversubscription(self) -> torch.Tensor:
        """(B,) present congestion summed over the board.

        The reward's congestion term is the *change* in this between
        macro-steps, so a policy is charged for creating contention and
        credited for resolving it.
        """
        return self.present.flatten(1).sum(dim=1)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"present": self.present, "history": self.history}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.present.copy_(state["present"])
        self.history.copy_(state["history"])
