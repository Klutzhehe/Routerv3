"""MZR -- simultaneous-growth PCB routing with Sampled/Gumbel MuZero.

See `mzr/DESIGN.md`. Every net grows from both pads at once in macro-steps, a
PathFinder congestion price makes nets negotiate instead of race, and search
(when it is switched on) branches over sampled joint moves of all live
frontiers.
"""

__all__ = ["world"]
