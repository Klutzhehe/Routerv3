"""Prints pcbnew.B_Cu's real numeric PCB_LAYER_ID.

Nothing in this repo has ever needed the back copper layer's numeric id
before now: pns_bridge.cpp uses F_Cu/B_Cu symbolically on the C++ side
(KiCad's own enum), and the only raw layer int anywhere in pcbworld/env/*
is layer=0 for F_Cu (Colab-verified repeatedly). scripts/
measure_layer_hop_rescue.py needs a real numeric id to call switch_layer()
with -- guessing KiCad's PCB_LAYER_ID enum layout risks wasting a Colab
round chasing a "layer hop doesn't help" result that's actually just a
wrong constant, so this prints the real thing instead.

Deliberately uses the *system* pcbnew module, run as its own process --
same constraint as every other script here that touches pcbnew (see
docs/performance.md): never in a process that also has
pcbworld_pns_bridge loaded.
"""

import pcbnew

if __name__ == "__main__":
    print(pcbnew.B_Cu)
