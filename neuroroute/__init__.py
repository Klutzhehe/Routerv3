"""NeuroRoute -- pure-RL PCB routing for many nets on many layers.

Read `neuroroute/DESIGN.md` for the architecture and why the two existing
threads in this repo cannot reach the target; `neuroroute/README.md` for what
is verified and what is not.

Nothing is imported eagerly here. `world.engine` pulls in torch, and several
verification scripts are meant to be runnable without paying that cost.
"""

__all__ = ["world", "env", "models", "training", "eval"]
__version__ = "0.1.0"
