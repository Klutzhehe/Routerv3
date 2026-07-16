"""Thin Python wrapper around the compiled pcbworld_pns_bridge extension.

The extension (pcbworld/engine/cpp/) is not built as part of a normal
`pip install` -- it must be compiled inside a KiCad source checkout, see
notebooks/00_setup.ipynb. This module just re-exports it with a friendlier
import path and adds nothing else yet.
"""

try:
    from pcbworld_pns_bridge import (  # noqa: F401
        Candidate,
        PNSBridge,
        MODE_ROUTE_SINGLE,
        MODE_ROUTE_DIFF_PAIR,
        MODE_TUNE_SINGLE,
        MODE_TUNE_DIFF_PAIR,
        MODE_TUNE_DIFF_PAIR_SKEW,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pcbworld_pns_bridge is not built/importable. Run "
        "notebooks/00_setup.ipynb first (it compiles the bridge inside a "
        "KiCad source checkout and adds it to sys.path)."
    ) from exc
