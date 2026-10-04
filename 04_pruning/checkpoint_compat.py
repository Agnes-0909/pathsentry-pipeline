"""Compatibility aliases for checkpoints created before the directory rename."""

from __future__ import annotations

import sys
import types


def register_pickle_aliases() -> None:
    """Make the historical ``prune.architectures`` pickle path resolve locally."""
    import architectures

    package = sys.modules.setdefault("prune", types.ModuleType("prune"))
    setattr(package, "architectures", architectures)
    sys.modules.setdefault("prune.architectures", architectures)
