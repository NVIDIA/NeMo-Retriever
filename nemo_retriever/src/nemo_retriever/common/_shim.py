# SPDX-License-Identifier: Apache-2.0
"""Runtime helper for auto-generated deprecation shims (see SHIMS.md)."""
from __future__ import annotations

import importlib
import sys
import warnings


def alias(old_name: str, new_name: str) -> None:
    """Make the deprecated module ``old_name`` resolve to ``new_name``."""
    warnings.warn(
        f"{old_name} has moved to {new_name}; update your import.",
        DeprecationWarning,
        stacklevel=3,
    )
    sys.modules[old_name] = importlib.import_module(new_name)
