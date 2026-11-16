"""Repository root, resolved once. Every script locates data/, configs/ and results/ from here.

Override with the environment variable MECHCONF_ROOT (e.g. when the package is installed
non-editably and run from a checkout elsewhere)."""
from __future__ import annotations

import os


def _find_root() -> str:
    env = os.environ.get("MECHCONF_ROOT")
    if env:
        return os.path.abspath(env)
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.exists(os.path.join(d, "pyproject.toml")) and os.path.isdir(os.path.join(d, "experiments")):
            return d
        d = os.path.dirname(d)
    return os.getcwd()


REPO_ROOT = _find_root()
