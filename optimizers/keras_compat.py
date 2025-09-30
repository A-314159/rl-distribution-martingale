# keras_compat.py
from __future__ import annotations

import importlib
import importlib.util
import os
import types
from typing import Any
from typing import TYPE_CHECKING

# Optional: set KERAS_COMPAT_PREFER_STANDALONE=1 to prefer standalone 'keras'
_PREFER_STANDALONE = os.environ.get("KERAS_COMPAT_PREFER_STANDALONE", "0") == "1"


def _has(modname: str) -> bool:
    try:
        return importlib.util.find_spec(modname) is not None
    except Exception:
        return False


def _load_backend() -> Any:
    # Prefer tf.keras unless env says otherwise
    if not _PREFER_STANDALONE and _has("tensorflow"):
        tf = importlib.import_module("tensorflow")
        k = getattr(tf, "keras", None)
        if k is not None:
            return k
        if _has("tensorflow.keras"):
            return importlib.import_module("tensorflow.keras")
    if _has("keras"):
        return importlib.import_module("keras")

    class _Missing:
        def __getattr__(self, name):
            raise ImportError(
                "Neither 'tensorflow.keras' nor 'keras' is available on this environment."
            )

    return _Missing()


class _Lazy(types.ModuleType):
    def __init__(self, name: str):
        super().__init__(name)
        self._target = None

    def _ensure(self):
        if self._target is None:
            self._target = _load_backend()

    def __getattr__(self, name: str) -> Any:
        self._ensure()
        return getattr(self._target, name)

    def __dir__(self):
        self._ensure()
        return sorted(set(dir(self._target)))


# Expose a module-like object
keras = _Lazy("keras_compat.keras")


if TYPE_CHECKING:
    from types import ModuleType
    keras: ModuleType
