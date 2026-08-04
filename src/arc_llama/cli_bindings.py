"""Dynamic forwarding helpers for monkey-patchable CLI bindings.

Command modules import their dependencies through this module.  Each binding
reads from `arc_llama.cli` at access time, so tests that patch
`arc_llama.cli.<name>` continue to affect the command implementations even
after the split into per-command modules.
"""
from __future__ import annotations

from typing import Any

import arc_llama.cli as _cli


class _LazyBinding:
    """Callable proxy that resolves the bound name on `arc_llama.cli` each access."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _cli.__dict__[self._name](*args, **kwargs)

    def __get__(self, instance: object | None, owner: type | None) -> _LazyBinding:
        return self

    def __getattr__(self, name: str) -> Any:
        return getattr(_cli.__dict__[self._name], name)

    def __repr__(self) -> str:
        return f"<_LazyBinding arc_llama.cli.{self._name}>"


load_config = _LazyBinding("load_config")
default_config_path = _LazyBinding("default_config_path")
detect_gpus = _LazyBinding("detect_gpus")
init_config_from_detection = _LazyBinding("init_config_from_detection")
save_or_die = _LazyBinding("_save_or_die")
slugify_for_name = _LazyBinding("slugify_for_name")
resolve_llama_server = _LazyBinding("resolve_llama_server")
print_gpu_table = _LazyBinding("_print_gpu_table")
download_from_hf = _LazyBinding("download_from_hf")
add_local_model = _LazyBinding("add_local_model")
experimental_agent_enabled = _LazyBinding("experimental_agent_enabled")

__all__ = [
    "load_config",
    "default_config_path",
    "detect_gpus",
    "init_config_from_detection",
    "save_or_die",
    "slugify_for_name",
    "resolve_llama_server",
    "print_gpu_table",
    "download_from_hf",
    "add_local_model",
    "experimental_agent_enabled",
]
