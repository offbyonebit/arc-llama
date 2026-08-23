"""Pluggable text-to-speech engines behind `/v1/audio/speech`.

Importing this package registers every engine that ships with arc-llama. A new
one needs a module here that subclasses
:class:`~arc_llama.tts.base.TTSEngine`, calls
:func:`~arc_llama.tts.base.register_engine`, and is imported below — nothing
else in the tree dispatches on an engine name.
"""
from arc_llama.tts.base import (
    DEFAULT_TTS_HEALTH_TIMEOUT,
    TTSEngine,
    engine_names,
    engines,
    get_engine,
    register_engine,
    require_engine,
)
from arc_llama.tts.omnivoice import TTS_ENGINE_OMNIVOICE, OmniVoiceEngine

__all__ = [
    "DEFAULT_TTS_HEALTH_TIMEOUT",
    "TTSEngine",
    "TTS_ENGINE_OMNIVOICE",
    "OmniVoiceEngine",
    "engine_names",
    "engines",
    "get_engine",
    "register_engine",
    "require_engine",
]
