"""The llama.cpp build that Arc Llama installs by default.

Kept in its own module so the command line can show it without importing the
installer. Bump this one line to move new installs, on both backends and on
Linux and Windows, to a newer build; ``--runtime-version latest`` still asks for
the newest build that has the needed download.
"""

LLAMA_CPP_BUILD = "b12193"
