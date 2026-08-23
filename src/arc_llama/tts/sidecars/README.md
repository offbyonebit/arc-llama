Standalone backend scripts, launched by path under a *foreign* interpreter —
the one that has the engine's own dependencies installed.

Nothing in here may import `arc_llama`, and nothing else belongs in this
directory. Running a script puts its directory at the front of `sys.path`, so
any module placed here would shadow a same-named library or stdlib module for
every sidecar. That is exactly what happened when these scripts lived one level
up next to `omnivoice.py`: `from omnivoice import OmniVoice` picked up
arc-llama's engine module instead of the installed package.
