"""Isolated library transfer: terminating the process closes all file handles."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path


def main() -> None:
    from arc_llama.models import HFModelSpec, download_from_hf

    job = json.loads(sys.argv[1])
    with contextlib.redirect_stdout(sys.stderr):
        path = download_from_hf(
            HFModelSpec(repo=job["repo"], file=job["file"], quant=None),
            target_dir=Path(job["target_dir"]), progress=False,
            revision=job["revision"], staging_root=Path(job["staging_root"]),
        )
    print(json.dumps({"path": str(path)}))


if __name__ == "__main__":
    main()
