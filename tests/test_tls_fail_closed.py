"""Static guard: no code path may switch off TLS certificate verification.

Arc downloads runtime archives and models over HTTPS, and certificate failures on Windows
must be fixed by trust-store or CA-file configuration, never by disabling verification.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN = {
    "ssl.CERT_NONE": re.compile(r"\bCERT_NONE\b"),
    "verify=False": re.compile(r"\bverify\s*=\s*False\b"),
    "check_hostname=False": re.compile(r"\bcheck_hostname\s*=\s*False\b"),
    "unverified context": re.compile(r"_create_unverified_context|_create_default_https_context\s*="),
    "curl insecure": re.compile(r"(?:curl|wget)[^\n]*\s(?:-k|--insecure|--no-check-certificate)\b"),
}


def _sources() -> list[Path]:
    files: list[Path] = []
    for base in ("src", "scripts", "registry-repo/scripts"):
        files.extend(p for p in (ROOT / base).rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(files)


def test_no_source_disables_certificate_verification() -> None:
    hits = []
    for path in _sources():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            for label, pattern in FORBIDDEN.items():
                if pattern.search(code):
                    hits.append(f"{path.relative_to(ROOT)}:{lineno}: {label}")
    assert not hits, "TLS verification must stay on:\n" + "\n".join(hits)


def test_guard_detects_each_forbidden_pattern() -> None:
    samples = {
        "ssl.CERT_NONE": "ctx.verify_mode = ssl.CERT_NONE",
        "verify=False": "httpx.get(url, verify=False)",
        "check_hostname=False": "ctx.check_hostname = False",
        "unverified context": "ssl._create_unverified_context()",
        "curl insecure": "subprocess.run('curl -k https://example.test')",
    }
    for label, sample in samples.items():
        assert FORBIDDEN[label].search(sample), label
