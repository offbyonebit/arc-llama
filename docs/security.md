# Security and remote access

Arc Llama binds to `127.0.0.1` by default. Keep that default for a workstation
used by one person.

Binding to `0.0.0.0`, a LAN address, or a public interface exposes the inference
API to that network. The admin routes require the generated admin token, while
the OpenAI-compatible inference routes are intended for trusted local clients
and do not provide complete internet-facing authentication. Put a reverse proxy
with TLS, authentication, request-size limits, and rate limits in front of Arc
Llama before allowing remote access. A VPN or SSH tunnel is usually simpler.

Set `ARC_LLAMA_ADMIN_TOKEN` to a long random value when running as a service.
Do not place it in browser URLs, logs, screenshots, or a public configuration
repository. The bundled UI can request the token only from a loopback peer and
rejects cross-origin token requests. Configure `ARC_LLAMA_CORS_ORIGINS` with an
explicit comma-separated allowlist when a separate trusted web client needs
browser access.

Runtime downloads are checked against the release asset size and SHA-256 digest
when GitHub supplies one. Archive paths and special files are validated before
extraction, and a failed install does not replace the working runtime.

Community recipe registries are untrusted input. Downloads have a 16 MiB limit,
full schema and value validation, safe recipe-field allowlists, and atomic
installation. A shared recipe is still advisory: retain the provenance check and
local A/B benchmark unless you have independently verified the source.

Plugins run inside the Arc Llama process with the service account's filesystem
and network access. Install only code you trust and review its declared routes.
Model output is untrusted too; the chat UI escapes raw HTML and filters unsafe
Markdown link and image protocols before inserting rendered content.
