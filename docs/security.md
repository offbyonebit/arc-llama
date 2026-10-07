# Security and remote access

Arc Llama binds to `127.0.0.1` by default. Keep that default for a workstation
used by one person.

Binding to `0.0.0.0`, a LAN address, or a public interface exposes the inference
API to that network. llama-server backends always listen on `127.0.0.1`
regardless of `server.host`, so only Arc Llama's own routes are reachable.

## API keys and LAN mode

`arc-llama serve --lan` binds every interface after asking for confirmation
(`--yes` skips it), creates an API key named `lan` if none exists, and prints
the addresses other devices can use. Manage keys with:

```bash
arc-llama keys create phone     # prints the key once; only its hash is stored
arc-llama keys list             # usage counts and last use
arc-llama keys revoke <id>
```

Keys live in `<state_dir>/api-keys.json` (mode 0600), not in `config.toml`, and
a running server picks up changes immediately. The same operations are
available to admin-token holders at `GET/POST /admin/api-keys` and
`DELETE /admin/api-keys/{id}`.

The rule for `/v1/*` and `/api/*`:

- Loopback callers never need a key.
- Remote callers need `Authorization: Bearer <key>` (or the admin token) once
  at least one key exists.
- With no keys at all, remote callers are let through. This keeps existing
  Docker and LAN setups working; create a key to close it.

The bundled chat page asks for a key the first time a remote browser is
refused and remembers it in that browser.

Keys are bearer secrets sent in clear text over plain HTTP. On anything but a
trusted home network, put a reverse proxy with TLS and rate limits in front of
Arc Llama, or use a VPN or SSH tunnel.

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
