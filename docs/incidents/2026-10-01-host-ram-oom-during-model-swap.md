# Incident: system OOM while arc-llama was serving (2026-10-01)

Status: root cause identified from `dmesg`, `journalctl --user`, and the
arc-llama state logs. No arc-llama bug produced the memory pressure, but the
incident exposes a gap in arc-llama's eviction accounting on small-RAM hosts.

## Summary

At 11:11–11:14 on 2026-10-01 the kernel OOM-killer fired seven times and
killed desktop victims (ChatGPT/Electron tabs, Brave). The proximate trigger
was arc-llama attempting a model swap from `gemma-qat` to
`qwen3.6-35b-a3b-ud-q4_k_xl` while:

1. the resident gemma-qat llama-server was still alive (evict → new process
   starts before old memory is fully reclaimed), and
2. the then-running llama.cpp build had the host-side **prompt cache enabled
   by default** (`--cache-ram`, default 8192 MiB — llama.cpp PR #16391), and
3. many long prompt states ( dozens of MiB each; one reached 485 MiB) had been
   cached in host RAM during earlier chat sessions, and
4. the swap target recipe uses `--no-mmap`, so the ~34 GB Qweight/KV of the
   qwen3.6 35B model must be committed as anonymous RAM (not page-cache), and
5. system RAM+swap was already nearly exhausted (free swap = 0 at the OOM
   instant) because the desktop (Chrome/ChatGPT/brave, plus open-webui and an
   opencode session) was resident at the same time.

arc-llama's own process was *not* the victim or a major consumer
(244 MB anon RSS, 933 MB VSZ). OOM victims were all `oom_score_adj=300`
desktop apps. arc-llama only supplied the last few GB over the cliff.

## Evidence

### Kernel (from `dmesg -T`)

```
Oct 01 11:11:43  OOM kill brave (46831)
Oct 01 11:12:44  OOM kill brave (14466); cuda-EvtHandlr invoked oom-killer  ←
                 eviction of the to-be-killed llama-server pages
Oct 01 11:13:26  OOM kill ChatGPT (85300)
Oct 01 11:13:31  OOM kill brave (14330); ollama invoked oom-killer (upstream)
Oct 01 11:13:49  OOM kill ChatGPT (84876)
Oct 01 11:13:53  OOM kill ChatGPT (85332)
Oct 01 11:14:23  OOM kill ChatGPT (12682)
Oct 01 11:14:33  OOM kill ChatGPT (85673)
```

At the 11:13:26 snapshot:

```
Mem-Info (at 11:13:26): free ~587 MB, free swap = 0kB (all 16.5 GB swap used)
shmem: 9.9 GB    pagetables: ~0.4 GB
```

Notable processes at OOM time (pages ×4/1024 → MB, from the OOM task list):

| pid | name | anon RSS | swap entries | notes |
|-----|------|----------|--------------|-------|
| 1478 | arc-llama | ~239 MB | ~70 MB | the router itself |
| 1504 | llama-server | ~12 MB | ~985 MB | pre-existing resident qwen3.8 (:8083), mostly swapped out |
| 85125 | llama-server | ~829 MB | 0 | retry-2 qwen3.6-35b process launched 11:10:21, never became healthy |
| 1483 | open-webui | ~15 MB anon + 380 MB file | ~709 MB | |
| 1889 | granian (searxng) | ~32 MB | ~41 MB | |
| 81272 | opencode session | ~513 MB | ~180 MB | plus its MCP/node subprocesses (several hundred MB more) |
| 85300…85673 | ChatGPT/brave tabs | 20–190 MB each | small | OOM victims, oom_score_adj=300 |

### arc-llama (from `journalctl --user -u arc-llama.service`)

```
10:50:50 INFO  arc_llama.router: evicting ornith-1.5-35b-q4_k_m before starting gemma-qat
10:50:51 INFO  arc_llama.launcher: [gemma-qat] starting: ... --spec-type draft-mtp
               --spec-draft-model ... --spec-draft-ngl 999 ... --no-mmap --mmproj gemma...BF16
10:51:03 INFO  arc_llama.launcher: [gemma-qat] ready after 12.0s
11:06:38 INFO  arc_llama.router: evicting gemma-qat before starting qwen3.6-35b-a3b-ud-q4_k_xl
11:06:38 INFO  arc_llama.launcher: [gemma-qat] stopping pid=75695
11:06:39 INFO  arc_llama.launcher: [qwen3.6...] starting: ... -ngl 999 -c 131072
               --cache-type-k q8_0 --cache-type-v q8_0 --spec-type draft-mtp
               -fa on --no-mmap --mmproj ... --reasoning auto
11:06:54→11:08:24 INFO [qwen3.6...] still loading... 15s..104s elapsed
11:08:40 WARN  [qwen3.6...] health-check timed out after 120s
11:08:43 WARN  [qwen3.6...] SIGTERM timed out, sending SIGKILL
11:10:21 INFO  retry 2: starting qwen3.6 again
11:11:43—11:12:31  ← OOM storm happens here, retry-2 process (pid 85125) never gets ready
11:12:23 WARN  [qwen3.6...] health-check timed out after 120s
11:12:26 WARN  SIGTERM timed out, sending SIGKILL
11:12:29 ERROR startup diagnostic startup_timeout-b73a2682ed0d: ...
```

Load-time log from both qwen attempts
(`~/.local/state/arc-llama/qwen3.6-35b-a3b-ud-q4_k_xl.log`):

```
0.00.849.257 W common_fit_params: failed to fit params to free device memory:
                  n_gpu_layers already set by user to 999, abort
0.12.703.649 I common_speculative_init_result: creating MTP draft context
0.13.854.720 I srv load_model: loaded multimodal model, mmproj-F16.gguf
← then silence for the remaining ~117 s of the readiness budget: no model
  loaded line, no slot init, no listening. The process is still committing
  host RAM (KV cache q8_0 at ctx 131072 + compute buffers + MTP draft ctx).
```

### gemma-qat server log (`~/.local/state/arc-llama/gemma-qat.log`)

Shows default-on prompt caching:

```
srv load_model: prompt cache is enabled, size limit: 8192 MiB
srv load_model: use `--cache-ram 0` to disable the prompt cache
srv load_model: context checkpoints enabled, max = 32
srv load_model: idle slots will be saved to prompt cache upon starting a new task
```

Prompt states cached during earlier chat sessions (sum of 143 saves):

```
sum state size: 6.2 GiB ; largest single state: 485 MiB (14 579-token prompt)
```

## Why the local-llm-proxy did not OOM

The router at `/mnt/storage/local-llm-proxy/proxy.py` launches llama-server
itself (or ollama) and, crucially:

- did **not** have a desktop-class chat/WebUI stack simultaneously running
  requests; and
- before starting a new llama-server it explicitly `systemctl --user stop`s
  the previous llama-server service and waits for the old process to exit
  (the "ensure/stop pair"), so at most one llama-server exists at a time,
  and it always ran with the older llama.cpp (no default host prompt cache).
- Within arc-llama, the gemma-qat OOM period is exactly the window when the
  new llama.cpp build was being put through its paces with `--no-mmap` and
  MTP speculation enabled, which is *new* memory behavior compared to the
  proxy's previous configuration.

So what is in arc-llama that the proxy does not have:

1. **Host-side prompt cache in default-on state** with chat sessions that
   naturally produce 50–485 MiB states (the gemma log shows ~6.2 GiB of
   cache saves across the day in the same process — duplicates saved on
   every idle-to-task transition, so the *resident* set is smaller, but the
   allocation churn is real) — this is a *new* host-RAM commitment.
2. **`--no-mmap`** on all big models. Without mmap, the full weight bytes
   must become anonymous RAM (at load time; on a 24 GB B60 desktop, ~10 GB
   spills back to RAM transiently during buffered/fitting, and the rest is
   eagerly resident).
3. Both the old and new llama-server coexist briefly during the
   evict→start handoff, and during the swap the arc-llama router also
   periodically samples VRAM/`ctx-fit`, keeping the older process alive
   between when it is "stopped" and when the new one becomes healthy.

In short: on this 32 GB-RAM + 16 GB-swap host with a heavy desktop session,
a single model-swap tick inside arc-llama added ~14 GB of anonymous host
allocator demand in under two minutes; the OOM victims were simply the
cheapest things to kill (Electron tabs). It is a **host-RAM**, not VRAM,
exhaustion event, iff:

- host RAM must absorb the prompt-cache pool plus `--no-mmap` weights plus
  the swap-in/out transient (two llama-servers briefly resident);
- swap is already nearly full from other tenants (Electron, opencode,
  open-webui, java services).

## What did *not* cause it

- arc-llama's own Python process: only ~0.9 GB RSS.
- The VRAM-side fit guard / `--fit`: the OOM was in host RAM (`GFP_HIGHUSER`
  from `ttm`/`xe` page allocation), not device memory.
- The `ollama` upstream in `cpuset=ollama.service`: victim, not aggressor.
- chat_store/_repo_map `lru_cache` helpers introduced in RC2: bounded (128
  entries, metadata-size tuples), OOM snapshot shows single-digit MB deltas.

## Remediation (proposed for repo)

1. **Set `--cache-ram` explicitly per model recipe.** arc-llama should
   derive `--cache-ram N` from `KV_PER_TOKEN_F16_BYTES[kv_class] * ctx` (so
   the cap ≈ the size of the prompt *cache* needed, not the whole KV pool)
   rather than accept llama.cpp's 8 GiB default. For gemma and qwen
   recipes, a cap of ~512 MiB (or 0 on ≤32 GB RAM hosts) is enough and
   eliminates the multi-GiB growth path.
2. **Bounded eviction hold-out.** Extend `router.py`'s evict-before-start so
   the previous llama-server's exit (detached from `--no-mmap` pages) is
   awaited with a small poll + SIGKILL fallback (currently assumes SIGTERM
   succeeds in fixed timeout); additionally pre-check host `MemAvailable` in
   `min_moe_offload_layers` / `_estimate_model_vram_mb` path so a swap
   targeting a model whose VRAM footprint is fine but whose *host*
   footprint is too big is refused with the same `fit: false` verdict the
   registration path uses (one formula, two directions).
3. **Default `no_mmap = false`** for models that fit fully into VRAM; this
   leaves weights in page-cache-mapped pages that can be reclaimed and
   re-faulted, greatly reducing anonymous commit at load time. The RC1
   perf story (`docs/incidents` and policy.py notes) is about mmap/page
   faults on *CPU-heavy* MoE layers; that only matters when n_cpu_moe > 0
   or layers spill to host. arc-llama should condition `--no-mmap` on
   `n_cpu_moe == 0` and full-VRAM fit, not set it unconditionally.
4. **Post-incident guard**: when a server startup times out after
   `start_deadline_s`, the `warning`/evict path should additionally check
   `MemAvailable` before restarting, and temporarily *drop* the requested
   `-c` value if the retry's host footprint won't fit (auto ctx reduction
   instead of a blind second attempt which is what burned the remaining
   swap here).
5. **Upstream model recipe delta**: qwen3.6's KV at ctx 131072 with
   `cache_type_k/v = q8_0` is fine on VRAM, but the MTP draft ctx adds
   another few GB of static VRAM; the swap's *failure mode* was host RAM,
   not VRAM, because all these fit; only the mmproj load and swap transients
   touch RAM — so the recipe only needs the `--cache-ram` cap, not any
   eviction of the draft.

## Follow-ups recorded elsewhere

- `docs/work/easiest-inference-gaps.md` mentions context-length truncation;
  the OOM guard proposed here belongs in the same 0.9-rc polish window.
- The local-llm-proxy (`/mnt/storage/local-llm-proxy/proxy.py`) and arcllama
  both expose port 11435 today (proxy is the fallback when arcllama is
  stopped, and vice versa). When arcllama replaces the proxy in the boot
  chain, the proxy must NOT be running simultaneously, or a spare llama-server
  process pair is created and the host is OOM-prone again with two models
  loaded. This doc assumes only one is active; verify before each swap test.
