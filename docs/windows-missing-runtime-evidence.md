# Windows: `llama-server.exe` missing or inaccessible

The 2026-10-22 Windows validation saw `llama-server.exe` report absent while its sibling DLLs stayed, and the normal
installer recovered it twice (see `release-validation-windows-2026-10-22.md`). No cause was established.

`scripts/windows_runtime_evidence.ps1` collects read-only evidence the next time it happens: the file's metadata, ACL,
signature and Mark-of-the-Web stream; its sibling files; Defender detection events (1116 to 1119) and threat history;
Code Integrity, Smart App Control and AppLocker blocks; Sysmon file deletes if Sysmon is already installed; and the
Storage Sense policy. It changes no settings and never touches antivirus configuration.

```powershell
powershell -File scripts\windows_runtime_evidence.ps1 -Hours 24
```

Reading the output:

- An observation (file absent at time T, Defender event 1117 at time T) is not a cause. Absence with no Defender or
  code-integrity event leaves cleanup races, installer swaps and permission changes open.
- Polling cannot say which process acted. Only a deletion audit or Sysmon, if the machine owner already runs it, can.
- Hashing the file can itself trigger a scan, so compare hashes only after the evidence above is saved.

The script is adapted from the `exewatch` harness (External R&D, reviewed 2026-10-28). It has not been run on a native
Windows host. exewatch's own polling monitor was not adopted: it adds a timeline of existence and hash changes, but its
Windows paths are untested and it would need a serial reproduction on the Windows machine to say anything new.
