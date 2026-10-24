"""A serial, durable agent queue independent of HTTP client lifetimes."""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from pathlib import Path
from typing import Any

import httpx

from arc_llama.agent.tools import TOOLS, ToolResult, list_directory, read_file
from arc_llama.workbench import attempts, files, sandbox, verify
from arc_llama.workbench.store import WorkOwner, WorkStore

READ_TOOLS = {"read_file", "list_directory", "search_files"}
EDIT_TOOLS = {"write_file", "apply_patch"}
ACTIVE = {"queued", "running", "waiting_approval", "preparing", "verifying", "settling"}
MAX_REPAIRS = 2


class WorkManager:
    def __init__(self, root: Path, base_url: str, token: str | None = None):
        self.store = WorkStore(root)
        self.base_url = base_url
        self.token = token
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.worker: asyncio.Task[None] | None = None
        self.active: str | None = None
        self.actions = asyncio.Lock()
        self.owner = WorkOwner(root / "worker.lock")

    async def start(self) -> None:
        self.owner.acquire()
        try:
            for job in self.store.records("job"):
                if job.get("copy_transition"):
                    await self._complete_copy_transition(job)
                if job["state"] != "discarded" and self.job_dir(job["id"]).is_dir():
                    self._recover_dirs(self.job_dir(job["id"]))
                if job["state"] == "preparing":
                    job.update(
                        state="failed",
                        message="Server stopped while copying the project. Discard this copy and create a new job.",
                    )
                    self.store.save("job", job)
                elif job["state"] == "running":
                    uncertain = any(p.get("state") == "executing" for p in job["pending"])
                    job["state"] = "interrupted" if uncertain else "paused"
                    job["message"] = (
                        "Server restarted. Review the project copy before resuming."
                        if uncertain
                        else "Server restarted; resume when ready."
                    )
                    self.store.save("job", job, {"type": "recovery", "message": job["message"]})
                elif job["state"] == "verifying":
                    job.update(
                        state="completed",
                        message="Server restarted before verification finished.",
                        verification=self._unverified(
                            "Server restarted before verification finished"
                        ),
                    )
                    self.store.save("job", job, {"type": "recovery", "message": job["message"]})
                elif job["state"] == "queued":
                    self.queue.put_nowait(job["id"])
            self.worker = asyncio.create_task(self._worker())
        except BaseException:
            self.owner.release()
            raise

    async def shutdown(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)
        self.owner.release()

    def workspace(self, body: dict[str, Any], identity: str | None = None) -> dict[str, Any]:
        current = self.store.get("workspace", identity) if identity else None
        root_text = body.get("root", current["root"] if current else "")
        if not isinstance(root_text, str) or not Path(root_text).is_absolute():
            raise ValueError("Choose an absolute project folder on this server.")
        root = Path(root_text).resolve()
        if (
            not root.is_dir()
            or root == self.store.root
            or root in self.store.root.parents
            or self.store.root in root.parents
        ):
            raise ValueError("Choose a project directory separate from Arc Llama's Work state.")
        if current and str(root) != current["root"]:
            raise ValueError("A workspace's project path cannot change. Create another workspace.")
        value = dict(current or {})
        for field, default, limit in (
            ("name", root.name, 100),
            ("instructions", "", 20_000),
            ("model", "", 200),
        ):
            text = body.get(field, value.get(field, default))
            if (
                not isinstance(text, str)
                or len(text) > limit
                or (field != "instructions" and not text.strip())
            ):
                raise ValueError(f"Invalid workspace {field}.")
            value[field] = text
        for flag in ("allow_edits", "allow_commands", "verify"):
            enabled = body.get(flag, value.get(flag, flag != "allow_commands"))
            if type(enabled) is not bool:
                raise ValueError(f"{flag} must be a boolean")
            value[flag] = enabled
        for field, number_default, low, high in (
            ("max_attempts", 1, 1, attempts.MAX_ATTEMPTS),
            ("time_budget_minutes", 30, 1, attempts.MAX_BUDGET_MINUTES),
        ):
            number = body.get(field, value.get(field, number_default))
            if type(number) is not int or not low <= number <= high:
                raise ValueError(f"{field} must be a whole number from {low} to {high}")
            value[field] = number
        if value["max_attempts"] > 1 and not value["verify"]:
            raise ValueError(
                "Several attempts need result checking, because that picks the best one."
            )
        if value["allow_commands"] and (not value["allow_edits"] or not sandbox.available()):
            raise ValueError("Commands require an editable project copy and the Linux OS sandbox.")
        value["root"] = str(root)
        value["source_identity"] = [root.stat().st_dev, root.stat().st_ino]
        if current:
            self.store.save("workspace", value)
            return value
        return self.store.create("workspace", **value)

    async def create_job(
        self, workspace_id: str, task: str, model: str | None = None
    ) -> dict[str, Any]:
        workspace = self.store.get("workspace", workspace_id)
        if not isinstance(task, str) or not task.strip() or len(task) > 20_000:
            raise ValueError("Task must contain 1 to 20000 characters.")
        if sum(j["state"] != "discarded" for j in self.store.records("job")) >= 20:
            raise ValueError(
                "Discard an old job before creating more (20 retained project copies maximum)."
            )
        prompt = (
            "You are working in an isolated project copy. Complete the user's task using the available tools. Read before editing. Never claim tests ran unless a tool ran them. Explain any unfinished work. Project instructions:\n"
            + workspace["instructions"]
        )
        job = self.store.create(
            "job",
            workspace=workspace,
            task=task,
            model=model or workspace["model"],
            state="preparing",
            message="Copying project",
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": task}],
            pending=[],
            turns=0,
            pause_requested=False,
            apply_state="none",
            receipt={},
            repairs=0,
            attempt=1,
            attempts=[],
        )
        self.job_dir(job["id"]).mkdir(parents=True)
        self.queue.put_nowait(job["id"])
        return job

    def job_dir(self, identity: str) -> Path:
        if len(identity) != 32 or any(c not in "0123456789abcdef" for c in identity):
            raise ValueError("Invalid job identifier")
        return self.store.root / "copies" / identity

    def summary(self, job: dict[str, Any]) -> dict[str, Any]:
        return {
            k: v for k, v in job.items() if k not in ("messages", "verification_by_attempt")
        } | {"project_copy": str(self.job_dir(job["id"]) / "tree")}

    async def action(self, identity: str, action: str, body: dict[str, Any]) -> dict[str, Any]:
        async with self.actions:
            job = self.store.get("job", identity)
            running = self.active == identity or job["state"] in ACTIVE
            if action == "pause":
                if job["state"] == "running":
                    job["pause_requested"] = True
                    job["message"] = "Pausing after the current step"
                elif job["state"] in ("queued", "waiting_approval"):
                    job["state"] = "paused"
                elif job["state"] == "verifying":
                    raise ValueError("Verification is finishing; it cannot be paused.")
                else:
                    raise ValueError("This job is not active.")
            elif action == "resume":
                if job["state"] not in ("paused", "interrupted"):
                    raise ValueError("Only paused or interrupted jobs can resume.")
                if not (self.job_dir(identity) / "tree").is_dir():
                    raise ValueError("The project copy is missing. Create a new job.")
                if any(p.get("state") == "executing" for p in job["pending"]):
                    if body.get("acknowledge_uncertain") is not True:
                        raise ValueError(
                            "Review the project copy and acknowledge the uncertain operation before resuming."
                        )
                    for pending in job["pending"]:
                        if pending["state"] == "executing":
                            self._result(
                                job,
                                pending,
                                "The server stopped during this operation. Its outcome is uncertain. Inspect files before taking further action; it was not replayed.",
                                True,
                            )
                job.update(state="queued", pause_requested=False, message="Resuming")
            elif action == "approve":
                if job["state"] != "waiting_approval" or not job["pending"]:
                    raise ValueError("No tool is awaiting approval.")
                pending = job["pending"][0]
                if body.get("call_id") != pending["id"] or type(body.get("approved")) is not bool:
                    raise ValueError(
                        "Approval must match the pending tool call and contain a boolean decision."
                    )
                pending["approved"] = body["approved"]
                job.update(state="queued", message="Approval recorded")
            elif action in ("apply", "undo", "discard"):
                if running:
                    raise ValueError("Pause or finish the job before changing its files.")
                root = self.job_dir(identity)
                source = Path(job["workspace"]["root"])
                if action != "discard" and (
                    source.is_symlink()
                    or [source.stat().st_dev, source.stat().st_ino]
                    != job["workspace"]["source_identity"]
                ):
                    raise ValueError(
                        "Project root changed independently. Resolve it before applying or undoing."
                    )
                if action == "discard":
                    if job["apply_state"] in ("applied", "applying"):
                        raise ValueError(
                            "Undo the applied source changes before discarding the copy."
                        )
                    await asyncio.to_thread(shutil.rmtree, root, True)
                    job.update(
                        state="discarded", message="Project copy discarded; source left unchanged"
                    )
                elif action == "apply":
                    if job["apply_state"] != "none":
                        raise ValueError("Undo the previous application first.")
                    receipt, _ = await asyncio.to_thread(
                        files.changes, root / "baseline", root / "tree"
                    )
                    if not receipt:
                        raise ValueError("There are no file changes to apply.")
                    paths = sorted(receipt)
                    if any(b.startswith(a + "/") for a, b in zip(paths, paths[1:], strict=False)):
                        raise ValueError("File/directory type changes require manual resolution.")
                    if body.get("review_revision") != files.revision(receipt):
                        raise ValueError("Review the current changes before applying them.")
                    job.update(receipt=receipt, apply_state="applying")
                    self.store.save("job", job, {"type": "apply_started", "files": list(receipt)})
                    try:
                        await asyncio.to_thread(
                            files.source_update,
                            Path(job["workspace"]["root"]),
                            root / "baseline",
                            root / "tree",
                            receipt,
                        )
                    except files.WorkConflictError:
                        job.update(apply_state="none", receipt={})
                        self.store.save("job", job)
                        raise
                    except BaseException:
                        # Keep the intent journal for explicit conflict-aware undo.
                        self.store.save("job", job)
                        raise
                    job["apply_state"] = "applied"
                else:
                    if job["apply_state"] not in ("applied", "applying"):
                        raise ValueError("No source changes have been applied.")
                    await asyncio.to_thread(
                        files.source_update,
                        Path(job["workspace"]["root"]),
                        root / "baseline",
                        root / "tree",
                        job["receipt"],
                        undo=True,
                    )
                    job.update(apply_state="none", receipt={})
            else:
                raise ValueError("Unknown job action")
            self.store.save("job", job, {"type": action, "message": job["message"]})
            if job["state"] == "queued":
                self.queue.put_nowait(identity)
            return self.summary(job)

    async def generate(self, job: dict[str, Any]) -> dict[str, Any]:
        names = (
            READ_TOOLS
            | (EDIT_TOOLS if job["workspace"]["allow_edits"] else set())
            | ({"run_command"} if job["workspace"]["allow_commands"] else set())
        )
        definitions = [d for d in TOOLS.definitions if d["function"]["name"] in names]
        definitions = [
            {
                **d,
                "function": {
                    **d["function"],
                    "description": "Find literal text in project files; returns at most 100 matches.",
                },
            }
            if d["function"]["name"] == "search_files"
            else d
            for d in definitions
        ]
        message = await self._chat(job["model"], job["messages"], definitions, 2048)
        return message

    async def _chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        # Queued recovery can begin just before the HTTP listener is ready.
        transport = httpx.AsyncHTTPTransport(retries=3)
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = tools
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(120, connect=5), headers=headers, transport=transport
        ) as client:
            response = await client.post(self.base_url + "/v1/chat/completions", json=body)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]

    async def ask(self, job: dict[str, Any], system: str, user: str, max_tokens: int = 1500) -> str:
        """One tool-free question in a fresh context, so a reviewer never inherits the agent's reasoning."""
        message = await self._chat(
            job["model"],
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            None,
            max_tokens,
            0.0,
        )
        content = message.get("content")
        return content if isinstance(content, str) else ""

    @staticmethod
    def _unverified(reason: str) -> dict[str, Any]:
        return {
            "verdict": "unverified",
            "label": verify.LABELS["unverified"],
            "reason": reason,
            "results": [],
        }

    async def plan_criteria(self, job: dict[str, Any]) -> list[dict[str, Any]]:
        listing = await asyncio.to_thread(files.inventory, self.job_dir(job["id"]) / "tree")
        names = sorted(listing)
        shown = "\n".join(names[:300]) + (
            f"\n... and {len(names) - 300} more" if len(names) > 300 else ""
        )
        reply = await self.ask(
            job,
            verify.PLAN_PROMPT,
            f"Task:\n{job['task']}\n\nProject instructions:\n{job['workspace']['instructions'][:2000]}\n\nFiles now present:\n{shown or '(none)'}",
        )
        return verify.normalise(verify.parse_json(reply))

    async def verification(self, job: dict[str, Any]) -> dict[str, Any]:
        root = self.job_dir(job["id"])

        def examine() -> tuple[dict[str, str], dict[str, str], dict[str, str | None], str]:
            before, after = files.inventory(root / "baseline"), files.inventory(root / "tree")
            receipt, diff = files.changes(root / "baseline", root / "tree")
            return before, after, receipt, diff

        before, after, receipt, diff = await asyncio.to_thread(examine)
        criteria = job.get("criteria") or []
        results = [
            verify.grade_machine(c, after, root / "tree")
            for c in criteria
            if c["type"] in verify.MACHINE
        ]
        results += await asyncio.to_thread(
            verify.safety_checks, criteria, before, after, root / "baseline", root / "tree"
        )
        judged = [c for c in criteria if c["type"] == "judge"]
        if judged:
            moved = "\n".join(f"{'+' if after.get(n) else '-'} {n}" for n in receipt)
            final = next(
                (
                    m.get("content") or ""
                    for m in reversed(job["messages"])
                    if m.get("role") == "assistant"
                ),
                "",
            )
            material = (
                "Criteria:\n"
                + "\n".join(f"{c['id']}. {c['text']}" for c in judged)
                + f"\n\nOriginal task:\n{job['task']}\n\nChanged paths (+ present, - removed):\n{moved[:4000] or '(none)'}"
                + f"\n\nAll files in the result:\n{chr(10).join(sorted(after)[:300])}"
                + f"\n\nDiff:\n{diff[:20000]}\n\nThe worker's own closing statement (not evidence):\n{final[:2000]}"
            )
            try:
                reply = await self.ask(job, verify.JUDGE_PROMPT, material)
            except (httpx.HTTPError, ValueError, KeyError):
                reply = None
            results += verify.grade_judged(judged, reply, material)
        outcome = verify.verdict(results)
        return {
            "verdict": outcome,
            "label": verify.LABELS[outcome],
            "results": results,
            "counts": {
                k: sum(r["kind"] == k for r in results) for k in ("machine", "judge", "safety")
            },
            "reason": "" if criteria else "No acceptance criteria were available for this job.",
        }

    @staticmethod
    def _recover_dirs(root: Path) -> None:
        """Finish a copy swap that a crash interrupted."""
        staged = root / "tree.new"
        if staged.is_dir():
            if (root / "tree").exists():
                shutil.rmtree(staged, True)
            else:
                staged.rename(root / "tree")

    @staticmethod
    def _copy_step(root: Path, transition: dict[str, Any]) -> None:
        """One idempotent filesystem step, with its intent already saved in SQLite."""
        kind, phase = transition["kind"], transition["phase"]
        tree, staged, best = root / "tree", root / "tree.new", root / "best"
        if phase == "prepare":
            if not staged.exists():
                building = root / "tree.build"
                if building.exists():
                    shutil.rmtree(building)
                shutil.copytree(root / "baseline", building)
                building.rename(staged)
        elif phase == "retire":
            if kind == "rotate" and transition["keep"]:
                if tree.exists():
                    if best.exists():
                        shutil.rmtree(best)
                    tree.rename(best)
            else:
                if tree.exists():
                    shutil.rmtree(tree)
        elif phase == "install":
            if not tree.exists():
                (staged if kind == "rotate" else best).rename(tree)
        elif phase == "cleanup":
            if best.exists():
                shutil.rmtree(best)

    async def _complete_copy_transition(self, job: dict[str, Any]) -> None:
        transition = job["copy_transition"]
        phases = {
            "rotate": ["prepare", "retire", "install"],
            "adopt": ["retire", "install"],
            "cleanup": ["cleanup"],
        }[transition["kind"]]
        for phase in phases[phases.index(transition["phase"]) :]:
            transition["phase"] = phase
            self.store.save("job", job)
            task = asyncio.create_task(
                asyncio.to_thread(self._copy_step, self.job_dir(job["id"]), transition.copy())
            )
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                # A cancelled await does not stop its filesystem thread. Keep ownership
                # until it finishes; recovery can repeat the saved step safely.
                await task
                raise
        job["state"] = transition["state"]
        del job["copy_transition"]
        self.store.save("job", job)

    async def _transition_copy(self, job: dict[str, Any], kind: str, *, keep: bool = False) -> None:
        job["copy_transition"] = {
            "kind": kind,
            "keep": keep,
            "phase": "prepare" if kind == "rotate" else "retire" if kind == "adopt" else "cleanup",
            "state": job["state"],
        }
        job["state"] = "settling"
        self.store.save("job", job)
        await self._complete_copy_transition(job)

    async def _finish_attempt(self, identity: str, incomplete: bool = False) -> bool:
        """Record this attempt; start another on a fresh copy, or settle on the best one."""
        job = self.store.get("job", identity)
        root = self.job_dir(identity)
        now = time.time()
        report = job.get("verification") or self._unverified("Result checking is off")
        entry = attempts.entry_for(job, report, now, incomplete)
        job["attempts"].append(entry)
        job.setdefault("verification_by_attempt", {})[str(entry["n"])] = report
        previous = job["attempts"][:-1]
        if attempts.may_start_another(job, entry, now):
            keep = not previous or attempts.best(job["attempts"]) is entry
            job["best_attempt"] = entry["n"] if keep else job.get("best_attempt")
            job.update(
                attempt=job["attempt"] + 1,
                attempt_started_at=now,
                messages=job["messages"][:2],
                pending=[],
                turns=0,
                repairs=0,
                verification=None,
                state="running",
                message=f"Attempt {entry['n']} was {entry['verdict']}; starting attempt {job['attempt'] + 1} on a fresh copy",
            )
            await self._transition_copy(job, "rotate", keep=keep)
            self.store.save(
                "job", job, {"type": "attempt", "attempt": entry, "message": job["message"]}
            )
            return True
        chosen = attempts.best(job["attempts"])
        if chosen is not entry:
            job["verification"] = job["verification_by_attempt"][str(chosen["n"])]
        job["chosen_attempt"] = chosen["n"]
        if len(job["attempts"]) > 1:
            job["message"] = (
                f"Kept attempt {chosen['n']} of {len(job['attempts'])}: {chosen['label']}. Review its changes."
            )
        if chosen["incomplete"]:
            job.update(
                state="failed", message="Reached the 30-turn limit. Review the project copy."
            )
        else:
            job["state"] = "completed"
        if chosen is not entry or (root / "best").exists():
            await self._transition_copy(job, "adopt" if chosen is not entry else "cleanup")
        self.store.save(
            "job",
            job,
            {"type": "attempts_settled", "chosen": chosen["n"], "attempts": job["attempts"]},
        )
        return False

    async def _verify(self, identity: str) -> bool:
        """Grade a finished job. Return True when failures were fed back and work should continue."""
        job = self.store.get("job", identity)
        if not job["workspace"].get("verify", True):
            return False
        job.update(state="verifying", message="Checking the result against the acceptance criteria")
        self.store.save("job", job, {"type": "verifying", "message": job["message"]})
        try:
            report = await self.verification(job)
        except Exception as exc:  # a broken reviewer must never hide a finished job
            report = self._unverified(
                f"Verification could not run: {str(exc)[:200] or type(exc).__name__}"
            )
        job["verification"] = report
        retry = (
            report["verdict"] == "failed"
            and job.get("repairs", 0) < MAX_REPAIRS
            and job["turns"] < 25
        )
        if retry:
            problems = verify.failures(report["results"])
            job["repairs"] = job.get("repairs", 0) + 1
            job["messages"].append(
                {
                    "role": "user",
                    "content": "An independent check of your work found problems:\n"
                    + problems
                    + "\nFix them using the tools, then say what you changed.",
                }
            )
            job.update(
                state="running", message=f"Fixing verification failures (attempt {job['repairs']})"
            )
        else:
            job["message"] = f"{report['label']}. Review its changes."
        self.store.save(
            "job",
            job,
            {"type": "verification", "verdict": report["verdict"], "results": report["results"]},
        )
        return retry

    def _result(self, job: dict[str, Any], call: dict[str, Any], output: str, error: bool) -> None:
        text = output[:32_000] + ("\n[tool output truncated]" if len(output) > 32_000 else "")
        job["messages"].append({"role": "tool", "tool_call_id": call["id"], "content": text})
        call["state"] = "done"
        self.store.save(
            "job",
            job,
            {
                "type": "tool_result",
                "call_id": call["id"],
                "name": call["name"],
                "content": text,
                "error": error,
            },
        )

    async def execute(self, job: dict[str, Any], call: dict[str, Any]) -> tuple[str, bool]:
        name, args = call["name"], call["arguments"]
        root = self.job_dir(job["id"]) / "tree"
        if name == "run_command":
            if not job["workspace"]["allow_commands"]:
                return "Commands are disabled for this workspace.", True
            if not isinstance(args.get("command"), str) or len(args["command"]) > 16_000:
                return "Invalid command", True
            try:
                return await sandbox.run(root, args["command"])
            except (ValueError, OSError, asyncio.TimeoutError) as exc:
                return f"Isolated command failed: {exc}", True
        if name not in READ_TOOLS | EDIT_TOOLS or (
            name in EDIT_TOOLS and not job["workspace"]["allow_edits"]
        ):
            return "This tool is not permitted for the workspace.", True
        try:
            path = args.get("path", "." if name in ("list_directory", "search_files") else "")
            if not isinstance(path, str):
                raise ValueError("path must be a string")
            if path == "." and name in ("list_directory", "search_files"):
                target = root
            else:
                target = files.safe_path(root, path)
            if target.is_file() and target.stat().st_size > files.MAX_FILE:
                raise ValueError("File exceeds the Work file size limit")
            if name == "read_file":
                result = read_file(path, root)
            elif name == "list_directory":
                result = list_directory(path, root)
            elif name == "search_files":
                pattern = str(args.get("pattern", ""))
                matches = []
                for relative in files.inventory(root):
                    candidate = files.safe_path(root, relative)
                    if not candidate.match(str(args.get("path_glob", "*"))):
                        continue
                    for number, line in enumerate(
                        candidate.read_text(encoding="utf-8", errors="replace").splitlines(), 1
                    ):
                        if pattern in line:
                            matches.append(f"{relative}:{number}: {line[:500]}")
                        if len(matches) >= 100:
                            break
                    if len(matches) >= 100:
                        break
                result = ToolResult("\n".join(matches) or "No matches")
            elif name == "write_file":
                content = args.get("content")
                if not isinstance(content, str) or len(content.encode()) > files.MAX_FILE:
                    raise ValueError("Invalid or oversized file content")
                files.atomic_file(
                    target,
                    content.encode(),
                    target.stat().st_mode & 0o777 if target.exists() else 0o644,
                )
                result = ToolResult(f"Wrote {path}")
            else:
                old, new = args.get("old_string"), args.get("new_string")
                if not isinstance(old, str) or not isinstance(new, str) or not old:
                    raise ValueError("Patch requires nonempty old_string and a new_string")
                original = target.read_text(encoding="utf-8")
                if old not in original or (
                    original.count(old) > 1 and args.get("replace_all") is not True
                ):
                    raise ValueError("Patch text was missing or ambiguous")
                content = original.replace(old, new, -1 if args.get("replace_all") is True else 1)
                if len(content.encode()) > files.MAX_FILE:
                    raise ValueError("Patched file exceeds the Work size limit")
                files.atomic_file(target, content.encode(), target.stat().st_mode & 0o777)
                result = ToolResult(f"Patched {path}")
            return result.content, result.error
        except (ValueError, OSError) as exc:
            return str(exc), True

    async def _worker(self) -> None:
        while True:
            identity = await self.queue.get()
            self.active = identity
            try:
                await self._run(identity)
            except asyncio.CancelledError:
                job = self.store.get("job", identity)
                if job["state"] == "running":
                    job.update(
                        state="interrupted"
                        if any(p["state"] == "executing" for p in job["pending"])
                        else "paused",
                        message="Server stopped; resume after reviewing progress",
                    )
                    self.store.save("job", job)
                if job["state"] == "verifying":
                    job.update(
                        state="completed",
                        message="Server stopped before verification finished.",
                        verification=self._unverified(
                            "Server stopped before verification finished"
                        ),
                    )
                    self.store.save("job", job)
                raise
            except Exception as exc:
                job = self.store.get("job", identity)
                state = (
                    "settling"
                    if job.get("copy_transition")
                    else "failed"
                    if job["state"] == "preparing"
                    else "interrupted"
                    if any(p["state"] == "executing" for p in job["pending"])
                    else "paused"
                )
                job.update(
                    state=state, message=f"Job {state}: {str(exc)[:400] or type(exc).__name__}"
                )
                self.store.save("job", job, {"type": "error", "message": job["message"]})
            finally:
                self.active = None
                self.queue.task_done()

    async def _run(self, identity: str) -> None:
        job = self.store.get("job", identity)
        if job["state"] == "preparing":
            root = self.job_dir(identity)

            def prepare() -> None:
                files.copy_project(Path(job["workspace"]["root"]), root / "baseline")
                shutil.copytree(root / "baseline", root / "tree")

            preparation = asyncio.create_task(asyncio.to_thread(prepare))
            try:
                await asyncio.shield(preparation)
            except asyncio.CancelledError:
                # Keep queue ownership until the filesystem thread has stopped writing.
                await preparation
                raise
        elif job["state"] != "queued":
            return
        now = time.time()
        job.setdefault("attempt", 1)
        job.setdefault("attempts", [])
        job.setdefault("started_at", now)
        job.setdefault("attempt_started_at", now)
        job.update(state="running", message="Working in the isolated copy")
        self.store.save("job", job)
        if job["workspace"].get("verify", True) and "criteria" not in job:
            try:
                job["criteria"] = await self.plan_criteria(job)
            except Exception:  # the plan is advisory; work continues without it
                job["criteria"] = []
            job["pause_requested"] = self.store.get("job", identity)["pause_requested"]
            self.store.save("job", job, {"type": "criteria", "criteria": job["criteria"]})
        while True:
            job = self.store.get("job", identity)
            if job["pause_requested"]:
                job.update(state="paused", message="Paused between steps")
                self.store.save("job", job)
                return
            job["pending"] = [p for p in job["pending"] if p["state"] != "done"]
            if job["pending"]:
                call = job["pending"][0]
                if call["name"] == "run_command" and not job["workspace"]["allow_commands"]:
                    self._result(job, call, "Commands are not enabled for this workspace.", True)
                    continue
                if call["name"] == "run_command" and "approved" not in call:
                    job.update(
                        state="waiting_approval", message="An isolated command needs your approval"
                    )
                    self.store.save("job", job, {"type": "approval_required", "call": call})
                    return
                if call.get("approved") is False:
                    self._result(job, call, "The user denied this tool call.", True)
                    continue
                call["state"] = "executing"
                self.store.save("job", job, {"type": "tool_call", "call": call})
                output, error = await self.execute(job, call)
                # Re-read pause requests that arrived while the tool awaited.
                job["pause_requested"] = self.store.get("job", identity)["pause_requested"]
                self._result(job, call, output, error)
                continue
            if job["turns"] >= 30:
                job["verification"] = job.get("verification") or self._unverified(
                    "Reached the 30-turn limit"
                )
                job["verification"].update(verdict="failed", label=verify.LABELS["failed"])
                self.store.save("job", job)
                if await self._finish_attempt(identity, incomplete=True):
                    continue
                return
            message = await self.generate(job)
            if not isinstance(message, dict):
                raise ValueError("Invalid model response")
            job["pause_requested"] = self.store.get("job", identity)["pause_requested"]
            job["messages"].append(message)
            job["turns"] += 1
            finished = False
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list) or len(calls) > 16:
                raise ValueError("Invalid tool calls")
            for index, call in enumerate(calls):
                args = call["function"].get("arguments", "{}")
                args = json.loads(args) if isinstance(args, str) else args
                if not isinstance(args, dict):
                    raise ValueError("Tool arguments must be an object")
                call_id = f"{job['id']}-{job.get('attempt', 1)}-{job['turns']}-{index}"
                call["id"] = call_id
                job["pending"].append(
                    {
                        "id": call_id,
                        "name": call["function"]["name"],
                        "arguments": args,
                        "state": "pending",
                    }
                )
            if not calls:
                if not isinstance(message.get("content"), str) or not message["content"].strip():
                    raise ValueError("Model returned an empty answer")
                finished = True
                job["message"] = "Agent finished; review its changes"
            self.store.save(
                "job", job, {"type": "assistant", "content": message.get("content") or ""}
            )
            if finished:
                if await self._verify(identity):
                    continue
                if await self._finish_attempt(identity):
                    continue
                return
