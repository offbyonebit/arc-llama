import assert from "node:assert/strict";

export async function testWorkProjectsJobsAndReview({ page, origin }) {
  let projects = [], jobs = [], selected = null;
  const actions = [];
  const json = data => ({ status: 200, contentType: "application/json", body: JSON.stringify(data) });
  await page.route("**/admin/work**", route => {
    const request = route.request(), path = new URL(request.url()).pathname;
    if (path === "/admin/work") return route.fulfill(json({ workspaces: projects, jobs, commands_available: false }));
    if (path === "/admin/work/workspaces" && request.method() === "POST") {
      projects = [{ id: "p", ...request.postDataJSON() }]; return route.fulfill(json(projects[0]));
    }
    if (path === "/admin/work/jobs" && request.method() === "POST") {
      selected = { id: "j", workspace: projects[0], task: request.postDataJSON().task, state: "completed", message: "Finished in copy", pending: [], project_copy: "/isolated/copy", apply_state: "none", events: [{ seq: 1, type: "tool_result", content: "Changed note.txt" }] };
      jobs = [selected]; return route.fulfill(json(selected));
    }
    if (path.endsWith("/changes")) return route.fulfill(json({ files: ["note.txt"], diff: "--- before/note.txt\n+++ after/note.txt\n-original\n+changed", revision: "reviewed-hash" }));
    if (request.method() === "POST") {
      actions.push({ action: path.split("/").at(-1), body: request.postDataJSON() });
      selected.apply_state = path.endsWith("/apply") ? "applied" : "none";
      return route.fulfill(json(selected));
    }
    return route.fulfill(json(selected));
  });
  await page.goto(origin + "/workbench");
  await page.waitForSelector("#workspace-model option", { state: "attached" });
  await page.fill("#workspace-name", "Private project <script>");
  await page.fill("#workspace-root", "/projects/private");
  await page.fill("#workspace-instructions", "Do not change the API");
  await page.fill("#max-attempts", "3");
  await page.fill("#time-budget-minutes", "20");
  await page.click("#save-workspace");
  await page.waitForFunction(() => !document.querySelector("#start-job").disabled);
  assert.equal(projects[0].max_attempts, 3);
  assert.equal(projects[0].time_budget_minutes, 20);
  assert.equal(await page.locator("#allow-commands").isDisabled(), true);
  await page.fill("#task", "Update the release note");
  await page.click("#start-job");
  await page.getByRole("button", { name: "Apply reviewed changes" }).waitFor();
  assert.equal(await page.getByRole("button", { name: "Apply reviewed changes" }).isDisabled(), true);
  await page.reload();
  await page.locator(".work-job").click();
  await page.waitForFunction(() => document.querySelector("#activity").textContent.includes("tool result"));
  await page.click("#review-changes");
  await page.waitForFunction(() => document.querySelector("#diff").textContent.includes("+changed"));
  await page.getByRole("button", { name: "Apply reviewed changes" }).click();
  await page.getByRole("button", { name: "Undo applied changes" }).waitFor();
  await page.getByRole("button", { name: "Undo applied changes" }).click();
  await page.getByRole("button", { name: "Apply reviewed changes" }).waitFor();
  assert.equal(actions[0].body.review_revision, "reviewed-hash");
  assert.deepEqual(actions.map(a => a.action), ["apply", "undo"]);
  assert.equal(await page.locator("#job-list script").count(), 0);
}


export async function testWorkManualTokenLoadsModels({ page, origin }) {
  await page.route("**/admin/session-token", route => route.fulfill({ status: 401, contentType: "application/json", body: JSON.stringify({ detail: "Token required" }) }));
  await page.route("**/v1/models", route => {
    if (route.request().headers().authorization === "Bearer manual-token") return route.fallback();
    return route.fulfill({ status: 401, contentType: "application/json", body: JSON.stringify({ detail: "Token required" }) });
  });
  await page.route("**/admin/work", route => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ workspaces: [], jobs: [], commands_available: false }) }));
  await page.goto(origin + "/workbench");
  await page.locator("#work-auth").waitFor();
  await page.fill("#work-token", "manual-token");
  await page.locator("#work-auth button").click();
  await page.waitForSelector("#workspace-model option", { state: "attached" });
  assert.equal(await page.locator("#work-auth").isHidden(), true);
  assert.equal(await page.locator("#work-error").isHidden(), true);
  assert.equal(await page.evaluate(() => sessionStorage.getItem("arc-work-token")), "manual-token");
}


export async function testWorkAttemptSettingsAndResults({ page, origin }) {
  const project = { id: "p", name: "Retry project", root: "/projects/retry", model: "lfm2", instructions: "", allow_edits: true, allow_commands: false, verify: true, max_attempts: 3, time_budget_minutes: 45 };
  const entry = (n, verdict) => ({ n, verdict, label: verdict === "failed" ? "Failed verification" : "Partly verified", passed: n, repairs: 2 });
  let selected = { id: "j", workspace: { ...project }, task: "Make a report", state: "verifying", message: "Checking results", pending: [], project_copy: "/isolated/copy", apply_state: "none", events: [], attempt: 2, attempts: [entry(1, "failed")], criteria: [{ text: "report.txt exists", type: "exists" }] };
  const saves = [];
  const json = data => ({ status: 200, contentType: "application/json", body: JSON.stringify(data) });
  // Keep the test independent of whichever model names the shared fixtures use.
  await page.route("**/v1/models", route => route.fulfill(json({ data: [{ id: "lfm2", owned_by: "arc-llama" }] })));
  await page.route("**/admin/work**", route => {
    const req = route.request(), path = new URL(req.url()).pathname;
    if (path === "/admin/work") return route.fulfill(json({ workspaces: [project], jobs: [selected], commands_available: false }));
    if (path === "/admin/work/workspaces/p" && req.method() === "PUT") {
      saves.push(req.postDataJSON()); Object.assign(project, req.postDataJSON());
      return route.fulfill(json(project));
    }
    if (path.endsWith("/changes")) return route.fulfill(json({ files: ["report.txt"], diff: "+report", revision: "revision" }));
    return route.fulfill(json(selected));
  });
  await page.goto(origin + "/workbench");
  await page.waitForFunction(() => document.querySelector("#workspace-picker").value === "p");
  await page.locator("#project-editor summary").click();
  assert.equal(await page.inputValue("#max-attempts"), "3");
  assert.equal(await page.inputValue("#time-budget-minutes"), "45");
  await page.fill("#max-attempts", "4");
  await page.fill("#time-budget-minutes", "60");
  await page.click("#save-workspace");
  await page.waitForFunction(() => !document.querySelector("#project-editor").open);
  assert.equal(saves[0].max_attempts, 4);
  assert.equal(saves[0].time_budget_minutes, 60);
  assert.equal(typeof saves[0].max_attempts, "number");
  // Reload demonstrates server settings, rather than only a retained form value.
  await page.reload();
  await page.waitForFunction(() => document.querySelector("#max-attempts").value === "4");
  await page.locator("#project-editor summary").click();
  await page.uncheck("#verify-work");
  assert.equal(await page.locator("#max-attempts").isDisabled(), true);
  assert.equal(await page.locator("#time-budget-minutes").isDisabled(), true);
  await page.check("#verify-work");
  assert.equal(await page.inputValue("#max-attempts"), "4");
  await page.fill("#time-budget-minutes", "");
  await page.uncheck("#verify-work");
  await page.click("#save-workspace");
  await page.waitForFunction(() => !document.querySelector("#project-editor").open);
  assert.equal(saves[1].verify, false);
  assert.equal(saves[1].max_attempts, 1);
  assert.equal(saves[1].time_budget_minutes, 30);
  await page.locator(".work-job").click();
  await page.waitForFunction(() => document.querySelector("#attempt-progress").textContent.includes("Attempt 2 of up to 3"));
  assert.equal(await page.getByRole("button", { name: "Apply reviewed changes" }).count(), 0);
  assert.equal(await page.getByRole("button", { name: "Discard project copy" }).count(), 0);
  assert.equal(await page.getByRole("button", { name: "Pause job" }).count(), 0);
  assert.equal(await page.textContent("#verdict"), "Checking the result…");
  selected = { ...selected, state: "settling", message: "Selecting the best copy" };
  await page.locator(".work-job").click();
  await page.waitForFunction(() => document.querySelector("#job-state").textContent.includes("settling"));
  assert.equal(await page.locator("#review-changes").isDisabled(), true);
  assert.equal(await page.getByRole("button", { name: "Apply reviewed changes" }).count(), 0);
  assert.equal(await page.getByRole("button", { name: "Discard project copy" }).count(), 0);
  selected = { ...selected, state: "completed", message: "Kept attempt 1", attempts: [entry(1, "partial"), entry(2, "failed")], chosen_attempt: 1, verification: { verdict: "partial", label: "Partly verified", results: [{ text: "report.txt exists", status: "pass", kind: "machine", evidence: "report.txt" }] } };
  // Reselect to reload deterministically, without relying on the polling interval.
  await page.locator(".work-job").click();
  await page.waitForFunction(() => document.querySelector("#attempt-progress").textContent.includes("Kept attempt 1 of 2"));
  assert.match(await page.textContent("#attempt-list"), /Attempt 1 \(kept\): Partly verified/);
  assert.match(await page.textContent("#attempt-list"), /Attempt 2: Failed verification/);
  assert.equal(await page.getByRole("button", { name: "Apply reviewed changes" }).isDisabled(), true);
  await page.click("#review-changes");
  await page.waitForFunction(() => ![...document.querySelectorAll("#job-actions button")].find(b => b.textContent === "Apply reviewed changes").disabled);
}
