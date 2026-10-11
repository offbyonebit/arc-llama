// Browser regression runner. Importing shared harness/cases starts no browser or server.
import { chromium } from "playwright";
import { installApiMocks, recordSessionTokenCalls, startStaticServer } from "./harness.mjs";
import {
  testChatSelection,
  testChatHistoryComponents,
  testChatSend,
  testStructuredLoadFailure,
  testRefreshPreservesEdits,
  testKeyboardNavigation,
  testSettingsSurviveStatusPoll,
  testRetrySendsOriginalTurnOnce,
  testMemoryFitAndSavedSettings,
  testRegenerateAndEdit,
  testStopKeepsPartialAnswer,
  testBudgetShownBeforeSending,
  testPresetShapesRequest,
  testTextAttachmentExtensionFallback,
  testImagesNeedVisionModel,
} from "./chat-cases.mjs";
import {
  testDashboardMeasurements,
  testDashboardFitAndEmptyMeasurements,
  testDashboardPluginLayoutController,
  testFrontendIntegrationController,
  testDashboardGenerationTrend,
  testDashboardNavigationAndBlockedModel,
  testFirstRunJourney,
  testFirstRunBlockers,
  testPollingEfficiency,
  testQueuedStatusRefresh,
  testHiddenTabPolling,
} from "./dashboard-cases.mjs";
import {
  testCompatibilityAbandonedChecks,
  testRuntimeCompatibilityGuidance,
  testLibraryPublisherAvatars,
  testLibrarySearchAndDownload,
  testModelChoiceExplanations,
  testDownloadRecovery,
  testMissingFileRecovery,
} from "./library-cases.mjs";

const TESTS = [
  ["recovery-download", testDownloadRecovery],
  ["recovery-missing-file", testMissingFileRecovery],
  ["polling-efficiency", testPollingEfficiency],
  ["polling-queued-status", testQueuedStatusRefresh],
  ["polling-hidden-tab", testHiddenTabPolling],
  ["model-choice-explanations", testModelChoiceExplanations],
  ["first-run-journey", testFirstRunJourney],
  ["first-run-blockers", testFirstRunBlockers],
  ["chat-selection", testChatSelection],
  ["chat-history-components", testChatHistoryComponents],
  ["chat-send", testChatSend],
  ["structured-load-failure", testStructuredLoadFailure],
  ["refresh-preserves-edits", testRefreshPreservesEdits],
  ["keyboard-navigation", testKeyboardNavigation],
  ["settings-survive-status-poll", testSettingsSurviveStatusPoll],
  ["retry-original-turn-once", testRetrySendsOriginalTurnOnce],
  ["dashboard-measurements", testDashboardMeasurements],
  ["memory-fit-and-saved-settings", testMemoryFitAndSavedSettings],
  ["dashboard-fit-and-empty-measurements", testDashboardFitAndEmptyMeasurements],
  ["dashboard-navigation-and-blocked-model", testDashboardNavigationAndBlockedModel],
  ["dashboard-generation-trend", testDashboardGenerationTrend],
  ["compatibility-abandoned-checks", testCompatibilityAbandonedChecks],
  ["runtime-compatibility-guidance", testRuntimeCompatibilityGuidance],
  ["library-publisher-avatars", testLibraryPublisherAvatars],
  ["library-search-and-download", testLibrarySearchAndDownload],
  ["dashboard-plugin-layout-controller", testDashboardPluginLayoutController],
  ["frontend-integration-controller", testFrontendIntegrationController],
  ["regenerate-and-edit", testRegenerateAndEdit],
  ["stop-keeps-partial-answer", testStopKeepsPartialAnswer],
  ["budget-before-sending", testBudgetShownBeforeSending],
  ["preset-shapes-request", testPresetShapesRequest],
  ["text-attachment-extension-fallback", testTextAttachmentExtensionFallback],
  ["images-need-vision-model", testImagesNeedVisionModel],
];

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

function parseFilter(argv) {
  const idx = argv.indexOf("--only");
  if (idx === -1 || idx + 1 >= argv.length) return null;
  return argv[idx + 1];
}

async function main() {
  const filter = parseFilter(process.argv);
  if (filter && !TESTS.some(([name]) => name.includes(filter))) {
    console.error(`Unknown test filter: ${filter}`);
    process.exitCode = 1;
    return;
  }
  const { server, port } = await startStaticServer();
  const origin = `http://127.0.0.1:${port}`;
  const launchOptions = { headless: true, args: ["--no-sandbox", "--disable-dev-shm-usage"] };
  let browser;
  const results = [];
  try {
    browser = await chromium.launch({ ...launchOptions, channel: "chrome" }).catch(() =>
      chromium.launch(launchOptions)
    );
    for (const [name, fn] of TESTS) {
      if (filter && !name.includes(filter)) continue;
      const context = await browser.newContext();
      await installApiMocks(context);
      const page = await context.newPage();
      page.setDefaultTimeout(5000);
      // Chromium cold navigation can outlast interaction waits on busy hosts.
      page.setDefaultNavigationTimeout(30000);
      const pageErrors = [];
      page.on("pageerror", error => pageErrors.push(error.message));
      recordSessionTokenCalls(page);
      const started = Date.now();
      try {
        await fn({ page, origin });
        if (pageErrors.length) throw new Error(pageErrors.join("; "));
        results.push({ name, ok: true, ms: Date.now() - started });
        console.log(`ok   ${name} (${Date.now() - started}ms)`);
      } catch (err) {
        results.push({ name, ok: false, ms: Date.now() - started, error: String(err && err.message || err) });
        console.error(`FAIL ${name}: ${err && err.message ? err.message : err}`);
      } finally {
        await context.close().catch(() => {});
      }
    }
  } catch (err) {
    console.error(`fatal: ${err && err.message ? err.message : err}`);
    process.exitCode = 1;
  } finally {
    if (browser) await browser.close().catch(() => {});
    server.close();
  }
  const failed = results.filter((r) => !r.ok);
  console.log(`\n${results.length - failed.length}/${results.length} passed`);
  if (failed.length) process.exitCode = 1;
}

main();
