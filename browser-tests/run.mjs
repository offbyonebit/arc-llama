import { testGuidedDynamicCatalogStates, testGuidedFirstAnswer, testGuidedPauseRetryAndErrors, testGuidedHardwareAndSmallScreen, testWelcomePreservesDraftAndDoesNotClaimFailureAsSuccess, testWelcomeNeedsCompleteNonemptyReply } from "./onboarding-cases.mjs";
import { testChatSseFraming, testChatSseErrorsKeepPartial } from "./streaming-cases.mjs";
import { testWorkAgentInstallFlow, testWorkAgentInstallCancelKeepsSelection, testSystemAgentsPanel, testSystemAgentsOffWithoutExperimental } from "./harness-install-cases.mjs";
import { testPaperPickerListVisibility, testPaperDiscoverNavigation, testPaperShellAndModelPicker, testPaperMobileSheets, testPaperRealClaimsAndContext, testPaperModelLoadStop, testPaperModelDelete, testPaperSystemSectionsAligned, testPaperWorkModelPicker } from "./paper-cases.mjs";
import { testPerModelRuntimeSettings, testRuntimeOverridesSurviveUnrelatedEdits } from "./runtime-cases.mjs";
// Browser regression runner. Importing shared harness/cases starts no browser or server.
import { chromium } from "playwright";
import { installApiMocks, recordSessionTokenCalls, startStaticServer } from "./harness.mjs";
import {
  testChatSingleReplyLifecycle,
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
  testWorkGenerationSpeedCounter,
  testWorkNativeHarnessHistory,
  testWorkSessionDuplicatesAreExpandable,
  testWorkRunCarriesItsStableConversationId,
  testWorkLinkedHarnessSessionsStayAccessible,
  testWorkHermesHarness,
  testWorkHermesUnavailable,
  testWorkViewHiddenUnlessExperimental,
  testWorkResumeAndVisibleContext,
  testWorkLiveActivityAndFullPrompt,
  testChatLiveActivity,
  testWorkContextIgnoresStaleBudgets,
  testWorkModeApprovalAndUndo,
  testWorkModeDeniedAndStopped,
  testWorkModeServerRejectsMissingFolder,
  testWorkModeEndedWithoutResultIsAnError,
  testWorkModeOffersClosestFolder,
  testWorkModeAutoApprove,
  testWorkModeAutoApproveHiddenOnOldServer,
  testWorkModeKeepsTheConversation,
  testWorkModeOutsideFolderIsSaidPlainly,
  testWorkModeCompactReplacesEarlierTurns,
  testWorkSlashMenuListsHarnessCommands,
  testWorkSlashCommandRunsInTheHarness,
  testWorkSessionsAreSavedAndReopened,
  testWorkToolProfilesLimitWhatARunCanUse,
  testWorkProfilesRecoverFromDiscoveryFailure,
  testWorkProfilesIgnoreClosedDiscovery,
} from "./work-cases.mjs";
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

import {
  testCompactionUndoFailureAndEdit,
  testCompactionRequiresPersistenceAck,
  testCompactionRoundTrip,
  testCompactionReloadAndClear,
  testCompactionFailures,
  testCompactionSaveFailure,
  testCompactionCancelAndGuards,
  testCompactionOversizedAndImages,
  testCompactionRecentTooLarge,
} from "./compaction-cases.mjs";

import { testInterruptedAnswerPersists, testStoppedAnswerPersists, testExactBudgetAndReplyLimit,
  testBudgetDiscardsStaleResponses, testDownloadFolderAndRetry } from "./release-cases.mjs";

import { testWorkProjectsJobsAndReview, testWorkManualTokenLoadsModels, testWorkAttemptSettingsAndResults } from "./workbench-cases.mjs";

const TESTS = [
  ["guided-dynamic-catalog", testGuidedDynamicCatalogStates],
  ["guided-first-answer", testGuidedFirstAnswer],
  ["guided-pause-retry-errors", testGuidedPauseRetryAndErrors],
  ["guided-hardware-small-screen", testGuidedHardwareAndSmallScreen],
  ["welcome-preserves-draft-failure", testWelcomePreservesDraftAndDoesNotClaimFailureAsSuccess],
  ["welcome-complete-nonempty", testWelcomeNeedsCompleteNonemptyReply],
  ["chat-sse-framing", testChatSseFraming],
  ["chat-sse-errors-keep-partial", testChatSseErrorsKeepPartial],
  ["work-hermes-harness", testWorkHermesHarness],
  ["work-hermes-unavailable", testWorkHermesUnavailable],
  ["work-live-activity-full-prompt", testWorkLiveActivityAndFullPrompt],
  ["chat-single-reply-lifecycle", testChatSingleReplyLifecycle],
  ["chat-live-activity", testChatLiveActivity],
  ["work-resume-visible-context", testWorkResumeAndVisibleContext],
  ["work-context-stale-budgets", testWorkContextIgnoresStaleBudgets],
  ["paper-picker-list-visibility", testPaperPickerListVisibility],
  ["paper-discover-navigation", testPaperDiscoverNavigation],
  ["paper-shell-model-picker", testPaperShellAndModelPicker],
  ["paper-mobile-sheets", testPaperMobileSheets],
  ["paper-claims-context", testPaperRealClaimsAndContext],
  ["paper-model-load-stop", testPaperModelLoadStop],
  ["paper-model-delete", testPaperModelDelete],
  ["paper-system-sections-aligned", testPaperSystemSectionsAligned],
  ["paper-work-model-picker", testPaperWorkModelPicker],
  ["work-attempt-settings-and-results", testWorkAttemptSettingsAndResults],
  ["work-projects-jobs-and-review", testWorkProjectsJobsAndReview],
  ["work-manual-token-loads-models", testWorkManualTokenLoadsModels],
  ["interrupted-answer-persists", testInterruptedAnswerPersists],
  ["stopped-answer-persists", testStoppedAnswerPersists],
  ["exact-budget-and-reply-limit", testExactBudgetAndReplyLimit],
  ["budget-discards-stale-responses", testBudgetDiscardsStaleResponses],
  ["download-folder-and-retry", testDownloadFolderAndRetry],
  ["compaction-requires-persistence-ack", testCompactionRequiresPersistenceAck],
  ["compaction-undo-failure-and-edit", testCompactionUndoFailureAndEdit],
  ["compaction-roundtrip", testCompactionRoundTrip],
  ["compaction-reloadandclear", testCompactionReloadAndClear],
  ["compaction-failures", testCompactionFailures],
  ["compaction-savefailure", testCompactionSaveFailure],
  ["compaction-cancelandguards", testCompactionCancelAndGuards],
  ["compaction-oversizedandimages", testCompactionOversizedAndImages],
  ["compaction-recenttoolarge", testCompactionRecentTooLarge],

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
  ["per-model-runtime-settings", testPerModelRuntimeSettings],
  ["per-model-runtime-unrelated-edits", testRuntimeOverridesSurviveUnrelatedEdits],
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
  ["work-hidden-unless-experimental", testWorkViewHiddenUnlessExperimental],
  ["work-approval-and-undo", testWorkModeApprovalAndUndo],
  ["work-denied-and-stopped", testWorkModeDeniedAndStopped],
  ["work-server-rejects-missing-folder", testWorkModeServerRejectsMissingFolder],
  ["work-ended-without-result", testWorkModeEndedWithoutResultIsAnError],
  ["work-auto-approve", testWorkModeAutoApprove],
  ["work-auto-approve-hidden-on-old-server", testWorkModeAutoApproveHiddenOnOldServer],
  ["work-keeps-the-conversation", testWorkModeKeepsTheConversation],
  ["work-outside-folder-is-said-plainly", testWorkModeOutsideFolderIsSaidPlainly],
  ["work-offers-closest-folder", testWorkModeOffersClosestFolder],
  ["work-compact-replaces-earlier-turns", testWorkModeCompactReplacesEarlierTurns],
  ["work-slash-menu-lists-harness-commands", testWorkSlashMenuListsHarnessCommands],
  ["work-slash-command-runs-in-the-harness", testWorkSlashCommandRunsInTheHarness],
  ["work-sessions-saved-and-reopened", testWorkSessionsAreSavedAndReopened],
  ["work-native-harness-history", testWorkNativeHarnessHistory],
  ["work-session-duplicates", testWorkSessionDuplicatesAreExpandable],
  ["work-stable-conversation-id", testWorkRunCarriesItsStableConversationId],
  ["work-linked-harness-history", testWorkLinkedHarnessSessionsStayAccessible],
  ["work-generation-speed-counter", testWorkGenerationSpeedCounter],
  ["work-tool-profiles-limit-a-run", testWorkToolProfilesLimitWhatARunCanUse],
  ["work-profiles-discovery-recovery", testWorkProfilesRecoverFromDiscoveryFailure],
  ["work-profiles-stale-discovery", testWorkProfilesIgnoreClosedDiscovery],
  ["work-agent-install-flow", testWorkAgentInstallFlow],
  ["work-agent-install-cancel", testWorkAgentInstallCancelKeepsSelection],
  ["system-agents-panel", testSystemAgentsPanel],
  ["system-agents-off-without-experimental", testSystemAgentsOffWithoutExperimental],
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
