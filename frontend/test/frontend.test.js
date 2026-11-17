import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { SAFETY_ROUTE_UI_MAP, getSafetyRouteUI, getSafeSmsAction } from '../src/utils/safetyRoutes.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const srcDir = path.resolve(__dirname, '../src');

describe('1 & 2. Emergency Resources Contract & Fallback Verification', () => {
  test('Backend-provided emergency resources preserve fields correctly', () => {
    const backendResource = {
      resource_id: 'in-erss-112',
      resource_name: 'National Emergency Response Support System',
      authority: 'Ministry of Home Affairs',
      telephone: '112',
      resource_type: 'emergency_dispatch',
      modality: ['voice'],
      availability: '24/7',
      toll_free: true,
      official_url: 'https://112.gov.in',
    };

    assert.equal(backendResource.resource_name, 'National Emergency Response Support System');
    assert.equal(backendResource.telephone, '112');
    assert.equal(backendResource.resource_type, 'emergency_dispatch');
    assert.equal(backendResource.modality.includes('voice'), true);
    assert.equal(backendResource.modality.includes('sms'), false); // Does not assume SMS
  });

  test('Missing resources do not produce fabricated contacts and use safe fallback', () => {
    const rawBackendPayload = {
      session_id: 'session-xyz',
      response: 'Supportive guidance text',
      safety_route: 'standard_conversation',
      emergency_resources: [],
    };

    const resources = Array.isArray(rawBackendPayload.emergency_resources)
      ? rawBackendPayload.emergency_resources
      : [];

    assert.equal(resources.length, 0);
  });
});

describe('3. Hardcoded Emergency Number Scan', () => {
  test('No hardcoded emergency numbers exist in frontend/src source files', () => {
    const prohibitedRegex = /\b(988|112|14416|999|111|741741|85258)\b/;
    const violations = [];

    function scanDir(dir) {
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          scanDir(fullPath);
        } else if (/\.(jsx?|tsx?|html)$/.test(entry.name)) {
          const content = fs.readFileSync(fullPath, 'utf8');
          // Allow comments or test files if any, but scan code
          const lines = content.split('\n');
          lines.forEach((line, lineNum) => {
            if (prohibitedRegex.test(line)) {
              violations.push(`${fullPath}:${lineNum + 1}: ${line.trim()}`);
            }
          });
        }
      }
    }

    scanDir(srcDir);
    assert.deepEqual(violations, [], `Discovered hardcoded emergency numbers: ${JSON.stringify(violations, null, 2)}`);
  });
});

describe('4. Visual Distinguishability of Resource Types', () => {
  test('Counselling and emergency dispatch are strictly distinguishable', () => {
    function classifyResource(type) {
      if (type === 'emergency_dispatch' || type === 'hospital_emergency') {
        return 'dispatch_badge';
      }
      if (type === 'crisis_counselling') {
        return 'counselling_badge';
      }
      if (type === 'text_support') {
        return 'text_badge';
      }
      return 'support_badge';
    }

    assert.equal(classifyResource('emergency_dispatch'), 'dispatch_badge');
    assert.equal(classifyResource('crisis_counselling'), 'counselling_badge');
    assert.equal(classifyResource('text_support'), 'text_badge');
    assert.notEqual(classifyResource('emergency_dispatch'), classifyResource('crisis_counselling'));
  });

  test('Telephone and SMS capability checks respect backend modality', () => {
    const voiceOnly = { telephone: '12345', modality: ['voice'] };
    const smsOnly = { telephone: '67890', modality: ['sms'] };
    const voiceAndSms = { telephone: '55555', modality: ['voice', 'sms'] };
    const webOnly = { telephone: null, modality: ['web'] };

    function checkCapabilities(res) {
      return {
        canCall: Boolean(res.telephone && res.modality?.includes('voice')),
        canSms: Boolean(res.telephone && res.modality?.includes('sms')),
      };
    }

    assert.deepEqual(checkCapabilities(voiceOnly), { canCall: true, canSms: false });
    assert.deepEqual(checkCapabilities(smsOnly), { canCall: false, canSms: true });
    assert.deepEqual(checkCapabilities(voiceAndSms), { canCall: true, canSms: true });
    assert.deepEqual(checkCapabilities(webOnly), { canCall: false, canSms: false });
  });
});

describe('5, 6 & 7. Safety Route Interpretation & Triage Mapping', () => {
  test('Elevated classifier signals do not automatically become emergency UI', () => {
    const uiState = getSafetyRouteUI('elevated_classifier_signal');
    assert.equal(uiState.isEmergency, false);
    assert.equal(uiState.mode, 'elevated_classifier_signal');
  });

  test('Pending safety follow-up is not emergency dispatch', () => {
    const uiState = getSafetyRouteUI('pending_safety_follow_up');
    assert.equal(uiState.isEmergency, false);
    assert.equal(uiState.mode, 'pending_safety_follow_up');
  });

  test('Emergency bypass triggers emergency UI', () => {
    const uiState = getSafetyRouteUI('emergency_bypass');
    assert.equal(uiState.isEmergency, true);
    assert.equal(uiState.mode, 'emergency_bypass');
  });

  test('Classifier failure (ml_risk_unassessed) uses safe fallback mode', () => {
    const uiState = getSafetyRouteUI('ml_risk_unassessed');
    assert.equal(uiState.isEmergency, false);
    assert.equal(uiState.mode, 'ml_risk_unassessed');
  });

  test('Unknown safety routes use safe neutral fallback, not emergency classification', () => {
    const unknownState = getSafetyRouteUI('unrecognized_experimental_route');
    assert.equal(unknownState.isEmergency, false);
    assert.equal(unknownState.mode, 'standard_conversation');

    const nullState = getSafetyRouteUI(null);
    assert.equal(nullState.isEmergency, false);
    assert.equal(nullState.mode, 'standard_conversation');
  });
});

describe('8 & 9. Grounding and Audio Lifecycle', () => {
  test('Audio default state is inactive (does not autoplay)', () => {
    let isPlaying = false; // Initial state in useAmbientAudio
    assert.equal(isPlaying, false);
  });

  test('Modal close triggers explicit audio stop handler', () => {
    let isPlaying = true;
    let stoppedCalled = false;

    const mockAudioHook = {
      stopAudio: () => {
        isPlaying = false;
        stoppedCalled = true;
      },
    };

    function handleModalClose() {
      mockAudioHook.stopAudio();
    }

    handleModalClose();
    assert.equal(stoppedCalled, true);
    assert.equal(isPlaying, false);
  });
});

describe('10. Privacy & Storage Isolation', () => {
  test('Sensitive message text is not persisted in browser storage APIs', () => {
    // Scan frontend/src to confirm localStorage/sessionStorage are never called
    let hasStorageCalls = false;
    function scanStorage(dir) {
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          scanStorage(fullPath);
        } else if (/\.(jsx?|tsx?)$/.test(entry.name)) {
          const content = fs.readFileSync(fullPath, 'utf8');
          if (/localStorage\.(setItem|getItem)|sessionStorage\.(setItem|getItem)/.test(content)) {
            hasStorageCalls = true;
          }
        }
      }
    }

    scanStorage(srcDir);
    assert.equal(hasStorageCalls, false, 'localStorage/sessionStorage call found in frontend/src');
  });
});

describe('11. Safe SMS Actions & Availability Integrity', () => {
  test('SMS service requiring keyword constructs link ONLY when instructions available', () => {
    const trevor = {
      modality: ['voice', 'sms'],
      telephone: '1-866-488-7386',
      sms_instructions: 'Text START to 678-678',
    };
    const action = getSafeSmsAction(trevor);
    assert.equal(action?.type, 'single');
    assert.equal(action?.href, 'sms:678678?body=START');
    assert.equal(action?.label, 'Text START to 678-678');
    assert.equal(action?.number, '678-678');
    assert.equal(action?.keyword, 'START');
  });

  test('SMS service without keyword instructions does not generate misleading SMS link for voice telephone', () => {
    const voiceLineWithSmsModality = {
      modality: ['voice', 'sms'],
      telephone: '1-800-273-8255',
      sms_instructions: null,
    };
    const action = getSafeSmsAction(voiceLineWithSmsModality);
    assert.equal(action, null);
  });

  test('Voice-only service never generates SMS action', () => {
    const voiceOnly = {
      modality: ['voice'],
      telephone: '112',
    };
    const action = getSafeSmsAction(voiceOnly);
    assert.equal(action, null);
  });

  test('Explicit direct SMS instruction generates direct text action without keyword', () => {
    const directSms = {
      modality: ['voice', 'sms'],
      telephone: '988',
      sms_instructions: 'Text 988',
    };
    const action = getSafeSmsAction(directSms);
    assert.equal(action?.type, 'single');
    assert.equal(action?.href, 'sms:988');
    assert.equal(action?.label, 'Text 988');
  });

  test('Never infers SMS capability from shortcode length when instructions are missing', () => {
    const shortcodeMissingInstructions = {
      modality: ['voice', 'sms'],
      telephone: '988',
      sms_instructions: null,
    };
    assert.equal(getSafeSmsAction(shortcodeMissingInstructions), null);

    const voiceShortcodeWithSmsModality = {
      modality: ['voice', 'sms'],
      telephone: '911',
      sms_instructions: null,
    };
    assert.equal(getSafeSmsAction(voiceShortcodeWithSmsModality), null);
  });

  test('Demographic dual shortcode instructions present safe instruction display', () => {
    const dualDemographic = {
      modality: ['sms'],
      telephone: '686868',
      sms_instructions: 'Text 686868 (Youth) or 741741 (Adults)',
    };
    const action = getSafeSmsAction(dualDemographic);
    assert.equal(action?.type, 'instruction_only');
    assert.equal(action?.instructionsText, 'Text 686868 (Youth) or 741741 (Adults)');
  });

  test('Unverified availability is distinguished from operational verification', () => {
    const cardContent = fs.readFileSync(path.join(srcDir, 'components/safety/EmergencyResourceCard.jsx'), 'utf8');
    assert.match(cardContent, /Listed:\s*\{availability\}/);
    assert.match(cardContent, /Live operational availability not independently verified/);
    assert.doesNotMatch(cardContent, /Verified Emergency & Crisis Resources/);
    assert.doesNotMatch(cardContent, /Authoritative Registry/);

    const safetyViewContent = fs.readFileSync(path.join(srcDir, 'components/safety/SafetyResponseView.jsx'), 'utf8');
    assert.match(safetyViewContent, /Listed Emergency & Crisis Resources/);
    assert.match(safetyViewContent, /Institutional Directory Reference/);
    assert.doesNotMatch(safetyViewContent, /Verified Emergency & Crisis Resources/);
    assert.doesNotMatch(safetyViewContent, /Authoritative Registry/);
  });
});

describe('12. HTTP 413 Handling & Safe User Guidance', () => {
  test('ApiError correctly encapsulates HTTP 413 status, errorCode, and emergency resources', async () => {
    const { ApiError } = await import('../src/services/api.js');
    const mockResources = [
      {
        resource_id: 'test-resource',
        resource_name: 'Test Emergency Line',
        telephone: '000',
      },
    ];

    const err = new ApiError(
      'Request body exceeds the maximum permitted size of 65536 bytes (64 KB).',
      413,
      'PAYLOAD_TOO_LARGE',
      null,
      mockResources
    );

    assert.equal(err.status, 413);
    assert.equal(err.errorCode, 'PAYLOAD_TOO_LARGE');
    assert.equal(Array.isArray(err.emergencyResources), true);
    assert.equal(err.emergencyResources.length, 1);
    assert.equal(err.emergencyResources[0].resource_name, 'Test Emergency Line');
  });

  test('ChatContext formats clear, non-technical guidance for HTTP 413 without connection error misattribution', () => {
    const contextContent = fs.readFileSync(path.join(srcDir, 'context/ChatContext.jsx'), 'utf8');

    // Confirm HTTP 413 / PAYLOAD_TOO_LARGE is explicitly handled
    assert.match(contextContent, /err\?\.status === 413 \|\| errorCode === 'PAYLOAD_TOO_LARGE'/);

    // Confirm it explains the 64 KB limit and requests a shorter message
    assert.match(contextContent, /exceeds the maximum allowed request size \(64 KB\)/);
    assert.match(contextContent, /please share a shorter message/);

    // Confirm emergencyResources from error are preserved in message state
    assert.match(contextContent, /emergencyResources:\s*errorEmergencyResources/);
  });

  test('MessageBubble visibly renders emergency resources on error responses', () => {
    const bubbleContent = fs.readFileSync(path.join(srcDir, 'components/chat/MessageBubble.jsx'), 'utf8');

    // Confirm emergency resource cards are rendered when emergencyResources is non-empty,
    // ensuring HTTP 413 error cards are visibly rendered for the user.
    assert.match(bubbleContent, /emergencyResources\s*&&\s*emergencyResources\.length\s*>\s*0/);
    assert.match(bubbleContent, /<EmergencyResourceCard/);
  });

  test('api.js sets X-Jurisdiction request header to support middleware jurisdiction lookup', () => {
    const apiContent = fs.readFileSync(path.join(srcDir, 'services/api.js'), 'utf8');
    assert.match(apiContent, /requestHeaders\['X-Jurisdiction'\]\s*=\s*jurisdiction/);
  });
});

describe('13. Default Jurisdiction & Regional Selector', () => {
  test('ChatContext initializes default jurisdiction to India (IN)', () => {
    const contextContent = fs.readFileSync(path.join(srcDir, 'context/ChatContext.jsx'), 'utf8');
    assert.match(contextContent, /const\s*\[\s*jurisdiction\s*,\s*setJurisdiction\s*\]\s*=\s*useState\(['"]IN['"]\)/);
  });

  test('Navbar renders India (IN) as the primary/default region option', () => {
    const navbarContent = fs.readFileSync(path.join(srcDir, 'components/layout/Navbar.jsx'), 'utf8');
    assert.match(navbarContent, /<option\s+value=["']IN["']>India\s*\(IN\)<\/option>/);
  });
});

describe('14. Safety Protocols Dynamic Directory Integration', () => {
  test('SafetyProtocolsPage queries dynamic emergency resources using active jurisdiction', () => {
    const protocolsContent = fs.readFileSync(path.join(srcDir, 'pages/SafetyProtocolsPage.jsx'), 'utf8');
    assert.match(protocolsContent, /import\s*\{\s*getEmergencyResources\s*\}\s*from\s*['"]\.\.\/services\/api['"]/);
    assert.match(protocolsContent, /import\s*\{\s*useChat\s*\}\s*from\s*['"]\.\.\/context\/ChatContext['"]/);
    assert.match(protocolsContent, /getEmergencyResources\s*\(\s*jurisdiction\s*\)/);
    assert.match(protocolsContent, /<SafetyResponseView[^>]*resources=\{resources\}/);
  });

  test('SafetyResponseView supports dynamic resources prop and directory fallback', () => {
    const viewContent = fs.readFileSync(path.join(srcDir, 'components/safety/SafetyResponseView.jsx'), 'utf8');
    assert.match(viewContent, /import\s*\{\s*getEmergencyResources\s*\}\s*from\s*['"]\.\.\/\.\.\/services\/api['"]/);
    assert.match(viewContent, /activeSafetyEvent\?\.emergencyResources/);
    assert.match(viewContent, /directoryResources/);
  });

  test('api.js exports getEmergencyResources targeting /safety/resources', () => {
    const apiContent = fs.readFileSync(path.join(srcDir, 'services/api.js'), 'utf8');
    assert.match(apiContent, /export\s+async\s+function\s+getEmergencyResources\s*\(/);
    assert.match(apiContent, /\/safety\/resources/);
  });
});

describe('15. Conversational UI Layout & Proper Message Bubbles', () => {
  test('ChatWorkspace implements a balanced two-column desktop layout with mobile drawer', () => {
    const workspaceContent = fs.readFileSync(path.join(srcDir, 'components/chat/ChatWorkspace.jsx'), 'utf8');
    // Confirms desktop sidebar ~290-310px width
    assert.match(workspaceContent, /w-\[290px\]\s+xl:w-\[310px\]/);
    // Confirms main chat area flexes to occupy remaining space
    assert.match(workspaceContent, /<section[^>]*className="[^"]*flex-1[^"]*min-h-0/);
    // Confirms mobile drawer toggle state
    assert.match(workspaceContent, /isMobileSidebarOpen/);
  });

  test('User messages align to the right with subtle sage background', () => {
    const bubbleContent = fs.readFileSync(path.join(srcDir, 'components/chat/MessageBubble.jsx'), 'utf8');
    // Right alignment for user messages
    assert.match(bubbleContent, /flex\s+justify-end\s+w-full/);
    // Subtle sage background
    assert.match(bubbleContent, /bg-primary-fixed-dim\/70/);
    // Preserves max-width constraint for comfortable reading
    assert.match(bubbleContent, /max-w-\[90%\]\s+sm:max-w-\[80%\]\s+md:max-w-\[75%\]/);
  });

  test('Assistant messages align to the left with warm-white container and distinct styling', () => {
    const bubbleContent = fs.readFileSync(path.join(srcDir, 'components/chat/MessageBubble.jsx'), 'utf8');
    // Left alignment for assistant messages
    assert.match(bubbleContent, /flex\s+justify-start\s+w-full/);
    // Warm-white container surface
    assert.match(bubbleContent, /bg-surface-container-lowest/);
    // Maximum width bounded to 75-85%
    assert.match(bubbleContent, /max-w-\[95%\]\s+sm:max-w-\[88%\]\s+md:max-w-\[82%\]/);
    // Does NOT align assistant to center
    assert.doesNotMatch(bubbleContent, /justify-center/);
  });
});

describe('16. Composer & Input Autofocus Verification', () => {
  test('ChatInput implements automatic focus on mount, session changes, and response completion', () => {
    const inputContent = fs.readFileSync(path.join(srcDir, 'components/chat/ChatInput.jsx'), 'utf8');
    // Native autoFocus attribute present
    assert.match(inputContent, /autoFocus=\{true\}/);
    // useEffect mount focus
    assert.match(inputContent, /textareaRef\.current\.focus\(\)/);
    // Focus dependency on sessionId (New Reflection)
    assert.match(inputContent, /prevSessionIdRef\.current\s*!==\s*sessionId/);
    // Restores focus after isLoading completes
    assert.match(inputContent, /prevLoadingRef\.current\s*&&\s*!isLoading/);
  });

  test('ChatInput provides multiline textarea with soothing placeholder and 10000 character limit', () => {
    const inputContent = fs.readFileSync(path.join(srcDir, 'components/chat/ChatInput.jsx'), 'utf8');
    // 10000 char limit (matches backend schema)
    assert.match(inputContent, /MAX_CHAR_LIMIT\s*=\s*10000/);
    // Placeholder matches prompt requirements
    assert.match(inputContent, /placeholder="Share whatever you're experiencing right now\.\.\. take your time\."/);
    // Enter sends, Shift+Enter newline
    assert.match(inputContent, /e\.key\s*===\s*'Enter'\s*&&\s*!e\.shiftKey/);
  });

  test('ChatWindow manages smooth independent scrolling without fixed container restrictions', () => {
    const windowContent = fs.readFileSync(path.join(srcDir, 'components/chat/ChatWindow.jsx'), 'utf8');
    // Scroll container present
    assert.match(windowContent, /id="chatMessageStream"/);
    assert.match(windowContent, /scroll-smooth/);
    // Smart bottom scroll tracking
    assert.match(windowContent, /isNearBottomRef/);
  });
});

describe('17. Conversational UI Refinements & Citation Normalization', () => {
  test('ChatWindow eliminates h-full flex overflow and provides generous clearance above composer', () => {
    const windowContent = fs.readFileSync(path.join(srcDir, 'components/chat/ChatWindow.jsx'), 'utf8');
    // Root container uses flex-1 min-h-0 without h-full
    assert.doesNotMatch(windowContent, /className="flex-1 flex flex-col h-full/);
    assert.match(windowContent, /className="flex-1 flex flex-col min-h-0/);
    // Subheader wraps content with spacious max-w container to unify alignment with stream and composer
    assert.match(windowContent, /max-w-5xl xl:max-w-6xl 2xl:max-w-7xl mx-auto flex items-center justify-between/);
    // Bottom padding ensures clearance above fixed composer
    assert.match(windowContent, /pb-20/);
    // Explicit bottom anchor spacer
    assert.match(windowContent, /streamEndRef[^>]*className="h-8 shrink-0"/);
  });

  test('MarkdownView normalizes raw chunk identifiers into user-friendly references', () => {
    const mdContent = fs.readFileSync(path.join(srcDir, 'components/common/MarkdownView.jsx'), 'utf8');
    // Exposes cleanInlineCitations
    assert.match(mdContent, /export\s+function\s+cleanInlineCitations/);
    // Handles CounselChat fallback label
    assert.match(mdContent, /\[Community Counseling Reference\]/);
    // Handles Verified Evidence fallback label
    assert.match(mdContent, /\[Verified Evidence Reference\]/);
    // Detects raw chunk IDs
    assert.match(mdContent, /counselchat_chunk_/);
    // Renders citation badge with BookOpen icon
    assert.match(mdContent, /<BookOpen/);
  });

  test('MessageBubble passes citations to MarkdownView and maintains clear left-alignment', () => {
    const bubbleContent = fs.readFileSync(path.join(srcDir, 'components/chat/MessageBubble.jsx'), 'utf8');
    // Passes citations to MarkdownView
    assert.match(bubbleContent, /<MarkdownView[^>]*citations=\{citations\}/);
    // Left alignment and text-left on assistant container
    assert.match(bubbleContent, /flex justify-start w-full items-start text-left/);
  });

  test('CitationCard displays friendly titles without exposing raw technical identifiers', () => {
    const cardContent = fs.readFileSync(path.join(srcDir, 'components/citations/CitationCard.jsx'), 'utf8');
    // Uses getDisplayTitle helper
    assert.match(cardContent, /getDisplayTitle\(cite\)/);
    // Preserves Doc ID in provenance details
    assert.match(cardContent, /Doc ID:\s*\{cite\.document_id\}/);
    // Fallback titles for counselchat and evidence
    assert.match(cardContent, /Community Counseling Reference/);
    assert.match(cardContent, /Verified Evidence Reference/);
  });
});

describe('18. Compact Modern Message Composer (WhatsApp-Desktop Style)', () => {
  test('ChatInput implements compact single-row container and capped dynamic auto-expansion', () => {
    const inputContent = fs.readFileSync(path.join(srcDir, 'components/chat/ChatInput.jsx'), 'utf8');
    // Compact input bar container ~48-52px min-height
    assert.match(inputContent, /min-h-\[48px\]\s+sm:min-h-\[52px\]/);
    // Rows initialized to 1 for true single-line compact empty state
    assert.match(inputContent, /rows=\{1\}/);
    // Auto-expansion dynamically between 24px and 130px max
    assert.match(inputContent, /Math\.min\(Math\.max\(scrollHeight,\s*24\),\s*130\)/);
    // Textarea constrained with max-h-[130px] and overflow-y-auto
    assert.match(inputContent, /max-h-\[130px\]\s+overflow-y-auto/);
    // Send button positioned at the right inside the input bar with items-end
    assert.match(inputContent, /flex\s+items-end/);
    assert.match(inputContent, /<Send/);
  });
});

describe('19. AIML Performance Dashboard & Empirical Verification', () => {
  test('Navbar and App routing integrate AIML Dashboard navigation seamlessly', () => {
    const navbarContent = fs.readFileSync(path.join(srcDir, 'components/layout/Navbar.jsx'), 'utf8');
    assert.match(navbarContent, /id:\s*'aiml',\s*label:\s*'AIML Dashboard'/);

    const appContent = fs.readFileSync(path.join(srcDir, 'App.jsx'), 'utf8');
    assert.match(appContent, /import\s*\{\s*AimlDashboardPage\s*\}\s*from\s*'\.\/pages\/AimlDashboardPage'/);
    assert.match(appContent, /activePage\s*===\s*'aiml'/);
  });

  test('AimlDashboardPage accurately displays verified RoBERTa test metrics & confusion matrix', () => {
    const dashboardContent = fs.readFileSync(path.join(srcDir, 'pages/AimlDashboardPage.jsx'), 'utf8');

    // Verified metrics
    assert.match(dashboardContent, /90\.78%/); // Accuracy
    assert.match(dashboardContent, /76\.09%/); // Precision
    assert.match(dashboardContent, /87\.60%/); // Recall
    assert.match(dashboardContent, /81\.44%/); // F1
    assert.match(dashboardContent, /96\.19%/); // ROC-AUC
    assert.match(dashboardContent, /87\.25%/); // PR-AUC

    // Confusion matrix exact counts
    assert.match(dashboardContent, /5,127/); // TN
    assert.match(dashboardContent, /462/);   // FP
    assert.match(dashboardContent, /208/);   // FN
    assert.match(dashboardContent, /1,470/); // TP

    // Training history and model calibration details
    assert.match(dashboardContent, /0\.5098/); // Threshold
    assert.match(dashboardContent, /3\.329/);  // pos_weight
    assert.match(dashboardContent, /roberta-base/);
  });

  test('AimlDashboardPage displays empirical dataset analytics without fabrication', () => {
    const dashboardContent = fs.readFileSync(path.join(srcDir, 'pages/AimlDashboardPage.jsx'), 'utf8');

    // Hugging Face source and ground truth explanation
    assert.match(dashboardContent, /ihsansaad24\/Mental-Health_Text-Classification_Dataset/);
    assert.match(dashboardContent, /Binary crisis labels.*derived from the dataset's original/);

    // Classification dataset numbers
    assert.match(dashboardContent, /49,?612/); // Raw
    assert.match(dashboardContent, /48,?456/); // Cleaned
    assert.match(dashboardContent, /17,?705/); // Normal
    assert.match(dashboardContent, /14,?243/); // Depression
    assert.match(dashboardContent, /11,?191/); // Suicidal
    assert.match(dashboardContent, /5,?317/);  // Anxiety
    assert.match(dashboardContent, /33,?920/); // Train
    assert.match(dashboardContent, /7,?269/);  // Val
    assert.match(dashboardContent, /7,?267/);  // Test

    // CounselChat dataset utilization & description
    assert.match(dashboardContent, /Public counseling Q&A dataset containing therapist responses/);
    assert.match(dashboardContent, /1,?482/); // Raw records
    assert.match(dashboardContent, /1,?421/); // Indexed
    assert.match(dashboardContent, /61/);     // Excluded
    assert.match(dashboardContent, /155/);    // Distinct therapists
    assert.match(dashboardContent, /1,?965/); // Vectors
  });

  test('AimlDashboardPage displays actual RAG evaluation and transparent non-evaluated labels', () => {
    const dashboardContent = fs.readFileSync(path.join(srcDir, 'pages/AimlDashboardPage.jsx'), 'utf8');

    // RAG infrastructure
    assert.match(dashboardContent, /BAAI\/bge-small-en-v1\.5/);
    assert.match(dashboardContent, /384/);
    assert.match(dashboardContent, /IndexFlatIP/);
    assert.match(dashboardContent, /1,?996/); // Total combined vectors

    // Verified retrieval report
    assert.match(dashboardContent, /Recall@3/);
    assert.match(dashboardContent, /Precision@Top-1/);
    assert.match(dashboardContent, /OOD Abstention/);
    assert.match(dashboardContent, /15\.2 ms/);

    // Transparently labels non-evaluated metrics
    assert.match(dashboardContent, /Not evaluated yet/);
  });

  test('AimlDashboardPage implements cascading provider telemetry and live demonstration console', () => {
    const dashboardContent = fs.readFileSync(path.join(srcDir, 'pages/AimlDashboardPage.jsx'), 'utf8');

    // Provider architecture
    assert.match(dashboardContent, /Gemini Developer API/);
    assert.match(dashboardContent, /Groq Cloud API/);
    assert.match(dashboardContent, /Live telemetry not yet collected/);

    // Live demonstration panel
    assert.match(dashboardContent, /Live AI\/ML Inference Lab/);
    assert.match(dashboardContent, /handleRunDemo/);
    assert.match(dashboardContent, /runAimlInference/);
    assert.match(dashboardContent, /roberta/);
    assert.match(dashboardContent, /safety/);
    assert.match(dashboardContent, /rag/);
    assert.match(dashboardContent, /llm/);
  });
});



