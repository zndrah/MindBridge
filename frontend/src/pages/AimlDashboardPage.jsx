import React, { useState, useEffect } from 'react';
import {
  Brain,
  Cpu,
  Database,
  Search,
  ArrowRight,
  ShieldCheck,
  Activity,
  Layers,
  GitBranch,
  Terminal,
  RefreshCw,
  CheckCircle,
  AlertTriangle,
  Info,
  BarChart2,
  BookOpen,
  Send,
  Sparkles,
  Server,
  Zap,
  ExternalLink,
  ShieldAlert,
  Sliders,
  Check,
} from 'lucide-react';
import { getReady, runAimlInference } from '../services/api';
import { useChat } from '../context/ChatContext';

export function AimlDashboardPage({ onStartChat }) {
  const { jurisdiction } = useChat();
  const [activeTab, setActiveTab] = useState('classifier'); // 'classifier' | 'dataset' | 'rag' | 'llm' | 'architecture' | 'demo'
  const [backendHealth, setBackendHealth] = useState(null);
  const [isHealthLoading, setIsHealthLoading] = useState(false);

  // Live demonstration state
  const [demoQuery, setDemoQuery] = useState(
    'What are some evidence-based grounding techniques for coping with exam anxiety?'
  );
  const [isDemoRunning, setIsDemoRunning] = useState(false);
  const [demoResult, setDemoResult] = useState(null);
  const [demoLatency, setDemoLatency] = useState(null);
  const [demoError, setDemoError] = useState(null);

  // Fetch /ready status on mount
  const refreshBackendDiagnostics = async () => {
    setIsHealthLoading(true);
    try {
      const data = await getReady();
      setBackendHealth(data);
    } catch {
      setBackendHealth({ isReady: false, status: 'error' });
    } finally {
      setIsHealthLoading(false);
    }
  };

  useEffect(() => {
    refreshBackendDiagnostics();
  }, []);

  const handleRunDemo = async (customPrompt) => {
    const textToSend = customPrompt || demoQuery;
    if (!textToSend.trim() || isDemoRunning) return;

    setIsDemoRunning(true);
    setDemoError(null);
    setDemoResult(null);
    const startTime = performance.now();

    try {
      const result = await runAimlInference({
        message: textToSend,
        jurisdiction: jurisdiction || 'IN',
      });
      const elapsed = Math.round(performance.now() - startTime);
      setDemoLatency(elapsed);
      setDemoResult(result);
    } catch (err) {
      const elapsed = Math.round(performance.now() - startTime);
      setDemoLatency(elapsed);

      const status = typeof err?.status === 'number' ? err.status : 0;
      const code = err?.errorCode || 'API_ERROR';
      const rawMsg = err?.message || 'Diagnostic inference failed.';

      let category = 'INFERENCE_ERROR';
      let remediation = 'An unexpected error occurred during pipeline execution. Verify backend server logs for detailed stack traces.';

      if (status === 0 || code === 'NETWORK_ERROR' || code === 'BACKEND_OFFLINE') {
        category = 'BACKEND_OFFLINE';
        remediation = 'MindBridge FastAPI backend is unreachable on port 8000. Start or verify the server using the command below.';
      } else if (status === 408 || code === 'REQUEST_TIMEOUT' || code === 'INFERENCE_TIMEOUT') {
        category = 'INFERENCE_TIMEOUT';
        remediation = 'Pipeline execution exceeded the 45-second execution budget (e.g. during LLM multi-provider fallback). Retry the diagnostic query.';
      } else if (status === 503 || code === 'MODEL_UNAVAILABLE') {
        category = 'MODEL_UNAVAILABLE';
        remediation = 'RoBERTa classifier weights or FAISS vector index are unavailable or still initializing. Check the /ready probe diagnostics.';
      } else if (status === 400 || status === 422 || code === 'VALIDATION_ERROR' || code === 'EMPTY_MESSAGE') {
        category = 'VALIDATION_ERROR';
        remediation = 'Payload validation failed. Ensure the input prompt contains valid non-empty text without illegal tokens.';
      }

      setDemoError({
        category,
        message: rawMsg,
        status,
        code,
        remediation,
      });
    } finally {
      setIsDemoRunning(false);
    }
  };

  const demoPresets = [
    {
      label: 'Academic Stress (Psychoeducation RAG)',
      text: 'What are some evidence-based grounding techniques for coping with exam anxiety?',
    },
    {
      label: 'Somatic Grounding (5-4-3-2-1 Technique)',
      text: 'How does the 5-4-3-2-1 sensory orientation technique help de-escalate panic attacks?',
    },
    {
      label: 'Depressive Inertia & Coping',
      text: 'I feel completely drained and unmotivated to get out of bed today. What can I do?',
    },
    {
      label: 'Suicidal Intent (Crisis Bypass Trigger)',
      text: 'I am going to kill myself tonight with pills',
    },
  ];

  return (
    <div className="w-full min-h-screen bg-surface text-on-surface py-6 sm:py-8 md:py-10 px-4 sm:px-6 lg:px-8">
      <div className="w-full max-w-[1560px] mx-auto space-y-6 sm:space-y-8">
        {/* Header Banner */}
        <div className="bg-surface-container-lowest border border-surface-container rounded-2xl p-6 sm:p-8 shadow-sm">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div className="space-y-2">
              <div className="inline-flex items-center gap-2 px-3 py-0.5 rounded-full bg-primary/10 text-primary text-xs font-semibold tracking-wide uppercase">
                <Sparkles className="w-3.5 h-3.5" />
                <span>AIML Research & Model Verification Console</span>
              </div>
              <h1 className="text-xl sm:text-2xl md:text-3xl font-bold font-headline text-on-surface tracking-tight">
                MindBridge Core AI/ML Performance Dashboard
              </h1>
              <p className="text-on-surface-variant text-xs sm:text-sm max-w-3xl leading-relaxed">
                Empirical evaluation metrics, Hugging Face dataset lineage, FAISS vector retrieval benchmarks, and cascading Gemini/Groq LLM orchestration telemetry verified on held-out test splits and live backend probes.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2.5">
              <button
                onClick={refreshBackendDiagnostics}
                disabled={isHealthLoading}
                className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl border border-surface-container-high bg-surface hover:bg-surface-container text-xs font-medium text-on-surface transition-colors shadow-2xs"
                title="Query live /ready diagnostics"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${isHealthLoading ? 'animate-spin text-primary' : 'text-on-surface-variant'}`} />
                <span>{isHealthLoading ? 'Querying...' : 'Refresh Telemetry'}</span>
              </button>

              <button
                onClick={onStartChat}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-primary text-on-primary text-xs font-semibold hover:bg-primary-container transition-all shadow-sm active:scale-95"
              >
                <span>Open Chat Assistant</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* Quick Telemetry Chips */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3.5 sm:gap-4 mt-6 pt-6 border-t border-surface-container">
            <div className="bg-surface-container-low rounded-xl p-3.5 sm:p-4 border border-surface-container/60">
              <div className="text-[11px] font-medium text-on-surface-variant uppercase tracking-wider">RoBERTa Classifier</div>
              <div className="text-base font-bold text-primary mt-0.5">90.78% Accuracy</div>
              <div className="text-[11px] text-on-surface-variant">87.60% Crisis Recall (N=7,267)</div>
            </div>
            <div className="bg-surface-container-low rounded-xl p-3.5 sm:p-4 border border-surface-container/60">
              <div className="text-[11px] font-medium text-on-surface-variant uppercase tracking-wider">RAG Vector Store</div>
              <div className="text-base font-bold text-secondary mt-0.5">1,996 Total Vectors</div>
              <div className="text-[11px] text-on-surface-variant">31 Clinical + 1,965 CounselChat</div>
            </div>
            <div className="bg-surface-container-low rounded-xl p-3.5 sm:p-4 border border-surface-container/60">
              <div className="text-[11px] font-medium text-on-surface-variant uppercase tracking-wider">Primary LLM Provider</div>
              <div className="text-base font-bold text-on-surface mt-0.5 flex items-center gap-1.5">
                <span className={`w-2 h-2 rounded-full ${backendHealth?.models?.gemini_generator ? 'bg-secondary' : 'bg-outline'}`} />
                Gemini Developer API
              </div>
              <div className="text-[11px] text-on-surface-variant">Fallback: Groq Cloud API</div>
            </div>
            <div className="bg-surface-container-low rounded-xl p-3.5 sm:p-4 border border-surface-container/60">
              <div className="text-[11px] font-medium text-on-surface-variant uppercase tracking-wider">Active Jurisdiction</div>
              <div className="text-base font-bold text-on-surface mt-0.5">India (IN) Default</div>
              <div className="text-[11px] text-on-surface-variant">Multi-region hotline directory</div>
            </div>
          </div>
        </div>

        {/* Dashboard Navigation Tabs */}
        <div className="flex items-center gap-2 overflow-x-auto pb-1.5 pt-1 scrollbar-none border-b border-surface-container">
          {[
            { id: 'classifier', label: 'RoBERTa Classifier', icon: Brain },
            { id: 'dataset', label: 'Dataset Analytics', icon: Database },
            { id: 'rag', label: 'RAG Performance', icon: Search },
            { id: 'llm', label: 'LLM Orchestration', icon: Cpu },
            { id: 'architecture', label: 'System Architecture', icon: GitBranch },
            { id: 'demo', label: 'Live AI/ML Inference Lab', icon: Terminal },
          ].map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-3.5 py-2 rounded-t-xl text-xs sm:text-sm font-semibold transition-all whitespace-nowrap border-b-2 ${
                  isActive
                    ? 'border-primary text-primary bg-surface-container-lowest shadow-2xs'
                    : 'border-transparent text-on-surface-variant hover:text-on-surface hover:bg-surface-container-low'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-primary' : 'text-on-surface-variant'}`} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* TAB 1: RoBERTa Crisis Classifier */}
        {activeTab === 'classifier' && (
          <div className="space-y-5">
            {/* Top Stat Cards */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
              {[
                { label: 'Overall Accuracy', value: '90.78%', sub: '7,267 test samples', color: 'text-primary' },
                { label: 'Crisis Precision', value: '76.09%', sub: 'PPV for crisis risk', color: 'text-primary' },
                { label: 'Crisis Recall', value: '87.60%', sub: 'Sensitivity on crisis', color: 'text-secondary' },
                { label: 'Crisis F1-Score', value: '81.44%', sub: 'Harmonic mean', color: 'text-secondary' },
                { label: 'ROC-AUC Score', value: '0.96188 (96.19%)', sub: 'Area under ROC curve', color: 'text-primary' },
                { label: 'PR-AUC Score', value: '0.87247 (87.25%)', sub: 'Area under PR curve', color: 'text-secondary' },
              ].map((card, idx) => (
                <div key={idx} className="bg-surface-container-lowest border border-surface-container rounded-xl p-3.5 shadow-2xs">
                  <div className="text-[11px] font-semibold text-on-surface-variant uppercase tracking-wider">{card.label}</div>
                  <div className={`text-xl sm:text-2xl font-bold font-sans mt-1 ${card.color}`}>{card.value}</div>
                  <div className="text-[11px] text-on-surface-variant mt-0.5">{card.sub}</div>
                </div>
              ))}
            </div>

            {/* Confusion Matrix & Calibration Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
              {/* Confusion Matrix Card */}
              <div className="lg:col-span-7 bg-surface-container-lowest border border-surface-container rounded-2xl p-5 sm:p-6 shadow-sm">
                <div className="flex items-center justify-between mb-3.5">
                  <div>
                    <h3 className="text-base font-bold text-on-surface font-headline">
                      Confusion Matrix (Held-out Test Split, N=7,267)
                    </h3>
                    <p className="text-xs text-on-surface-variant mt-0.5">
                      Target: Binary Crisis Risk (Positive: Suicidal Intent / Severe Self-Harm; Negative: Normal / Depressive Non-Crisis)
                    </p>
                  </div>
                  <span className="text-xs px-2.5 py-1 rounded-md bg-surface-container text-on-surface-variant font-mono">
                    Threshold: 0.5098
                  </span>
                </div>

                {/* Heatmap Grid */}
                <div className="overflow-x-auto">
                  <table className="w-full text-center border-collapse">
                    <thead>
                      <tr>
                        <th className="p-2 text-xs font-medium text-on-surface-variant w-1/4"></th>
                        <th className="p-2 text-xs font-semibold text-on-surface bg-surface-container-low rounded-t-lg w-3/8" colSpan={1}>
                          Predicted Non-Crisis (0)
                        </th>
                        <th className="p-2 text-xs font-semibold text-on-surface bg-surface-container-low rounded-t-lg w-3/8" colSpan={1}>
                          Predicted Crisis-Risk (1)
                        </th>
                      </tr>
                    </thead>
                    <tbody className="text-xs sm:text-sm">
                      <tr>
                        <td className="p-3 text-left font-semibold text-on-surface bg-surface-container-low border-r border-surface-container">
                          Actual Non-Crisis (N=5,589)
                        </td>
                        <td className="p-4 bg-primary/10 border border-surface-container text-primary font-bold">
                          <div className="text-lg font-mono">5,127</div>
                          <div className="text-[11px] font-normal text-on-surface-variant">True Negative (91.73%)</div>
                        </td>
                        <td className="p-4 bg-tertiary-container/30 border border-surface-container text-tertiary font-bold">
                          <div className="text-lg font-mono">462</div>
                          <div className="text-[11px] font-normal text-on-surface-variant">False Positive (8.27% benign flag)</div>
                        </td>
                      </tr>
                      <tr>
                        <td className="p-3 text-left font-semibold text-on-surface bg-surface-container-low border-r border-surface-container">
                          Actual Crisis-Risk (N=1,678)
                        </td>
                        <td className="p-4 bg-error-container/30 border border-surface-container text-error font-bold">
                          <div className="text-lg font-mono">208</div>
                          <div className="text-[11px] font-normal text-on-surface-variant">False Negative (12.40% rule caught)</div>
                        </td>
                        <td className="p-4 bg-secondary-container/60 border border-surface-container text-secondary font-bold">
                          <div className="text-lg font-mono">1,470</div>
                          <div className="text-[11px] font-normal text-on-surface-variant">True Positive (87.60% sensitivity)</div>
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                <div className="mt-3.5 pt-3 border-t border-surface-container flex flex-wrap items-center justify-between text-xs text-on-surface-variant gap-2">
                  <span>Non-Crisis Specificity: <strong>91.73%</strong></span>
                  <span>Crisis Sensitivity (Recall): <strong>87.60%</strong></span>
                  <span>Crisis Precision: <strong>76.09%</strong></span>
                  <span>Balanced Accuracy: <strong>89.67%</strong></span>
                </div>
              </div>

              {/* Model Specifications Card */}
              <div className="lg:col-span-5 bg-surface-container-lowest border border-surface-container rounded-2xl p-5 sm:p-6 shadow-sm flex flex-col justify-between">
                <div>
                  <h3 className="text-base font-bold text-on-surface font-headline mb-3">
                    Architecture & Calibration Details
                  </h3>
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between py-1.5 border-b border-surface-container">
                      <span className="text-on-surface-variant">Base Architecture</span>
                      <span className="font-mono font-medium text-on-surface">roberta-base (125M params)</span>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-surface-container">
                      <span className="text-on-surface-variant">Max Token Sequence Length</span>
                      <span className="font-mono font-medium text-on-surface">256 tokens (evaluated to 384)</span>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-surface-container">
                      <span className="text-on-surface-variant">Calibrated Decision Threshold</span>
                      <span className="font-mono font-medium text-primary">0.5098 (F1-optimal on validation)</span>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-surface-container">
                      <span className="text-on-surface-variant">Loss Function</span>
                      <span className="font-mono font-medium text-on-surface">BCEWithLogitsLoss (pos_weight=3.329)</span>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-surface-container">
                      <span className="text-on-surface-variant">Optimizer & Learning Rate</span>
                      <span className="font-mono font-medium text-on-surface">AdamW, lr=2e-5, warmup=10%</span>
                    </div>
                    <div className="flex justify-between py-1.5 border-b border-surface-container">
                      <span className="text-on-surface-variant">Weight Decay & Batch Size</span>
                      <span className="font-mono font-medium text-on-surface">decay=0.01, effective batch=32</span>
                    </div>
                    <div className="flex justify-between py-1.5">
                      <span className="text-on-surface-variant">Best Checkpoint</span>
                      <span className="font-mono font-medium text-secondary">Epoch 2 (val_f1=79.43%)</span>
                    </div>
                  </div>
                </div>

                <div className="mt-3.5 p-3 bg-surface-container-low rounded-xl border border-surface-container text-xs text-on-surface-variant">
                  <div className="font-semibold text-on-surface flex items-center gap-1.5 mb-1">
                    <Info className="w-3.5 h-3.5 text-primary" />
                    <span>Safety Layer Defense-in-Depth</span>
                  </div>
                  False negatives (12.4%) from RoBERTa are caught by our deterministic rule-based safety orchestrator, which triggers immediate emergency bypass on explicit self-harm keywords before LLM generation.
                </div>
              </div>
            </div>

            {/* Training & Validation Progression */}
            <div className="bg-surface-container-lowest border border-surface-container rounded-2xl p-5 sm:p-6 shadow-sm">
              <div className="flex items-center justify-between mb-3.5">
                <div>
                  <h3 className="text-base font-bold text-on-surface font-headline">
                    Training & Validation Progression (3 Fine-Tuning Epochs)
                  </h3>
                  <p className="text-xs text-on-surface-variant mt-0.5">
                    Source: experiments/logs/training_history.json (Tracked across 33,920 train and 7,269 validation samples)
                  </p>
                </div>
                <span className="text-xs px-2.5 py-1 rounded-md bg-secondary/10 text-secondary font-semibold">
                  Checkpoint Saved: Epoch 2
                </span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs sm:text-sm border-collapse">
                  <thead>
                    <tr className="border-b border-surface-container bg-surface-container-low text-on-surface font-semibold">
                      <th className="py-2 px-3">Epoch</th>
                      <th className="py-2 px-3">Train Loss</th>
                      <th className="py-2 px-3">Val Loss</th>
                      <th className="py-2 px-3">Val Accuracy</th>
                      <th className="py-2 px-3">Val Crisis F1</th>
                      <th className="py-2 px-3">Val Recall</th>
                      <th className="py-2 px-3">Val Precision</th>
                      <th className="py-2 px-3">Val PR-AUC</th>
                      <th className="py-2 px-3">Epoch Time</th>
                      <th className="py-2 px-3">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-surface-container">
                    <tr className="hover:bg-surface-container-low/50">
                      <td className="py-2.5 px-3 font-mono font-medium">Epoch 1</td>
                      <td className="py-2.5 px-3 font-mono">0.6038</td>
                      <td className="py-2.5 px-3 font-mono">0.4578</td>
                      <td className="py-2.5 px-3 font-mono">88.39%</td>
                      <td className="py-2.5 px-3 font-mono">77.25%</td>
                      <td className="py-2.5 px-3 font-mono">79.52%</td>
                      <td className="py-2.5 px-3 font-mono">75.10%</td>
                      <td className="py-2.5 px-3 font-mono">84.77%</td>
                      <td className="py-2.5 px-3 font-mono text-on-surface-variant">610.7 s</td>
                      <td className="py-2.5 px-3 text-on-surface-variant">Baseline</td>
                    </tr>
                    <tr className="bg-primary/5 hover:bg-primary/10 font-medium">
                      <td className="py-2.5 px-3 font-mono text-primary font-bold">Epoch 2</td>
                      <td className="py-2.5 px-3 font-mono text-primary font-bold">0.4025</td>
                      <td className="py-2.5 px-3 font-mono text-primary font-bold">0.3542</td>
                      <td className="py-2.5 px-3 font-mono text-primary font-bold">89.85%</td>
                      <td className="py-2.5 px-3 font-mono text-primary font-bold">79.43%</td>
                      <td className="py-2.5 px-3 font-mono text-primary font-bold">79.58%</td>
                      <td className="py-2.5 px-3 font-mono text-primary font-bold">79.28%</td>
                      <td className="py-2.5 px-3 font-mono text-primary font-bold">87.31%</td>
                      <td className="py-2.5 px-3 font-mono text-on-surface-variant">611.5 s</td>
                      <td className="py-2.5 px-3 text-primary font-bold">Best Checkpoint</td>
                    </tr>
                    <tr className="hover:bg-surface-container-low/50">
                      <td className="py-2.5 px-3 font-mono font-medium">Epoch 3</td>
                      <td className="py-2.5 px-3 font-mono">0.3165</td>
                      <td className="py-2.5 px-3 font-mono">0.3540</td>
                      <td className="py-2.5 px-3 font-mono">89.68%</td>
                      <td className="py-2.5 px-3 font-mono">79.04%</td>
                      <td className="py-2.5 px-3 font-mono">78.98%</td>
                      <td className="py-2.5 px-3 font-mono">79.11%</td>
                      <td className="py-2.5 px-3 font-mono">87.11%</td>
                      <td className="py-2.5 px-3 font-mono text-on-surface-variant">611.1 s</td>
                      <td className="py-2.5 px-3 text-on-surface-variant">Overfit plateau</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            {/* Experimental Safeguard Disclaimer */}
            <div className="bg-surface-container-low border border-surface-container rounded-xl p-3.5 flex items-start gap-3">
              <ShieldAlert className="w-5 h-5 text-tertiary shrink-0 mt-0.5" />
              <div className="text-xs text-on-surface-variant leading-relaxed">
                <strong className="text-on-surface">Experimental Safeguard Disclaimer:</strong> These performance metrics reflect held-out test splits from curated online mental health forum corpora. MindBridge is an automated peer first-response triage tool and does not constitute a certified medical device, diagnostic instrument, or clinical psychiatric assessment.
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: Dataset Analytics */}
        {activeTab === 'dataset' && (
          <div className="space-y-5">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
              {/* Dataset 1: Mental Health Classification */}
              <div className="bg-surface-container-lowest border border-surface-container rounded-2xl p-5 sm:p-6 shadow-sm space-y-3.5">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-base font-bold text-on-surface font-headline">
                      Mental Health Classification Dataset
                    </h3>
                    <p className="text-xs text-primary font-medium mt-0.5">
                      Hugging Face: <code className="font-mono bg-surface-container px-1 py-0.5 rounded">ihsansaad24/Mental-Health_Text-Classification_Dataset</code>
                    </p>
                  </div>
                  <span className="text-xs px-2.5 py-1 rounded-full bg-primary/10 text-primary font-semibold">
                    48,456 Cleaned
                  </span>
                </div>

                {/* Sample breakdown */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1">
                  <div className="p-2.5 bg-surface-container-low rounded-xl text-center">
                    <div className="text-xs text-on-surface-variant">Raw Samples</div>
                    <div className="text-base font-bold font-mono text-on-surface">49,612</div>
                  </div>
                  <div className="p-2.5 bg-surface-container-low rounded-xl text-center">
                    <div className="text-xs text-on-surface-variant">Cleaned</div>
                    <div className="text-base font-bold font-mono text-primary">48,456</div>
                  </div>
                  <div className="p-2.5 bg-surface-container-low rounded-xl text-center">
                    <div className="text-xs text-on-surface-variant">Duplicates Purged</div>
                    <div className="text-base font-bold font-mono text-tertiary">1,156</div>
                  </div>
                  <div className="p-2.5 bg-surface-container-low rounded-xl text-center">
                    <div className="text-xs text-on-surface-variant">pos_weight</div>
                    <div className="text-base font-bold font-mono text-secondary">3.329</div>
                  </div>
                </div>

                {/* Class Distribution Bars */}
                <div className="space-y-2.5 pt-1">
                  <div className="text-xs font-semibold text-on-surface">Class Label Distribution:</div>
                  {[
                    { label: 'Normal / Neutral Sentiment', count: 17705, pct: '36.54%', color: 'bg-primary' },
                    { label: 'Depression Symptoms', count: 14243, pct: '29.39%', color: 'bg-secondary' },
                    { label: 'Suicidal Intent / Ideation', count: 11191, pct: '23.09%', color: 'bg-error' },
                    { label: 'Anxiety & Panic Symptoms', count: 5317, pct: '10.97%', color: 'bg-tertiary' },
                  ].map((cls, idx) => (
                    <div key={idx} className="space-y-1">
                      <div className="flex justify-between text-xs">
                        <span className="text-on-surface font-medium">{cls.label}</span>
                        <span className="text-on-surface-variant font-mono">{cls.count.toLocaleString()} ({cls.pct})</span>
                      </div>
                      <div className="w-full h-2 bg-surface-container rounded-full overflow-hidden">
                        <div className={`h-full ${cls.color}`} style={{ width: cls.pct }}></div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Splits Breakdown */}
                <div className="pt-3 border-t border-surface-container">
                  <div className="text-xs font-semibold text-on-surface mb-2">Stratified Split Allocation:</div>
                  <div className="grid grid-cols-3 gap-2 text-center text-xs">
                    <div className="p-2 rounded-lg bg-surface-container-low">
                      <div className="text-on-surface-variant">Train (70%)</div>
                      <div className="font-mono font-bold text-on-surface">33,920</div>
                    </div>
                    <div className="p-2 rounded-lg bg-surface-container-low">
                      <div className="text-on-surface-variant">Validation (15%)</div>
                      <div className="font-mono font-bold text-on-surface">7,269</div>
                    </div>
                    <div className="p-2 rounded-lg bg-surface-container-low">
                      <div className="text-on-surface-variant">Test (15%)</div>
                      <div className="font-mono font-bold text-on-surface">7,267</div>
                    </div>
                  </div>
                </div>

                <div className="p-3 bg-surface-container-low rounded-xl border border-surface-container text-xs text-on-surface-variant leading-relaxed">
                  <strong className="text-on-surface">Dataset Source & Ground Truth Transparency:</strong> Binary crisis labels (Crisis-Risk vs Non-Crisis) are derived from the dataset's original categorical annotations (Suicidal/Depression vs Normal/Anxiety). These labels reflect self-reported subjective distress from online user text and do <strong>not</strong> represent clinically validated ground truth or formal psychiatric diagnoses.
                </div>
              </div>

              {/* Dataset 2: CounselChat Knowledge Corpus */}
              <div className="bg-surface-container-lowest border border-surface-container rounded-2xl p-5 sm:p-6 shadow-sm space-y-3.5 flex flex-col justify-between">
                <div className="space-y-3.5">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-base font-bold text-on-surface font-headline">
                        CounselChat Expert Coping Knowledge Base
                      </h3>
                      <p className="text-xs text-on-surface-variant mt-0.5">
                        Public counseling Q&A dataset containing therapist responses; credentials are not independently verified.
                      </p>
                    </div>
                    <span className="text-xs px-2.5 py-1 rounded-full bg-secondary/10 text-secondary font-semibold">
                      1,421 Indexed Answers
                    </span>
                  </div>

                  {/* Utilization Metrics */}
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 pt-1">
                    <div className="p-2.5 bg-surface-container-low rounded-xl text-center">
                      <div className="text-xs text-on-surface-variant">Original Records</div>
                      <div className="text-base font-bold font-mono text-on-surface">1,482</div>
                    </div>
                    <div className="p-2.5 bg-surface-container-low rounded-xl text-center">
                      <div className="text-xs text-on-surface-variant">Distinct Therapists</div>
                      <div className="text-base font-bold font-mono text-primary">155</div>
                    </div>
                    <div className="p-2.5 bg-surface-container-low rounded-xl text-center">
                      <div className="text-xs text-on-surface-variant">Unique Questions</div>
                      <div className="text-base font-bold font-mono text-on-surface">780</div>
                    </div>
                    <div className="p-2.5 bg-surface-container-low rounded-xl text-center">
                      <div className="text-xs text-on-surface-variant">Indexed Coping Answers</div>
                      <div className="text-base font-bold font-mono text-secondary">1,421</div>
                    </div>
                    <div className="p-2.5 bg-surface-container-low rounded-xl text-center">
                      <div className="text-xs text-on-surface-variant">Excluded Unsafe Records</div>
                      <div className="text-base font-bold font-mono text-tertiary">61</div>
                    </div>
                    <div className="p-2.5 bg-surface-container-low rounded-xl text-center">
                      <div className="text-xs text-on-surface-variant">FAISS Vector Count</div>
                      <div className="text-base font-bold font-mono text-primary">1,965</div>
                    </div>
                  </div>

                  {/* Exclusions Breakdown */}
                  <div className="p-3 bg-surface-container-low rounded-xl border border-surface-container text-xs space-y-1.5">
                    <div className="font-semibold text-on-surface">Safety Exclusion Breakdown (61 records):</div>
                    <div className="flex justify-between text-on-surface-variant">
                      <span>• Acute Crisis & Self-Harm Restriction</span>
                      <span className="font-mono font-medium">49 records</span>
                    </div>
                    <div className="flex justify-between text-on-surface-variant">
                      <span>• Prescription Drug / Medical Prescribing</span>
                      <span className="font-mono font-medium">8 records</span>
                    </div>
                    <div className="flex justify-between text-on-surface-variant">
                      <span>• Malformed / Short Non-Informative Answers</span>
                      <span className="font-mono font-medium">4 records</span>
                    </div>
                  </div>
                </div>

                <div className="p-3 bg-tertiary-container/20 border border-tertiary-container rounded-xl text-xs text-on-surface-variant leading-relaxed">
                  <strong className="text-on-surface">Dataset Utilization Notice:</strong> CounselChat is unverified community Q&A content used for non-clinical conversational support, anxiety reframing, and stress coping perspectives—not clinical recommendations or verified medical facts.
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: RAG Performance */}
        {activeTab === 'rag' && (
          <div className="space-y-5">
            {/* Vector Store Overview */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-3.5">
              <div className="bg-surface-container-lowest border border-surface-container rounded-xl p-3.5 shadow-2xs">
                <div className="text-[11px] font-semibold text-on-surface-variant uppercase tracking-wider">Clinical KB Vectors</div>
                <div className="text-2xl font-bold font-mono text-primary mt-1">31</div>
                <div className="text-[11px] text-on-surface-variant mt-0.5">22 markdown protocol documents</div>
              </div>
              <div className="bg-surface-container-lowest border border-surface-container rounded-xl p-3.5 shadow-2xs">
                <div className="text-[11px] font-semibold text-on-surface-variant uppercase tracking-wider">CounselChat Vectors</div>
                <div className="text-2xl font-bold font-mono text-secondary mt-1">1,965</div>
                <div className="text-[11px] text-on-surface-variant mt-0.5">1,421 indexed therapist answers</div>
              </div>
              <div className="bg-surface-container-lowest border border-surface-container rounded-xl p-3.5 shadow-2xs">
                <div className="text-[11px] font-semibold text-on-surface-variant uppercase tracking-wider">Total Combined Vectors</div>
                <div className="text-2xl font-bold font-mono text-on-surface mt-1">1,996</div>
                <div className="text-[11px] text-on-surface-variant mt-0.5">Dual FAISS vector stores</div>
              </div>
              <div className="bg-surface-container-lowest border border-surface-container rounded-xl p-3.5 shadow-2xs">
                <div className="text-[11px] font-semibold text-on-surface-variant uppercase tracking-wider">Embedding & Search</div>
                <div className="text-sm font-bold font-mono text-primary mt-1">BAAI/bge-small-en-v1.5</div>
                <div className="text-[11px] text-on-surface-variant mt-0.5">384-d • FAISS IndexFlatIP Cosine</div>
              </div>
            </div>

            {/* Benchmark Retrieval Evaluation Results */}
            <div className="bg-surface-container-lowest border border-surface-container rounded-2xl p-5 sm:p-6 shadow-sm">
              <div className="flex items-center justify-between mb-3.5">
                <div>
                  <h3 className="text-base font-bold text-on-surface font-headline">
                    Empirical Retrieval Evaluation Report (13 Verified Benchmark Queries)
                  </h3>
                  <p className="text-xs text-on-surface-variant mt-0.5">
                    Source: data/experimental/counselchat/retrieval_evaluation_report.json
                  </p>
                </div>
                <span className="text-xs px-2.5 py-1 rounded-md bg-primary/10 text-primary font-semibold">
                  Zero Hallucination Target
                </span>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
                <div className="p-3 bg-surface-container-low rounded-xl border border-surface-container">
                  <div className="text-[11px] text-on-surface-variant">Recall@3</div>
                  <div className="text-xl font-bold font-mono text-primary mt-0.5">100.0%</div>
                  <div className="text-[10px] text-on-surface-variant">Target doc in top 3 hits</div>
                </div>
                <div className="p-3 bg-surface-container-low rounded-xl border border-surface-container">
                  <div className="text-[11px] text-on-surface-variant">Precision@Top-1</div>
                  <div className="text-xl font-bold font-mono text-primary mt-0.5">87.5%</div>
                  <div className="text-[10px] text-on-surface-variant">Direct topic alignment</div>
                </div>
                <div className="p-3 bg-surface-container-low rounded-xl border border-surface-container">
                  <div className="text-[11px] text-on-surface-variant">OOD Abstention</div>
                  <div className="text-xl font-bold font-mono text-secondary mt-0.5">100.0%</div>
                  <div className="text-[10px] text-on-surface-variant">Out-of-domain rejection</div>
                </div>
                <div className="p-3 bg-surface-container-low rounded-xl border border-surface-container">
                  <div className="text-[11px] text-on-surface-variant">Crisis Bypass Rate</div>
                  <div className="text-xl font-bold font-mono text-secondary mt-0.5">100.0%</div>
                  <div className="text-[10px] text-on-surface-variant">Zero LLM leakage on risk</div>
                </div>
                <div className="p-3 bg-surface-container-low rounded-xl border border-surface-container">
                  <div className="text-[11px] text-on-surface-variant">Citation Compliance</div>
                  <div className="text-xl font-bold font-mono text-primary mt-0.5">100.0%</div>
                  <div className="text-[10px] text-on-surface-variant">Strict document attribution</div>
                </div>
                <div className="p-3 bg-surface-container-low rounded-xl border border-surface-container">
                  <div className="text-[11px] text-on-surface-variant">Warm Retrieval Latency</div>
                  <div className="text-xl font-bold font-mono text-on-surface mt-0.5">~15.2 ms</div>
                  <div className="text-[10px] text-on-surface-variant">Cold index load: ~398 ms</div>
                </div>
              </div>

              {/* Non-evaluated benchmarks */}
              <div className="mt-4 pt-3.5 border-t border-surface-container grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
                <div className="p-2.5 bg-surface-container-low/60 rounded-lg">
                  <span className="text-on-surface-variant">Mean Reciprocal Rank (MRR): </span>
                  <span className="font-semibold text-outline">Not evaluated yet</span>
                </div>
                <div className="p-2.5 bg-surface-container-low/60 rounded-lg">
                  <span className="text-on-surface-variant">Context Relevance Benchmark: </span>
                  <span className="font-semibold text-outline">Not evaluated yet</span>
                </div>
                <div className="p-2.5 bg-surface-container-low/60 rounded-lg">
                  <span className="text-on-surface-variant">Faithfulness / Groundedness Metric: </span>
                  <span className="font-semibold text-outline">Not evaluated yet</span>
                </div>
              </div>
            </div>

            {/* Visual RAG Pipeline Architecture */}
            <div className="bg-surface-container-lowest border border-surface-container rounded-2xl p-5 sm:p-6 shadow-sm">
              <h3 className="text-base font-bold text-on-surface font-headline mb-3.5">
                End-to-End RAG Knowledge Synthesis Pipeline
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-6 gap-3">
                {[
                  { step: '1', title: 'User Query', desc: 'Sanitized input & intent classification', icon: Terminal },
                  { step: '2', title: 'Dense Embedder', desc: 'BAAI/bge-small-en-v1.5 (384-d normalized)', icon: Brain },
                  { step: '3', title: 'FAISS Search', desc: 'Clinical KB (31) + CounselChat (1,965)', icon: Search },
                  { step: '4', title: 'Context Augmentation', desc: 'Top-k passages formatted with citations', icon: BookOpen },
                  { step: '5', title: 'LLM Synthesis', desc: 'Gemini Primary (Groq fallback cascade)', icon: Cpu },
                  { step: '6', title: 'Verification Gate', desc: 'Grounding check & attribution guardrails', icon: ShieldCheck },
                ].map((item, idx) => {
                  const Icon = item.icon;
                  return (
                    <div key={idx} className="relative p-3 bg-surface-container-low rounded-xl border border-surface-container flex flex-col justify-between">
                      <div>
                        <div className="flex items-center justify-between mb-1.5">
                          <span className="w-5 h-5 rounded-full bg-primary text-on-primary text-[11px] font-bold flex items-center justify-center">
                            {item.step}
                          </span>
                          <Icon className="w-4 h-4 text-primary" />
                        </div>
                        <div className="text-xs font-bold text-on-surface">{item.title}</div>
                        <div className="text-[11px] text-on-surface-variant mt-1 leading-snug">{item.desc}</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: LLM Multi-Provider Architecture */}
        {activeTab === 'llm' && (
          <div className="space-y-5">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
              {/* Primary: Gemini Developer API */}
              <div className="bg-surface-container-lowest border border-surface-container rounded-2xl p-5 sm:p-6 shadow-sm space-y-3.5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <div className="w-9 h-9 rounded-xl bg-primary/10 flex items-center justify-center text-primary">
                      <Sparkles className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="text-base font-bold text-on-surface font-headline">Gemini Developer API</h3>
                      <div className="text-xs text-on-surface-variant">Primary Generation Engine</div>
                    </div>
                  </div>
                  <span className={`text-xs px-2.5 py-1 rounded-full font-semibold ${backendHealth?.models?.gemini_generator ? 'bg-secondary/15 text-secondary' : 'bg-outline/20 text-outline'}`}>
                    {backendHealth?.models?.gemini_generator ? 'Online & Active' : 'Checking'}
                  </span>
                </div>

                <div className="space-y-2 text-xs">
                  <div className="flex justify-between py-1.5 border-b border-surface-container">
                    <span className="text-on-surface-variant">Configured Model Identifier</span>
                    <span className="font-mono font-medium text-on-surface">gemini-2.5-flash / gemini-3.5-flash</span>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-surface-container">
                    <span className="text-on-surface-variant">Role in Architecture</span>
                    <span className="font-medium text-on-surface">Primary Grounded Response Synthesis</span>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-surface-container">
                    <span className="text-on-surface-variant">Request Timeout Limit</span>
                    <span className="font-mono font-medium text-on-surface">30,000 ms</span>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-surface-container">
                    <span className="text-on-surface-variant">Cascading Trigger</span>
                    <span className="font-medium text-on-surface">Rate limit (429), Network timeout, 5xx</span>
                  </div>
                </div>

                <div className="p-3 bg-surface-container-low rounded-xl border border-surface-container text-xs text-on-surface-variant">
                  <strong>Capabilities:</strong> Native JSON structured output, long context window for multi-document RAG synthesis, and calibrated empathy instructions aligned with safety guardrails.
                </div>
              </div>

              {/* Fallback: Groq Cloud API */}
              <div className="bg-surface-container-lowest border border-surface-container rounded-2xl p-5 sm:p-6 shadow-sm space-y-3.5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2.5">
                    <div className="w-9 h-9 rounded-xl bg-secondary/10 flex items-center justify-center text-secondary">
                      <Zap className="w-5 h-5" />
                    </div>
                    <div>
                      <h3 className="text-base font-bold text-on-surface font-headline">Groq Cloud API</h3>
                      <div className="text-xs text-on-surface-variant">Automated Fallback Engine</div>
                    </div>
                  </div>
                  <span className={`text-xs px-2.5 py-1 rounded-full font-semibold ${backendHealth?.models?.groq_generator ? 'bg-secondary/15 text-secondary' : 'bg-outline/20 text-outline'}`}>
                    {backendHealth?.models?.groq_generator ? 'Standby Ready' : 'Checking'}
                  </span>
                </div>

                <div className="space-y-2 text-xs">
                  <div className="flex justify-between py-1.5 border-b border-surface-container">
                    <span className="text-on-surface-variant">Configured Model Identifier</span>
                    <span className="font-mono font-medium text-on-surface">openai/gpt-oss-120b</span>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-surface-container">
                    <span className="text-on-surface-variant">Role in Architecture</span>
                    <span className="font-medium text-on-surface">High-speed resilience fallback</span>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-surface-container">
                    <span className="text-on-surface-variant">Inference Architecture</span>
                    <span className="font-medium text-on-surface">Language Processing Unit (LPU)</span>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-surface-container">
                    <span className="text-on-surface-variant">Fallback Trigger Condition</span>
                    <span className="font-medium text-on-surface">Automatic seamless switchover</span>
                  </div>
                </div>

                <div className="p-3 bg-surface-container-low rounded-xl border border-surface-container text-xs text-on-surface-variant">
                  <strong>Resilience Role:</strong> If Gemini experiences quota depletion, server timeout, or network latency, the CascadingLLMGenerator immediately reroutes the synthesis to Groq without crashing the user session.
                </div>
              </div>
            </div>

            {/* Provider Telemetry Status */}
            <div className="bg-surface-container-lowest border border-surface-container rounded-2xl p-5 sm:p-6 shadow-sm">
              <div className="flex items-center justify-between mb-3.5">
                <div>
                  <h3 className="text-base font-bold text-on-surface font-headline">
                    Provider Execution Telemetry & Health Probe
                  </h3>
                  <p className="text-xs text-on-surface-variant mt-0.5">
                    Live probe data from backend /ready endpoint
                  </p>
                </div>
                <div className="text-xs text-on-surface-variant">
                  LLM Provider Mode: <strong className="text-primary font-mono">{backendHealth?.models?.llm_provider || 'gemini'}</strong>
                </div>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs sm:text-sm border-collapse">
                  <thead>
                    <tr className="border-b border-surface-container bg-surface-container-low text-on-surface font-semibold">
                      <th className="py-2 px-3">Provider</th>
                      <th className="py-2 px-3">Target Model</th>
                      <th className="py-2 px-3">Live Health</th>
                      <th className="py-2 px-3">Historical Requests</th>
                      <th className="py-2 px-3">Success Rate</th>
                      <th className="py-2 px-3">Average Latency</th>
                      <th className="py-2 px-3">Fallback Trigger Count</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-surface-container text-xs">
                    <tr>
                      <td className="py-2.5 px-3 font-semibold text-on-surface flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full bg-secondary"></span>
                        Gemini (Primary)
                      </td>
                      <td className="py-2.5 px-3 font-mono">gemini-2.5-flash</td>
                      <td className="py-2.5 px-3 text-secondary font-semibold">
                        {backendHealth?.models?.gemini_generator ? 'Operational' : 'Unavailable'}
                      </td>
                      <td className="py-2.5 px-3 text-outline">Live telemetry not yet collected</td>
                      <td className="py-2.5 px-3 text-outline">Live telemetry not yet collected</td>
                      <td className="py-2.5 px-3 text-outline">Live telemetry not yet collected</td>
                      <td className="py-2.5 px-3 text-outline">Live telemetry not yet collected</td>
                    </tr>
                    <tr>
                      <td className="py-2.5 px-3 font-semibold text-on-surface flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full bg-secondary"></span>
                        Groq Cloud (Fallback)
                      </td>
                      <td className="py-2.5 px-3 font-mono">openai/gpt-oss-120b</td>
                      <td className="py-2.5 px-3 text-secondary font-semibold">
                        {backendHealth?.models?.groq_generator ? 'Operational' : 'Unavailable'}
                      </td>
                      <td className="py-2.5 px-3 text-outline">Live telemetry not yet collected</td>
                      <td className="py-2.5 px-3 text-outline">Live telemetry not yet collected</td>
                      <td className="py-2.5 px-3 text-outline">Live telemetry not yet collected</td>
                      <td className="py-2.5 px-3 text-outline">Live telemetry not yet collected</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div className="mt-3 text-[11px] text-on-surface-variant">
                * Note on Telemetry: MindBridge chat sessions are designed to be strictly ephemeral in memory for privacy protection. Historical telemetry counters are not persisted to a database to prevent user footprint tracking.
              </div>
            </div>
          </div>
        )}

        {/* TAB 5: System Architecture */}
        {activeTab === 'architecture' && (
          <div className="space-y-5">
            <div className="bg-surface-container-lowest border border-surface-container rounded-2xl p-5 sm:p-7 shadow-sm">
              <div className="max-w-3xl mb-5">
                <h3 className="text-lg sm:text-xl font-bold text-on-surface font-headline">
                  MindBridge Dual-Path Safety & RAG Architecture
                </h3>
                <p className="text-xs sm:text-sm text-on-surface-variant mt-1 leading-relaxed">
                  MindBridge implements a strict safety-first architectural topology. Imminent suicidal intent or severe self-harm signals immediately bypass generative AI, triggering deterministic emergency safe harbor protocols.
                </p>
              </div>

              {/* Architecture Diagram Visualization */}
              <div className="space-y-5">
                {/* Node 1: User Input */}
                <div className="flex flex-col items-center">
                  <div className="w-full max-w-md p-3.5 bg-surface-container-high rounded-xl border border-surface-container text-center">
                    <div className="text-[11px] uppercase tracking-wider text-on-surface-variant font-semibold">Stage 1: User Request Ingestion</div>
                    <div className="text-sm font-bold text-on-surface mt-0.5">User Natural Language Message</div>
                    <div className="text-[11px] text-on-surface-variant">XSS Sanitization • Tokenization • Jurisdiction Binding</div>
                  </div>
                  <div className="w-0.5 h-5 bg-outline-variant"></div>
                </div>

                {/* Node 2: Safety Classification Gate */}
                <div className="flex flex-col items-center">
                  <div className="w-full max-w-lg p-3.5 bg-primary/10 rounded-xl border-2 border-primary/30 text-center shadow-xs">
                    <div className="text-[11px] uppercase tracking-wider text-primary font-bold">Stage 2: Safety & Crisis Triage Node</div>
                    <div className="text-sm font-bold text-on-surface mt-0.5">RoBERTa Classifier + Deterministic Rule Safety Guard</div>
                    <div className="text-xs text-on-surface-variant mt-1">
                      Evaluates crisis risk probability against calibrated threshold (0.5098)
                    </div>
                  </div>
                  <div className="w-0.5 h-5 bg-outline-variant"></div>
                </div>

                {/* Node 3: Decision Router */}
                <div className="w-full grid grid-cols-1 md:grid-cols-2 gap-5 pt-1">
                  {/* Left Path: Crisis Bypass */}
                  <div className="p-4 sm:p-5 bg-error-container/20 rounded-2xl border-2 border-error-container flex flex-col justify-between">
                    <div>
                      <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-error-container text-on-error-container text-[11px] font-bold uppercase mb-2">
                        <ShieldAlert className="w-3.5 h-3.5" />
                        <span>Path A: Imminent Crisis Bypass</span>
                      </div>
                      <h4 className="text-sm font-bold text-on-surface">Deterministic Safe Harbor Protocol</h4>
                      <p className="text-xs text-on-surface-variant mt-1.5 leading-relaxed">
                        Triggered when RoBERTa probability ≥ 0.5098 or rule guard detects explicit self-harm intent.
                      </p>
                      <div className="mt-3.5 space-y-2 text-xs">
                        <div className="p-2 bg-surface-container-lowest rounded-lg border border-surface-container flex items-center justify-between">
                          <span className="text-on-surface-variant">Generative LLM Call</span>
                          <span className="font-bold text-error">Bypassed (0% Hallucination)</span>
                        </div>
                        <div className="p-2 bg-surface-container-lowest rounded-lg border border-surface-container flex items-center justify-between">
                          <span className="text-on-surface-variant">RAG Retrieval</span>
                          <span className="font-bold text-error">Bypassed</span>
                        </div>
                        <div className="p-2 bg-surface-container-lowest rounded-lg border border-surface-container flex items-center justify-between">
                          <span className="text-on-surface-variant">Response Output</span>
                          <span className="font-bold text-secondary">Emergency Resources Directory</span>
                        </div>
                      </div>
                    </div>
                    <div className="mt-3.5 pt-3 border-t border-surface-container/60 text-[11px] text-on-surface-variant">
                      Renders regional emergency dispatch contacts dynamically based on active jurisdiction (e.g. India National Emergency lines).
                    </div>
                  </div>

                  {/* Right Path: Non-Crisis RAG & LLM */}
                  <div className="p-4 sm:p-5 bg-secondary-container/20 rounded-2xl border-2 border-secondary-container/60 flex flex-col justify-between">
                    <div>
                      <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-secondary-container text-on-secondary-container text-[11px] font-bold uppercase mb-2">
                        <Sparkles className="w-3.5 h-3.5" />
                        <span>Path B: Empathetic First Response</span>
                      </div>
                      <h4 className="text-sm font-bold text-on-surface">Dual-Knowledge RAG + Cascading LLM</h4>
                      <p className="text-xs text-on-surface-variant mt-1.5 leading-relaxed">
                        Triggered for coping queries, academic stress, panic de-escalation, and emotional exploration.
                      </p>
                      <div className="mt-3.5 space-y-2 text-xs">
                        <div className="p-2 bg-surface-container-lowest rounded-lg border border-surface-container flex items-center justify-between">
                          <span className="text-on-surface-variant">Dense FAISS Search</span>
                          <span className="font-bold text-primary">Clinical KB (31) + CounselChat (1,965)</span>
                        </div>
                        <div className="p-2 bg-surface-container-lowest rounded-lg border border-surface-container flex items-center justify-between">
                          <span className="text-on-surface-variant">Cascading Generation</span>
                          <span className="font-bold text-primary">Gemini 2.5 Flash → Groq Fallback</span>
                        </div>
                        <div className="p-2 bg-surface-container-lowest rounded-lg border border-surface-container flex items-center justify-between">
                          <span className="text-on-surface-variant">Verification Gate</span>
                          <span className="font-bold text-secondary">Citation Grounding & Attribution</span>
                        </div>
                      </div>
                    </div>
                    <div className="mt-3.5 pt-3 border-t border-surface-container/60 text-[11px] text-on-surface-variant">
                      Returns empathetic, structured response strictly grounded in verified psychoeducation documents with clear citations.
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 6: Live AI/ML Inference Lab */}
        {activeTab === 'demo' && (
          <div className="space-y-6 sm:space-y-8">
            <div className="bg-surface-container-lowest border border-surface-container rounded-2xl p-6 sm:p-8 shadow-sm">
              <div className="max-w-3xl mb-6">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 text-primary text-xs font-semibold uppercase mb-2">
                  <Terminal className="w-3.5 h-3.5" />
                  <span>Real-Time Inference Telemetry</span>
                </div>
                <h3 className="text-lg sm:text-2xl font-bold text-on-surface font-headline">
                  Live AI/ML Inference Lab
                </h3>
                <p className="text-xs sm:text-sm text-on-surface-variant mt-1.5 leading-relaxed">
                  Execute sample user utterances directly through the compiled MindBridge pipeline. Inspect genuine RoBERTa classification probabilities, safety routing branches, FAISS cosine similarity scores, cascading LLM generation latency, and response grounding in real time.
                </p>
              </div>

              {/* Preset Scenario Buttons */}
              <div className="space-y-2.5 mb-6">
                <div className="text-xs font-semibold text-on-surface-variant">Evaluator Test Presets:</div>
                <div className="flex flex-wrap gap-2.5">
                  {demoPresets.map((preset, idx) => (
                    <button
                      key={idx}
                      onClick={() => {
                        setDemoQuery(preset.text);
                        handleRunDemo(preset.text);
                      }}
                      className="px-3.5 py-2 rounded-xl bg-surface-container-low hover:bg-surface-container border border-surface-container text-xs font-medium text-on-surface transition-colors cursor-pointer"
                    >
                      {preset.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Input Area */}
              <div className="space-y-4">
                <div className="relative">
                  <textarea
                    value={demoQuery}
                    onChange={(e) => setDemoQuery(e.target.value)}
                    rows={3}
                    placeholder="Enter test user utterance..."
                    className="w-full p-4 text-xs sm:text-sm bg-surface-container-lowest border border-surface-container-high rounded-xl focus:outline-none focus:border-primary text-on-surface resize-none font-sans"
                  />
                  <div className="absolute right-3.5 bottom-3.5 flex items-center gap-2">
                    <button
                      onClick={() => handleRunDemo()}
                      disabled={isDemoRunning || !demoQuery.trim()}
                      className="inline-flex items-center gap-1.5 px-5 py-2.5 rounded-xl bg-primary text-on-primary text-xs font-semibold hover:bg-primary-container disabled:opacity-50 transition-all shadow-sm cursor-pointer"
                    >
                      {isDemoRunning ? (
                        <>
                          <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                          <span>Executing Pipeline...</span>
                        </>
                      ) : (
                        <>
                          <Send className="w-3.5 h-3.5" />
                          <span>Run Inference</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </div>

              {/* Error Alert */}
              {demoError && (
                <div className="mt-4 p-4 sm:p-5 rounded-2xl bg-error-container/15 border border-error-container text-on-surface text-xs space-y-3">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4 text-error shrink-0" />
                      <span className="font-bold text-error uppercase tracking-wider font-mono text-[11px]">
                        [{demoError.category}] Diagnostic Inference Failed
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      {demoError.status > 0 && (
                        <span className="px-2 py-0.5 rounded bg-surface-container font-mono text-[10px] text-on-surface-variant font-semibold">
                          HTTP {demoError.status}
                        </span>
                      )}
                      <span className="px-2 py-0.5 rounded bg-surface-container font-mono text-[10px] text-on-surface-variant font-semibold">
                        {demoError.code}
                      </span>
                    </div>
                  </div>

                  <p className="text-on-surface text-xs leading-relaxed font-medium">
                    {demoError.message}
                  </p>

                  <div className="p-3 bg-surface-container-low rounded-xl border border-surface-container space-y-1">
                    <div className="text-[10px] font-semibold text-on-surface-variant uppercase tracking-wider">
                      Actionable Remediation
                    </div>
                    <p className="text-xs text-on-surface leading-relaxed">
                      {demoError.remediation}
                    </p>
                    {demoError.category === 'BACKEND_OFFLINE' && (
                      <div className="mt-2 flex items-center gap-2 p-2 rounded-lg bg-surface-container-highest text-primary font-mono text-[11px] select-all">
                        <Terminal className="w-3.5 h-3.5 text-secondary shrink-0" />
                        <span>uv run uvicorn backend.api.main:app --port 8000</span>
                      </div>
                    )}
                  </div>

                  <div className="flex justify-end pt-1">
                    <button
                      onClick={() => handleRunDemo()}
                      disabled={isDemoRunning}
                      className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl bg-error hover:bg-error-container text-on-error hover:text-on-error-container font-semibold text-xs transition-colors shadow-2xs cursor-pointer"
                    >
                      <RefreshCw className={`w-3 h-3 ${isDemoRunning ? 'animate-spin' : ''}`} />
                      <span>Retry Diagnostic</span>
                    </button>
                  </div>
                </div>
              )}

              {/* Demonstration Telemetry Output */}
              {demoResult && (
                <div className="mt-6 space-y-5 pt-6 border-t border-surface-container">
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="text-sm sm:text-base font-bold text-on-surface font-headline">
                        Empirical Inference Telemetry & Subsystem Telemetry
                      </h4>
                      <p className="text-[11px] sm:text-xs text-on-surface-variant">Session: {demoResult.session_id}</p>
                    </div>
                    <span className="text-xs px-3 py-1 rounded-md bg-primary/10 text-primary font-mono font-semibold">
                      Pipeline Latency: {demoResult.total_pipeline_latency_ms || demoLatency} ms
                    </span>
                  </div>

                  {/* 1. RoBERTa Classifier Telemetry */}
                  <div className="bg-surface-container-low rounded-xl p-5 border border-surface-container space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-xs sm:text-sm font-bold text-on-surface">
                        <Brain className="w-4 h-4 text-primary" />
                        <span>1. RoBERTa Crisis Classifier Telemetry</span>
                      </div>
                      <span className={`text-[11px] sm:text-xs px-2.5 py-0.5 rounded font-mono font-bold ${
                        demoResult.roberta?.crisis_flag
                          ? 'bg-error-container text-on-error-container'
                          : 'bg-primary/10 text-primary'
                      }`}>
                        Predicted: {demoResult.roberta?.predicted_label || 'Non-Crisis'}
                      </span>
                    </div>

                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs pt-1">
                      <div className="p-2.5 bg-surface-container-lowest rounded-lg">
                        <div className="text-[10px] text-on-surface-variant uppercase">Crisis Probability</div>
                        <div className="font-mono font-bold text-on-surface mt-0.5">
                          {demoResult.roberta?.probability !== null && demoResult.roberta?.probability !== undefined
                            ? `${(demoResult.roberta.probability * 100).toFixed(2)}%`
                            : 'N/A (Rule Override)'}
                        </div>
                      </div>
                      <div className="p-2.5 bg-surface-container-lowest rounded-lg">
                        <div className="text-[10px] text-on-surface-variant uppercase">Decision Threshold</div>
                        <div className="font-mono font-bold text-primary mt-0.5">
                          {demoResult.roberta?.threshold || 0.5098}
                        </div>
                      </div>
                      <div className="p-2.5 bg-surface-container-lowest rounded-lg">
                        <div className="text-[10px] text-on-surface-variant uppercase">Inference Latency</div>
                        <div className="font-mono font-bold text-secondary mt-0.5">
                          {demoResult.roberta?.latency_ms || 0} ms
                        </div>
                      </div>
                      <div className="p-2.5 bg-surface-container-lowest rounded-lg">
                        <div className="text-[10px] text-on-surface-variant uppercase">Triage Decision</div>
                        <div className="font-mono font-bold text-on-surface mt-0.5">
                          {demoResult.roberta?.decision || 'safe_to_route'}
                        </div>
                      </div>
                    </div>
                    <div className="text-[10px] sm:text-xs text-on-surface-variant italic">
                      * {demoResult.roberta?.note || 'Experimental dataset-derived signal; not clinical diagnostic risk assessment.'}
                    </div>
                  </div>

                  {/* 2. Safety Routing Breakdown */}
                  <div className="bg-surface-container-low rounded-xl p-5 border border-surface-container space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-xs sm:text-sm font-bold text-on-surface">
                        <GitBranch className="w-4 h-4 text-secondary" />
                        <span>2. Safety Routing Decision</span>
                      </div>
                      <span className="text-[11px] sm:text-xs font-mono px-2.5 py-0.5 rounded bg-surface-container text-on-surface font-semibold">
                        Route: {demoResult.safety?.safety_route}
                      </span>
                    </div>

                    <div className="grid grid-cols-3 gap-3 text-xs pt-1 text-center">
                      <div className="p-2.5 bg-surface-container-lowest rounded-lg">
                        <div className="text-[10px] text-on-surface-variant uppercase">Emergency Bypass</div>
                        <div className={`font-mono font-bold mt-0.5 ${demoResult.safety?.emergency_bypass_triggered ? 'text-error' : 'text-primary'}`}>
                          {demoResult.safety?.emergency_bypass_triggered ? 'TRIGGERED' : 'NO'}
                        </div>
                      </div>
                      <div className="p-2.5 bg-surface-container-lowest rounded-lg">
                        <div className="text-[10px] text-on-surface-variant uppercase">RAG Bypassed</div>
                        <div className={`font-mono font-bold mt-0.5 ${demoResult.safety?.rag_retrieval_bypassed ? 'text-error' : 'text-primary'}`}>
                          {demoResult.safety?.rag_retrieval_bypassed ? 'YES (0 Vectors Read)' : 'NO (Active Search)'}
                        </div>
                      </div>
                      <div className="p-2.5 bg-surface-container-lowest rounded-lg">
                        <div className="text-[10px] text-on-surface-variant uppercase">LLM Generation Skipped</div>
                        <div className={`font-mono font-bold mt-0.5 ${demoResult.safety?.llm_generation_skipped ? 'text-error' : 'text-primary'}`}>
                          {demoResult.safety?.llm_generation_skipped ? 'YES (0 Tokens Used)' : 'NO (Grounded Synthesis)'}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* 3. RAG Retrieval Telemetry */}
                  <div className="bg-surface-container-low rounded-xl p-5 border border-surface-container space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-xs sm:text-sm font-bold text-on-surface">
                        <Search className="w-4 h-4 text-primary" />
                        <span>3. FAISS Dense Vector Retrieval Telemetry</span>
                      </div>
                      <span className="text-[11px] sm:text-xs font-mono text-on-surface-variant">
                        Latency: {demoResult.rag?.latency_ms || 0} ms • Model: {demoResult.rag?.embedding_model}
                      </span>
                    </div>

                    {demoResult.rag?.chunks && demoResult.rag.chunks.length > 0 ? (
                      <div className="space-y-1.5 pt-1">
                        <div className="text-[11px] font-semibold text-on-surface-variant">
                          Retrieved Knowledge Chunks ({demoResult.rag.chunks_count}):
                        </div>
                        {demoResult.rag.chunks.map((chunk, idx) => (
                          <div key={idx} className="p-2.5 bg-surface-container-lowest rounded-lg border border-surface-container text-xs flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                            <div className="space-y-0.5">
                              <div className="font-semibold text-on-surface">{chunk.title}</div>
                              <div className="text-[10px] text-on-surface-variant font-mono">
                                ID: {chunk.document_id} • Category: {chunk.category}
                              </div>
                            </div>
                            <div className="flex items-center gap-2 shrink-0">
                              <span className="text-[11px] font-mono font-bold px-2 py-0.5 rounded bg-primary/10 text-primary" title="FAISS Cosine Similarity (IndexFlatIP)">
                                Cosine Score: {chunk.score.toFixed(4)}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="p-3 bg-surface-container-lowest rounded-lg text-xs text-on-surface-variant text-center">
                        {demoResult.safety?.emergency_bypass_triggered
                          ? 'Vector search bypassed due to active emergency safe harbor protocol.'
                          : 'No knowledge chunks met the retrieval similarity threshold.'}
                      </div>
                    )}
                  </div>

                  {/* 4. LLM Multi-Provider Cascading Telemetry */}
                  <div className="bg-surface-container-low rounded-xl p-5 border border-surface-container space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-xs sm:text-sm font-bold text-on-surface">
                        <Cpu className="w-4 h-4 text-secondary" />
                        <span>4. LLM Multi-Provider Cascade Telemetry</span>
                      </div>
                      <span className="text-[11px] sm:text-xs font-mono text-on-surface-variant">
                        Gen Latency: {demoResult.llm?.generation_latency_ms || 0} ms
                      </span>
                    </div>

                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs pt-1">
                      <div className="p-2.5 bg-surface-container-lowest rounded-lg">
                        <div className="text-[10px] text-on-surface-variant uppercase">Selected Provider</div>
                        <div className="font-mono font-bold text-primary uppercase mt-0.5">
                          {demoResult.llm?.provider || 'local'}
                        </div>
                      </div>
                      <div className="p-2.5 bg-surface-container-lowest rounded-lg">
                        <div className="text-[10px] text-on-surface-variant uppercase">Target Model</div>
                        <div className="font-mono font-bold text-on-surface truncate mt-0.5" title={demoResult.llm?.model_name}>
                          {demoResult.llm?.model_name || 'deterministic_grounded_fallback'}
                        </div>
                      </div>
                      <div className="p-2.5 bg-surface-container-lowest rounded-lg">
                        <div className="text-[10px] text-on-surface-variant uppercase">Fallback Triggered</div>
                        <div className={`font-mono font-bold mt-0.5 ${demoResult.llm?.fallback_triggered ? 'text-secondary' : 'text-on-surface'}`}>
                          {demoResult.llm?.fallback_triggered ? 'YES (Groq/Local)' : 'NO (Direct Primary)'}
                        </div>
                      </div>
                      <div className="p-2.5 bg-surface-container-lowest rounded-lg">
                        <div className="text-[10px] text-on-surface-variant uppercase">Execution Mode</div>
                        <div className="font-mono font-bold text-on-surface mt-0.5">
                          {demoResult.llm?.local_fallback_used ? 'Local Deterministic' : 'Remote Cloud API'}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* 5. Verification Gate & Grounding */}
                  <div className="bg-surface-container-low rounded-xl p-5 border border-surface-container space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-xs sm:text-sm font-bold text-on-surface">
                        <ShieldCheck className="w-4 h-4 text-primary" />
                        <span>5. Verification Gate & Grounding Check</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-[11px] sm:text-xs font-mono px-2.5 py-0.5 rounded bg-primary/10 text-primary font-semibold">
                          Verification: {demoResult.verification_status}
                        </span>
                        <span className="text-[11px] sm:text-xs font-mono px-2.5 py-0.5 rounded bg-secondary/10 text-secondary font-semibold">
                          Grounding: {demoResult.grounding_status || 'not_applicable'}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* 6. Assistant Generated Output & Provenance */}
                  <div className="p-5 bg-surface-container-low rounded-xl border border-surface-container space-y-3.5">
                    <div className="text-xs sm:text-sm font-semibold text-on-surface">6. Generated Pipeline Response:</div>
                    <div className="p-4 sm:p-5 bg-surface-container-lowest rounded-xl border border-surface-container text-xs sm:text-sm text-on-surface whitespace-pre-wrap leading-relaxed font-sans">
                      {demoResult.final_response}
                    </div>

                    {/* Citations if present */}
                    {Array.isArray(demoResult.citations) && demoResult.citations.length > 0 && (
                      <div className="pt-2">
                        <div className="text-xs font-semibold text-on-surface-variant mb-1">
                          Retrieved Evidence Citations ({demoResult.citations.length}):
                        </div>
                        <div className="space-y-1">
                          {demoResult.citations.map((cite, idx) => (
                            <div key={idx} className="p-2 bg-surface-container-lowest rounded border border-surface-container text-xs flex justify-between items-center">
                              <span className="font-medium text-on-surface">{cite.title}</span>
                              <span className="font-mono text-[10px] text-on-surface-variant">{cite.document_id}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Ephemeral Session Notice */}
                  <div className="text-[11px] text-on-surface-variant flex items-center gap-1.5">
                    <ShieldCheck className="w-3.5 h-3.5 text-primary" />
                    <span>Privacy Notice: This diagnostic inference session is ephemeral in memory. No user utterances, personal identifiers, or API credentials are logged or retained on disk.</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
