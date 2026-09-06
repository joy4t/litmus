import { useState } from 'react';
import { Loader2, AlertCircle, FlaskConical, RotateCcw } from 'lucide-react';

const SYSTEM_PROMPT = `You are an experiment readout critic. You evaluate a single A/B test and decide whether its result can be trusted and acted on. You are rigorous and skeptical; your job is to stop the user shipping on bad data.

You receive: (1) the product and the change tested, (2) the metric, (3) per-variant numbers, (4) intended traffic split, (5) duration in days, (6) optional guardrail metric, (7) whether the user peeked, (8) an optional prior from the user's own comparable past test, and (9) a PRECOMPUTED_STATS block (significance p-value, 95% CI, absolute/relative lift, SRM chi-square + p-value, and minimum detectable effect).

HARD RULES:
- Do NOT compute or recompute any statistics. Use the precomputed values exactly as given. Never produce your own p-values, z-scores, or any number not in the input.
- NEVER invent benchmarks, industry averages, or "typical" figures. The only numeric prior you may use is one the user explicitly provides. If none is given, skip the plausibility check and do not speculate about what's normal.
- Read the domain from the product description and tailor your qualitative judgment to it (weekly cycles for consumer apps, business-day patterns for B2B, payday effects for Indian e-commerce), but never invent domain numbers.

CHECKS (use inputs + precomputed stats):
1. Sample Ratio Mismatch — if SRM p < 0.001, assignment/logging is broken and groups aren't comparable. FATAL.
2. Statistical Power — if the result is not significant AND the observed effect is below the precomputed MDE, it's INCONCLUSIVE, not negative. Severe underpowering is fatal to any "no effect" claim.
3. Duration & Seasonality — does runtime cover a full business cycle for this domain (generally >=1 week, ideally 2)? Flag novelty/primacy risk for short runs on repeat-use products.
4. Metric Fit — is the metric right for the change, or a proxy that could move for the wrong reason?
5. Guardrail — if provided, factor it in; if absent, flag that shipping on the win metric alone is risky.
6. Peeking — if the user peeked and stopped at significance, the false-positive rate is inflated; downgrade confidence.
7. Effect-Size Plausibility — ONLY if a prior is provided. If the observed effect is implausibly large vs their own history, treat it as likely instrumentation error, not a real win (Twyman's law).

VERDICT GATING:
- If any fatal threat fires (SRM failure, or a claim resting on a broken/underpowered test), verdict is "INVALID": refuse to read the result, state what's broken and what to fix and rerun. Do NOT say ship/don't-ship.
- Otherwise: significant + meaningful effect + guardrails ok -> "SHIP"; adequately powered but not significant -> "DO_NOT_SHIP"; significant but trivial effect, or confidence undercut by peeking -> "INCONCLUSIVE".
- Always translate into plain English a non-technical stakeholder can act on.

Output ONLY valid JSON, nothing else, in exactly this shape:
{
 "verdict": "SHIP" | "DO_NOT_SHIP" | "INCONCLUSIVE" | "INVALID",
 "headline": "one plain-English sentence: the call",
 "checks": [
   {"name":"Sample Ratio Mismatch","status":"pass|warn|fail","note":"one sentence"},
   {"name":"Statistical Power","status":"pass|warn|fail","note":"..."},
   {"name":"Duration & Seasonality","status":"pass|warn|fail","note":"..."},
   {"name":"Metric Fit","status":"pass|warn|fail","note":"..."},
   {"name":"Guardrail","status":"pass|warn|fail","note":"..."},
   {"name":"Peeking","status":"pass|warn|fail","note":"..."},
   {"name":"Effect-Size Plausibility","status":"pass|warn|fail|skipped","note":"..."}
 ],
 "reasoning": "2-4 sentences tying the checks to the verdict",
 "recommended_next_step": "one concrete action"
}`;

interface FormData {
  productDescription: string;
  metricName: string;
  controlUsers: string;
  controlConversions: string;
  treatmentUsers: string;
  treatmentConversions: string;
  intendedSplit: '50/50' | '80/20' | '90/10' | 'Custom';
  customSplitA: string;
  customSplitB: string;
  durationDays: string;
  guardrailMetric: string;
  didPeek: 'yes' | 'no';
  priorText: string;
}

interface PrecomputedStats {
  pValue: number;
  significant: boolean;
  absoluteLift: number;
  relativeLift: number;
  ci95: [number, number];
  srmChi2: number;
  srmP: number;
  srmFail: boolean;
  mde: number;
  observedBelowMde: boolean;
}

interface CheckResult {
  name: string;
  status: 'pass' | 'warn' | 'fail' | 'skipped';
  note: string;
}

interface AnalysisResult {
  verdict: 'SHIP' | 'DO_NOT_SHIP' | 'INCONCLUSIVE' | 'INVALID';
  headline: string;
  checks: CheckResult[];
  reasoning: string;
  recommended_next_step: string;
}

function erf(x: number): number {
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;

  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);

  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

  return sign * y;
}

function normCdf(x: number): number {
  return 0.5 * (1 + erf(x / Math.sqrt(2)));
}

function computeStats(formData: FormData): PrecomputedStats | null {
  const n1 = parseInt(formData.controlUsers);
  const x1 = parseInt(formData.controlConversions);
  const n2 = parseInt(formData.treatmentUsers);
  const x2 = parseInt(formData.treatmentConversions);

  if (isNaN(n1) || isNaN(x1) || isNaN(n2) || isNaN(x2)) return null;
  if (n1 <= 0 || n2 <= 0) return null;
  if (x1 < 0 || x2 < 0) return null;
  if (x1 > n1 || x2 > n2) return null;

  const p1 = x1 / n1;
  const p2 = x2 / n2;
  const pooled = (x1 + x2) / (n1 + n2);

  const seUnpooled = Math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2);
  const sePooled = Math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2));

  const z = (p2 - p1) / sePooled;
  const pValue = 2 * (1 - normCdf(Math.abs(z)));
  const significant = pValue < 0.05;

  const absoluteLift = p2 - p1;
  const relativeLift = p1 !== 0 ? (p2 - p1) / p1 : 0;
  const ci95: [number, number] = [
    (p2 - p1) - 1.96 * seUnpooled,
    (p2 - p1) + 1.96 * seUnpooled
  ];

  let split1: number, split2: number;
  if (formData.intendedSplit === 'Custom') {
    const a = parseFloat(formData.customSplitA) || 50;
    const b = parseFloat(formData.customSplitB) || 50;
    split1 = a / (a + b);
    split2 = b / (a + b);
  } else {
    const splitMap: Record<string, [number, number]> = {
      '50/50': [0.5, 0.5],
      '80/20': [0.8, 0.2],
      '90/10': [0.9, 0.1]
    };
    [split1, split2] = splitMap[formData.intendedSplit];
  }

  const total = n1 + n2;
  const e1 = total * split1;
  const e2 = total * split2;
  const chi2 = Math.pow(n1 - e1, 2) / e1 + Math.pow(n2 - e2, 2) / e2;
  const srmP = 2 * (1 - normCdf(Math.sqrt(chi2)));
  const srmFail = srmP < 0.001;

  const mde = (1.96 + 0.8416) * Math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2));
  const observedBelowMde = Math.abs(absoluteLift) < mde;

  return {
    pValue,
    significant,
    absoluteLift,
    relativeLift,
    ci95,
    srmChi2: chi2,
    srmP,
    srmFail,
    mde,
    observedBelowMde
  };
}

function formatPercent(value: number): string {
  return (value * 100).toFixed(2) + '%';
}

function formatPValue(value: number): string {
  if (value < 0.0001) return '<0.0001';
  return value.toFixed(4);
}

const verdictStyles = {
  SHIP: 'bg-emerald-600 border-emerald-500',
  DO_NOT_SHIP: 'bg-amber-600 border-amber-500',
  INCONCLUSIVE: 'bg-stone-700 border-stone-600',
  INVALID: 'bg-red-600 border-red-500'
};

const statusDotStyles = {
  pass: 'bg-emerald-400',
  warn: 'bg-amber-500',
  fail: 'bg-red-500 ring-2 ring-red-500/40',
  skipped: 'bg-stone-500'
};

const statusCardStyles = {
  pass: 'bg-stone-800/60 border-stone-700',
  warn: 'bg-amber-950/40 border-amber-800/60',
  fail: 'bg-red-950/50 border-red-800/60 ring-1 ring-red-500/40',
  skipped: 'bg-stone-800/30 border-stone-700/50'
};

export default function App() {
  const [formData, setFormData] = useState<FormData>({
    productDescription: '',
    metricName: '',
    controlUsers: '',
    controlConversions: '',
    treatmentUsers: '',
    treatmentConversions: '',
    intendedSplit: '50/50',
    customSplitA: '50',
    customSplitB: '50',
    durationDays: '',
    guardrailMetric: '',
    didPeek: 'no',
    priorText: ''
  });

  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [stats, setStats] = useState<PrecomputedStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const updateField = (field: keyof FormData, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleReset = () => {
    setFormData({
      productDescription: '',
      metricName: '',
      controlUsers: '',
      controlConversions: '',
      treatmentUsers: '',
      treatmentConversions: '',
      intendedSplit: '50/50',
      customSplitA: '50',
      customSplitB: '50',
      durationDays: '',
      guardrailMetric: '',
      didPeek: 'no',
      priorText: ''
    });
    setResult(null);
    setStats(null);
    setError(null);
  };

  const validateForm = (): string | null => {
    const n1 = parseInt(formData.controlUsers);
    const x1 = parseInt(formData.controlConversions);
    const n2 = parseInt(formData.treatmentUsers);
    const x2 = parseInt(formData.treatmentConversions);
    const duration = parseInt(formData.durationDays);

    if (!formData.productDescription.trim()) return 'Product description is required';
    if (!formData.metricName.trim()) return 'Metric name is required';
    if (isNaN(n1) || n1 <= 0) return 'Control users must be a positive number';
    if (isNaN(x1) || x1 < 0) return 'Control conversions must be 0 or greater';
    if (x1 > n1) return 'Control conversions cannot exceed control users';
    if (isNaN(n2) || n2 <= 0) return 'Treatment users must be a positive number';
    if (isNaN(x2) || x2 < 0) return 'Treatment conversions must be 0 or greater';
    if (x2 > n2) return 'Treatment conversions cannot exceed treatment users';
    if (isNaN(duration) || duration <= 0) return 'Duration must be a positive number';

    if (formData.intendedSplit === 'Custom') {
      const a = parseFloat(formData.customSplitA);
      const b = parseFloat(formData.customSplitB);
      if (isNaN(a) || isNaN(b) || a <= 0 || b <= 0) {
        return 'Custom split percentages must be positive numbers';
      }
    }

    return null;
  };

  const handleAnalyze = async () => {
    const validationError = validateForm();
    if (validationError) {
      setError(validationError);
      return;
    }

    const precomputedStats = computeStats(formData);
    if (!precomputedStats) {
      setError('Failed to compute statistics');
      return;
    }

    setStats(precomputedStats);
    setLoading(true);
    setError(null);
    setResult(null);

    const apiKey = import.meta.env.VITE_GROQ_API_KEY;
    if (!apiKey) {
      setError('VITE_GROQ_API_KEY environment variable is not set');
      setLoading(false);
      return;
    }

    const userMessage = {
      productDescription: formData.productDescription,
      metricName: formData.metricName,
      controlUsers: parseInt(formData.controlUsers),
      controlConversions: parseInt(formData.controlConversions),
      treatmentUsers: parseInt(formData.treatmentUsers),
      treatmentConversions: parseInt(formData.treatmentConversions),
      intendedSplit: formData.intendedSplit,
      customSplitA: formData.intendedSplit === 'Custom' ? formData.customSplitA : null,
      customSplitB: formData.intendedSplit === 'Custom' ? formData.customSplitB : null,
      durationDays: parseInt(formData.durationDays),
      guardrailMetric: formData.guardrailMetric || null,
      didPeek: formData.didPeek === 'yes',
      priorText: formData.priorText || null,
      PRECOMPUTED_STATS: precomputedStats
    };

    try {
      const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: 'openai/gpt-oss-120b',
          messages: [
            { role: 'system', content: SYSTEM_PROMPT },
            { role: 'user', content: JSON.stringify(userMessage) }
          ],
          response_format: { type: 'json_object' },
          temperature: 0.2
        })
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`API error: ${response.status} - ${text}`);
      }

      const data = await response.json();
      const content = data.choices[0]?.message?.content;
      if (!content) {
        throw new Error('No response content from API');
      }

      const parsed: AnalysisResult = JSON.parse(content);
      setResult(parsed);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };

  const inputBase = "w-full px-3 py-2 bg-stone-900/80 border border-stone-700 rounded-lg text-sm text-stone-200 placeholder-stone-500 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500";
  const labelBase = "block text-xs font-medium text-stone-400 uppercase tracking-wide mb-1.5";
  const sectionHeader = "text-sm font-semibold uppercase tracking-widest text-stone-400 mb-5";

  return (
    <div className="min-h-screen bg-stone-950 text-stone-100 relative overflow-x-hidden">
      {/* Dot grid background - more visible */}
      <div
        className="fixed inset-0 opacity-[0.08] pointer-events-none"
        style={{
          backgroundImage: 'radial-gradient(circle, #78716c 1px, transparent 1px)',
          backgroundSize: '20px 20px'
        }}
      />

      {/* Emerald radial glow behind hero */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-emerald-500/15 blur-[120px] rounded-full pointer-events-none" />

      {/* Centered header */}
      <header className="relative pt-12 pb-8 text-center">
        <div className="flex items-center justify-center gap-3 mb-3">
          <FlaskConical className="w-7 h-7 text-emerald-400" />
          <span className="text-3xl font-bold tracking-tight text-stone-100">LITMUS</span>
        </div>
        <p className="text-base text-stone-500">Significance isn&apos;t proof</p>
      </header>

      <main className="relative max-w-6xl mx-auto px-6 pb-10 space-y-6">
        {/* INPUT SECTION - Full width horizontal */}
        <section className="bg-stone-900/80 border border-stone-800 rounded-xl p-6 shadow-lg shadow-black/20">
          <h2 className={sectionHeader}>Experiment Details</h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Row 1: Product description - spans full width */}
            <div className="lg:col-span-4">
              <label className={labelBase}>Product & what changed</label>
              <textarea
                value={formData.productDescription}
                onChange={(e) => updateField('productDescription', e.target.value)}
                placeholder="e.g., E-commerce checkout — changed button color from blue to green"
                rows={2}
                className={inputBase}
              />
            </div>

            {/* Row 2: Metric, Duration, and Control inputs */}
            <div>
              <label className={labelBase}>Metric name</label>
              <input
                type="text"
                list="metric-suggestions"
                value={formData.metricName}
                onChange={(e) => updateField('metricName', e.target.value)}
                placeholder="e.g., completion rate"
                className={inputBase}
              />
              <datalist id="metric-suggestions">
                <option value="Conversion rate" />
                <option value="Signup completion rate" />
                <option value="Click-through rate" />
                <option value="Add-to-cart rate" />
                <option value="Checkout completion rate" />
                <option value="Purchase rate" />
                <option value="Activation rate" />
                <option value="Retention rate" />
                <option value="Engagement rate" />
                <option value="Bounce rate" />
              </datalist>
            </div>

            <div>
              <label className={labelBase}>Duration (days)</label>
              <input
                type="number"
                value={formData.durationDays}
                onChange={(e) => updateField('durationDays', e.target.value)}
                placeholder="14"
                className={inputBase}
              />
            </div>

            <div>
              <label className={labelBase}>Control — Users</label>
              <input
                type="number"
                value={formData.controlUsers}
                onChange={(e) => updateField('controlUsers', e.target.value)}
                placeholder="10000"
                className={inputBase}
              />
            </div>

            <div>
              <label className={labelBase}>Control — Conversions</label>
              <input
                type="number"
                value={formData.controlConversions}
                onChange={(e) => updateField('controlConversions', e.target.value)}
                placeholder="850"
                className={inputBase}
              />
            </div>

            {/* Row 3: Treatment inputs, split, guardrail */}
            <div>
              <label className={labelBase}>Treatment — Users</label>
              <input
                type="number"
                value={formData.treatmentUsers}
                onChange={(e) => updateField('treatmentUsers', e.target.value)}
                placeholder="10000"
                className={inputBase}
              />
            </div>

            <div>
              <label className={labelBase}>Treatment — Conversions</label>
              <input
                type="number"
                value={formData.treatmentConversions}
                onChange={(e) => updateField('treatmentConversions', e.target.value)}
                placeholder="920"
                className={inputBase}
              />
            </div>

            <div>
              <label className={labelBase}>Intended split</label>
              <select
                value={formData.intendedSplit}
                onChange={(e) => updateField('intendedSplit', e.target.value as FormData['intendedSplit'])}
                className={inputBase}
              >
                <option value="50/50">50/50</option>
                <option value="80/20">80/20</option>
                <option value="90/10">90/10</option>
                <option value="Custom">Custom</option>
              </select>
              {formData.intendedSplit === 'Custom' && (
                <div className="grid grid-cols-2 gap-2 mt-2">
                  <input
                    type="number"
                    value={formData.customSplitA}
                    onChange={(e) => updateField('customSplitA', e.target.value)}
                    placeholder="A %"
                    className={`${inputBase} text-xs py-1.5`}
                  />
                  <input
                    type="number"
                    value={formData.customSplitB}
                    onChange={(e) => updateField('customSplitB', e.target.value)}
                    placeholder="B %"
                    className={`${inputBase} text-xs py-1.5`}
                  />
                </div>
              )}
            </div>

            <div>
              <label className={labelBase}>Guardrail metric <span className="text-stone-600">(optional)</span></label>
              <input
                type="text"
                value={formData.guardrailMetric}
                onChange={(e) => updateField('guardrailMetric', e.target.value)}
                placeholder="e.g., refund rate"
                className={inputBase}
              />
            </div>

            {/* Row 4: Peek and Prior */}
            <div>
              <label className={labelBase}>Peeked & stopped at significance?</label>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => updateField('didPeek', 'yes')}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                    formData.didPeek === 'yes'
                      ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-500/20'
                      : 'bg-stone-800 text-stone-400 hover:bg-stone-700'
                  }`}
                >
                  Yes
                </button>
                <button
                  type="button"
                  onClick={() => updateField('didPeek', 'no')}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                    formData.didPeek === 'no'
                      ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-500/20'
                      : 'bg-stone-800 text-stone-400 hover:bg-stone-700'
                  }`}
                >
                  No
                </button>
              </div>
            </div>

            <div>
              <label className={labelBase}>Prior from past similar test <span className="text-stone-600">(optional)</span></label>
              <input
                type="text"
                value={formData.priorText}
                onChange={(e) => updateField('priorText', e.target.value)}
                placeholder='e.g., "past button changes moved ~1-2%"'
                className={inputBase}
              />
            </div>
          </div>

          {/* Analyze + Reset buttons and error */}
          <div className="mt-5 flex flex-col md:flex-row md:items-center gap-3">
            <button
              onClick={handleAnalyze}
              disabled={loading}
              className="w-full md:w-auto bg-emerald-600 hover:bg-emerald-500 disabled:bg-stone-700 disabled:text-stone-500 text-white font-semibold py-2.5 px-8 rounded-lg transition-all duration-200 flex items-center justify-center gap-2 shadow-lg shadow-emerald-500/10 hover:shadow-emerald-500/20"
            >
              {loading && <Loader2 className="w-4 h-4 animate-spin" />}
              {loading ? 'Analyzing...' : 'Analyze'}
            </button>

            <button
              onClick={handleReset}
              disabled={loading}
              className="w-full md:w-auto border border-stone-600 hover:border-stone-500 disabled:opacity-50 disabled:cursor-not-allowed text-stone-300 hover:text-stone-100 font-semibold py-2.5 px-8 rounded-lg transition-all duration-200 flex items-center justify-center gap-2 bg-stone-800/50 hover:bg-stone-800"
            >
              <RotateCcw className="w-4 h-4" />
              Reset
            </button>

            {error && (
              <div className="mt-3 bg-red-950/40 border border-red-900/60 rounded-lg p-3 flex items-start gap-2.5">
                <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
                <p className="text-red-300 text-sm">{error}</p>
              </div>
            )}
          </div>
        </section>

        {/* ANALYSIS SECTION - Full width */}
        <section className="bg-stone-900/80 border border-stone-800 rounded-xl p-6 shadow-lg shadow-black/20">
          <h2 className={sectionHeader}>Analysis</h2>

          {loading && (
            <div className="flex flex-col items-center justify-center py-20">
              <Loader2 className="w-10 h-10 text-emerald-500 animate-spin mb-4" />
              <p className="text-stone-500">Running diagnostics...</p>
            </div>
          )}

          {!loading && !result && (
            <div className="flex flex-col items-center justify-center py-20 text-center">
              <div className="w-16 h-16 rounded-full bg-stone-800 flex items-center justify-center mb-4">
                <FlaskConical className="w-8 h-8 text-stone-600" />
              </div>
              <p className="text-stone-500">Significance isn&apos;t proof</p>
            </div>
          )}

          {result && (
            <div className="space-y-5">
              {/* Verdict Banner */}
              <div className={`rounded-lg px-5 py-4 border-2 ${verdictStyles[result.verdict]}`}>
                <div className="text-2xl font-bold tracking-tight text-white">
                  {result.verdict.replace(/_/g, ' ')}
                </div>
                <div className="text-white/80 text-sm mt-1">{result.headline}</div>
              </div>

              {/* Check Cards - 3 column grid, last one centered */}
              <div className="grid grid-cols-2 lg:grid-cols-3 gap-2">
                {result.checks.slice(0, 6).map((check, idx) => (
                  <div
                    key={idx}
                    className={`rounded-lg px-3 py-2.5 border ${statusCardStyles[check.status]}`}
                  >
                    <div className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-full flex-shrink-0 ${statusDotStyles[check.status]}`} />
                      <div className="font-medium text-stone-200 text-xs">{check.name}</div>
                    </div>
                    <div className="text-stone-400 text-[11px] mt-1 leading-relaxed">{check.note}</div>
                  </div>
                ))}
              </div>
              {/* Effect-Size Plausibility - centered on its own row */}
              {result.checks[6] && (
                <div className="flex justify-center">
                  <div className={`rounded-lg px-3 py-2.5 border w-full max-w-sm ${statusCardStyles[result.checks[6].status]}`}>
                    <div className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-full flex-shrink-0 ${statusDotStyles[result.checks[6].status]}`} />
                      <div className="font-medium text-stone-200 text-xs">{result.checks[6].name}</div>
                    </div>
                    <div className="text-stone-400 text-[11px] mt-1 leading-relaxed">{result.checks[6].note}</div>
                  </div>
                </div>
              )}

              {/* Bottom row: Reasoning, Next Step, Stats - centered */}
              <div className="flex justify-center">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 w-full max-w-4xl">
                  {/* Reasoning */}
                  <div className="bg-stone-800/50 border border-stone-700/50 rounded-lg py-3">
                    <div className="text-[11px] uppercase tracking-wide text-stone-500 mb-1.5 px-3">Reasoning</div>
                    <p className="text-stone-300 text-xs leading-relaxed px-3">{result.reasoning}</p>
                  </div>

                  {/* Next Step */}
                  <div className="bg-emerald-950/30 border border-emerald-900/40 rounded-lg px-3 py-3">
                    <div className="text-[11px] uppercase tracking-wide text-emerald-400 mb-1.5">Next Step</div>
                    <p className="text-emerald-100 text-xs font-medium">{result.recommended_next_step}</p>
                  </div>

                  {/* Statistics - always open, no toggle */}
                  {stats && (
                    <div className="border border-stone-700 rounded-lg bg-stone-800/40">
                      <div className="py-2 border-b border-stone-700">
                        <span className="text-[11px] font-medium text-stone-400 uppercase tracking-wide px-3">Statistics</span>
                      </div>
                      <div className="py-3 text-[11px] space-y-1.5 px-3">
                        <div className="flex justify-between">
                          <span className="text-stone-500">p-value</span>
                          <span className="font-mono text-stone-300">{formatPValue(stats.pValue)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-stone-500">95% CI</span>
                          <span className="font-mono text-stone-300">[{formatPercent(stats.ci95[0])}, {formatPercent(stats.ci95[1])}]</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-stone-500">Lift</span>
                          <span className="font-mono text-stone-300">{formatPercent(stats.relativeLift)} rel</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-stone-500">SRM p</span>
                          <span className="font-mono text-stone-300">{formatPValue(stats.srmP)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-stone-500">MDE</span>
                          <span className="font-mono text-stone-300">{formatPercent(stats.mde)}</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
