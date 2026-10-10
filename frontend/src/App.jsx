import { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';
import StepInspector from './components/StepInspector';
import EvidenceMatrixView from './components/EvidenceMatrixView';
import ResearchGapLab from './components/ResearchGapLab';
import AblationDashboard from './components/AblationDashboard';
import CorpusExplorer from './components/CorpusExplorer';
import ReviewFeedbackPanel from './components/ReviewFeedbackPanel';
import Icon from './components/Icon';
import { exportToWord, exportToPdf, exportToMarkdown, copyToClipboard } from './exportUtils';

const API_BASE = 'http://127.0.0.1:3000/api';

// The agents of the LangGraph pipeline, in execution order (node names reported by the AI engine).
const AGENTS = [
  { key: 'search_agent', name: 'Scholarly Search', description: 'Ranks papers from MongoDB; fetches more from OpenAlex if too few match.' },
  { key: 'paper_extraction_agent', name: 'PDF Extraction', description: 'Downloads open-access PDFs and extracts full text.' },
  { key: 'paper_analysis_agent', name: 'Paper Analysis', description: 'Extracts methods, data, results, metrics and limitations.' },
  { key: 'cross_paper_analysis_agent', name: 'Cross-Paper Synthesis', description: 'Compares papers: agreements, contradictions, trade-offs.' },
  { key: 'research_gap_agent', name: 'Research-Gap Triad', description: 'Proposer, Skeptic and Verifier agree on grounded gaps.' },
  { key: 'summarization_agent', name: 'Summarisation', description: 'Condenses each paper into reusable evidence summaries.' },
  { key: 'taxonomy_agent', name: 'Taxonomy', description: 'Groups the papers into research themes.' },
  { key: 'citation_agent', name: 'Citation Structuring', description: 'Builds the numbered reference list from metadata.' },
  { key: 'writing_agent', name: 'Grounded Writer', description: 'Writes the citation-grounded literature review.' },
  { key: 'feedback_agent', name: 'Verification Feedback', description: 'Critic, fact-checker and reviser repair unsupported claims.' },
];

// the AI engine retrieves at most 20 papers per review (GenerationRequest.max_papers in app.py)
const MAX_PAPERS = 20;

const TABS = [
  { key: 'MANUSCRIPT', label: 'Manuscript', icon: 'document' },
  { key: 'DASHBOARD', label: 'Agent Pipeline', icon: 'pipeline' },
  { key: 'EVIDENCE', label: 'Evidence Matrix', icon: 'table' },
  { key: 'GAPS', label: 'Research Gaps', icon: 'scale' },
  { key: 'ABLATION', label: 'Evaluation', icon: 'chart' },
  { key: 'CORPUS', label: 'Paper Corpus', icon: 'library' },
];

// Converts the many status words used by the backend (completed, done, success, queued, running, error ...)
// into the five states the page understands: IDLE, PROCESSING, COMPLETED, FAILED, RATE_LIMITED.
//The server may say “done”, “completed” or “success”. This turns all of them into one word, COMPLETED. So the page only has to know 5 words: IDLE, PROCESSING, COMPLETED, FAILED, RATE_LIMITED.
function normalizeStatus(value) {
  if (!value) return 'IDLE';
  const status = String(value).toUpperCase();
  if (['COMPLETED', 'COMPLETE', 'DONE', 'SUCCESS'].includes(status)) return 'COMPLETED';
  if (['RATE_LIMITED', 'QUOTA_EXCEEDED'].includes(status)) return 'RATE_LIMITED';
  if (['FAILED', 'ERROR'].includes(status)) return 'FAILED';
  if (['QUEUED', 'PROCESSING', 'RUNNING', 'STARTED', 'PENDING'].includes(status)) return 'PROCESSING';
  return status;
}

// Reads a saved run object and picks out what the page needs: the review text, the research gaps, the papers,
// the progress list and the feedback history. The same data can sit in different places (run.x or run.data.x),
// so it tries each place in turn and falls back to an empty value.
//The server sends a big package. This picks out only what we need: review, papers, gaps, progress.
function getRunData(run) {
  if (!run) return null;
  return {
    final_review_md:
      run.final_review_md || run.final_review || run.data?.final_review_md || run.data?.final_review ||
      run.result?.final_review_md || run.result?.final_review || '',
    research_gap:
      run.research_gap || run.data?.research_gap || run.data?.research_gap_analysis || run.result?.research_gap || null,
    raw_papers_list:
      run.raw_papers_list || run.papers || run.data?.raw_papers_list || run.data?.papers ||
      run.result?.raw_papers_list || run.result?.papers || [],
    progress: run.progress || run.data?.progress || [],
    feedback_history: run.feedback_history || [],
  };
}

// attach the Paper Analysis Agent's structured evidence (data.analyzed_papers) to each retrieved paper
// Attaches each paper's structured evidence (from run.data.analyzed_papers) to the matching retrieved paper,
// found by paper_id or title, so the Evidence Matrix tab can show methodology, results and limitations.
//Puts each paper’s details (method, results) next to its paper, so the table can show them.  
function withAnalysis(run, data) {
  const analyzed = run?.data?.analyzed_papers || [];
  if (!data || !analyzed.length) return data;
  const byId = new Map(analyzed.map((a) => [a.paper_id || a.title, a.paper_analysis]));
  data.raw_papers_list = data.raw_papers_list.map((p) => ({
    ...p,
    paper_analysis: p.paper_analysis || byId.get(p.paper_id || p.title),
  }));
  return data;
}

// Turns a number of seconds into a clock text like 02:21 for the 'Running 02:21' badge.
function formatClock(seconds) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

// The whole web page (the main React component).
// It keeps the form values and the run state in React state variables, and does these jobs:
//  - loads the corpus size and the evaluation range for the tab badges when the page opens,
//  - reopens a saved run from the address ?run=<id>,
//  - keeps the address bar updated so a run can be bookmarked,
//  - counts the elapsed time while a run is processing,
//  - polls the backend every 3 seconds for progress and the finished review,
//  - shows the form, the 10-agent pipeline with green checks, and the six tabs
//    (Manuscript, Agent Pipeline, Evidence Matrix, Research Gaps, Evaluation, Paper Corpus).
export default function App() {
  const [topic, setTopic] = useState('');
  const [startYear, setStartYear] = useState('2022');
  const [endYear, setEndYear] = useState('2025');
  const [maxPapers, setMaxPapers] = useState('10');

  const [runId, setRunId] = useState(null);
  const [initialRun] = useState(() => new URLSearchParams(window.location.search).get('run'));
  const [status, setStatus] = useState('IDLE');
  const [runData, setRunData] = useState(null);
  const [viewMode, setViewMode] = useState(() => {
    const tab = new URLSearchParams(window.location.search).get('tab');
    return TABS.some((t) => t.key === tab) ? tab : 'MANUSCRIPT';
  });

  const [error, setError] = useState('');
  const [elapsedTime, setElapsedTime] = useState(0);

  const [theme, setTheme] = useState(() => localStorage.getItem('malrg_theme') || 'light');
  const [isInspectorOpen, setIsInspectorOpen] = useState(() => new URLSearchParams(window.location.search).get('inspector') === 'open');
  const [copyToast, setCopyToast] = useState(false);
  const [showSidebar, setShowSidebar] = useState(() => new URLSearchParams(window.location.search).get('panel') !== 'hidden');

  // live badge values (corpus size from MongoDB, configurations of the computed evaluation)
  const [corpusTotal, setCorpusTotal] = useState(null);
  const [ablationRange, setAblationRange] = useState(null);

  const light = theme === 'light';
  // Small helper: chooses the CSS classes for light mode or dark mode.
  const c = (lightCls, darkCls) => (light ? lightCls : darkCls);

  useEffect(() => {
    fetch(`${API_BASE}/corpus/stats`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => d && setCorpusTotal(d.total))
      .catch(() => {});
    fetch(`${API_BASE}/evaluation/benchmark`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        const ids = (d?.configs || []).map((cfg) => cfg.id);
        if (ids.length) setAblationRange(`${ids[0]}–${ids[ids.length - 1]}`);
      })
      .catch(() => {});
  }, []);

  // reopen a saved run from a shareable link: ?run=<id>&tab=<view>
  useEffect(() => {
    const savedRun = initialRun;
    if (!savedRun) return;
    fetch(`${API_BASE}/review/${savedRun}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`Run not found (HTTP ${r.status})`))))
      .then((run) => {
        setRunId(savedRun);
        if (run.topic) setTopic(run.topic);
        if (run.start_year) setStartYear(run.start_year);
        if (run.end_year) setEndYear(run.end_year);
        if (run.max_papers) setMaxPapers(run.max_papers);
        setRunData(withAnalysis(run, getRunData(run)));
        if (run.createdAt) setElapsedTime(Math.max(0, Math.floor((Date.now() - new Date(run.createdAt).getTime()) / 1000)));
        setStatus(normalizeStatus(run.status));
      })
      .catch((err) => setError(err.message));
  }, [initialRun]);

  // keep the address bar pointing at the current run so it can be bookmarked or shared
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (runId) params.set('run', runId);
    params.set('tab', viewMode);
    window.history.replaceState(null, '', `${window.location.pathname}?${params.toString()}`);
  }, [runId, viewMode]);

  // Switches between light and dark mode and remembers the choice in the browser (localStorage).
  const toggleTheme = () => {
    const next = light ? 'dark' : 'light';
    setTheme(next);
    localStorage.setItem('malrg_theme', next);
  };

  // elapsed-time clock while a run is in progress
  useEffect(() => {
    if (status !== 'PROCESSING') return;
    const timer = setInterval(() => setElapsedTime((v) => v + 1), 1000);
    return () => clearInterval(timer);
  }, [status]);

  // poll the run (Node API -> MongoDB, which mirrors the AI engine's real progress)
  useEffect(() => {
    if (!runId || status !== 'PROCESSING') return;
    let cancelled = false;

    // One polling step: GET /api/review/<runId> from the Node server (which reads MongoDB), save the data in state,
    // and set the status: COMPLETED (show the Manuscript), RATE_LIMITED or FAILED (show the error), else keep processing.
    // It is called at once and then every 3 seconds by setInterval.
    const poll = async () => {
      try {
        const response = await fetch(`${API_BASE}/review/${runId}`);
        if (!response.ok) throw new Error(`Backend returned HTTP ${response.status}`);
        const run = await response.json();
        if (cancelled) return;

        const normalized = normalizeStatus(run.status);
        setRunData(withAnalysis(run, getRunData(run)));

        if (normalized === 'COMPLETED') {
          setStatus('COMPLETED');
          setViewMode('MANUSCRIPT');
          return;
        }
        if (normalized === 'RATE_LIMITED') {
          setStatus('RATE_LIMITED');
          setError(run.error || run.reason || 'The language-model API quota was exceeded. Generation was halted.');
          return;
        }
        if (normalized === 'FAILED') {
          setStatus('FAILED');
          setError(run.error || run.message || 'The literature-review pipeline failed.');
          return;
        }
        setStatus('PROCESSING');
      } catch (err) {
        if (!cancelled) console.error('Run polling error:', err);
      }
    };

    poll();
    const interval = setInterval(poll, 3000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [runId, status]);

  // Runs when the user clicks 'Generate review'.
  //  1. Stops the normal form submit; trims the topic and converts the years and paper count to numbers.
  //  2. Checks: topic not empty, years valid (1900-2100, start <= end), papers between 1 and 20.
  //  3. Clears the old run and sets status PROCESSING (this shows the pipeline view).
  //  4. Sends POST {topic, start_year, end_year, max_papers} to the Node server /api/review/generate.
  //  5. Stores the returned run_id, which starts the polling. On any error it shows the message.
  const handleGenerate = async (event) => {
    event.preventDefault();
    const cleanTopic = topic.trim();
    if (!cleanTopic) {
      setError('Please enter a research topic.');
      return;
    }
    const start = Number(startYear);
    const end = Number(endYear);
    const count = Number(maxPapers);
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < 1900 || end > 2100 || start > end) {
      setError('Please enter a valid publication year range.');
      return;
    }
    if (!Number.isInteger(count) || count < 1 || count > MAX_PAPERS) {
      setError(`Number of papers must be between 1 and ${MAX_PAPERS}.`);
      return;
    }

    setError('');
    setRunId(null);
    setRunData(null);
    setElapsedTime(0);
    setViewMode('DASHBOARD');
    setStatus('PROCESSING');

    try {
      const response = await fetch(`${API_BASE}/review/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic: cleanTopic, start_year: start, end_year: end, max_papers: count }),
      });
      const data = await response.json();
      if (!response.ok || !data.success) {
        throw new Error(data.error || data.message || 'Unable to start the review-generation pipeline.');
      }
      if (!data.run_id) throw new Error('The API server did not return a run ID.');
      setRunId(data.run_id);
      if (data.status) setStatus(normalizeStatus(data.status));
    } catch (err) {
      console.error(err);
      setStatus('FAILED');
      setError(err.message);
    }
  };

  const papers = runData?.raw_papers_list || [];
  const researchGap = runData?.research_gap;
  const verifiedGaps = Array.isArray(researchGap?.verified_gaps) ? researchGap.verified_gaps : [];
  const finalReview = runData?.final_review_md || '';
  const progress = runData?.progress || [];
  const doneAgents = new Set(progress.map((p) => p.agent));

  // Decides how one agent card is drawn: 'done' (its name is in the progress list), 'active' (the next unfinished
  // agent while processing), 'failed' (the next unfinished agent after an error), 'pending' or 'idle'.
  const agentState = (agent) => {
    if (doneAgents.has(agent.key)) return 'done';
    if (status === 'COMPLETED') return 'done';
    if (status === 'PROCESSING') {
      const next = AGENTS.find((a) => !doneAgents.has(a.key));
      return next && next.key === agent.key ? 'active' : 'pending';
    }
    if (status === 'FAILED' || status === 'RATE_LIMITED') {
      const next = AGENTS.find((a) => !doneAgents.has(a.key));
      return next && next.key === agent.key ? 'failed' : 'pending';
    }
    return 'idle';
  };
  const completedCount = status === 'COMPLETED' ? AGENTS.length : AGENTS.filter((a) => doneAgents.has(a.key)).length;

  const statusPill = {
    IDLE: { label: 'Ready', cls: c('bg-slate-100 text-slate-600 ring-slate-200', 'bg-slate-800 text-slate-300 ring-slate-700'), icon: 'info' },
    PROCESSING: { label: `Running · ${formatClock(elapsedTime)}`, cls: c('bg-indigo-50 text-indigo-700 ring-indigo-200', 'bg-indigo-500/15 text-indigo-300 ring-indigo-500/30'), icon: 'spinner' },
    COMPLETED: { label: 'Completed', cls: c('bg-emerald-50 text-emerald-700 ring-emerald-200', 'bg-emerald-500/15 text-emerald-300 ring-emerald-500/30'), icon: 'checkCircle' },
    FAILED: { label: 'Failed', cls: c('bg-rose-50 text-rose-700 ring-rose-200', 'bg-rose-500/15 text-rose-300 ring-rose-500/30'), icon: 'alert' },
    RATE_LIMITED: { label: 'Quota reached', cls: c('bg-amber-50 text-amber-800 ring-amber-200', 'bg-amber-500/15 text-amber-300 ring-amber-500/30'), icon: 'alert' },
  }[status] || { label: status, cls: '', icon: 'info' };

  const card = c('bg-white border-slate-200 shadow-sm', 'bg-slate-900 border-slate-800');
  const muted = c('text-slate-500', 'text-slate-400');
  const strong = c('text-slate-900', 'text-slate-100');
  const inputCls = `w-full h-10 rounded-lg border px-3 text-sm outline-none transition focus:ring-2 disabled:opacity-60 ${c(
    'bg-white border-slate-300 text-slate-900 placeholder:text-slate-400 focus:border-indigo-500 focus:ring-indigo-100',
    'bg-slate-950 border-slate-700 text-slate-100 placeholder:text-slate-500 focus:border-indigo-400 focus:ring-indigo-500/20'
  )}`;
  const ghostBtn = `inline-flex items-center gap-1.5 h-8 px-3 rounded-lg border text-xs font-medium transition ${c(
    'bg-white border-slate-300 text-slate-700 hover:bg-slate-50',
    'bg-slate-900 border-slate-700 text-slate-200 hover:bg-slate-800'
  )}`;

  // Gives the small number shown next to a tab name: papers count on Evidence Matrix, finished-agents like 2/10 on
  // Agent Pipeline, M0-M5 on Evaluation, and the total papers on Paper Corpus.
  const tabBadge = (key) => {
    if (key === 'EVIDENCE' && papers.length) return papers.length;
    if (key === 'DASHBOARD' && status === 'PROCESSING') return `${completedCount}/${AGENTS.length}`;
    if (key === 'ABLATION') return ablationRange;
    if (key === 'CORPUS') return corpusTotal;
    return null;
  };

  return (
    <div className={`min-h-screen ${c('bg-slate-50 text-slate-900', 'bg-slate-950 text-slate-100')}`}>
      {/* Header */}
      <header className={`app-header sticky top-0 z-40 border-b ${c('bg-white/95 border-slate-200', 'bg-slate-950/95 border-slate-800')} backdrop-blur`}>
        <div className="max-w-[1400px] mx-auto px-6 h-16 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <div className="w-9 h-9 rounded-lg bg-indigo-600 text-white flex items-center justify-center shrink-0">
              <Icon name="logo" size={20} />
            </div>
            <div className="min-w-0">
              <div className="flex items-baseline gap-2">
                <h1 className="text-base font-semibold tracking-tight">MALRG</h1>
                <span className={`hidden sm:inline text-sm ${muted}`}>Multi-Agent Literature Review Generator</span>
              </div>
              <p className={`text-xs ${muted}`}>Evidence-grounded reviews with verified citations and research gaps</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className={`inline-flex items-center gap-1.5 h-8 px-3 rounded-full text-xs font-medium ring-1 ${statusPill.cls}`}>
              <Icon name={statusPill.icon} size={14} spin={status === 'PROCESSING'} />
              {statusPill.label}
            </span>
            <button type="button" onClick={() => setIsInspectorOpen(true)} className={ghostBtn} title="Open step inspector">
              <Icon name="inspect" size={14} /> Inspector
            </button>
            <button type="button" onClick={() => setShowSidebar((v) => !v)} className={ghostBtn}
              title={showSidebar ? 'Hide the parameter panel' : 'Show the parameter panel'}>
              <Icon name="sidebar" size={14} /> {showSidebar ? 'Hide panel' : 'Show panel'}
            </button>
            <button type="button" onClick={toggleTheme} className={`${ghostBtn} px-2`} title="Toggle light / dark mode"
              aria-label="Toggle light / dark mode">
              <Icon name={light ? 'moon' : 'sun'} size={15} />
            </button>
          </div>
        </div>

        {/* Tabs */}
        <nav className="app-nav max-w-[1400px] mx-auto px-6 flex items-center gap-1 overflow-x-auto">
          {TABS.map((tab) => {
            const active = viewMode === tab.key;
            const badge = tabBadge(tab.key);
            return (
              <button
                key={tab.key}
                type="button"
                onClick={() => setViewMode(tab.key)}
                className={`relative inline-flex items-center gap-2 px-3 h-11 text-sm font-medium whitespace-nowrap transition ${
                  active ? c('text-indigo-700', 'text-indigo-300') : c('text-slate-500 hover:text-slate-800', 'text-slate-400 hover:text-slate-200')
                }`}
              >
                <Icon name={tab.icon} size={16} />
                {tab.label}
                {badge !== null && badge !== undefined && (
                  <span className={`text-[11px] font-medium px-1.5 py-0.5 rounded-md ${
                    active ? c('bg-indigo-50 text-indigo-700', 'bg-indigo-500/20 text-indigo-200') : c('bg-slate-100 text-slate-600', 'bg-slate-800 text-slate-300')
                  }`}>{badge}</span>
                )}
                {tab.key === 'MANUSCRIPT' && finalReview && <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />}
                {active && <span className="absolute left-2 right-2 -bottom-px h-0.5 rounded-full bg-indigo-600" />}
              </button>
            );
          })}
        </nav>
      </header>

      <main className="app-main max-w-[1400px] mx-auto px-6 py-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left panel */}
        {showSidebar && (
          <aside className="lg:col-span-4 xl:col-span-3 space-y-5">
            <section className={`rounded-xl border p-5 ${card}`}>
              <h2 className={`text-sm font-semibold ${strong}`}>New literature review</h2>
              <p className={`text-xs mt-1 ${muted}`}>Describe the topic; the agents retrieve, analyse and synthesise the papers.</p>

              <form onSubmit={handleGenerate} className="mt-5 space-y-4">
                <div>
                  <label className={`block text-xs font-medium mb-1.5 ${c('text-slate-700', 'text-slate-300')}`}>Research topic</label>
                  <input type="text" value={topic} onChange={(e) => setTopic(e.target.value)} disabled={status === 'PROCESSING'}
                    placeholder="e.g. Deep learning for plant disease detection" className={inputCls} />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className={`block text-xs font-medium mb-1.5 ${c('text-slate-700', 'text-slate-300')}`}>From year</label>
                    <input type="number" min="1900" max="2100" value={startYear} onChange={(e) => setStartYear(e.target.value)}
                      disabled={status === 'PROCESSING'} className={`${inputCls} tabular-nums`} />
                  </div>
                  <div>
                    <label className={`block text-xs font-medium mb-1.5 ${c('text-slate-700', 'text-slate-300')}`}>To year</label>
                    <input type="number" min="1900" max="2100" value={endYear} onChange={(e) => setEndYear(e.target.value)}
                      disabled={status === 'PROCESSING'} className={`${inputCls} tabular-nums`} />
                  </div>
                </div>
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <label className={`text-xs font-medium ${c('text-slate-700', 'text-slate-300')}`}>Number of papers</label>
                    <span className={`text-[11px] ${muted}`}>1–{MAX_PAPERS}</span>
                  </div>
                  <input type="number" min="1" max={MAX_PAPERS} value={maxPapers} onChange={(e) => setMaxPapers(e.target.value)}
                    disabled={status === 'PROCESSING'} className={`${inputCls} tabular-nums`} />
                </div>
                <button type="submit" disabled={status === 'PROCESSING'}
                  className="w-full h-10 inline-flex items-center justify-center gap-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium shadow-sm transition disabled:opacity-60 disabled:cursor-not-allowed">
                  <Icon name={status === 'PROCESSING' ? 'spinner' : 'play'} size={15} spin={status === 'PROCESSING'} />
                  {status === 'PROCESSING' ? `Generating · ${formatClock(elapsedTime)}` : 'Generate review'}
                </button>
              </form>

              {runId && (
                <div className={`mt-4 pt-4 border-t text-xs ${c('border-slate-200', 'border-slate-800')}`}>
                  <span className={muted}>Run ID </span>
                  <span className={`font-mono ${c('text-slate-700', 'text-slate-300')} break-all`}>{runId}</span>
                </div>
              )}
            </section>

            {status !== 'IDLE' && (
              <section className={`rounded-xl border p-5 ${card}`}>
                <div className="flex items-center justify-between">
                  <h3 className={`text-sm font-semibold ${strong}`}>Pipeline progress</h3>
                  <span className={`text-xs tabular-nums ${muted}`}>{completedCount} / {AGENTS.length} agents</span>
                </div>
                <div className={`mt-3 h-1.5 rounded-full overflow-hidden ${c('bg-slate-100', 'bg-slate-800')}`}>
                  <div className="h-full bg-indigo-600 transition-all duration-500"
                    style={{ width: `${(completedCount / AGENTS.length) * 100}%` }} />
                </div>
                <ol className="mt-4 space-y-2.5">
                  {AGENTS.map((agent) => {
                    const st = agentState(agent);
                    return (
                      <li key={agent.key} className="flex items-center gap-3 text-sm">
                        <span className={`w-5 h-5 rounded-full flex items-center justify-center shrink-0 ${
                          st === 'done' ? 'bg-emerald-500 text-white'
                            : st === 'active' ? 'bg-indigo-600 text-white'
                              : st === 'failed' ? 'bg-rose-500 text-white'
                                : c('bg-slate-100 text-slate-400 ring-1 ring-slate-200', 'bg-slate-800 text-slate-500 ring-1 ring-slate-700')
                        }`}>
                          {st === 'done' && <Icon name="check" size={12} strokeWidth={2.6} />}
                          {st === 'active' && <Icon name="spinner" size={12} spin strokeWidth={2.6} />}
                          {st === 'failed' && <Icon name="x" size={12} strokeWidth={2.6} />}
                        </span>
                        <span className={st === 'pending' || st === 'idle' ? muted : strong}>{agent.name}</span>
                      </li>
                    );
                  })}
                </ol>
              </section>
            )}
          </aside>
        )}

        {/* Workspace */}
        <section className={showSidebar ? 'lg:col-span-8 xl:col-span-9 min-w-0' : 'lg:col-span-12 min-w-0'}>
          {status === 'RATE_LIMITED' && (
            <div className={`mb-4 flex gap-3 p-4 rounded-xl border text-sm ${c('bg-amber-50 border-amber-200 text-amber-900', 'bg-amber-500/10 border-amber-500/30 text-amber-200')}`}>
              <Icon name="alert" size={18} className="shrink-0 mt-0.5" />
              <div><strong className="font-semibold">API quota reached.</strong> The language-model provider's daily token quota is exhausted; the run will need to be restarted when the quota resets. {error}</div>
            </div>
          )}
          {error && status !== 'RATE_LIMITED' && (
            <div className={`mb-4 flex gap-3 p-4 rounded-xl border text-sm ${c('bg-rose-50 border-rose-200 text-rose-800', 'bg-rose-500/10 border-rose-500/30 text-rose-200')}`}>
              <Icon name="alert" size={18} className="shrink-0 mt-0.5" />
              <div><strong className="font-semibold">Something went wrong.</strong> {error}</div>
            </div>
          )}

          <div className={`app-workspace rounded-xl border min-h-[calc(100vh-11rem)] overflow-hidden ${card}`}>
            {/* Agent pipeline view (real progress reported by the AI engine) */}
            {viewMode === 'DASHBOARD' && (
              <div>
                <WorkspaceHeader title="Agent pipeline" subtitle="Live progress of the LangGraph multi-agent workflow" light={light}>
                  <button type="button" onClick={() => setIsInspectorOpen(true)} className={ghostBtn}>
                    <Icon name="inspect" size={14} /> Step inspector
                  </button>
                </WorkspaceHeader>
                <div className="p-6">
                  {status === 'IDLE' ? (
                    <EmptyState icon="pipeline" title="No run yet" light={light}
                      text="Start a review from the panel on the left. Each agent reports here as soon as it finishes." />
                  ) : (
                    <ol className="relative">
                      {AGENTS.map((agent, i) => {
                        const st = agentState(agent);
                        const event = progress.find((p) => p.agent === agent.key);
                        return (
                          <li key={agent.key} className="flex gap-4 pb-5 last:pb-0">
                            <div className="flex flex-col items-center">
                              <span className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-semibold ${
                                st === 'done' ? 'bg-emerald-500 text-white'
                                  : st === 'active' ? 'bg-indigo-600 text-white'
                                    : st === 'failed' ? 'bg-rose-500 text-white'
                                      : c('bg-slate-100 text-slate-500', 'bg-slate-800 text-slate-400')
                              }`}>
                                {st === 'done' ? <Icon name="check" size={15} strokeWidth={2.4} />
                                  : st === 'active' ? <Icon name="spinner" size={15} spin strokeWidth={2.4} /> : i + 1}
                              </span>
                              {i < AGENTS.length - 1 && <span className={`w-px flex-1 mt-1 ${c('bg-slate-200', 'bg-slate-800')}`} />}
                            </div>
                            <div className={`flex-1 rounded-lg border px-4 py-3 ${
                              st === 'active' ? c('border-indigo-200 bg-indigo-50/60', 'border-indigo-500/40 bg-indigo-500/10')
                                : c('border-slate-200', 'border-slate-800')
                            }`}>
                              <div className="flex items-center justify-between gap-3">
                                <div className={`text-sm font-medium ${strong}`}>{agent.name}</div>
                                <span className={`text-xs tabular-nums ${muted}`}>
                                  {event?.seconds !== undefined ? `${event.seconds.toFixed(1)} s`
                                    : st === 'active' ? 'running…' : st === 'failed' ? 'stopped' : st === 'done' ? 'done' : 'waiting'}
                                </span>
                              </div>
                              <div className={`text-xs mt-0.5 ${muted}`}>{agent.description}</div>
                            </div>
                          </li>
                        );
                      })}
                    </ol>
                  )}
                </div>
              </div>
            )}

            {/* Manuscript view */}
            {viewMode === 'MANUSCRIPT' && (
              <div>
                <WorkspaceHeader title="Manuscript" light={light}
                  subtitle={papers.length ? `${papers.length} reviewed papers · citations [1]–[${papers.length}]` : 'The generated literature review appears here'}>
                  {finalReview && (
                    <div className="no-print flex flex-wrap items-center gap-2">
                      <button type="button" onClick={() => exportToWord(finalReview, topic, papers)}
                        className="inline-flex items-center gap-1.5 h-8 px-3 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-medium">
                        <Icon name="download" size={14} /> Word
                      </button>
                      <button type="button" onClick={exportToPdf} className={ghostBtn}><Icon name="printer" size={14} /> PDF</button>
                      <button type="button" onClick={() => exportToMarkdown(finalReview, topic)} className={ghostBtn}>
                        <Icon name="download" size={14} /> Markdown
                      </button>
                      <button type="button" className={ghostBtn} onClick={async () => {
                        if (await copyToClipboard(finalReview)) {
                          setCopyToast(true);
                          setTimeout(() => setCopyToast(false), 2000);
                        }
                      }}>
                        <Icon name={copyToast ? 'check' : 'copy'} size={14} /> {copyToast ? 'Copied' : 'Copy'}
                      </button>
                    </div>
                  )}
                </WorkspaceHeader>

                <div className="p-6 lg:p-8">
                  {status === 'IDLE' && (
                    <EmptyState icon="document" title="Ready for a new review" light={light}
                      text="Enter a research topic, a publication-year range and the number of papers. The review, its sources and the verified research gaps will appear here." />
                  )}

                  {status === 'PROCESSING' && !finalReview && (
                    <div className="min-h-[420px] flex flex-col items-center justify-center text-center">
                      <Icon name="spinner" size={32} spin className="text-indigo-600" />
                      <div className={`mt-4 text-sm font-medium ${strong}`}>Generating the literature review</div>
                      <p className={`text-sm mt-1 ${muted}`}>
                        {completedCount} of {AGENTS.length} agents finished · {formatClock(elapsedTime)} elapsed
                      </p>
                      <button type="button" onClick={() => setViewMode('DASHBOARD')} className={`${ghostBtn} mt-5`}>
                        <Icon name="pipeline" size={14} /> Follow the agent pipeline
                      </button>
                    </div>
                  )}

                  {papers.length > 0 && (
                    <section className="mb-8">
                      <h2 className={`text-sm font-semibold ${strong}`}>Reviewed papers</h2>
                      <div className={`mt-3 divide-y rounded-lg border ${c('divide-slate-200 border-slate-200', 'divide-slate-800 border-slate-800')}`}>
                        {papers.map((paper, index) => (
                          <div key={paper.paper_id || paper.doi || index} className="flex items-start justify-between gap-4 px-4 py-3">
                            <div className="flex gap-3 min-w-0">
                              <span className={`text-xs font-semibold tabular-nums mt-0.5 ${c('text-indigo-700', 'text-indigo-300')}`}>[{index + 1}]</span>
                              <div className="min-w-0">
                                <div className={`text-sm font-medium ${strong}`}>{paper.title || 'Untitled paper'}</div>
                                <div className={`text-xs mt-0.5 ${muted}`}>
                                  {paper.authors?.slice(0, 3)?.join(', ') || 'Authors unavailable'}
                                  {paper.authors?.length > 3 ? ' et al.' : ''}
                                  {paper.year ? ` · ${paper.year}` : ''}
                                  {paper.venue ? ` · ${paper.venue}` : ''}
                                </div>
                                {String(paper.source || '').includes('online') && (
                                  <span className={`inline-flex items-center gap-1 mt-1.5 text-[11px] font-medium px-2 py-0.5 rounded-full ring-1 ${c('bg-sky-50 text-sky-700 ring-sky-200', 'bg-sky-500/15 text-sky-300 ring-sky-500/30')}`}>
                                    <Icon name="external" size={11} /> Fetched online from OpenAlex
                                  </span>
                                )}
                              </div>
                            </div>
                            {paper.pdf_url && (
                              <a href={paper.pdf_url} target="_blank" rel="noopener noreferrer"
                                className={`shrink-0 inline-flex items-center gap-1 text-xs font-medium ${c('text-indigo-700 hover:text-indigo-900', 'text-indigo-300 hover:text-indigo-200')}`}>
                                PDF <Icon name="external" size={12} />
                              </a>
                            )}
                          </div>
                        ))}
                      </div>
                    </section>
                  )}

                  {researchGap && (
                    <section className={`mb-8 rounded-lg border p-5 ${c('bg-indigo-50/50 border-indigo-100', 'bg-indigo-500/5 border-indigo-500/20')}`}>
                      <div className="flex items-center justify-between gap-3">
                        <h2 className={`text-sm font-semibold ${strong}`}>Verified research gap</h2>
                        <span className={`inline-flex items-center gap-1 text-xs font-medium ${c('text-emerald-700', 'text-emerald-300')}`}>
                          <Icon name="checkCircle" size={14} /> Evidence verified
                        </span>
                      </div>
                      <div className={`mt-3 text-sm leading-relaxed ${c('text-slate-700', 'text-slate-300')}`}>
                        {typeof researchGap === 'string' ? <p>{researchGap}</p> : verifiedGaps.length ? (
                          <>
                            <p>{verifiedGaps[0].gap_statement}</p>
                            <button type="button" onClick={() => setViewMode('GAPS')}
                              className={`mt-3 inline-flex items-center gap-1 text-xs font-medium ${c('text-indigo-700 hover:text-indigo-900', 'text-indigo-300 hover:text-indigo-200')}`}>
                              {verifiedGaps.length > 1 ? `View all ${verifiedGaps.length} verified gaps` : 'View the gap evidence'} <Icon name="chevronRight" size={13} />
                            </button>
                          </>
                        ) : (
                          <p className={muted}>No candidate gap passed verification for this run.</p>
                        )}
                      </div>
                    </section>
                  )}

                  {finalReview && (
                    <article className={`manuscript font-serif-doc max-w-3xl mx-auto ${c('text-slate-800', 'text-slate-200')}`}>
                      <ReactMarkdown
                        components={{
                          h1: (props) => <h1 className={strong} {...props} />,
                          h2: (props) => <h2 className={c('text-slate-900', 'text-slate-100')} {...props} />,
                          h3: (props) => <h3 className={c('text-slate-800', 'text-slate-200')} {...props} />,
                          p: ({ children, ...props }) => {
                            const text = String(children ?? '');
                            if (/^\[\d+\]/.test(text.trim())) {
                              return <p className={`!text-sm !text-left !leading-relaxed pl-8 -indent-8 ${muted}`} {...props}>{children}</p>;
                            }
                            return <p {...props}>{children}</p>;
                          },
                        }}
                      >
                        {finalReview}
                      </ReactMarkdown>
                    </article>
                  )}

                  {status === 'COMPLETED' && finalReview && runId && (
                    <div className="max-w-3xl mx-auto">
                      <ReviewFeedbackPanel runId={runId} theme={theme} history={runData?.feedback_history || []}
                        onRevised={(markdown) => setRunData((prev) => ({ ...(prev || {}), final_review_md: markdown }))} />
                    </div>
                  )}

                  {status === 'COMPLETED' && !finalReview && (
                    <div className={`p-4 rounded-lg border text-sm ${c('bg-amber-50 border-amber-200 text-amber-900', 'bg-amber-500/10 border-amber-500/30 text-amber-200')}`}>
                      The run completed, but the backend did not return the review text.
                    </div>
                  )}
                </div>
              </div>
            )}

            {viewMode === 'EVIDENCE' && <EvidenceMatrixView papers={papers} theme={theme} />}
            {viewMode === 'GAPS' && <ResearchGapLab researchGap={researchGap} papers={papers} theme={theme} />}
            {viewMode === 'ABLATION' && <AblationDashboard theme={theme} />}
            {viewMode === 'CORPUS' && <CorpusExplorer theme={theme} />}
          </div>
        </section>
      </main>

      <StepInspector
        isOpen={isInspectorOpen}
        onClose={() => setIsInspectorOpen(false)}
        topic={topic}
        startYear={startYear}
        endYear={endYear}
        maxPapers={maxPapers}
        status={status}
        runData={runData}
        theme={theme}
      />
    </div>
  );
}

// A small title bar (title and subtitle) shown above the content of a tab.
function WorkspaceHeader({ title, subtitle, light, children }) {
  return (
    <div className={`px-6 py-4 border-b flex flex-wrap items-center justify-between gap-3 ${light ? 'border-slate-200' : 'border-slate-800'}`}>
      <div>
        <h2 className={`text-base font-semibold ${light ? 'text-slate-900' : 'text-slate-100'}`}>{title}</h2>
        {subtitle && <p className={`text-xs mt-0.5 ${light ? 'text-slate-500' : 'text-slate-400'}`}>{subtitle}</p>}
      </div>
      {children}
    </div>
  );
}

// The friendly 'nothing here yet' box (icon, title and text) shown before a review has been generated.
function EmptyState({ icon, title, text, light }) {
  return (
    <div className="min-h-[420px] flex flex-col items-center justify-center text-center px-6">
      <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${light ? 'bg-indigo-50 text-indigo-600' : 'bg-indigo-500/15 text-indigo-300'}`}>
        <Icon name={icon} size={24} />
      </div>
      <h3 className={`mt-4 text-base font-semibold ${light ? 'text-slate-900' : 'text-slate-100'}`}>{title}</h3>
      <p className={`mt-1.5 text-sm max-w-md ${light ? 'text-slate-500' : 'text-slate-400'}`}>{text}</p>
    </div>
  );
}
