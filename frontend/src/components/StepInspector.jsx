import { useEffect, useState } from 'react';
import Icon from './Icon';

// Joins the items of a list into one text separated by '; ' (or returns the value if it is not a list).
const joinList = (value) => (Array.isArray(value) ? value.filter(Boolean).join('; ') : value || '');

/*
 * Side panel that shows the actual input and output of each stage of the current run.
 * Every value comes from the run record; nothing is filled in when a stage has no data.
 */
// The 'Inspector' side panel: for every pipeline step it shows whether it is Completed, Pending or Not started,
// how long it took, and some key numbers of that step. It is built from the run's progress list and data.
// Escape closes it.
export default function StepInspector({ isOpen, onClose, topic, startYear, endYear, maxPapers, status, runData, theme = 'light' }) {
  // ?step=N opens a specific stage (1-based), e.g. when sharing a link to the gap stage
  const [activeStep, setActiveStep] = useState(() => Math.min(5, Math.max(0, (Number(new URLSearchParams(window.location.search).get('step')) || 1) - 1)));
  const light = theme === 'light';
  const muted = light ? 'text-slate-500' : 'text-slate-400';
  const strong = light ? 'text-slate-900' : 'text-slate-100';
  const body = light ? 'text-slate-700' : 'text-slate-300';
  const border = light ? 'border-slate-200' : 'border-slate-800';
  const panel = light ? 'bg-slate-50 border-slate-200' : 'bg-slate-950 border-slate-800';

  useEffect(() => {
    if (!isOpen) return undefined;
    // Closes the inspector panel when the Escape key is pressed.
    const onKey = (e) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const papers = runData?.raw_papers_list || [];
  const gap = runData?.research_gap;
  const review = runData?.final_review_md || '';
  const progress = runData?.progress || [];
  // Adds up the seconds the given agents took, taken from the progress list; null if none of them finished.
  const timeOf = (...agents) => {
    const events = progress.filter((p) => agents.includes(p.agent));
    if (!events.length) return null;
    return events.reduce((s, e) => s + (Number(e.seconds) || 0), 0);
  };
  // True only if every one of the given agents already appears in the progress list.
  const doneAll = (...agents) => agents.every((a) => progress.some((p) => p.agent === a));

  const cited = papers.filter((_, i) => review.includes(`[${i + 1}]`)).length;
  const words = review ? review.split(/\s+/).filter(Boolean).length : 0;
  // Chooses the step label: 'Completed' when done (or when the whole run is completed), 'Pending' while processing,
  // otherwise 'Not started'.
  const stateOf = (done) => (done ? 'Completed' : status === 'PROCESSING' ? 'Pending' : status === 'COMPLETED' ? 'Completed' : 'Not started');

  const steps = [
    {
      title: 'Query and scope',
      agents: 'Input form',
      state: topic ? (status === 'IDLE' ? 'Ready' : 'Completed') : 'Not started',
      input: [['Topic', topic || '–'], ['Year range', `${startYear} – ${endYear}`], ['Papers requested', maxPapers]],
    },
    {
      title: 'Retrieval',
      agents: 'Scholarly Search',
      state: stateOf(papers.length > 0 || doneAll('search_agent')),
      seconds: timeOf('search_agent'),
      input: [['Sources', 'MongoDB corpus first; OpenAlex online if too few relevant papers'], ['Eligibility', `Peer-reviewed, ${startYear}–${endYear}`]],
      render: () => papers.length ? (
        <ol className="space-y-2">
          {papers.map((p, i) => (
            <li key={p.paper_id || i} className={`rounded-lg border p-3 ${light ? 'bg-white' : 'bg-slate-900'} ${border}`}>
              <div className="flex gap-2">
                <span className={`text-xs font-semibold tabular-nums ${light ? 'text-indigo-700' : 'text-indigo-300'}`}>[{i + 1}]</span>
                <div className="min-w-0">
                  <div className={`text-sm font-medium leading-snug ${strong}`}>{p.title || 'Untitled'}</div>
                  <div className={`text-xs mt-1 ${muted}`}>
                    {[Array.isArray(p.authors) ? p.authors.slice(0, 3).join(', ') : p.authors, p.year, p.venue].filter(Boolean).join(' · ')}
                  </div>
                </div>
              </div>
            </li>
          ))}
        </ol>
      ) : null,
    },
    {
      title: 'Evidence extraction',
      agents: 'PDF Extraction · Paper Analysis',
      state: stateOf(doneAll('paper_extraction_agent', 'paper_analysis_agent')),
      seconds: timeOf('paper_extraction_agent', 'paper_analysis_agent'),
      input: [['Papers', papers.length], ['Fields', 'methodology, findings, metrics, limitations']],
      render: () => papers.length ? (
        <div className="space-y-2">
          {papers.map((p, i) => {
            const a = p.paper_analysis || {};
            const rows = [
              ['Method', joinList(a.methodology) || p.methodology],
              ['Findings', joinList(a.key_findings) || p.results],
              ['Limitations', joinList(a.limitations) || p.limitations],
            ];
            return (
              <div key={p.paper_id || i} className={`rounded-lg border p-3 space-y-1.5 ${light ? 'bg-white' : 'bg-slate-900'} ${border}`}>
                <div className={`text-sm font-medium ${strong}`}>[{i + 1}] {p.title}</div>
                {rows.map(([k, v]) => (
                  <div key={k} className={`text-xs leading-relaxed ${body}`}>
                    <span className={`font-medium ${muted}`}>{k}: </span>{v || <span className={`italic ${muted}`}>Not reported</span>}
                  </div>
                ))}
              </div>
            );
          })}
        </div>
      ) : null,
    },
    {
      title: 'Synthesis',
      agents: 'Cross-Paper Synthesis · Summarisation · Taxonomy',
      state: stateOf(doneAll('cross_paper_analysis_agent', 'summarization_agent', 'taxonomy_agent')),
      seconds: timeOf('cross_paper_analysis_agent', 'summarization_agent', 'taxonomy_agent'),
      input: [['Evidence profiles', papers.length]],
      output: [['Output', 'Themes and comparisons passed to the Grounded Writer']],
    },
    {
      title: 'Research gaps',
      agents: 'Research-Gap Triad',
      state: stateOf(Boolean(gap) || doneAll('research_gap_agent')),
      seconds: timeOf('research_gap_agent'),
      input: [['Candidate gaps (Proposer)', gap?.proposed_gaps?.length ?? '–']],
      output: [
        ['Challenged (Skeptic)', gap?.challenged_gaps?.length ?? '–'],
        ['Verified (Grounded Verifier)', gap?.verified_gaps?.length ?? '–'],
      ],
      render: () => gap?.verified_gaps?.length ? (
        <ul className="space-y-2">
          {gap.verified_gaps.map((g, i) => (
            <li key={i} className={`flex gap-2 text-sm leading-relaxed ${body}`}>
              <Icon name="checkCircle" size={16} className="mt-0.5 shrink-0 text-emerald-600" /> {g.gap_statement}
            </li>
          ))}
        </ul>
      ) : null,
    },
    {
      title: 'Writing and verification',
      agents: 'Citation Structuring · Grounded Writer · Verification Feedback',
      state: stateOf(Boolean(review)),
      seconds: timeOf('citation_agent', 'writing_agent', 'feedback_agent'),
      input: [['Papers to cite', papers.length]],
      output: [
        ['Papers cited in text', papers.length ? `${cited} of ${papers.length} (${Math.round((cited / papers.length) * 100)}%)` : '–'],
        ['Manuscript length', review ? `${words.toLocaleString()} words` : '–'],
      ],
    },
  ];

  const current = steps[activeStep];
  // Chooses the colour of the step label: green for Completed, blue for Pending, grey otherwise.
  const stateTone = (s) => s === 'Completed'
    ? (light ? 'bg-emerald-50 text-emerald-700 ring-emerald-200' : 'bg-emerald-500/15 text-emerald-300 ring-emerald-500/30')
    : s === 'Pending'
      ? (light ? 'bg-indigo-50 text-indigo-700 ring-indigo-200' : 'bg-indigo-500/15 text-indigo-300 ring-indigo-500/30')
      : (light ? 'bg-slate-100 text-slate-600 ring-slate-200' : 'bg-slate-800 text-slate-300 ring-slate-700');

  // Draws a small key/value list (label on the left, value on the right) for one step's numbers.
  const kv = (rows) => (
    <dl className="divide-y divide-dashed divide-slate-300/60">
      {rows.map(([k, v]) => (
        <div key={k} className="flex items-start justify-between gap-4 py-2 text-sm">
          <dt className={muted}>{k}</dt>
          <dd className={`text-right ${strong}`}>{v}</dd>
        </div>
      ))}
    </dl>
  );

  const detail = current.render?.();

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-slate-900/40 backdrop-blur-[2px]" onClick={onClose}>
      <aside
        onClick={(e) => e.stopPropagation()}
        className={`w-full max-w-2xl h-full flex flex-col shadow-xl border-l ${light ? 'bg-white border-slate-200' : 'bg-slate-900 border-slate-800'}`}
      >
        <div className={`px-6 py-4 border-b flex items-start justify-between gap-4 ${border}`}>
          <div>
            <h2 className={`text-base font-semibold ${strong}`}>Step inspector</h2>
            <p className={`text-xs mt-0.5 ${muted}`}>Inputs and outputs recorded for each stage of this run</p>
          </div>
          <button type="button" onClick={onClose} aria-label="Close"
            className={`p-1.5 rounded-md ${light ? 'text-slate-500 hover:bg-slate-100' : 'text-slate-400 hover:bg-slate-800'}`}>
            <Icon name="x" size={18} />
          </button>
        </div>

        <nav className={`px-4 border-b flex gap-1 overflow-x-auto ${border}`}>
          {steps.map((step, idx) => (
            <button
              key={step.title}
              type="button"
              onClick={() => setActiveStep(idx)}
              className={`shrink-0 px-3 py-2.5 text-sm border-b-2 -mb-px transition ${
                activeStep === idx
                  ? (light ? 'border-indigo-600 text-indigo-700 font-medium' : 'border-indigo-400 text-indigo-300 font-medium')
                  : `border-transparent ${muted} ${light ? 'hover:text-slate-800' : 'hover:text-slate-200'}`
              }`}
            >
              <span className="tabular-nums opacity-60 mr-1">{idx + 1}.</span>{step.title}
            </button>
          ))}
        </nav>

        <div className="flex-1 overflow-y-auto p-6 space-y-5">
          <div className="flex items-start justify-between gap-4">
            <div>
              <div className={`text-xs ${muted}`}>Stage {activeStep + 1} · {current.agents}</div>
              <div className={`text-lg font-semibold mt-0.5 ${strong}`}>{current.title}</div>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              {current.seconds != null && (
                <span className={`inline-flex items-center gap-1 text-xs tabular-nums ${muted}`}>
                  <Icon name="clock" size={13} /> {current.seconds.toFixed(1)} s
                </span>
              )}
              <span className={`text-xs font-medium px-2 py-0.5 rounded-full ring-1 ${stateTone(current.state)}`}>{current.state}</span>
            </div>
          </div>

          <section className={`rounded-lg border p-4 ${panel}`}>
            <h3 className={`text-xs font-semibold uppercase tracking-wide mb-1 ${muted}`}>Input</h3>
            {kv(current.input)}
          </section>

          {(current.output || detail) && (
            <section className={`rounded-lg border p-4 ${panel}`}>
              <h3 className={`text-xs font-semibold uppercase tracking-wide mb-1 ${muted}`}>Output</h3>
              {current.output && kv(current.output)}
              {detail && <div className={current.output ? 'mt-3' : 'mt-2'}>{detail}</div>}
            </section>
          )}
          {!current.output && !detail && status !== 'COMPLETED' && (
            <p className={`text-sm ${muted}`}>The output appears here once this stage has run.</p>
          )}
        </div>

        <div className={`px-6 py-3 border-t flex items-center justify-between ${border}`}>
          <span className={`text-xs ${muted}`}>Stage {activeStep + 1} of {steps.length}</span>
          <div className="flex gap-2">
            <button type="button" disabled={activeStep === 0} onClick={() => setActiveStep((s) => s - 1)}
              className={`h-8 px-3 rounded-lg text-sm border disabled:opacity-40 ${light ? 'bg-white border-slate-300 text-slate-700 hover:bg-slate-50' : 'bg-slate-900 border-slate-700 text-slate-300 hover:bg-slate-800'}`}>
              Previous
            </button>
            <button type="button" disabled={activeStep === steps.length - 1} onClick={() => setActiveStep((s) => s + 1)}
              className="h-8 px-3 rounded-lg text-sm font-medium bg-indigo-600 hover:bg-indigo-700 text-white disabled:opacity-40">
              Next
            </button>
          </div>
        </div>
      </aside>
    </div>
  );
}
