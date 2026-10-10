import { useState } from 'react';
import Icon from './Icon';

const API_BASE = 'http://127.0.0.1:3000/api';

/*
 * Human-in-the-loop feedback: the reviewer's comments are combined with the
 * automatic Critic / Fact-Checker findings and applied by the Reviser agent.
 */
// The 'Reviewer feedback' box under the manuscript.
// The user types a comment and clicks 'Apply feedback'; the component sends it to the Node server
// (POST /api/review/<runId>/feedback), shows the revised review via onRevised, and displays the critic scores
// before and after (score, supported claims, paper coverage, open issues) plus the history of earlier rounds.
export default function ReviewFeedbackPanel({ runId, theme = 'light', onRevised, history = [] }) {
  const [comments, setComments] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  const light = theme === 'light';

  // Sends the feedback to the server: ignores empty text (less than 3 characters), shows a busy state,
  // POST {comments} to /api/review/<runId>/feedback, then stores the result, clears the text box and hands the
  // revised review to the page. Errors are shown under the box.
  const submit = async () => {
    if (!runId || comments.trim().length < 3) return;
    setBusy(true);
    setError('');
    try {
      const response = await fetch(`${API_BASE}/review/${runId}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ comments: comments.trim() }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Feedback revision failed');
      setResult(data);
      setComments('');
      onRevised?.(data.final_review_md);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  // scores of the latest round: the live response, or the last round stored with the run in MongoDB
  const last = history[history.length - 1];
  // Converts a critic report into the four numbers shown on screen.
  const fromCritic = (c) => c && { score: c.score, supported_fraction: c.stats?.supported_fraction,
    paper_coverage: c.stats?.paper_coverage, open_issues: c.issues?.length };
  const before = result ? fromCritic(result.critic_before) : last?.critic_before;
  const after = result ? fromCritic(result.critic_after) : last?.critic_after;
  const rounds = result?.history?.length ?? history.length;
  const lastComment = result ? null : last?.comments;

  return (
    <section className={`no-print mt-10 rounded-xl border p-5 font-sans ${light ? 'bg-slate-50 border-slate-200' : 'bg-slate-950 border-slate-800'}`}>
      <div className="flex items-start gap-3">
        <div className={`w-9 h-9 rounded-lg flex items-center justify-center shrink-0 ${light ? 'bg-white text-indigo-600 ring-1 ring-slate-200' : 'bg-slate-900 text-indigo-300 ring-1 ring-slate-800'}`}>
          <Icon name="message" size={18} />
        </div>
        <div>
          <h3 className={`text-sm font-semibold ${light ? 'text-slate-900' : 'text-slate-100'}`}>Reviewer feedback</h3>
          <p className={`text-xs mt-0.5 leading-relaxed ${light ? 'text-slate-500' : 'text-slate-400'}`}>
            Describe what should change, for example “compare [2] and [4] more directly” or “the dataset gap is too vague”.
            The Reviser agent applies it together with the automatic fact-check, using only evidence from the cited papers.
          </p>
        </div>
      </div>

      <textarea
        value={comments}
        onChange={(e) => setComments(e.target.value)}
        rows={3}
        maxLength={4000}
        placeholder="Your feedback on this review…"
        className={`mt-4 w-full rounded-lg border px-3 py-2.5 text-sm outline-none transition focus:ring-2 ${
          light
            ? 'bg-white border-slate-300 text-slate-900 placeholder:text-slate-400 focus:border-indigo-500 focus:ring-indigo-100'
            : 'bg-slate-900 border-slate-700 text-slate-100 placeholder:text-slate-500 focus:border-indigo-400 focus:ring-indigo-500/20'
        }`}
      />

      <div className="mt-3 flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={submit}
          disabled={busy || comments.trim().length < 3}
          className="inline-flex items-center gap-2 h-9 px-4 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm font-medium"
        >
          <Icon name={busy ? 'spinner' : 'send'} size={15} spin={busy} />
          {busy ? 'Revising…' : 'Apply feedback'}
        </button>
        {rounds > 0 && (
          <span className={`text-xs ${light ? 'text-slate-500' : 'text-slate-400'}`}>
            {rounds} feedback revision{rounds === 1 ? '' : 's'} applied to this review.
          </span>
        )}
        {error && <span className="text-xs text-rose-600">{error}</span>}
      </div>

      {lastComment && (
        <p className={`mt-4 text-sm ${light ? 'text-slate-600' : 'text-slate-300'}`}>
          <span className="font-medium">Last feedback: </span>“{lastComment}”
        </p>
      )}
      {before && after && (
        <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3">
          <Stat label="Critic score" before={before.score} after={after.score} light={light} />
          <Stat label="Supported claims" before={before.supported_fraction} after={after.supported_fraction} pct light={light} />
          <Stat label="Papers cited" before={before.paper_coverage} after={after.paper_coverage} pct light={light} />
          <Stat label="Open issues" before={before.open_issues} after={after.open_issues} light={light} lowerIsBetter />
        </div>
      )}
    </section>
  );
}

// Shows one before/after number (for example the critic score 95 -> 92) and colours it green when it improved
// (or red when it got worse).
function Stat({ label, before, after, pct = false, light, lowerIsBetter = false }) {
  // Formats a value for display: '-' if missing, a percentage when pct is on, otherwise a number with one decimal.
  const show = (v) => (v === undefined || v === null ? '–' : pct ? `${(v * 100).toFixed(0)}%` : `${Number(v).toFixed(1).replace(/\.0$/, '')}`);
  const improved = before !== undefined && after !== undefined && before !== null && after !== null &&
    (lowerIsBetter ? after < before : after > before);
  return (
    <div className={`rounded-lg border px-3 py-2.5 ${light ? 'bg-white border-slate-200' : 'bg-slate-900 border-slate-800'}`}>
      <div className={`text-xs ${light ? 'text-slate-500' : 'text-slate-400'}`}>{label}</div>
      <div className={`mt-1 text-sm tabular-nums ${light ? 'text-slate-900' : 'text-slate-100'}`}>
        <span className={light ? 'text-slate-400' : 'text-slate-500'}>{show(before)}</span>
        <span className="mx-1.5 text-slate-400">→</span>
        <span className={`font-semibold ${improved ? (light ? 'text-emerald-700' : 'text-emerald-300') : ''}`}>{show(after)}</span>
      </div>
    </div>
  );
}
