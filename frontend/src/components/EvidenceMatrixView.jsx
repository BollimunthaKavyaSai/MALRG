import { useState } from 'react';
import Icon from './Icon';

// the agent returns several evidence sentences per field; show the first one to keep rows readable
// Joins the first few items of a list into one text (or returns the value as it is when it is not a list).
const joinList = (value, max = 1) => (Array.isArray(value) ? value.filter(Boolean).slice(0, max).join(' ') : value || '');

// The Evidence Matrix tab: a table with one row per reviewed paper and the columns methodology, results,
// metrics and limitations. The values come from the Paper Analysis agent (paper_analysis), with the stored
// corpus fields as backup. A search box filters the rows; with no papers it shows an explanation.
export default function EvidenceMatrixView({ papers = [], theme = 'light' }) {
  const [searchTerm, setSearchTerm] = useState('');
  const light = theme === 'light';
  const muted = light ? 'text-slate-500' : 'text-slate-400';

  const rows = papers.map((paper, idx) => {
    const a = paper.paper_analysis || {};
    return {
      key: paper.paper_id || idx,
      n: idx + 1,
      title: paper.title || 'Untitled paper',
      meta: [paper.year, paper.venue].filter(Boolean).join(' · '),
      methodology: joinList(a.methodology) || paper.methodology || '',
      results: joinList(a.key_findings) || paper.results || '',
      metrics: (a.evaluation_metrics?.length ? a.evaluation_metrics : paper.evaluation_metrics) || [],
      limitations: joinList(a.limitations) || paper.limitations || '',
    };
  });

  const term = searchTerm.trim().toLowerCase();
  const filtered = term
    ? rows.filter((r) => [r.title, r.methodology, r.results, r.limitations, r.metrics.join(' ')].join(' ').toLowerCase().includes(term))
    : rows;

  if (!papers.length) {
    return (
      <div className="min-h-[420px] flex flex-col items-center justify-center text-center px-6">
        <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${light ? 'bg-indigo-50 text-indigo-600' : 'bg-indigo-500/15 text-indigo-300'}`}>
          <Icon name="table" size={24} />
        </div>
        <h3 className={`mt-4 text-base font-semibold ${light ? 'text-slate-900' : 'text-slate-100'}`}>No evidence extracted yet</h3>
        <p className={`mt-1.5 text-sm max-w-md ${muted}`}>
          Generate a review to compare the methodology, results, metrics and limitations of the retrieved studies side by side.
        </p>
      </div>
    );
  }

  const notReported = <span className={`italic ${muted}`}>Not reported</span>;

  return (
    <div>
      <div className={`px-6 py-4 border-b flex flex-wrap items-center justify-between gap-3 ${light ? 'border-slate-200' : 'border-slate-800'}`}>
        <div>
          <h2 className={`text-base font-semibold ${light ? 'text-slate-900' : 'text-slate-100'}`}>Evidence matrix</h2>
          <p className={`text-xs mt-0.5 ${muted}`}>Evidence extracted by the Paper Analysis Agent for {papers.length} studies</p>
        </div>
        <div className="relative w-full sm:w-72">
          <Icon name="search" size={15} className={`absolute left-3 top-1/2 -translate-y-1/2 ${muted}`} />
          <input
            type="text"
            placeholder="Filter by term or metric"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className={`w-full h-9 pl-9 pr-3 text-sm rounded-lg border outline-none transition focus:ring-2 ${
              light
                ? 'bg-white border-slate-300 text-slate-900 placeholder:text-slate-400 focus:border-indigo-500 focus:ring-indigo-100'
                : 'bg-slate-950 border-slate-700 text-slate-100 placeholder:text-slate-500 focus:border-indigo-400 focus:ring-indigo-500/20'
            }`}
          />
        </div>
      </div>

      <div className="p-6">
        <div className={`overflow-x-auto rounded-lg border ${light ? 'border-slate-200' : 'border-slate-800'}`}>
          <table className="w-full text-left text-sm border-collapse">
            <thead>
              <tr className={light ? 'bg-slate-50 text-slate-600' : 'bg-slate-950 text-slate-400'}>
                {['#', 'Study', 'Methodology', 'Results', 'Metrics', 'Limitations'].map((h, i) => (
                  <th key={h} className={`px-4 py-3 text-xs font-semibold uppercase tracking-wide border-b ${light ? 'border-slate-200' : 'border-slate-800'} ${
                    i === 0 ? 'w-12' : i === 1 ? 'min-w-[200px]' : 'min-w-[160px]'
                  }`}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className={`divide-y ${light ? 'divide-slate-200' : 'divide-slate-800'}`}>
              {filtered.map((r) => (
                <tr key={r.key} className={`align-top ${light ? 'bg-white hover:bg-slate-50' : 'bg-slate-900 hover:bg-slate-800/60'}`}>
                  <td className={`px-4 py-3 text-xs font-semibold tabular-nums ${light ? 'text-indigo-700' : 'text-indigo-300'}`}>[{r.n}]</td>
                  <td className="px-4 py-3">
                    <div className={`font-medium leading-snug ${light ? 'text-slate-900' : 'text-slate-100'}`}>{r.title}</div>
                    {r.meta && <div className={`text-xs mt-1 ${muted}`}>{r.meta}</div>}
                  </td>
                  <td className={`px-4 py-3 text-[13px] leading-relaxed ${light ? 'text-slate-700' : 'text-slate-300'}`}>{r.methodology || notReported}</td>
                  <td className={`px-4 py-3 text-[13px] leading-relaxed ${light ? 'text-slate-700' : 'text-slate-300'}`}>{r.results || notReported}</td>
                  <td className="px-4 py-3">
                    {r.metrics.length ? (
                      <div className="flex flex-wrap gap-1">
                        {r.metrics.slice(0, 5).map((m, i) => (
                          <span key={i} className={`text-[11px] px-2 py-0.5 rounded-md ${light ? 'bg-slate-100 text-slate-700' : 'bg-slate-800 text-slate-300'}`}>{m}</span>
                        ))}
                      </div>
                    ) : notReported}
                  </td>
                  <td className={`px-4 py-3 text-[13px] leading-relaxed ${light ? 'text-slate-700' : 'text-slate-300'}`}>{r.limitations || notReported}</td>
                </tr>
              ))}
              {!filtered.length && (
                <tr><td colSpan={6} className={`px-4 py-8 text-center text-sm ${muted}`}>No study matches “{searchTerm}”.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
