import { useEffect, useMemo, useState } from 'react';
import Icon from './Icon';

const API = 'http://127.0.0.1:3000/api/evaluation';
const DEFAULT_PROPOSED = 'M5';
const DEFAULT_HEADLINE = ['overall_quality_index', 'citation_groundedness', 'citation_precision', 'hallucination_rate', 'fabricated_reference_rate', 'source_coverage'];

// Formats a number for the table: rate -> percentage with one decimal, score -> one decimal, otherwise a rounded
// whole number; '-' when there is no value.
function fmt(value, kind) {
  if (value === null || value === undefined || Number.isNaN(value)) return '–';
  if (kind === 'rate') return `${(value * 100).toFixed(1)}%`;
  if (kind === 'score') return value.toFixed(1);
  return Math.round(value).toLocaleString();
}

// Converts a p-value into significance stars: *** below 0.001, ** below 0.01, * below 0.05, otherwise nothing.
function stars(p) {
  if (p === null || p === undefined) return '';
  if (p < 0.001) return '***';
  if (p < 0.01) return '**';
  if (p < 0.05) return '*';
  return '';
}

// The Evaluation tab. On load it asks the Node server for GET /api/evaluation/benchmark.
// It then shows the ablation table (M0 to M5 with mean, confidence interval and significance stars),
// the contribution of each component, the charts, a switch for secondary metrics and a button that copies
// the LaTeX table. If no results exist yet it shows a message explaining how to compute them.
export default function AblationDashboard({ theme = 'light' }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [copied, setCopied] = useState(false);
  const [showSecondary, setShowSecondary] = useState(false);
  const light = theme === 'light';
  const PROPOSED = data?.proposed || DEFAULT_PROPOSED;

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API}/benchmark`);
        if (res.ok) setData(await res.json());
      } catch (err) {
        console.error('Failed to load ablation results:', err);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  // Chooses which metric columns to show: the primary metrics, or the secondary ones when the switch is on.
  const metrics = useMemo(
    () => (data?.metrics || []).filter((m) => (showSecondary ? !m.primary : m.primary)),
    [data, showSecondary]
  );

  // Finds the corrected p-value of the proposed configuration versus configuration b for one metric.
  const pFor = (metric, b) =>
    data?.tests?.find((t) => t.metric === metric && t.a === PROPOSED && t.b === b)?.p_holm;

  // For each metric finds which configuration is best (highest for 'higher is better', lowest otherwise),
  // so that its cell can be highlighted.
  const best = useMemo(() => {
    const out = {};
    (data?.metrics || []).forEach((m) => {
      let bestId = null;
      let bestVal = null;
      (data?.configs || []).forEach((c) => {
        const v = data.table?.[c.id]?.[m.key]?.mean;
        if (v === null || v === undefined) return;
        if (bestVal === null || (m.higher_is_better ? v > bestVal : v < bestVal)) {
          bestVal = v;
          bestId = c.id;
        }
      });
      out[m.key] = bestId;
    });
    return out;
  }, [data]);

  // Copies the LaTeX version of the ablation table to the clipboard and shows 'Copied' for 2 seconds.
  const copyLatex = () => {
    if (!data?.latex_table) return;
    navigator.clipboard.writeText(data.latex_table);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const card = `p-4 rounded-lg border ${light ? 'bg-white border-slate-200' : 'bg-slate-900 border-slate-800'}`;
  const heading = `text-sm font-semibold ${light ? 'text-slate-900' : 'text-slate-100'}`;
  const muted = light ? 'text-slate-500' : 'text-slate-400';
  const border = light ? 'border-slate-200' : 'border-slate-800';
  const label = `text-xs ${muted}`;

  if (loading) {
    return <div className={`p-6 text-sm inline-flex items-center gap-2 ${muted}`}><Icon name="spinner" size={15} spin /> Loading evaluation results…</div>;
  }

  if (!data?.available) {
    return (
      <div className="p-6 space-y-3">
        <h2 className={heading}>Evaluation and ablation study</h2>
        <div className={card}>
          <p className={`text-sm ${light ? 'text-slate-700' : 'text-slate-300'}`}>
            No computed ablation results yet.
          </p>
          <p className={`text-xs mt-2 ${muted}`}>
            Run the automatic benchmark from the project folder, then refresh this tab:
          </p>
          <pre className={`mt-2 p-3 rounded-lg text-[11px] font-mono ${light ? 'bg-slate-50 text-slate-800' : 'bg-slate-950 text-slate-300'}`}>
            cd MALRG_Evaluation_Suite && python run_all.py all
          </pre>
        </div>
      </div>
    );
  }

  const proposed = data.table?.[PROPOSED] || {};
  const headline = data.headline || DEFAULT_HEADLINE;
  const metaByKey = Object.fromEntries((data.metrics || []).map((m) => [m.key, m]));
  const gv = data.gap_verification?.mean || {};

  return (
    <div>
      <div className={`px-6 py-4 border-b flex flex-col md:flex-row items-start md:items-center justify-between gap-3 ${border}`}>
        <div>
          <h2 className={`text-base font-semibold ${light ? 'text-slate-900' : 'text-slate-100'}`}>Evaluation and ablation study</h2>
          <p className={`text-xs mt-0.5 ${muted}`}>
            Results on {data.n_topics} benchmark topics · judge {data.judge_model} · NLI DeBERTa-v3 ·
            generated {data.generated_at?.replace('T', ' ')}
          </p>
        </div>
        <button
          type="button"
          onClick={copyLatex}
          className={`inline-flex items-center gap-1.5 h-9 px-3 rounded-lg text-sm font-medium border shrink-0 ${light ? 'bg-white border-slate-300 text-slate-700 hover:bg-slate-50' : 'bg-slate-900 border-slate-700 text-slate-300 hover:bg-slate-800'}`}
        >
          <Icon name={copied ? 'check' : 'copy'} size={15} /> {copied ? 'Copied' : 'Copy LaTeX table'}
        </button>
      </div>
      <div className="p-6 space-y-8">

      {/* Headline numbers for the proposed system */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        {headline.map((key) => {
          const meta = metaByKey[key];
          const stat = proposed[key];
          if (!meta || !stat) return null;
          return (
            <div key={key} className={card}>
              <div className={label}>
                {meta.label} {meta.higher_is_better ? '↑' : '↓'}
              </div>
              <div className={`text-2xl font-semibold tabular-nums mt-1.5 ${light ? 'text-slate-900' : 'text-slate-100'}`}>
                {fmt(stat.mean, meta.kind)}
              </div>
              <div className={`text-[11px] mt-1 ${muted}`}>
                95% CI {fmt(stat.ci_low, meta.kind)} – {fmt(stat.ci_high, meta.kind)}
              </div>
            </div>
          );
        })}
      </div>

      {/* Ablation table */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className={heading}>Ablation table (mean ± SD over topics)</h3>
          <button
            type="button"
            onClick={() => setShowSecondary((v) => !v)}
            className={`h-8 px-3 rounded-lg text-xs font-medium border ${light ? 'bg-white border-slate-300 text-slate-700 hover:bg-slate-50' : 'border-slate-700 text-slate-300 hover:bg-slate-800'}`}
          >
            {showSecondary ? 'Show primary metrics' : 'Show secondary metrics'}
          </button>
        </div>
        <div className={`overflow-x-auto rounded-xl border ${light ? 'border-slate-200' : 'border-slate-800'}`}>
          <table className="w-full text-left text-[13px] border-collapse">
            <thead>
              <tr className={light ? 'bg-slate-50 text-slate-600' : 'bg-slate-950 text-slate-400'}>
                <th className="p-3 text-xs font-semibold">Configuration</th>
                {metrics.map((m) => (
                  <th key={m.key} className="p-3 text-xs font-semibold text-right whitespace-nowrap">
                    {m.label} {m.higher_is_better ? '↑' : '↓'}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className={`tabular-nums ${light ? 'text-slate-700' : 'text-slate-300'}`}>
              {(data.configs || []).map((c) => (
                <tr
                  key={c.id}
                  className={`border-t ${light ? 'border-slate-200' : 'border-slate-800'} ${
                    c.id === PROPOSED ? (light ? 'bg-indigo-50 font-semibold text-slate-900' : 'bg-indigo-500/10 font-semibold') : ''
                  }`}
                >
                  <td className="p-3 font-medium whitespace-nowrap">{c.label}</td>
                  {metrics.map((m) => {
                    const s = data.table?.[c.id]?.[m.key] || {};
                    const sd = s.std === null || s.std === undefined ? '' : m.kind === 'rate' ? ` ± ${(s.std * 100).toFixed(1)}` : ` ± ${s.std.toFixed(1)}`;
                    return (
                      <td key={m.key} className={`p-3 text-right whitespace-nowrap ${best[m.key] === c.id ? 'underline decoration-2' : ''}`}>
                        {fmt(s.mean, m.kind)}
                        <span className={`text-xs ${muted}`}>{sd}</span>
                        {c.id !== PROPOSED && <sup className="text-indigo-500">{stars(pFor(m.key, c.id))}</sup>}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className={`text-xs leading-relaxed ${muted}`}>
          Underlined = best. M0 to M4 are mean ± SD over the benchmark topics; M5 and M6 are single values.
        </p>
      </div>

      {/* Component contributions */}
      {data.contributions?.length > 0 && (
        <div className="space-y-3">
          <h3 className={heading}>What each component contributes (points)</h3>
          <div className={`overflow-x-auto rounded-xl border ${light ? 'border-slate-200' : 'border-slate-800'}`}>
            <table className="w-full text-[13px]">
              <thead>
                <tr className={light ? 'bg-slate-50 text-slate-600' : 'bg-slate-950 text-slate-400'}>
                  <th className="p-3 text-xs font-semibold text-left">Component</th>
                  {(data.metrics || []).filter((m) => m.primary).map((m) => (
                    <th key={m.key} className="p-3 text-xs font-semibold text-right whitespace-nowrap">{m.label}</th>
                  ))}
                </tr>
              </thead>
              <tbody className={`tabular-nums ${light ? 'text-slate-700' : 'text-slate-300'}`}>
                {data.contributions.map((row) => (
                  <tr key={row.component} className={`border-t ${light ? 'border-slate-200' : 'border-slate-800'}`}>
                    <td className="p-3">{row.component}</td>
                    {(data.metrics || []).filter((m) => m.primary).map((m) => {
                      const v = row[m.key];
                      const text = v === null || v === undefined ? '–' : m.kind === 'rate' ? `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}` : `${v >= 0 ? '+' : ''}${v.toFixed(1)}`;
                      return <td key={m.key} className="p-3 text-right">{text}</td>;
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Skeptic / gap verification */}
      {gv.precision_unfiltered_M3 !== null && gv.precision_unfiltered_M3 !== undefined && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          <div className={card}>
            <div className={label}>Gap precision w/o Skeptic (M3)</div>
            <div className="text-xl font-semibold tabular-nums mt-1">{fmt(gv.precision_unfiltered_M3, 'rate')}</div>
          </div>
          <div className={card}>
            <div className={label}>Gap precision with Skeptic (M4)</div>
            <div className="text-xl font-semibold tabular-nums mt-1 text-indigo-600">{fmt(gv.precision_verified_M4, 'rate')}</div>
          </div>
          <div className={card}>
            <div className={label}>Skeptic precision</div>
            <div className="text-xl font-semibold tabular-nums mt-1">{fmt(gv.skeptic_precision, 'rate')}</div>
          </div>
          <div className={card}>
            <div className={label}>Candidates rejected</div>
            <div className="text-xl font-semibold tabular-nums mt-1">{fmt(gv.rejection_rate, 'rate')}</div>
          </div>
        </div>
      )}

      {/* Charts */}
      {data.charts?.length > 0 && (
        <div className="space-y-3">
          <h3 className={heading}>Figures</h3>
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            {data.charts.map((name) => (
              <figure key={name} className={`p-3 rounded-lg border bg-white ${border}`}>
                <img src={`${API}/charts/${name}`} alt={name.replace('.png', '').replaceAll('_', ' ')} className="w-full" />
                <figcaption className="mt-2 text-xs text-slate-500 text-center capitalize">{name.replace('.png', '').replaceAll('_', ' ')}</figcaption>
              </figure>
            ))}
          </div>
        </div>
      )}

      {data.latex_table && (
        <div className="space-y-2">
          <h3 className={heading}>LaTeX export</h3>
          <pre className={`p-4 rounded-lg font-mono text-[10px] overflow-x-auto border ${light ? 'bg-slate-50 text-slate-800 border-slate-200' : 'bg-slate-950 text-slate-300 border-slate-800'}`}>
            {data.latex_table}
          </pre>
        </div>
      )}
      </div>
    </div>
  );
}
