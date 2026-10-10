import { useState, useEffect } from 'react';
import Icon from './Icon';

const API_BASE = 'http://127.0.0.1:3000/api';
const PAGE_SIZE = 15;

/* Browses the paper corpus stored in MongoDB through the Express API. */
// The Paper Corpus tab: a browser for the papers stored in MongoDB.
// It shows statistic cards, a search box, a paginated list (GET /api/corpus/papers) and a detail panel for the
// selected paper (abstract, authors, DOI, links). Escape closes the detail panel.
export default function CorpusExplorer({ theme = 'light' }) {
  const [papers, setPapers] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [search, setSearch] = useState('');
  const [selectedPaper, setSelectedPaper] = useState(null);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState('');

  const light = theme === 'light';
  const muted = light ? 'text-slate-500' : 'text-slate-400';
  const strong = light ? 'text-slate-900' : 'text-slate-100';
  const body = light ? 'text-slate-700' : 'text-slate-300';
  const border = light ? 'border-slate-200' : 'border-slate-800';

  // Loads one page of papers from the Node server: GET /api/corpus/papers?page=..&limit=..&q=..,
  // then saves the papers, total count and page numbers in state. Shows an error text if the server is unreachable.
  const fetchPapers = async (currentPage = page, currentSearch = search) => {
    setLoading(true);
    setError('');
    try {
      const query = new URLSearchParams({ page: currentPage, limit: PAGE_SIZE, q: currentSearch });
      const res = await fetch(`${API_BASE}/corpus/papers?${query.toString()}`);
      if (!res.ok) throw new Error(`Corpus API returned ${res.status}`);
      const data = await res.json();
      setPapers(data.papers || []);
      setTotal(data.total || 0);
      setTotalPages(data.totalPages || 1);
      setPage(data.page || 1);
    } catch (err) {
      setError('Could not reach the corpus API. Is the Node server running?');
      console.error('Failed to fetch corpus papers:', err);
    } finally {
      setLoading(false);
    }
  };

  // initial load: corpus statistics and the first page of papers
  useEffect(() => {
    fetch(`${API_BASE}/corpus/stats`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => d && setStats(d))
      .catch((err) => console.error('Failed to fetch stats:', err));
    fetch(`${API_BASE}/corpus/papers?page=1&limit=${PAGE_SIZE}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`Corpus API returned ${r.status}`))))
      .then((data) => {
        setPapers(data.papers || []);
        setTotal(data.total || 0);
        setTotalPages(data.totalPages || 1);
      })
      .catch(() => setError('Could not reach the corpus API. Is the Node server running?'))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!selectedPaper) return undefined;
    // Closes the paper detail panel when the Escape key is pressed.
    const onKey = (e) => e.key === 'Escape' && setSelectedPaper(null);
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selectedPaper]);

  // When the search form is submitted: stop the page reload and load page 1 of the results for the typed text.
  const handleSearchSubmit = (e) => {
    e.preventDefault();
    fetchPapers(1, search);
  };

  // Draws one small statistic box (icon, label and value).
  const statCard = (icon, label, value) => (
    <div className={`rounded-lg border px-4 py-3 ${light ? 'bg-white' : 'bg-slate-900'} ${border}`}>
      <div className={`flex items-center gap-1.5 text-xs ${muted}`}><Icon name={icon} size={13} /> {label}</div>
      <div className={`mt-1 text-xl font-semibold tabular-nums ${strong}`}>{value ?? '–'}</div>
    </div>
  );

  // Draws the small badge 'Preprint' (amber) or 'Peer-reviewed' (green) for a paper.
  const typeBadge = (p) => {
    const preprint = p.is_preprint;
    return (
      <span className={`text-[11px] font-medium px-2 py-0.5 rounded-full ring-1 whitespace-nowrap ${
        preprint
          ? (light ? 'bg-amber-50 text-amber-700 ring-amber-200' : 'bg-amber-500/15 text-amber-300 ring-amber-500/30')
          : (light ? 'bg-emerald-50 text-emerald-700 ring-emerald-200' : 'bg-emerald-500/15 text-emerald-300 ring-emerald-500/30')
      }`}>
        {preprint ? 'Preprint' : 'Peer-reviewed'}
      </span>
    );
  };

  // Returns the styling of the previous/next page buttons (faded when disabled).
  const navBtn = (disabled) => `inline-flex items-center gap-1 h-8 px-3 rounded-lg text-sm border transition ${
    disabled ? 'opacity-40 cursor-not-allowed' : light ? 'hover:bg-slate-50' : 'hover:bg-slate-800'
  } ${light ? 'bg-white border-slate-300 text-slate-700' : 'bg-slate-900 border-slate-700 text-slate-300'}`;

  // Draws one labelled block in the detail panel; nothing is drawn when the value is empty.
  const field = (label, value, tone = '') => value ? (
    <div>
      <h4 className={`text-xs font-semibold uppercase tracking-wide mb-1 ${tone || muted}`}>{label}</h4>
      <p className={`text-sm leading-relaxed ${body}`}>{Array.isArray(value) ? value.join(', ') : value}</p>
    </div>
  ) : null;

  return (
    <div>
      <div className={`px-6 py-4 border-b ${border}`}>
        <h2 className={`text-base font-semibold ${strong}`}>Paper corpus</h2>
        <p className={`text-xs mt-0.5 ${muted}`}>
          Papers stored in MongoDB (collection <code className="font-mono">paper</code>). The Search Agent only uses peer-reviewed papers.
        </p>
      </div>

      <div className="p-6 space-y-5">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {statCard('database', 'Papers in database', stats?.total ?? total)}
          {statCard('checkCircle', 'Peer-reviewed (eligible)', stats?.eligible_published)}
          {statCard('calendar', 'Earliest year', stats?.year_range?.min)}
          {statCard('calendar', 'Latest year', stats?.year_range?.max)}
        </div>

        <form onSubmit={handleSearchSubmit} className="flex flex-wrap gap-2">
          <div className="relative flex-1 min-w-[220px]">
            <Icon name="search" size={15} className={`absolute left-3 top-1/2 -translate-y-1/2 ${muted}`} />
            <input
              type="text"
              placeholder="Search titles, abstracts and venues"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className={`w-full h-9 pl-9 pr-3 text-sm rounded-lg border outline-none transition focus:ring-2 ${
                light
                  ? 'bg-white border-slate-300 text-slate-900 placeholder:text-slate-400 focus:border-indigo-500 focus:ring-indigo-100'
                  : 'bg-slate-950 border-slate-700 text-slate-100 placeholder:text-slate-500 focus:border-indigo-400 focus:ring-indigo-500/20'
              }`}
            />
          </div>
          <button type="submit" className="h-9 px-4 rounded-lg text-sm font-medium bg-indigo-600 hover:bg-indigo-700 text-white">
            Search
          </button>
          {search && (
            <button type="button" onClick={() => { setSearch(''); fetchPapers(1, ''); }} className={navBtn(false)}>
              Clear
            </button>
          )}
        </form>

        {error && (
          <div className={`flex items-center gap-2 text-sm rounded-lg px-3 py-2 ${light ? 'bg-rose-50 text-rose-700' : 'bg-rose-500/10 text-rose-300'}`}>
            <Icon name="alert" size={15} /> {error}
          </div>
        )}

        <div className={`overflow-x-auto rounded-lg border ${border}`}>
          <table className="w-full text-left text-sm border-collapse">
            <thead>
              <tr className={light ? 'bg-slate-50 text-slate-600' : 'bg-slate-950 text-slate-400'}>
                {['Year', 'Title and authors', 'Venue', 'Type', ''].map((h, i) => (
                  <th key={i} className={`px-4 py-3 text-xs font-semibold uppercase tracking-wide border-b ${border} ${
                    i === 0 ? 'w-16' : i === 2 ? 'w-56' : i === 3 ? 'w-32' : i === 4 ? 'w-20' : ''
                  }`}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className={`divide-y ${light ? 'divide-slate-200' : 'divide-slate-800'}`}>
              {loading ? (
                <tr><td colSpan={5} className={`px-4 py-10 text-center ${muted}`}>
                  <span className="inline-flex items-center gap-2"><Icon name="spinner" size={15} spin /> Loading papers…</span>
                </td></tr>
              ) : papers.length === 0 ? (
                <tr><td colSpan={5} className={`px-4 py-10 text-center ${muted}`}>No matching papers found.</td></tr>
              ) : (
                papers.map((p) => (
                  <tr
                    key={p.paper_id}
                    onClick={() => setSelectedPaper(p)}
                    className={`cursor-pointer align-top ${light ? 'bg-white hover:bg-slate-50' : 'bg-slate-900 hover:bg-slate-800/60'}`}
                  >
                    <td className={`px-4 py-3 tabular-nums ${muted}`}>{p.year || '–'}</td>
                    <td className="px-4 py-3">
                      <div className={`font-medium leading-snug ${strong}`}>{p.title}</div>
                      <div className={`text-xs mt-1 ${muted}`}>
                        {p.authors?.slice(0, 3)?.join(', ')}{p.authors?.length > 3 ? ' et al.' : ''}
                      </div>
                    </td>
                    <td className={`px-4 py-3 text-[13px] ${body}`}>{p.venue || <span className={`italic ${muted}`}>Not recorded</span>}</td>
                    <td className="px-4 py-3">{typeBadge(p)}</td>
                    <td className="px-4 py-3 text-right">
                      <span className={`inline-flex items-center gap-0.5 text-xs font-medium ${light ? 'text-indigo-600' : 'text-indigo-300'}`}>
                        View <Icon name="chevronRight" size={13} />
                      </span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        <div className="flex items-center justify-between">
          <div className={`text-sm ${muted}`}>Page {page} of {totalPages} · {total} peer-reviewed papers listed</div>
          <div className="flex gap-2">
            <button type="button" disabled={page <= 1 || loading} onClick={() => fetchPapers(page - 1, search)} className={navBtn(page <= 1)}>
              Previous
            </button>
            <button type="button" disabled={page >= totalPages || loading} onClick={() => fetchPapers(page + 1, search)} className={navBtn(page >= totalPages)}>
              Next
            </button>
          </div>
        </div>
      </div>

      {selectedPaper && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/50 backdrop-blur-[2px]" onClick={() => setSelectedPaper(null)}>
          <div
            onClick={(e) => e.stopPropagation()}
            className={`w-full max-w-2xl max-h-[85vh] overflow-y-auto rounded-xl border shadow-xl ${light ? 'bg-white border-slate-200' : 'bg-slate-900 border-slate-800'}`}
          >
            <div className={`sticky top-0 flex items-start justify-between gap-4 px-6 py-4 border-b ${border} ${light ? 'bg-white' : 'bg-slate-900'}`}>
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  {typeBadge(selectedPaper)}
                  <span className={`text-xs ${muted}`}>{[selectedPaper.year, selectedPaper.venue].filter(Boolean).join(' · ')}</span>
                </div>
                <h3 className={`text-base font-semibold mt-2 leading-snug ${strong}`}>{selectedPaper.title}</h3>
                <p className={`text-xs mt-1 ${muted}`}>{selectedPaper.authors?.join(', ')}</p>
              </div>
              <button type="button" onClick={() => setSelectedPaper(null)} aria-label="Close"
                className={`p-1.5 rounded-md ${light ? 'text-slate-500 hover:bg-slate-100' : 'text-slate-400 hover:bg-slate-800'}`}>
                <Icon name="x" size={18} />
              </button>
            </div>

            <div className="px-6 py-5 space-y-4">
              {field('Abstract', selectedPaper.abstract || 'Not available')}
              {field('Methodology', selectedPaper.methodology)}
              {field('Results', selectedPaper.results)}
              {field('Evaluation metrics', selectedPaper.evaluation_metrics?.length ? selectedPaper.evaluation_metrics : null)}
              {field('Limitations', selectedPaper.limitations, light ? 'text-amber-700' : 'text-amber-300')}

              <div className={`pt-3 border-t flex flex-wrap items-center gap-3 ${border}`}>
                {selectedPaper.doi && <span className={`text-xs font-mono ${muted}`}>DOI: {selectedPaper.doi}</span>}
                {selectedPaper.source && <span className={`text-xs ${muted}`}>Source: {selectedPaper.source}</span>}
                {(selectedPaper.pdf_url || selectedPaper.url) && (
                  <a href={selectedPaper.pdf_url || selectedPaper.url} target="_blank" rel="noopener noreferrer"
                    className="ml-auto inline-flex items-center gap-1.5 h-8 px-3 rounded-lg text-sm font-medium bg-indigo-600 hover:bg-indigo-700 text-white">
                    Open paper <Icon name="external" size={14} />
                  </a>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
