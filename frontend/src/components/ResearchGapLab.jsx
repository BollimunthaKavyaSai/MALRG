import Icon from './Icon';

/*
 * Research gaps produced by the adversarial triad: Proposer -> Skeptic -> Grounded Verifier.
 * Shows the real outputs of each stage (no placeholder text).
 */
// The Research Gaps tab: shows the three-step debate of the gap agents.
// 1) Proposer - how many candidate gaps were proposed, 2) Skeptic - how many were accepted without objection and
// how many got concerns, 3) Verifier - how many were confirmed. Then it lists the verified gaps and every
// Skeptic assessment (concerns and the suggested safer wording). If none passed it says so.
export default function ResearchGapLab({ researchGap, theme = 'light' }) {
  const light = theme === 'light';
  const muted = light ? 'text-slate-500' : 'text-slate-400';
  const strong = light ? 'text-slate-900' : 'text-slate-100';
  const card = light ? 'bg-white border-slate-200' : 'bg-slate-900 border-slate-800';

  const verified = Array.isArray(researchGap?.verified_gaps) ? researchGap.verified_gaps
    : Array.isArray(researchGap) ? researchGap : [];
  const proposed = researchGap?.proposed_gaps || [];
  const challenged = researchGap?.challenged_gaps || [];
  const survived = challenged.filter((g) => g.challenge_valid === true).length;
  const rejected = challenged.filter((g) => g.challenge_valid === false).length;

  const header = (
    <div className={`px-6 py-4 border-b ${light ? 'border-slate-200' : 'border-slate-800'}`}>
      <h2 className={`text-base font-semibold ${strong}`}>Research gaps</h2>
      <p className={`text-xs mt-0.5 ${muted}`}>Adversarial triad: candidates are proposed, challenged by a Skeptic and confirmed by a Grounded Verifier</p>
    </div>
  );

  if (!researchGap) {
    return (
      <div>
        {header}
        <div className="min-h-[380px] flex flex-col items-center justify-center text-center px-6">
          <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${light ? 'bg-indigo-50 text-indigo-600' : 'bg-indigo-500/15 text-indigo-300'}`}>
            <Icon name="scale" size={24} />
          </div>
          <h3 className={`mt-4 text-base font-semibold ${strong}`}>No research gaps yet</h3>
          <p className={`mt-1.5 text-sm max-w-md ${muted}`}>Generate a review to see the proposed, challenged and verified research gaps.</p>
        </div>
      </div>
    );
  }

  // Draws one numbered stage card (Proposer, Skeptic or Verifier) with its title, big number and a note.
  const stage = (n, title, value, note, tone) => (
    <div className={`rounded-lg border p-4 ${card}`}>
      <div className="flex items-center gap-2">
        <span className={`w-6 h-6 rounded-full text-xs font-semibold flex items-center justify-center ${tone}`}>{n}</span>
        <span className={`text-sm font-medium ${strong}`}>{title}</span>
      </div>
      <div className={`mt-3 text-2xl font-semibold tabular-nums ${strong}`}>{value}</div>
      <div className={`text-xs mt-0.5 ${muted}`}>{note}</div>
    </div>
  );

  return (
    <div>
      {header}
      <div className="p-6 space-y-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {stage(1, 'Proposer', proposed.length, 'candidate gaps from paper limitations', light ? 'bg-slate-100 text-slate-700' : 'bg-slate-800 text-slate-200')}
          {stage(2, 'Skeptic', challenged.length ? `${survived} / ${challenged.length}` : '–',
            challenged.length ? `accepted without objection${rejected ? ` · concerns raised on ${rejected}` : ''}` : 'no assessment returned',
            light ? 'bg-amber-100 text-amber-800' : 'bg-amber-500/20 text-amber-200')}
          {stage(3, 'Grounded Verifier', verified.length, 'confirmed after weighing the concerns', light ? 'bg-emerald-100 text-emerald-800' : 'bg-emerald-500/20 text-emerald-200')}
        </div>

        <section>
          <h3 className={`text-sm font-semibold ${strong}`}>Verified research gaps</h3>
          {verified.length ? (
            <div className="mt-3 space-y-3">
              {verified.map((gap, idx) => (
                <article key={idx} className={`rounded-lg border p-4 ${card}`}>
                  <div className="flex items-start gap-3">
                    <span className={`mt-0.5 shrink-0 ${light ? 'text-emerald-600' : 'text-emerald-400'}`}><Icon name="checkCircle" size={18} /></span>
                    <div className="min-w-0 space-y-2">
                      <p className={`text-sm font-medium leading-relaxed ${strong}`}>{gap.gap_statement || gap.verified_gap || gap.gap}</p>
                      {gap.verification_reason && <p className={`text-sm leading-relaxed ${muted}`}>{gap.verification_reason}</p>}
                      {gap.supporting_papers?.length > 0 && (
                        <div className="flex flex-wrap gap-1.5 pt-1">
                          {gap.supporting_papers.map((p, i) => (
                            <span key={i} className={`inline-flex items-center gap-1 text-xs px-2 py-1 rounded-md ${light ? 'bg-slate-100 text-slate-700' : 'bg-slate-800 text-slate-300'}`}>
                              <Icon name="document" size={12} /> {p}
                            </span>
                          ))}
                        </div>
                      )}
                      {gap.evidence_basis?.length > 0 && (
                        <ul className={`text-xs leading-relaxed list-disc pl-4 space-y-1 ${muted}`}>
                          {gap.evidence_basis.map((ev, i) => <li key={i}>{ev}</li>)}
                        </ul>
                      )}
                    </div>
                  </div>
                </article>
              ))}
            </div>
          ) : (
            <p className={`mt-3 text-sm ${muted}`}>
              No candidate passed verification for this run, so the review discusses only the limitations reported by the papers.
            </p>
          )}
        </section>

        {challenged.length > 0 && (
          <section>
            <h3 className={`text-sm font-semibold ${strong}`}>Skeptic assessments</h3>
            <p className={`text-xs mt-0.5 ${muted}`}>The Verifier makes the final decision: it confirms a candidate only when the cited evidence answers the Skeptic's concerns.</p>
            <div className={`mt-3 divide-y rounded-lg border ${light ? 'divide-slate-200 border-slate-200' : 'divide-slate-800 border-slate-800'}`}>
              {challenged.map((ch, idx) => {
                const verdict = ch.challenge_valid === true ? 'No objection' : ch.challenge_valid === false ? 'Concerns raised' : 'Not reviewed';
                const tone = ch.challenge_valid === true
                  ? (light ? 'bg-emerald-50 text-emerald-700 ring-emerald-200' : 'bg-emerald-500/15 text-emerald-300 ring-emerald-500/30')
                  : ch.challenge_valid === false
                    ? (light ? 'bg-amber-50 text-amber-800 ring-amber-200' : 'bg-amber-500/15 text-amber-200 ring-amber-500/30')
                    : (light ? 'bg-slate-100 text-slate-600 ring-slate-200' : 'bg-slate-800 text-slate-300 ring-slate-700');
                return (
                  <div key={idx} className="px-4 py-3">
                    <div className="flex items-start justify-between gap-3">
                      <p className={`text-sm leading-relaxed ${strong}`}>{ch.gap_statement}</p>
                      <span className={`shrink-0 text-xs font-medium px-2 py-0.5 rounded-full ring-1 ${tone}`}>{verdict}</span>
                    </div>
                    {ch.concerns?.some((cn) => String(cn).trim()) && (
                      <ul className={`mt-1.5 text-xs leading-relaxed list-disc pl-4 space-y-0.5 ${muted}`}>
                        {ch.concerns.filter((cn) => String(cn).trim()).map((cn, i) => <li key={i}>{cn}</li>)}
                      </ul>
                    )}
                    {ch.revised_gap_statement && ch.revised_gap_statement !== ch.gap_statement && (
                      <p className={`mt-1.5 text-xs ${muted}`}><span className="font-medium">Suggested revision: </span>{ch.revised_gap_statement}</p>
                    )}
                  </div>
                );
              })}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
