"""
LangGraph orchestration. One graph is compiled per ablation configuration; the
configuration flags decide which agents are wired in. Every configuration of a
topic uses the same frozen source set and the same writer model and prompt, so
differences in the output are attributable to the removed component only.
"""

import json
import time
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

import config
from malrg import agents, llm
from malrg.retrieval import build_benchmark


class RunState(TypedDict, total=False):
    topic: str
    config_id: str
    papers: List[Dict[str, Any]]
    analyses: List[Dict[str, Any]]
    cross: Optional[Dict[str, Any]]
    groups: Optional[List[Dict[str, Any]]]
    gap_result: Optional[Dict[str, Any]]
    draft_md: str
    review_body: str
    review_md: str
    raw_writer_output: str
    feedback_history: List[Dict[str, Any]]
    timings: Dict[str, float]


def _timed(name, fn):
    def wrapper(state):
        t = time.time()
        out = fn(state) or {}
        timings = dict(state.get("timings", {}))
        timings[name] = round(time.time() - t, 2)
        out["timings"] = timings
        return out
    return wrapper


def build_graph(cfg, bench, log=print):
    papers = bench["papers"]
    topic = bench["topic"]

    def n_retrieve(state):
        return {"papers": papers if cfg["retrieval"] else []}

    def n_analyze(state):
        analyses = []
        for p in papers:
            a = agents.analyze_paper(p)
            analyses.append(a)
        log(f"    [analysis] {len(analyses)} papers, evidence fidelity "
            f"{sum(a['evidence_fidelity'] for a in analyses) / max(1, len(analyses)):.2f}")
        return {"analyses": analyses}

    def n_cross(state):
        return {"cross": agents.cross_paper_analysis(topic, state["analyses"])}

    def n_taxonomy(state):
        return {"groups": agents.taxonomy(state["analyses"])}

    def n_gaps(state):
        res = agents.research_gap_triad(topic, state["analyses"], state.get("cross"), papers,
                                        adversarial=cfg["gap_triad"])
        log(f"    [gaps] proposed={len(res['proposed'])} final={len(res['final'])} ({res['mode']})")
        return {"gap_result": res}

    def n_write(state):
        model = cfg.get("writer_model")
        prefs = ""
        if cfg.get("preferences"):
            from malrg import preferences
            store = cfg.get("preference_store") or preferences.EXPERIMENT_STORE
            if store == preferences.EXPERIMENT_STORE:
                preferences.ensure_experiment_preferences()
            prefs = preferences.preferences_block(store)
        if not cfg["retrieval"]:
            raw = agents.write_review(topic, "vanilla", model=model)
        elif not cfg["analysis"]:
            raw = agents.write_review(topic, "retrieval", papers=papers)
        else:
            gaps = (state.get("gap_result") or {}).get("final") if cfg["gaps"] else None
            raw = agents.write_review(topic, "evidence", papers=papers, analyses=state["analyses"],
                                      cross=state.get("cross"), gaps=gaps, groups=state.get("groups"),
                                      model=model, prompt_variant=cfg.get("prompt"), preferences=prefs)
        body = agents.clean_review(raw)
        return {"raw_writer_output": raw, "draft_md": body, "review_body": body}

    def n_feedback(state):
        from evaluation.nli import PassageIndex, get_verifier
        index = PassageIndex(papers)
        body, history = agents.refine_review(topic, state["review_body"], papers, state.get("analyses"),
                                             index, get_verifier("critic"), log=log)
        return {"review_body": body, "feedback_history": history}

    def n_cite(state):
        body = state["review_body"]
        if cfg["retrieval"]:
            md = f"# Literature Review: {topic}\n\n{body}\n\n{agents.references_block(papers)}"
        else:
            md = f"# Literature Review: {topic}\n\n{state['raw_writer_output']}"
        return {"review_md": md}

    g = StateGraph(RunState)
    g.add_node("retrieve", _timed("retrieve", n_retrieve))
    g.add_node("write", _timed("write", n_write))
    g.add_node("cite", _timed("cite", n_cite))
    chain = ["retrieve"]
    if cfg["analysis"]:
        g.add_node("analyze", _timed("analyze", n_analyze))
        g.add_node("taxonomy", _timed("taxonomy", n_taxonomy))
        chain += ["analyze", "taxonomy"]
        if cfg["cross_paper"]:
            g.add_node("cross_paper", _timed("cross_paper", n_cross))
            chain.append("cross_paper")
        if cfg["gaps"]:
            g.add_node("gaps", _timed("gaps", n_gaps))
            chain.append("gaps")
    chain.append("write")
    if cfg["feedback"]:
        g.add_node("feedback", _timed("feedback", n_feedback))
        chain.append("feedback")
    chain.append("cite")
    g.add_edge(START, chain[0])
    for a, b in zip(chain, chain[1:]):
        g.add_edge(a, b)
    g.add_edge(chain[-1], END)
    return g.compile(), chain


def run_config(topic, config_id, log=print, force=False, cfg=None):
    from malrg.retrieval import slugify
    out_dir = config.RUNS_DIR / slugify(topic)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{config_id}.json"
    if out_file.exists() and not force:
        return json.loads(out_file.read_text(encoding="utf-8"))

    bench = build_benchmark(topic, log=log)
    cfg = cfg or config.CONFIGS[config_id]
    graph, chain = build_graph(cfg, bench, log=log)
    log(f"  [{config_id}] {cfg['name']}  ->  {' -> '.join(chain)}")
    llm.USAGE.reset()
    t0 = time.time()
    state = graph.invoke({"topic": topic, "config_id": config_id, "timings": {}})
    record = {
        "topic": topic,
        "config_id": config_id,
        "config_name": cfg["name"],
        "agent_chain": chain,
        "review_md": state["review_md"],
        "draft_md": state.get("draft_md"),
        "raw_writer_output": state.get("raw_writer_output"),
        "analyses": state.get("analyses"),
        "cross": state.get("cross"),
        "groups": state.get("groups"),
        "gap_result": state.get("gap_result"),
        "feedback_history": state.get("feedback_history"),
        "timings": state.get("timings"),
        "wall_time_s": round(time.time() - t0, 2),
        "usage": llm.USAGE.summary(),
        "n_papers": len(bench["papers"]),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_file.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / f"{config_id}_review.md").write_text(state["review_md"], encoding="utf-8")
    return record
