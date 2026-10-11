"""
Import human-written literature reviews and evaluate them with the same metrics as the systems.

For each topic put the review as  human_reviews/<topic-slug>.md  using the same [n] citation numbers as the
5 benchmark papers of that topic (see data/benchmark/<topic-slug>.json, ref_id 1..5) and a final
"References" section. Then:

    python make_human_runs.py            # creates runs/<slug>/H0.json and evaluates it (NLI metrics)
"""
import json
import time

import config
from evaluation import runner
from malrg.retrieval import slugify

for topic in config.TOPICS:
    src = config.ROOT / "human_reviews" / f"{slugify(topic)}.md" if hasattr(config, "ROOT") else None
    from pathlib import Path
    src = Path(__file__).parent / "human_reviews" / f"{slugify(topic)}.md"
    if not src.exists():
        print("missing:", src.name)
        continue
    d = config.RUNS_DIR / slugify(topic)
    rec = {"topic": topic, "config_id": "H0", "config_name": "H0: Human-written review",
           "agent_chain": ["human"], "review_md": src.read_text(encoding="utf-8"), "n_papers": 5,
           "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (d / "H0.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    res = runner.evaluate_run(topic, "H0", log=print, use_judge=False)
    print(topic, {k: round(res["nli"][k], 3) for k in ("citation_groundedness", "hallucination_rate")})
