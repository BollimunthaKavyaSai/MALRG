"""
Key-finding coverage of M0-M6 against LLM-written reference reviews (10 topics).
The reference reviews/key findings were written by an LLM from the benchmark abstracts (not human).
A review covers a finding when every group of required terms has at least one hit (case-insensitive).
"""
import json, os, glob
import numpy as np

KF = {
"deep_learning_for_plant": [[["afd"],["98.7"]],[["tomato"],["inception"]],[["278"]],[["plantdoc"]],[["hyperparameter"]],[["time-consuming","error-prone","non-destructive"]]],
"federated_learning": [[["96.1"],["1.9"]],[["flops","mlops"]],[["trade-off","tradeoff"],["image"]],[["homomorphic"]],[["fairness"],["iid","heterogene"]],[["hipaa","regulat"]]],
"large_language_model_based": [[["macnet","directed acyclic","dag"],["thousand"]],[["scaling law","logistic"]],[["survey"],["communicat"]],[["incident"],["backdoors","centralized"]],[["commllm","6g"]],[["memory"],["reflection"]]],
"artificial_intelligence_in_edu": [[["history","historical"]],[["generative"],["question generation","feedback"]],[["stem"],["physics"]],[["personalized learning","personalised learning"],["time management"]],[["workload"]],[["pedagogical"],["bias"]]],
"explainable_artificial_intelli": [[["systematic"]],[["regulat"]],[["end-to-end"]],[["darpa"]],[["helpfulness","helpful"]],[["safety assurance","inscrutable","trust"]]],
"graph_neural_networks_for_drug": [[["repositioning","retrosynthesis"]],[["descriptor"],["bond","atoms"]],[["11 public","11 datasets","11 benchmark","eleven"]],[["virtual screening","molecular generation"]],[["advprop"]],[["drug-drug","drug–drug"]]],
"internet_of_things_and_machine": [[["yield"],["iot"]],[["humidity"]],[["43"]],[["moisture"],["sensor"]],[["fusion"]],[["resource"],["sustainab"]]],
"multimodal_large_language_mode": [[["clinical decision"]],[["asclepius"]],[["pneumonia"]],[["pandemic"]],[["mmir","tongue"]],[["label"],["rare","unlabel"]]],
"retrieval_augmented_generation": [[["vision-language","lvlm"]],[["structured"],["workflow"]],[["rowen"]],[["mega-rag","mega"]],[["hybrid"],["bm25"]],[["irrelevant","exacerbat","external hallucination"]]],
"transformer_models_for_time_se": [[["persistence"]],[["autoformer"]],[["long-term","ltsf"],["review","survey"]],[["psformer"]],[["ister"]],[["quadratic"]]],
}
def cov(text, finds):
    t = text.lower()
    return float(np.mean([all(any(k in t for k in g) for g in f) for f in finds]))
cfgs = ["M0","M1","M2","M3","M4","M5"]
res = {c: {} for c in cfgs + ["M6"]}
for key, finds in KF.items():
    d = next(os.path.join("runs", x) for x in os.listdir("runs") if x.startswith(key))
    for c in cfgs:
        res[c][key] = cov(json.load(open(f"{d}/{c}.json", encoding="utf-8"))["review_md"], finds)
    rows = json.load(open(f"{d}/M5_eval.json", encoding="utf-8"))["sentences"]
    kept = " ".join(r["sentence"] for r in rows if not (r["is_claim"] and not r.get("any_support")))
    res["M6"][key] = cov(kept, finds)
print("config  per-topic coverage %                      mean  sd")
for c, v in res.items():
    a = np.array(list(v.values())) * 100
    print(c, [round(x) for x in a], round(a.mean(), 1), round(a.std(ddof=1), 1))
json.dump(res, open("runs/_reference_coverage.json", "w"), indent=1)
