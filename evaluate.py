"""Run tests.json through the pipeline and write results.csv for manual scoring."""
import csv, json

import rag

tests = json.loads(open("tests.json", encoding="utf-8").read())
rows = []
for t in tests:
    res = rag.answer(t["question"])
    top = res["sources"][0]["file"] if res["sources"] else None
    rows.append({
        "question": t["question"],
        "should_answer": t["answerable"],
        "abstained": res["abstained"],
        "fallback_correct": res["abstained"] != t["answerable"],
        "top_source": top,
        "source_correct": (top == t["expected_source"]) if t["answerable"] else "",
        "latency_s": res["latency_s"],
        "answer": res["answer"],
        "relevance_1to5": "", "factual_consistency_1to5": "", "completeness_1to5": "",  # fill by hand
    })
    print(("OK  " if rows[-1]["fallback_correct"] else "FAIL"), t["question"])

with open("results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print("Saved results.csv - add your manual scores, then use them in your report.")
