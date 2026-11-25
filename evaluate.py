"""Run tests.json through the pipeline and write results.csv for manual scoring."""
import csv, json

import rag

tests = json.loads(open("tests.json", encoding="utf-8").read())
rows = []
for t in tests:
    res = rag.answer(t["question"])
    files = [s["file"] for s in res["sources"]]
    top = files[0] if files else None
    exp = t["expected_source"]
    rows.append({
        "question": t["question"],
        "should_answer": t["answerable"],
        "abstained": res["abstained"],
        "fallback_correct": res["abstained"] != t["answerable"],
        "top_source": top,
        "source_correct_top1": (top == exp) if t["answerable"] else "",
        "source_in_results": (exp in files) if t["answerable"] else "",
        "latency_s": res["latency_s"],
        "answer": res["answer"],
        "relevance_1to5": "", "factual_consistency_1to5": "", "completeness_1to5": "",  # fill by hand
    })
    print(("OK  " if rows[-1]["fallback_correct"] else "FAIL"), t["question"])

with open("results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)

ans = [r for r in rows if r["should_answer"]]
una = [r for r in rows if not r["should_answer"]]
lat = sorted(r["latency_s"] for r in rows)
print("\n--- SUMMARY ---")
print(f"Answerable questions answered:   {sum(not r['abstained'] for r in ans)}/{len(ans)}")
print(f"Correct source ranked first:     {sum(r['source_correct_top1'] is True for r in ans)}/{len(ans)}")
print(f"Correct source among retrieved:  {sum(r['source_in_results'] is True for r in ans)}/{len(ans)}")
print(f"Unanswerable correctly refused:  {sum(r['abstained'] for r in una)}/{len(una)}")
print(f"Median latency: {lat[len(lat)//2]}s")
print("Saved results.csv - now read each answer and fill the three score columns by hand.")
