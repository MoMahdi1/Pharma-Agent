import json
from .agent import validate_grounding, ask_question
from .schema import RESPONSE, validate_data


def evaluate(provider, store, cases, top_k=4, min_score=0.35):
    results = []
    for case in cases:
        try:
            hits = []
            answer = ask_question(case["query"], provider, store, top_k, min_score, retrieved=hits)
            validate_data(answer, RESPONSE)
            ids = {h["source_id"] for h in hits}
            expected = set(case["expected_sources"])
            text = " ".join(c['text'] for c in answer[case["field"]]).lower()
            claims = answer['summary'] + answer['risks'] + answer['interactions']
            checks = {
                "structured_validity":True,
                "expected_status":answer['status'] == case["status"],
                "source_recall_at_k":len(ids & expected)/len(expected) if expected else None,
                "quote_and_citation_validity":validate_grounding(answer, hits),
                "expected_content":all(t.lower() in text for t in case["required_terms"]),
                "abstention_has_no_claims":not claims if case["status"] != "answered" else True,
            }
            passed = all(v for k,v in checks.items() if k != "source_recall_at_k") and (checks["source_recall_at_k"] in (None, 1.0))
            results.append({"id":case["id"],"query":case["query"],"passed":passed,"checks":checks,"retrieved":hits,"answer":answer,"human_faithfulness_review":"pending"})
        except Exception as exc:
            # Do not serialize provider exceptions: they can contain request details.
            results.append({"id":case["id"],"passed":False,"error_type":type(exc).__name__})
    return {"mode":"live_gemini","passed":sum(r["passed"] for r in results),"total":len(results),"cases":results,"note":"Automatic citation checks are not semantic faithfulness proof. Review each claim against its source manually."}
