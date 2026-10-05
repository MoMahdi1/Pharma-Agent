import argparse
import json
import os
import sys
from pathlib import Path
from .agent import ask_question
from .provider import connect_gemini
from .retrieval import open_store, index_documents, check_index, close_store, load_documents
from .evaluation import evaluate
from .errors import error_message

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Gemini Pharma Drug Research Assistant")
    parser.add_argument("command", choices=["index","ask","demo","evaluate"])
    parser.add_argument("query", nargs="?")
    parser.add_argument("--db", type=Path, default=ROOT / "chroma_db")
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation-results.json")
    args = parser.parse_args()
    store = None
    try:
        provider = connect_gemini()
        docs = load_documents(ROOT / "knowledge")
        store = open_store(args.db)
        if args.command == "index":
            index_documents(store, docs, provider)
            print(f"Indexed {len(docs)} section chunks with {provider['embedding_model']}.")
            return 0
        check_index(store, docs, provider)
        k = int(os.getenv("RETRIEVAL_TOP_K","4"))
        threshold = float(os.getenv("RETRIEVAL_MIN_SCORE","0.35"))
        if k < 1 or not -1 <= threshold <= 1:
            raise ValueError("Invalid retrieval settings.")
        if args.command == "ask":
            if not args.query:
                parser.error("ask requires a quoted query")
            print(json.dumps(ask_question(args.query, provider, store, k, threshold), indent=2))
        else:
            cases = json.loads((ROOT / "evaluation/cases.json").read_text())
            if args.command == "demo":
                for case in cases[:3] + [cases[4]]:
                    print(json.dumps({"query":case["query"],"response":ask_question(case["query"], provider, store, k, threshold)}, indent=2))
            else:
                report = evaluate(provider, store, cases, k, threshold)
                args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(f"{report['passed']}/{report['total']} automatic cases passed; report: {args.output}")
                return 0 if report["passed"] == report["total"] else 1
        return 0
    except Exception as exc:
        print(error_message(exc), file=sys.stderr)
        return 1
    finally:
        if store:
            close_store(store)


if __name__ == "__main__":
    raise SystemExit(main())
