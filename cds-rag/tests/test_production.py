"""
Production test runner for the CDS RAG agent.

This runner does NOT modify the RAG system.
It sends the test questions through the existing chain and saves
the raw results for evaluation.

Run from the project root:

    python -m tests.test_production
"""

import json
from pathlib import Path

from app.rag.rag_chain import get_rag_chain
from app.rag.vector_store import get_vector_store


ROOT = Path(__file__).resolve().parents[1]

QUESTIONS_FILE = (
    ROOT / "tests" / "production_questions.json"
)

RESULTS_FILE = (
    ROOT / "tests" / "production_results.json"
)


def get_sources(question: str):

    vector_store = get_vector_store()

    documents = vector_store.similarity_search(
        question,
        k=5,
    )

    sources = []

    for document in documents:

        sources.append(
            {
                "source": document.metadata.get(
                    "source"
                ),
                "page": document.metadata.get(
                    "page"
                ),
            }
        )

    return sources


def main():

    with QUESTIONS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        tests = json.load(file)

    print("=" * 80)
    print("CDS PRODUCTION TEST SUITE")
    print("=" * 80)

    print(
        f"Total tests: {len(tests)}"
    )

    print()

    # Load the RAG chain only once.
    chain = get_rag_chain()

    results = []

    for index, test in enumerate(
        tests,
        start=1,
    ):

        question = test["question"]

        print("=" * 80)

        print(
            f"[{index}/{len(tests)}] "
            f"{question}"
        )

        print("=" * 80)

        try:

            answer = chain.invoke(
                question
            )

            if not isinstance(
                answer,
                str,
            ):

                answer = str(answer)

            sources = get_sources(
                question
            )

            result = {

                "id": test["id"],

                "category": test[
                    "category"
                ],

                "question": question,

                "expected": test[
                    "expected"
                ],

                "answer": answer,

                "sources": sources,

                "status": "COMPLETED",
            }

            print(
                "\nANSWER:"
            )

            print(answer)

            print(
                "\nSOURCES:"
            )

            seen = set()

            for source in sources:

                key = (
                    source["source"],
                    source["page"],
                )

                if key in seen:
                    continue

                seen.add(key)

                print(
                    f"- {source['source']} "
                    f"(page {source['page']})"
                )

        except Exception as error:

            result = {

                "id": test["id"],

                "category": test[
                    "category"
                ],

                "question": question,

                "expected": test[
                    "expected"
                ],

                "answer": None,

                "sources": [],

                "status": "ERROR",

                "error": repr(error),
            }

            print(
                "\nERROR:"
            )

            print(error)

        results.append(result)

        print()

    with RESULTS_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 80)
    print("PRODUCTION TEST RUN COMPLETE")
    print("=" * 80)

    print(
        f"Results saved to: "
        f"{RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()