from app.rag.vector_store import get_vector_store


def search(
    query: str,
    k: int = 20,
):

    vector_store = get_vector_store()

    results = vector_store.similarity_search_with_score(
        query=query,
        k=k,
    )

    return results


def search_study_material(
    query: str,
    k: int = 20,
):

    vector_store = get_vector_store()

    results = vector_store.similarity_search_with_score(
        query=query,
        k=k,
        filter={
            "must": [
                {
                    "key": "metadata.content_type",
                    "match": {
                        "value": "study_material"
                    },
                }
            ]
        },
    )

    return results


if __name__ == "__main__":

    query = input(
        "\nEnter your CDS question: "
    ).strip()

    if not query:
        raise ValueError(
            "Question cannot be empty."
        )

    results = search_study_material(
        query,
        k=20,
    )

    print("\n" + "=" * 100)
    print(f"QUERY: {query}")
    print("=" * 100)

    for i, (document, score) in enumerate(
        results,
        start=1,
    ):

        print(f"\nRESULT {i}")
        print("-" * 100)

        print(f"SCORE: {score}")

        print(
            f"CONTENT TYPE: "
            f"{document.metadata.get('content_type')}"
        )

        print("\nCONTENT:")
        print(document.page_content)

        print("\nMETADATA:")
        print(document.metadata)