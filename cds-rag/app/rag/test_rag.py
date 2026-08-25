from app.rag.rag_chain import (
    get_rag_chain,
    get_sources,
)


def main():

    question = input(
        "\nEnter your CDS question: "
    ).strip()

    if not question:

        print(
            "Question cannot be empty."
        )

        return

    print(
        "\nGenerating answer...\n"
    )

    chain = get_rag_chain()

    answer = chain.invoke(
        question
    )

    sources = get_sources(
        question
    )

    print("=" * 80)
    print("ANSWER")
    print("=" * 80)

    print(answer)

    print("\n")
    print("=" * 80)
    print("SOURCES")
    print("=" * 80)

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


if __name__ == "__main__":
    main()