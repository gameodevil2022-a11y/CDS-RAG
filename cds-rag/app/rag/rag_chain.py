from functools import lru_cache

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser

from app.rag.vector_store import get_vector_store
from app.rag.prompt import get_rag_prompt


TOP_K = 5


@lru_cache(maxsize=1)
def get_llm():

    return ChatNVIDIA(
        model="meta/llama-3.1-8b-instruct",
        temperature=0.1,
        max_tokens=512,
    )


def format_documents(documents):

    formatted = []

    for index, document in enumerate(
        documents,
        start=1,
    ):

        source = document.metadata.get(
            "source",
            "Unknown source",
        )

        page = document.metadata.get(
            "page",
            "Unknown page",
        )

        formatted.append(
            f"""
--- CONTEXT {index} ---

SOURCE: {source}
PAGE: {page}

CONTENT:
{document.page_content}
"""
        )

    return "\n\n".join(formatted)


@lru_cache(maxsize=256)
def retrieve_documents(question):

    vector_store = get_vector_store()

    return tuple(
        vector_store.similarity_search(
            question,
            k=TOP_K,
        )
    )


def get_rag_chain():

    prompt = get_rag_prompt()
    llm = get_llm()

    def retrieve(question):

        documents = retrieve_documents(
            question
        )

        return format_documents(
            documents
        )

    chain = (
        {
            "context": retrieve,
            "question": lambda x: x,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def get_sources(question):

    documents = retrieve_documents(
        question
    )

    sources = []

    seen = set()

    for document in documents:

        source = document.metadata.get(
            "source"
        )

        page = document.metadata.get(
            "page"
        )

        key = (
            source,
            page,
        )

        if key in seen:
            continue

        seen.add(key)

        sources.append(
            {
                "source": source,
                "page": page,
            }
        )

    return sources