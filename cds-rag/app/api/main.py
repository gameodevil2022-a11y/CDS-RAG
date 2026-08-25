from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag.rag_chain import (
    get_rag_chain,
    get_sources,
)


app = FastAPI(
    title="CDS Agent API",
    description="RAG-powered CDS exam assistant",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
    )


class Source(BaseModel):
    source: str | None
    page: int | str | None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]


@lru_cache(maxsize=1)
def get_chain():
    return get_rag_chain()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "cds-agent",
    }


@app.get("/api/v1/health")
def agent_health():
    return {
        "status": "ok",
        "service": "cds-agent",
        "rag": "ready",
    }


@app.post(
    "/api/v1/ask",
    response_model=AskResponse,
)
def ask(request: AskRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:

        chain = get_chain()

        answer = chain.invoke(question)

        sources = get_sources(question)

        return AskResponse(
            question=question,
            answer=answer.strip(),
            sources=[
                Source(
                    source=item.get("source"),
                    page=item.get("page"),
                )
                for item in sources
            ],
        )

    except Exception as exc:

        print(
            f"CDS Agent error: {type(exc).__name__}: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to process the question.",
        )