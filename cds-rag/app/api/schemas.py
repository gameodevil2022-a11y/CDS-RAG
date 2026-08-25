from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        description="CDS question",
    )


class Source(BaseModel):

    source: str | None = None
    page: int | None = None


class QuestionResponse(BaseModel):

    answer: str
    sources: list[Source]