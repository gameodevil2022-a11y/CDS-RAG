from pathlib import Path

import fitz

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from app.rag.vector_store import (
    get_vector_store,
)


def load_pdf(
    pdf_path: str,
) -> list[Document]:

    pdf = fitz.open(pdf_path)

    documents = []

    for page_number, page in enumerate(
        pdf
    ):

        text = page.get_text(
            "text"
        ).strip()

        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(
                        pdf_path
                    ),
                    "page": page_number + 1,
                    "document_type": "pdf",
                    "exam": "CDS",
                    "content_type": "study_material",
                },
            )
        )

    pdf.close()

    print(
        f"Loaded pages: {len(documents)}"
    )

    return documents


def split_documents(
    documents: list[Document],
) -> list[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"Created chunks: {len(chunks)}"
    )

    return chunks


def ingest_pdf(
    pdf_path: str,
):

    documents = load_pdf(
        pdf_path
    )

    chunks = split_documents(
        documents
    )

    vector_store = get_vector_store()

    vector_store.add_documents(
        documents=chunks
    )

    print(
        f"Indexed {len(chunks)} chunks."
    )


if __name__ == "__main__":

    pdf = Path(
        "data/raw/cds_history.pdf"
    )

    if not pdf.exists():

        raise FileNotFoundError(
            f"PDF not found: {pdf}"
        )

    ingest_pdf(
        str(pdf)
    )