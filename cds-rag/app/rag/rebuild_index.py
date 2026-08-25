from pathlib import Path

from app.rag.vector_store import (
    recreate_collection,
)
from app.rag.ingestion import (
    ingest_pdf,
)


def main():

    print("=" * 80)
    print("REBUILDING CDS KNOWLEDGE BASE")
    print("=" * 80)

    pdf = Path(
        "data/raw/cds_history.pdf"
    )

    if not pdf.exists():

        raise FileNotFoundError(
            f"PDF not found: {pdf}"
        )

    recreate_collection()

    ingest_pdf(
        str(pdf)
    )

    print("=" * 80)
    print("REBUILD COMPLETE")
    print("=" * 80)


if __name__ == "__main__":

    main()