import os
from functools import lru_cache
from functools import lru_cache
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings


load_dotenv()


COLLECTION_NAME = "cds_knowledge"
NVIDIA_COLLECTION_NAME = "cds_knowledge_nvidia"


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url:
        raise RuntimeError(
            "QDRANT_URL is missing"
        )

    if not api_key:
        raise RuntimeError(
            "QDRANT_API_KEY is missing"
        )

    return QdrantClient(
        url=url,
        api_key=api_key,
    )


@lru_cache(maxsize=1)
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
        encode_kwargs={
            "normalize_embeddings": True
        },
    )


@lru_cache(maxsize=1)
def get_nvidia_embeddings():

    api_key = os.getenv(
        "NVIDIA_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is missing"
        )

    return NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        api_key=api_key,
    )


@lru_cache(maxsize=1)
def get_vector_store():

    client = get_qdrant_client()
    embeddings = get_embeddings()

    vector_size = len(
        embeddings.embed_query(
            "sample text"
        )
    )

    if not client.collection_exists(
        COLLECTION_NAME
    ):

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

        print(
            f"Created Qdrant collection: "
            f"{COLLECTION_NAME}"
        )

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )


@lru_cache(maxsize=1)
def get_nvidia_vector_store():

    client = get_qdrant_client()
    embeddings = get_nvidia_embeddings()

    vector_size = len(
        embeddings.embed_query(
            "sample text"
        )
    )

    if not client.collection_exists(
        NVIDIA_COLLECTION_NAME
    ):

        client.create_collection(
            collection_name=NVIDIA_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

        print(
            f"Created Qdrant collection: "
            f"{NVIDIA_COLLECTION_NAME}"
        )

    return QdrantVectorStore(
        client=client,
        collection_name=NVIDIA_COLLECTION_NAME,
        embedding=embeddings,
    )


def recreate_collection():

    client = get_qdrant_client()

    if client.collection_exists(
        COLLECTION_NAME
    ):

        print(
            f"Deleting existing collection: "
            f"{COLLECTION_NAME}"
        )

        client.delete_collection(
            collection_name=COLLECTION_NAME
        )

        print(
            "Existing collection deleted."
        )

    embeddings = get_embeddings()

    vector_size = len(
        embeddings.embed_query(
            "sample text"
        )
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )

    print(
        f"Created fresh collection: "
        f"{COLLECTION_NAME}"
    )

    # The cached vector store may point to
    # the old collection state, so clear it.
    get_vector_store.cache_clear()