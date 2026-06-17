"""db-memory — a vector-DB MCP server for conversational memory.

Stores solved problems + their solutions as vectors and resurfaces the most
relevant ones when a new request looks similar to something solved before.

Switch the vector store with one env var — no code change:
    VECTOR_BACKEND=local   -> Chroma, embedded on disk (default, offline)
    VECTOR_BACKEND=cloud   -> Qdrant Cloud (needs QDRANT_URL + QDRANT_API_KEY)

Embeddings run locally with a small, fast model (no API key, no network).
"""

import os
import uuid

from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------- config
BACKEND = os.environ.get("VECTOR_BACKEND", "local").lower()
COLLECTION = os.environ.get("COLLECTION_NAME", "solved_issues")
# Small + fast: 384-dim, ~90MB, runs on CPU in milliseconds per query.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "memory_db"))
# Cosine similarity in [0, 1]; matches below this are dropped as irrelevant.
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.30"))

mcp = FastMCP("db-memory")

# ---------------------------------------------------------------- embeddings
# Load lazily so `claude mcp add` / --help don't pay the model load.
_model: SentenceTransformer | None = None


def _embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed(text: str) -> list[float]:
    # normalize_embeddings=True -> vectors are unit length, so cosine works cleanly.
    return _embedder().encode(text, normalize_embeddings=True).tolist()


def embed_dim() -> int:
    return _embedder().get_sentence_embedding_dimension()


# ---------------------------------------------------------------- stores
# Each store returns query hits as (problem, solution, similarity) where
# similarity is cosine similarity in [0, 1] (higher = closer).


class ChromaStore:
    """Local, embedded, file-based. No server, no account."""

    def __init__(self) -> None:
        import chromadb

        client = chromadb.PersistentClient(path=DB_PATH)
        self.col = client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def add(self, vector, problem, solution) -> str:
        doc_id = str(uuid.uuid4())
        self.col.add(
            ids=[doc_id],
            embeddings=[vector],
            documents=[solution],
            metadatas=[{"problem": problem}],
        )
        return doc_id

    def query(self, vector, top_k):
        res = self.col.query(query_embeddings=[vector], n_results=top_k)
        if not res["ids"] or not res["ids"][0]:
            return []
        out = []
        for meta, solution, dist in zip(
            res["metadatas"][0], res["documents"][0], res["distances"][0]
        ):
            out.append((meta["problem"], solution, 1.0 - dist))  # cosine dist -> sim
        return out

    def count(self) -> int:
        return self.col.count()


class QdrantStore:
    """Managed cloud cluster. Shared across machines."""

    def __init__(self, dim: int) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        url = os.environ.get("QDRANT_URL")
        if not url:
            raise RuntimeError(
                "VECTOR_BACKEND=cloud requires QDRANT_URL (and usually QDRANT_API_KEY)."
            )
        self.client = QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"))
        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(
                COLLECTION,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def add(self, vector, problem, solution) -> str:
        from qdrant_client.models import PointStruct

        doc_id = str(uuid.uuid4())
        self.client.upsert(
            COLLECTION,
            points=[
                PointStruct(
                    id=doc_id,
                    vector=vector,
                    payload={"problem": problem, "solution": solution},
                )
            ],
        )
        return doc_id

    def query(self, vector, top_k):
        hits = self.client.query_points(
            COLLECTION, query=vector, limit=top_k
        ).points
        return [(h.payload["problem"], h.payload["solution"], h.score) for h in hits]

    def count(self) -> int:
        return self.client.count(COLLECTION).count


_store = None


def store():
    global _store
    if _store is None:
        if BACKEND == "cloud":
            _store = QdrantStore(embed_dim())
        else:
            _store = ChromaStore()
    return _store


# ---------------------------------------------------------------- MCP tools
@mcp.tool()
def save_memory(problem: str, solution: str) -> str:
    """Store a solved issue and its solution for future retrieval.

    Call this after you resolve a user's problem, so it can be resurfaced
    if a similar problem comes up later.

    Args:
        problem: A short description of the problem that was solved.
        solution: The solution / fix / answer, in enough detail to reuse.
    """
    doc_id = store().add(embed(problem), problem, solution)
    return f"Saved memory {doc_id} (backend={BACKEND}, total={store().count()})."


@mcp.tool()
def search_memory(query: str, top_k: int = 3) -> str:
    """Search past solved issues relevant to the current request.

    Call this when the user's question may have been solved before; use any
    returned solution as context for your answer.

    Args:
        query: The current user request or problem to look up.
        top_k: How many past solutions to retrieve (default 3).
    """
    hits = store().query(embed(query), top_k)
    blocks = [
        f"Past problem: {problem}\nSolution: {solution}\n(similarity {score:.2f})"
        for problem, solution, score in hits
        if score >= MIN_SCORE
    ]
    if not blocks:
        return "No sufficiently relevant past solutions found."
    return "\n\n---\n\n".join(blocks)


@mcp.tool()
def memory_stats() -> str:
    """Report which vector backend is active and how many memories are stored."""
    return f"backend={BACKEND}, collection={COLLECTION}, model={EMBED_MODEL}, count={store().count()}"


if __name__ == "__main__":
    mcp.run()
