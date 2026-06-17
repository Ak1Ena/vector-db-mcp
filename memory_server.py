"""db-memory — a vector-DB MCP server for conversational memory.

Stores solved problems + their solutions as vectors and resurfaces the most
relevant ones when a new request looks similar to something solved before.

Switch the vector store with one env var — no code change:
    VECTOR_BACKEND=local   -> Chroma, embedded on disk (default, offline)
    VECTOR_BACKEND=cloud   -> Qdrant Cloud (needs QDRANT_URL + QDRANT_API_KEY)

Embeddings run locally with a small, fast model (no API key, no network).

save_memory is FIRE-AND-FORGET: it queues the write to a background worker and
returns instantly, so the model never blocks on a save. If a background write
fails, the failure (with the topic that failed) is logged to stderr AND
surfaced on the next tool call. search_memory stays synchronous — the model
needs the results.
"""

import os
import queue
import sys
import threading
import time
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

mcp = FastMCP(
    "db-memory",
    instructions=(
        "Long-term memory for solved problems and reusable knowledge. Use this "
        "instead of writing notes or Memory/knowledge files.\n"
        "BEFORE starting a task: call search_memory with the user's request. It "
        "returns headers (id + title + similarity) only; then call "
        "get_memory(ids=[...]) to fetch full text for just the entries you need.\n"
        "AFTER solving a problem or producing reusable knowledge: call save_memory "
        "(problem = what was solved, solution = the fix in reusable detail). It is "
        "fire-and-forget and reports any failed write on the next call.\n"
        "Only store durable, reusable knowledge — not conversation-specific details."
    ),
)

# One lock serializes all embed + store I/O across the main thread (search) and
# the background save worker, so the model and the DB are never touched concurrently.
_lock = threading.Lock()

# ---------------------------------------------------------------- embeddings
_model: SentenceTransformer | None = None


def _embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed(text: str) -> list[float]:
    # normalize_embeddings=True -> unit vectors, so cosine works cleanly.
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
        for doc_id, meta, solution, dist in zip(
            res["ids"][0], res["metadatas"][0], res["documents"][0], res["distances"][0]
        ):
            out.append((doc_id, meta["problem"], solution, 1.0 - dist))  # cosine dist -> sim
        return out

    def get(self, ids):
        res = self.col.get(ids=ids)
        return list(zip(res["ids"], (m["problem"] for m in res["metadatas"]), res["documents"]))

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
        hits = self.client.query_points(COLLECTION, query=vector, limit=top_k).points
        return [(str(h.id), h.payload["problem"], h.payload["solution"], h.score) for h in hits]

    def get(self, ids):
        recs = self.client.retrieve(COLLECTION, ids=ids, with_payload=True)
        return [(str(r.id), r.payload["problem"], r.payload["solution"]) for r in recs]

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


# ---------------------------------------------------------------- background saves
_save_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
_failures: list[str] = []  # topics whose background write failed
_failures_lock = threading.Lock()
_worker_started = False
_worker_lock = threading.Lock()


def _worker() -> None:
    while True:
        problem, solution = _save_queue.get()
        try:
            with _lock:
                store().add(embed(problem), problem, solution)
        except Exception as e:  # noqa: BLE001 — must not kill the worker
            note = f"{problem!r}: {type(e).__name__}: {e}"
            with _failures_lock:
                _failures.append(note)
            print(f"[db-memory] BACKGROUND SAVE FAILED for {note}", file=sys.stderr, flush=True)
        finally:
            _save_queue.task_done()


def _ensure_worker() -> None:
    global _worker_started
    if not _worker_started:
        with _worker_lock:
            if not _worker_started:
                threading.Thread(target=_worker, name="db-memory-save", daemon=True).start()
                _worker_started = True


def _drain_failures() -> str:
    """Pop any recorded background-save failures so the model sees them on the
    next tool call. Returns a prefix string (empty if none)."""
    with _failures_lock:
        if not _failures:
            return ""
        lines = "\n".join(f"  - {f}" for f in _failures)
        _failures.clear()
    return (
        "⚠️ BACKGROUND SAVE FAILED for the following topic(s) — they were "
        f"NOT stored:\n{lines}\nConsider re-saving them.\n\n"
    )


def _flush_on_exit() -> None:
    # Give queued writes a few seconds to finish when the process is shutting down.
    end = time.time() + 5.0
    while not _save_queue.empty() and time.time() < end:
        time.sleep(0.05)


import atexit  # noqa: E402

atexit.register(_flush_on_exit)


# ---------------------------------------------------------------- MCP tools
@mcp.tool()
def save_memory(problem: str, solution: str) -> str:
    """Store a solved issue and its solution for future retrieval.

    Call this after you resolve a user's problem, so it can be resurfaced
    if a similar problem comes up later. This returns immediately; the write
    happens in the background. If a prior background write failed, this call
    reports it (with the failed topic).

    Args:
        problem: A short description of the problem that was solved.
        solution: The solution / fix / answer, in enough detail to reuse.
    """
    _ensure_worker()
    _save_queue.put((problem, solution))
    return (
        _drain_failures()
        + f"Queued '{problem[:60]}' for background save (backend={BACKEND}). "
        "It will be stored shortly; any failure is reported on the next call."
    )


@mcp.tool()
def search_memory(query: str, top_k: int = 5) -> str:
    """Search past solved issues — returns lightweight HEADERS only (id +
    problem title + similarity), NOT the full solutions, to save tokens.

    Read the headers, then call get_memory(ids=[...]) to fetch the full
    solution(s) for ONLY the ones you actually need.

    Args:
        query: The current user request or problem to look up.
        top_k: How many headers to return (default 5).
    """
    prefix = _drain_failures()
    with _lock:
        hits = store().query(embed(query), top_k)
    headers = [
        f"[{doc_id}] {problem[:120]}  (similarity {score:.2f})"
        for doc_id, problem, _solution, score in hits
        if score >= MIN_SCORE
    ]
    if not headers:
        return prefix + "No sufficiently relevant past solutions found."
    return (
        prefix
        + "Matches (headers only — call get_memory with the ids you want):\n"
        + "\n".join(headers)
    )


@mcp.tool()
def get_memory(ids: list[str]) -> str:
    """Fetch the full problem + solution text for memory ids from search_memory.

    Args:
        ids: One or more memory ids (the [id] shown in search_memory headers).
    """
    prefix = _drain_failures()
    with _lock:
        rows = store().get(ids)
    if not rows:
        return prefix + "No memories found for those ids."
    return prefix + "\n\n---\n\n".join(
        f"[{doc_id}] Past problem: {problem}\nSolution: {solution}"
        for doc_id, problem, solution in rows
    )


@mcp.tool()
def memory_stats() -> str:
    """Report backend, stored count, pending background writes, and failed saves."""
    prefix = _drain_failures()
    with _lock:
        count = store().count()
    with _failures_lock:
        failed = len(_failures)
    return (
        prefix
        + f"backend={BACKEND}, collection={COLLECTION}, model={EMBED_MODEL}, "
        f"count={count}, pending_writes={_save_queue.qsize()}, failed_saves={failed}"
    )


if __name__ == "__main__":
    mcp.run()
