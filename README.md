# db-memory

A vector-DB MCP server that gives Claude **conversational memory**: it stores
solved problems + solutions as vectors and resurfaces the most relevant ones
when a new request looks similar to something solved before.

- **Switchable backend** — `VECTOR_BACKEND=local|cloud`, no code change.
  - `local` → [Chroma](https://www.trychroma.com), embedded on disk. Offline, no account.
  - `cloud` → [Qdrant Cloud](https://qdrant.tech), managed, shared across machines.
- **Small/fast local embeddings** — `all-MiniLM-L6-v2` (384-dim). No API key, runs on CPU in ms.

## Tools exposed

| Tool | What it does |
|------|--------------|
| `search_memory(query, top_k)` | Find past solved issues similar to the current request |
| `save_memory(problem, solution)` | Store a solved issue for future retrieval |
| `memory_stats()` | Show active backend + how many memories are stored |

## Setup

```bash
cd ~/code/mcp/db-memory
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # edit if you want cloud
```

### Run / test standalone

```bash
mcp dev memory_server.py    # opens the MCP Inspector UI
```

### Register with Claude Code

Local backend (default):

```bash
claude mcp add db-memory -- \
  ~/code/mcp/db-memory/.venv/bin/python ~/code/mcp/db-memory/memory_server.py
```

Then `/mcp` in Claude Code shows the three tools. Claude will call
`search_memory` when a request resembles a past one, and `save_memory` after
solving something.

## Switching to cloud

1. Create a free cluster at [cloud.qdrant.io](https://cloud.qdrant.io), copy the URL + API key.
2. In `.env` (or the MCP env):

   ```bash
   VECTOR_BACKEND=cloud
   QDRANT_URL=https://YOUR-CLUSTER.cloud.qdrant.io:6333
   QDRANT_API_KEY=...
   ```

Same embeddings, same tools — only the storage moves. (The two backends don't
share data; re-save or migrate if you switch with existing memories.)

## Notes

- **Embedding dimension is fixed by the model** (384 for MiniLM). If you change
  `EMBED_MODEL`, delete the old Chroma `memory_db/` or use a fresh Qdrant
  collection — vectors of different sizes can't mix.
- An MCP tool only runs when Claude invokes it; it can't passively log every
  message. For guaranteed full capture, log each turn to the same DB from your
  app (or a Claude Code `Stop` hook) and keep this server for retrieval.
