# db-memory

A vector-DB MCP server that gives Claude **conversational memory**: it stores
solved problems + solutions as vectors and resurfaces the most relevant ones
when a new request looks similar to something solved before.

- **Switchable backend** — `VECTOR_BACKEND=local|cloud`, no code change.
  - `local` → [Chroma](https://www.trychroma.com), embedded on disk. Offline, no account.
  - `cloud` → [Qdrant Cloud](https://qdrant.tech), managed, shared across machines.
- **Small/fast local embeddings** — `all-MiniLM-L6-v2` (384-dim). No API key, runs on CPU in ms.
- **Optional web dashboard** — flip on `WEB_UI=on` to browse the store in your browser. Off by default.

## Tools exposed

| Tool | What it does |
|------|--------------|
| `search_memory(query, top_k)` | Find past solved issues — returns lightweight **headers** (id + title + similarity) to save tokens |
| `get_memory(ids)` | Fetch the full problem + solution text for the ids you actually want |
| `save_memory(problem, solution)` | Store a solved issue for future retrieval (fire-and-forget background write). Claude confirms with you — "Did this solve your problem?" — before saving |
| `update_memory(id, problem?, solution?)` | Edit a memory in place; re-embeds if the problem text changes |
| `delete_memory(ids)` | Permanently remove out-of-date or wrong entries |
| `memory_stats()` | Show active backend + how many memories are stored |

Retrieval is **two-stage**: `search_memory` returns only headers, then you call
`get_memory` for the few you need — so a search never dumps every full solution
into the context.

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

Then `/mcp` in Claude Code shows the tools. Claude will call `search_memory`
when a request resembles a past one, and `save_memory` after solving something.

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

## Web dashboard (optional)

A read-only page for browsing what's in the store — searchable, auto-refresh,
shows every memory. **Off by default**; you turn it on and pick the port purely
with env vars. When on, it **autostarts with the MCP server** (which Claude Code
launches), running in a daemon thread — no extra process, no new dependencies
(Python stdlib only).

Set these in the `env` block where the server is registered (e.g. the
`db-memory` entry in `~/.claude.json`), then restart Claude Code:

```json
"env": { "WEB_UI": "on", "WEB_PORT": "8765", "WEB_HOST": "127.0.0.1" }
```

| Env | Default | Meaning |
|-----|---------|---------|
| `WEB_UI` | `off` | `1`/`on`/`true`/`yes` enables it; anything else keeps it off |
| `WEB_PORT` | `8765` | Port to serve on |
| `WEB_HOST` | `127.0.0.1` | Bind address — localhost-only by default; set `0.0.0.0` only to expose on your network |

Then open **http://localhost:8765**. The page is a plain template at
`web/index.html` that you can edit freely; the server just serves it plus two
read-only JSON endpoints it calls (`/api/stats`, `/api/memories`).

## Notes

- **Embedding dimension is fixed by the model** (384 for MiniLM). If you change
  `EMBED_MODEL`, delete the old Chroma `memory_db/` or use a fresh Qdrant
  collection — vectors of different sizes can't mix.
- An MCP tool only runs when Claude invokes it; it can't passively log every
  message. For guaranteed full capture, log each turn to the same DB from your
  app (or a Claude Code `Stop` hook) and keep this server for retrieval.
