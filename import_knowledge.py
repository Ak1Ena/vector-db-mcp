"""Bulk-import knowledge into db-memory before any conversation.

Pre-seeds the vector store so search_memory can hit it on day one. Reuses the
same backend (local Chroma / cloud Qdrant) and embedding model as the server,
so it honors VECTOR_BACKEND and the rest of your env / .env.

Two input shapes:

  # CSV with a `problem` column and a `solution` column
  python import_knowledge.py --csv tickets.csv

  # A folder of .md / .txt docs (one memory per file, large files chunked)
  python import_knowledge.py --docs ./knowledge

Options:
  --chunk-chars N   max chars per chunk when importing docs (default 1500; 0 = no chunking)
"""

import argparse
import csv
import glob
import os

import memory_server as m  # reuses embed() + store() + the configured backend


def add(problem: str, solution: str) -> None:
    problem = problem.strip()
    solution = solution.strip()
    if not solution:
        return
    if not problem:
        problem = solution[:80]  # fall back to a snippet as the "problem" label
    m.store().add(m.embed(problem), problem, solution)


def import_csv(path: str) -> int:
    n = 0
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "solution" not in (reader.fieldnames or []):
            raise SystemExit(
                f"CSV needs at least a 'solution' column (and ideally 'problem'). "
                f"Found: {reader.fieldnames}"
            )
        for row in reader:
            add(row.get("problem", ""), row.get("solution", ""))
            n += 1
    return n


def chunk(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    # split on blank lines (paragraphs), then pack into <= max_chars chunks
    paras = [p for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if buf and len(buf) + len(p) + 2 > max_chars:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    return chunks


def title_of(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line.lstrip("# ").strip() or fallback  # first non-empty line / heading
    return fallback


def import_docs(folder: str, chunk_chars: int) -> int:
    paths = sorted(
        glob.glob(os.path.join(folder, "**", "*.md"), recursive=True)
        + glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True)
    )
    if not paths:
        raise SystemExit(f"No .md or .txt files found under {folder}")
    n = 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            continue
        base = os.path.splitext(os.path.basename(path))[0]
        doc_title = title_of(text, base)
        parts = chunk(text, chunk_chars)
        for i, part in enumerate(parts):
            label = doc_title if len(parts) == 1 else f"{doc_title} (part {i + 1}/{len(parts)})"
            add(label, part)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Bulk-import knowledge into db-memory.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="CSV file with problem,solution columns")
    src.add_argument("--docs", help="Folder of .md/.txt files to import")
    ap.add_argument("--chunk-chars", type=int, default=1500,
                    help="Max chars per chunk for docs (0 = no chunking)")
    args = ap.parse_args()

    before = m.store().count()
    if args.csv:
        rows = import_csv(args.csv)
        print(f"Imported {rows} CSV rows.")
    else:
        rows = import_docs(args.docs, args.chunk_chars)
        print(f"Imported {rows} chunks from docs.")

    print(f"Backend={m.BACKEND}  collection={m.COLLECTION}  "
          f"count: {before} -> {m.store().count()}")


if __name__ == "__main__":
    main()
