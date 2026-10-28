"""Reviewed HTML ingestion and bounded, read-only SQLite FTS5 retrieval."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import ssl
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit
from uuid import uuid4

import httpx
from bs4 import BeautifulSoup

from schemas import Evidence, Passage, ReadInput, SearchInput

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "data/sources.json"
DEFAULT_CORPUS = ROOT / "data/knowledge.sqlite"
HOST = "www.sydneywater.com.au"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def manifest() -> list[dict]:
    sources = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not sources or len({s["id"] for s in sources}) != len(sources):
        raise ValueError("Invalid source manifest")
    for source in sources:
        validate_url(source["url"])
        if not re.fullmatch(r"[a-z0-9_]+", source["id"]):
            raise ValueError("Invalid source ID")
    return sources


def validate_url(url: str) -> None:
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname != HOST or parsed.username
            or parsed.password or parsed.port not in (None, 443) or parsed.fragment):
        raise ValueError("Source URL is not approved")


def fetch(client: httpx.Client, url: str, allowed: set[str]) -> str:
    for _ in range(4):
        validate_url(url)
        if url not in allowed:
            raise ValueError("Redirect is outside the manifest")
        with client.stream("GET", url, follow_redirects=False) as response:
            if response.is_redirect:
                url = urljoin(url, response.headers["location"])
                continue
            response.raise_for_status()
            if "text/html" not in response.headers.get("content-type", ""):
                raise ValueError("Expected an HTML page")
            chunks, size = [], 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > 2_000_000:
                    raise ValueError("Source exceeds 2 MB")
                chunks.append(chunk)
            return b"".join(chunks).decode(response.encoding or "utf-8")
    raise ValueError("Too many source redirects")


def clean_sections(html: str, source: dict, allowed: set[str]) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    canonical = soup.find("link", rel="canonical")
    if canonical and canonical.get("href"):
        target = urljoin(source["url"], canonical["href"])
        validate_url(target)
        if target not in allowed:
            raise ValueError("Canonical URL is outside the manifest")
    root = soup.select_one(source["selector"])
    if root is None:
        raise ValueError(f"Missing reviewed selector for {source['id']}")
    for element in root.select("script,style,nav,form,footer,noscript,iframe,.sw-advertisement"):
        element.decompose()
    for line_break in root.select("br"):
        line_break.replace_with(" ")
    def plain(node) -> str:
        return re.sub(r"\s+", " ", node.get_text("", strip=False)).strip()
    groups: list[tuple[str, list[str]]] = []
    heading, blocks = source["title"], []
    for node in root.select("h1,h2,h3,h4,p,ul,ol,table"):
        if node.find_parent(["ul", "ol", "table"]) is not None:
            continue
        if node.name in ("h1", "h2", "h3", "h4"):
            if blocks:
                groups.append((heading, blocks))
            heading, blocks = plain(node), []
            continue
        if node.name == "table":
            rows = [" | ".join(plain(cell) for cell in row.find_all(["th", "td"]))
                    for row in node.find_all("tr")]
            text = "\n".join(rows)
            # Large directories retain whole rows and repeat their column headings.
            if len(text) > 3500:
                header = rows[0]
                blocks.extend(header + "\n" + row for row in rows[1:] if row)
                continue
        elif node.name in ("ul", "ol"):
            text = "\n".join(f"{i}. {plain(li)}" for i, li
                in enumerate(node.find_all("li", recursive=False), 1))
        else:
            text = plain(node)
        if text:
            blocks.append(text)
    if blocks:
        groups.append((heading, blocks))
    sections, seen = [], set()
    for heading, blocks in groups:
        identity = re.sub(r"\W", "", heading + "".join(blocks)).casefold()
        if identity in seen:
            continue
        seen.add(identity)
        parts, pending = [], []
        for block in blocks:
            if len(block) > 8000:
                raise ValueError("A source block needs manual bounded sectioning")
            if pending and (len("\n\n".join(pending + [block])) > 8000
                            or len(" ".join(pending).split()) >= 450):
                parts.append(pending)
                pending = []
            pending.append(block)
        if pending:
            parts.append(pending)
        for part in parts:
            text = "\n\n".join(part)
            # Whole first paragraphs; the model must read for omitted conditions.
            excerpt_parts = [part[0]]
            for block in part[1:]:
                if len("\n\n".join(excerpt_parts + [block])) > 800:
                    break
                excerpt_parts.append(block)
            excerpt = "\n\n".join(excerpt_parts)
            if len(excerpt) > 3500:
                raise ValueError("Search excerpt needs manual paragraph sectioning")
            sections.append({"section_id": f"{source['id']}-{len(sections)+1:03d}",
                             "heading": heading, "text": text, "excerpt": excerpt})
    if not sections or sum(len(s["text"]) for s in sections) < 100:
        raise ValueError("Empty or implausibly small extraction")
    return sections


def ingest(path: Path = DEFAULT_CORPUS) -> dict:
    sources = manifest()
    allowed = {s["url"] for s in sources}
    snapshots, review = ROOT / "data/snapshots", ROOT / "data/review"
    snapshots.mkdir(parents=True, exist_ok=True)
    review.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".building.sqlite")
    temporary.unlink(missing_ok=True)
    try:
        with closing(sqlite3.connect(temporary)) as db, db, httpx.Client(timeout=15, verify=ssl.create_default_context(), headers={
                "User-Agent": "StudentInformationPOC/0.1 (public document review)"}) as client:
            db.executescript("""
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE documents (id TEXT PRIMARY KEY, topic TEXT, title TEXT,
                    url TEXT, fetched_at TEXT, reviewed_at TEXT, reviewer TEXT);
                CREATE TABLE sections (section_id TEXT PRIMARY KEY, document_id TEXT,
                    heading TEXT, text TEXT, excerpt TEXT, related TEXT);
                CREATE VIRTUAL TABLE search_index USING fts5(section_id UNINDEXED,
                    title, heading, text, tokenize='porter unicode61');
            """)
            db.execute("INSERT INTO metadata VALUES ('corpus_version', ?)", (str(uuid4()),))
            for source in sources:
                html = fetch(client, source["url"], allowed)
                sections = clean_sections(html, source, allowed)
                (snapshots / f"{source['id']}.html").write_text(html, encoding="utf-8")
                (review / f"{source['id']}.json").write_text(
                    json.dumps(sections, indent=2, ensure_ascii=False), encoding="utf-8")
                db.execute("INSERT INTO documents VALUES (?,?,?,?,?,?,?)",
                    (source["id"], source["topic"], source["title"], source["url"], now(), "", ""))
                for i, section in enumerate(sections):
                    related = [s["section_id"] for s in sections[max(0,i-1):i+2]
                               if s["section_id"] != section["section_id"]]
                    db.execute("INSERT INTO sections VALUES (?,?,?,?,?,?)",
                        (section["section_id"], source["id"], section["heading"],
                         section["text"], section["excerpt"], json.dumps(related)))
                    db.execute("INSERT INTO search_index VALUES (?,?,?,?)",
                        (section["section_id"], source["title"], section["heading"], section["text"]))
                time.sleep(0.3)
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Index integrity check failed")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"documents": len(sources), "status": "awaiting_review"}


def connection(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def readiness(path: Path = DEFAULT_CORPUS) -> dict:
    with closing(connection(path)) as db:
        rows = db.execute("SELECT * FROM documents").fetchall()
        approved = {s["id"]: s["url"] for s in manifest()}
        if not rows:
            raise ValueError("Knowledge is unavailable")
        for row in rows:
            if approved.get(row["id"]) != row["url"]:
                raise ValueError("Index sources differ from the approved manifest")
            if not row["reviewed_at"]:
                raise ValueError("Knowledge awaits source review")
            age = datetime.now(timezone.utc) - datetime.fromisoformat(row["reviewed_at"])
            if not 0 <= age.total_seconds() <= 30 * 86400:
                raise ValueError("Knowledge review has expired")
        version = db.execute("SELECT value FROM metadata WHERE key='corpus_version'").fetchone()[0]
        return {"corpus_version": version, "documents": len(rows),
                "reviewed_at": min(r["reviewed_at"] for r in rows)}


def review_sources(ids: list[str], reviewer: str, path: Path = DEFAULT_CORPUS) -> None:
    """Explicit operator action after comparing extraction with the saved HTML."""
    with closing(sqlite3.connect(path)) as db, db:
        for source_id in ids:
            cursor = db.execute("UPDATE documents SET reviewed_at=?, reviewer=? WHERE id=?",
                                (now(), reviewer, source_id))
            if cursor.rowcount != 1:
                raise ValueError("Unknown document")


def passage(row: sqlite3.Row, *, full: bool) -> Passage:
    text = row["text"] if full else row["excerpt"]
    return Passage(evidence_id=row["section_id"] + (":full" if full else ":excerpt"),
        document_id=row["document_id"], section_id=row["section_id"], title=row["title"],
        heading=row["heading"], canonical_url=row["url"], fetched_at=row["fetched_at"],
        reviewed_at=row["reviewed_at"], text=text, is_excerpt=text != row["text"],
        related_section_ids=json.loads(row["related"]))


STOPWORDS = set("a an the what how is are i my me can do does it to of for and with in on about please water sydney tell where".split())
SYNONYMS = {"afford": "payment assistance hardship", "instalments": "payment plan",
            "overdue": "overdue arrangement", "estimated": "estimate reading",
            "mean": "cover", "parts": "fixed usage charges", "plans": "instalments",
            "ways": "options choose", "methods": "options choose",
            "higher": "usage charges period estimated", "high": "usage charges period estimated",
            "difficulty": "hardship assistance", "prices": "pricing typical home",
            "pricing": "prices typical home", "access": "view register"}


def search_docs(query: str, topic: str = "all", limit: int = 3,
                path: Path = DEFAULT_CORPUS) -> Evidence:
    request = SearchInput(query=query, topic=topic, limit=limit)
    readiness(path)
    words = re.findall(r"[\w]+", request.query.lower(), flags=re.UNICODE)
    tokens = [w for w in words if w not in STOPWORDS][:24]
    for token in list(tokens):
        tokens.extend(SYNONYMS.get(token, "").split())
    tokens = list(dict.fromkeys(tokens))[:32]
    if not tokens:
        return Evidence(status="no_results")
    expression = " OR ".join('"' + word.replace('"', '""') + '"' for word in tokens)
    with closing(connection(path)) as db:
        rows = db.execute("""SELECT s.*, d.title, d.url, d.fetched_at, d.reviewed_at
            FROM search_index f JOIN sections s ON s.section_id=f.section_id
            JOIN documents d ON d.id=s.document_id
            WHERE search_index MATCH ? AND (?='all' OR d.topic=?)
            ORDER BY bm25(search_index, 0, 3, 2, 1), s.section_id LIMIT ?""",
            (expression, request.topic, request.topic, request.limit)).fetchall()
    passages, used = [], 0
    for row in rows:
        item = passage(row, full=False)
        if used + len(item.text) <= 3500:
            passages.append(item)
            used += len(item.text)
    return Evidence(status="ok" if passages else "no_results", passages=passages)


def read_section(section_id: str, path: Path = DEFAULT_CORPUS) -> Evidence:
    request = ReadInput(section_id=section_id)
    readiness(path)
    with closing(connection(path)) as db:
        row = db.execute("""SELECT s.*, d.title, d.url, d.fetched_at, d.reviewed_at
            FROM sections s JOIN documents d ON d.id=s.document_id WHERE s.section_id=?""",
            (request.section_id,)).fetchone()
    return Evidence(status="ok", passages=[passage(row, full=True)]) if row else Evidence(status="not_found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["ingest", "review", "status", "search"])
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--reviewer")
    parser.add_argument("--query", default="usage charge")
    args = parser.parse_args()
    if args.action == "ingest":
        print(json.dumps(ingest()))
    elif args.action == "review":
        if not args.ids or not args.reviewer:
            parser.error("Review requires --ids and --reviewer after inspecting snapshots/extraction")
        review_sources(args.ids, args.reviewer)
        print("Review recorded; restart all conversations before serving the rebuilt index.")
    elif args.action == "status":
        print(json.dumps(readiness()))
    else:
        print(search_docs(args.query).model_dump_json(indent=2))
