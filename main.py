"""
Real-Time BFS Wikipedia Crawler with FastAPI and WebSocket UI

Backend: FastAPI + Uvicorn
- Serves a minimal frontend (index.html)
- WebSocket endpoint for real-time crawl updates
- BFS crawler over Wikipedia article links with depth limit (3)
- Saves article content to ./results/{title}.txt asynchronously

Requirements (install via pip):
    fastapi uvicorn[standard] requests beautifulsoup4 aiofiles python-multipart

Run locally:
    python main.py
Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import asyncio
import os
import re
import math
import json
from typing import Optional
from collections import deque
from typing import Deque, Set
from urllib.parse import urljoin, urlparse

import aiofiles
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

# ---------- FastAPI app ----------
app = FastAPI(title="Real-Time BFS Wikipedia Crawler")

# Path helpers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "index.html")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------- Relevance Engine Utilities ----------
WORD_RE = re.compile(r"[A-Za-z0-9]+")

# Optional heavy dependencies (transformers + BM25). Use graceful fallback if unavailable.
try:  # numpy is common, used for cosine/array ops
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - fallback safe
    np = None  # type: ignore

try:
    from rank_bm25 import BM25Okapi  # type: ignore
except Exception:
    BM25Okapi = None  # type: ignore

_EMBEDDER = None  # lazy-initialized global
_CROSS_ENCODER = None  # lazy-initialized global

def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        # Small, fast, good quality general-purpose model
        _EMBEDDER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return _EMBEDDER
    except Exception:
        return None


def get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is not None:
        return _CROSS_ENCODER
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
        # Lightweight cross-encoder suitable for reranking
        _CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return _CROSS_ENCODER
    except Exception:
        return None


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return 0.5


def cross_encoder_refine_score(text: str, seeds: list[str], top_k: int = 8) -> float:
    """Use a cross-encoder to compute a relevance score over top chunks.
    Chooses up to top_k chunks (sentences/paragraphs) and returns max score in [0,1].
    Falls back to 0.0 if unavailable.
    """
    model = get_cross_encoder()
    if model is None:
        return 0.0
    # Build chunks
    sentences = [s.strip() for s in re.split(r"(?<=[\.!?])\s+", text) if s.strip()]
    if not sentences:
        sentences = [text[:1500]]
    # Pick a subset (first N for simplicity; could preselect via BM25)
    chunks = sentences[:max(1, top_k)]
    query = ", ".join(seeds)
    try:
        pairs = [(query, c) for c in chunks]
        scores = model.predict(pairs)
        if isinstance(scores, (list, tuple)):
            raw = float(max(scores))
        else:
            # assume numpy array
            raw = float(scores.max())
        return max(0.0, min(1.0, _sigmoid(raw)))
    except Exception:
        return 0.0


def sigmoid_scale(p: float, k: float = 7.0, center: float = 0.5) -> float:
    """Sharpen probabilities with a logistic squashing around `center`.
    p in [0,1] -> sigma(k*(p-center)) in (0,1).
    Larger k increases separation between low/high relevance.
    """
    try:
        return 1.0 / (1.0 + math.exp(-k * (float(p) - center)))
    except Exception:
        return p


def embed_texts(texts: list[str]):
    if not texts:
        return None
    embedder = get_embedder()
    if embedder is None or np is None:
        return None
    try:
        vecs = embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vecs
    except Exception:
        return None


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [t.lower() for t in WORD_RE.findall(text)]


def compute_relevance_score(
    text: str,
    title: str,
    seeds: list[str],
    df_map: dict[str, int],
    doc_count: int,
) -> float:
    """Stronger, research-grade relevance scoring with multiple signals:
    - Transformer embeddings similarity (SentenceTransformer all-MiniLM-L6-v2)
    - BM25 lexical relevance over sentences
    - Phrase position/presence
    Fallbacks gracefully if heavy deps are not available.
    Returns a score clamped to [0,1].
    """
    # Prepare tokens
    tokens = tokenize(text)
    title_tokens = tokenize(title)
    if not tokens:
        return 0.0

    # Seed phrases and tokens
    seed_phrases: list[list[str]] = []
    seed_token_set: set[str] = set()
    for s in seeds:
        sp = tokenize(s)
        if sp:
            seed_phrases.append(sp)
            seed_token_set.update(sp)
    if not seed_phrases:
        return 1.0

    # --- Position/presence component (fast, always available) ---
    pos_scores = []
    n = len(tokens)
    index_map: dict[str, list[int]] = {}
    for i, tok in enumerate(tokens):
        index_map.setdefault(tok, []).append(i)
    for phrase in seed_phrases:
        if len(phrase) == 1:
            positions = index_map.get(phrase[0], [])
            if positions:
                first = positions[0]
                pos_scores.append(1.0 - (first / max(n, 1)))
            elif phrase[0] in title_tokens:
                pos_scores.append(1.0)
        else:
            found_idx = None
            for i in range(0, max(n - len(phrase) + 1, 0)):
                if tokens[i : i + len(phrase)] == phrase:
                    found_idx = i
                    break
            if found_idx is not None:
                pos_scores.append(1.0 - (found_idx / max(n, 1)))
            else:
                for j in range(0, max(len(title_tokens) - len(phrase) + 1, 0)):
                    if title_tokens[j : j + len(phrase)] == phrase:
                        pos_scores.append(1.0)
                        break
    pos_score = sum(pos_scores) / len(pos_scores) if pos_scores else 0.0

    # --- BM25 over sentences (lexical) ---
    bm25_score = 0.0
    if BM25Okapi is not None:
        # Split to sentences roughly
        sentences = re.split(r"(?<=[\.!?])\s+", text)
        docs = [tokenize(s) for s in sentences if s.strip()]
        if docs:
            bm25 = BM25Okapi(docs)
            query = list(seed_token_set)
            try:
                scores = bm25.get_scores(query)
                if np is not None:
                    bm25_raw = float(np.mean(scores))
                else:
                    bm25_raw = sum(float(s) for s in scores) / max(len(scores), 1)
                bm25_score = 1.0 - math.exp(-0.1 * bm25_raw)
            except Exception:
                bm25_score = 0.0

    # --- Transformer embeddings (semantic) ---
    emb_score = 0.0
    emb_vecs = None
    if np is not None:
        # Chunk article to paragraphs/windows for robust max similarity
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        # fallback to big chunks if paragraphs too small
        if not paragraphs:
            paragraphs = [text[:1500]]
        # Limit chunks to avoid too-heavy compute
        chunks = paragraphs[:20]
        # Build seed repr as average embedding of seed phrases
        seed_strings = [" ".join(ph) for ph in seed_phrases]
        # Try embed, fallback silently
        seed_embs = embed_texts(seed_strings)
        chunk_embs = embed_texts(chunks)
        if seed_embs is not None and chunk_embs is not None:
            try:
                seed_vec = seed_embs.mean(axis=0)
                # cosine sim = dot since normalized
                sims = chunk_embs @ seed_vec
                emb_score = float(np.clip(np.max(sims), -1.0, 1.0))
                emb_score = 0.5 * (emb_score + 1.0)  # map [-1,1] -> [0,1]
            except Exception:
                emb_score = 0.0

    # If heavy deps missing, approximate with simple TF factor to avoid zeroing out
    if emb_score == 0.0 and bm25_score == 0.0:
        # Simple fallback using term overlap
        tf = {t: 0 for t in seed_token_set}
        for t in tokens:
            if t in tf:
                tf[t] += 1
        overlap = sum(1 for t, c in tf.items() if c > 0)
        emb_score = min(1.0, overlap / max(len(seed_token_set), 1))

    # Combine signals
    # Prioritize semantic similarity, then lexical BM25, then position
    combined = 0.6 * emb_score + 0.25 * bm25_score + 0.15 * pos_score
    return max(0.0, min(1.0, combined))


def update_df_stats(text: str, df_map: dict[str, int], seeds: list[str]) -> None:
    """Update document frequency counts for seed tokens based on this document."""
    tokens = set(tokenize(text))
    seed_tokens = set()
    for s in seeds:
        seed_tokens.update(tokenize(s))
    for tok in tokens.intersection(seed_tokens):
        df_map[tok] = df_map.get(tok, 0) + 1


async def safe_send(ws: WebSocket, text: str) -> bool:
    """Send a text message over WebSocket, returning False if the client disconnected.

    Catches WebSocketDisconnect and RuntimeError raised when the connection is closed.
    """
    try:
        await ws.send_text(text)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False
    except Exception:
        # Any unexpected error while sending should stop the crawl gracefully
        return False


def is_valid_wikipedia_link(url: str) -> bool:
    """Validate that a URL is a crawlable English Wikipedia article link.

    Rules:
    - scheme: http or https
    - netloc: en.wikipedia.org
    - path starts with /wiki/
    - path does not contain ':' (filters special pages like Category:, File:, Help:, etc.)
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return False
        if parsed.netloc != "en.wikipedia.org":
            return False
        if not parsed.path.startswith("/wiki/"):
            return False
        if ":" in parsed.path:
            return False
        return True
    except Exception:
        return False


def sanitize_filename(name: str) -> str:
    """Convert a page title into a safe filename: spaces -> underscores, remove bad chars."""
    safe = name.replace(" ", "_")
    # Remove characters that are commonly invalid in filenames on Windows
    bad = '<>:"/\\|?*\n\r\t'
    for ch in bad:
        safe = safe.replace(ch, "")
    # Trim and default if empty
    safe = safe.strip("._ ") or "untitled"
    return safe + ".txt"


async def save_content_async(filename: str, content: str) -> None:
    """Asynchronously write content to a file in RESULTS_DIR with utf-8 encoding."""
    path = os.path.join(RESULTS_DIR, filename)
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(content)


def fetch_page(url: str, timeout: int = 15) -> tuple[str, BeautifulSoup]:
    """Fetch a page synchronously using requests and return (final_url, soup).

    This runs in a threadpool when called from async code via asyncio.to_thread to
    avoid blocking the event loop.
    """
    resp = requests.get(url, timeout=timeout, headers={
        "User-Agent": "RT-BFS-Crawler/1.0 (+https://github.com/)"
    })
    resp.raise_for_status()
    # Parse with lxml if available, else html.parser
    soup = BeautifulSoup(resp.text, "html.parser")
    return (resp.url, soup)




async def crawl_website(websocket: WebSocket, start_url: str) -> None:
    """Perform a BFS crawl up to depth 3 starting from start_url, sending updates via WS.

    This variant expects that seeds and threshold are provided via the first message
    from the client in websocket_endpoint (JSON). For backward compatibility, if
    seeds are not provided, all pages are treated as relevant.
    """
    # Validate seed URL first
    if not is_valid_wikipedia_link(start_url):
        await safe_send(websocket, "Error: Please provide a valid English Wikipedia article URL.")
        return

    # Runtime state: relevance engine corpus stats
    # Keep lightweight DF counts across pages in this crawl session
    df_map: dict[str, int] = {}
    doc_count: int = 0

    # Retrieve seeds and threshold from connection state if present
    seeds = getattr(websocket, "_seed_words", None) or []
    threshold = getattr(websocket, "_relevance_threshold", None)
    mode = getattr(websocket, "_analysis_mode", "balanced")  # fast | balanced | best
    if threshold is None:
        threshold = 0.3  # default threshold in [0,1]

    # BFS data structures
    queue: Deque[tuple[str, int]] = deque()
    visited: Set[str] = set()

    queue.append((start_url, 0))
    visited.add(start_url)

    current_level = 0

    # Process while queue is not empty
    while queue:
        # Determine level size (all nodes with same depth)
        level_size = sum(1 for _url, depth in list(queue) if depth == current_level)
        if level_size == 0:
            # Move to the next level present in the queue (if any)
            if queue:
                current_level = min(depth for _url, depth in queue)
                level_size = sum(1 for _url, depth in list(queue) if depth == current_level)
            else:
                break

        # Depth limit check: start of processing this level
        if current_level >= 3:
            if not await safe_send(websocket, "Crawl depth limit reached (3 levels). Stopping."):
                return
            break

        if not await safe_send(websocket, f"--- Processing Level {current_level} --- ({level_size} links)"):
            return

        for _ in range(level_size):
            try:
                url, depth = queue.popleft()
            except IndexError:
                break

            # Guard: ensure we are on the intended level
            if depth != current_level:
                # Put it back for later level
                queue.append((url, depth))
                continue

            # Include the BFS depth in the log so the UI shows the level of each link
            if not await safe_send(websocket, f"Crawling: {url} (level {depth})"):
                return

            # Fetch page without blocking event loop
            try:
                final_url, soup = await asyncio.to_thread(fetch_page, url)
            except requests.RequestException as e:
                if not await safe_send(websocket, f"Failed to fetch {url}: {e}"):
                    return
                await asyncio.sleep(0.1)
                continue
            except Exception as e:
                if not await safe_send(websocket, f"An error occurred while processing {url}: {e}"):
                    return
                await asyncio.sleep(0.1)
                continue

            # Extract title
            title_tag = soup.find("h1", id="firstHeading")
            title = title_tag.get_text(strip=True) if title_tag else "Untitled"

            # Extract main content text from mw-content-text
            content_div = soup.find("div", id="mw-content-text")
            text_content = ""
            if content_div:
                text_content = content_div.get_text(separator="\n", strip=True)

            # Relevance gate: if seeds provided, compute a relevance score and prune if below threshold
            is_relevant = True
            score = 1.0
            if seeds:
                # Progress logs (terminal + WS)
                print(f"[relevance] Base scoring (mode={mode}) for: {title}")
                await safe_send(websocket, "Analyzing relevance: base scoring...")
                score = compute_relevance_score(text_content, title, seeds, df_map, doc_count)
                # Optional cross-encoder refinement in 'best' mode
                if mode == "best":
                    await safe_send(websocket, "Analyzing relevance: cross-encoder refine...")
                    print(f"[relevance] Cross-encoder refine for: {title}")
                    ce_score = cross_encoder_refine_score(text_content, seeds, top_k=8)
                    # Blend with base score
                    score = 0.7 * score + 0.3 * ce_score
                # Apply sigmoid scaling to diversify relevance extremes
                k_map = {"fast": 5.0, "balanced": 7.0, "best": 9.0}
                k_val = k_map.get(mode, 7.0)
                score = sigmoid_scale(score, k=k_val, center=0.5)
                is_relevant = score >= float(threshold)
                # Update corpus stats after scoring this doc
                try:
                    update_df_stats(text_content, df_map, seeds)
                    doc_count += 1
                except Exception:
                    pass

            if is_relevant:
                # Save the content asynchronously
                filename = sanitize_filename(title)
                try:
                    await save_content_async(filename, text_content)
                    if not await safe_send(websocket, f"Saved content to {filename} (level {depth}) | relevance={score:.2f}"):
                        return
                except Exception as e:
                    if not await safe_send(websocket, f"An error occurred while saving content for {url}: {e}"):
                        return
            else:
                if not await safe_send(websocket, f"Pruned (low relevance {score:.2f}) -> {url} (level {depth})"):
                    return

            # Extract and enqueue valid links for next level (only if relevant)
            next_depth = depth + 1
            if is_relevant and next_depth <= 3 and content_div:
                enq_limit = 20  # cap: explore only first 20 valid anchors from this page
                enq_count = 0
                for a in content_div.find_all("a", href=True):
                    abs_url = urljoin(final_url, a.get("href"))
                    if is_valid_wikipedia_link(abs_url) and abs_url not in visited:
                        visited.add(abs_url)
                        queue.append((abs_url, next_depth))
                        enq_count += 1
                        if enq_count >= enq_limit:
                            break

            # Queue updates removed (no UI queue rendering)

            # Politeness delay
            await asyncio.sleep(0.1)

        # Move to next level
        current_level += 1

    await safe_send(websocket, "--- Crawl Finished ---")


@app.get("/")
async def root() -> FileResponse:
    """Serve the single-page frontend."""
    return FileResponse(INDEX_PATH)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket handler: receives seed URL, runs BFS crawl, streams updates."""
    await websocket.accept()
    try:
        while True:
            # Wait for a seed URL or JSON config from the client
            raw = await websocket.receive_text()
            raw = raw.strip()
            start_url = raw
            # Try to parse JSON envelope: {url: str, seeds: [str], threshold?: float, mode?: str}
            try:
                cfg = json.loads(raw)
                if isinstance(cfg, dict) and "url" in cfg:
                    start_url = str(cfg.get("url", "")).strip()
                    seeds = cfg.get("seeds", [])
                    threshold = cfg.get("threshold", None)
                    mode = cfg.get("mode", None)
                    # attach to websocket for this crawl session
                    websocket._seed_words = [s.strip() for s in seeds if isinstance(s, str) and s.strip()]
                    if threshold is not None:
                        websocket._relevance_threshold = float(threshold)
                    if mode in {"fast", "balanced", "best"}:
                        websocket._analysis_mode = mode
                else:
                    # Treat as plain URL string
                    websocket._seed_words = []
                    websocket._relevance_threshold = 0.3
                    websocket._analysis_mode = "balanced"
            except Exception:
                # Not JSON; fall back to plain URL
                websocket._seed_words = []
                websocket._relevance_threshold = 0.3
                websocket._analysis_mode = "balanced"

            await crawl_website(websocket, start_url)
    except WebSocketDisconnect:
        # Client disconnected; simply end handler
        return


if __name__ == "__main__":
    import uvicorn

    # Run uvicorn server; host 0.0.0.0 for broader access, port 8000
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
