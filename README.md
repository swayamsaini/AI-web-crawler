# ARACHNE-Net: Adaptive Relevance-Aware Crawling via Hybrid Neural and Lexical Evaluation

Real-time focused web crawler built with FastAPI, WebSockets, and a three-signal relevance engine combining semantic embeddings (`all-MiniLM-L6-v2`), lexical ranking (`BM25Okapi`), positional heuristics, and optional Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`).

## Key Highlights
- **Hybrid Scoring:** 60% Semantic + 25% BM25 + 15% Positional with sigmoid normalization and optional Cross-Encoder reranking (`fast`, `balanced`, `best` modes).
- **Controlled Async Traversal:** Depth-bounded BFS (max depth 3, max 20 links per relevant page) with live WebSocket log streaming.
- **Evaluated Performance:** 0.959 Pearson correlation with human relevance scores and 90% classification accuracy at threshold `0.40`.

## Quick Start

    python main.py

Open `http://127.0.0.1:8000` in your browser. Saved articles are written to `./results/`.

## Author
**Swayam Saini**
