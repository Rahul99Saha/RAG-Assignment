# RAG: Baseline (Bi-Encoder) vs. Re-Ranking (Cross-Encoder)

All code is in **`RAG_Baseline_vs_Reranking.ipynb`**.

- **Document:** `data/29 CFR 1910 - General Industry Special Requirements 2026.pdf` (OSHA, 378 pages, 757 chunks)
- **Bi-encoder:** `sentence-transformers/all-MiniLM-L6-v2` + FAISS `IndexFlatIP` (cosine, normalised vectors)
- **Cross-encoder:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM:** Gemini (`gemini-3.5-flash-lite` by default, override with `GEMINI_MODEL`), temperature 0, same prompt for both pipelines

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key_here" > .env
jupyter notebook RAG_Baseline_vs_Reranking.ipynb   # Run All
```
Outputs (rank shifts, answers, latency, scaling) are written to `results/`.
