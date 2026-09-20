# Comparative benchmark — OmniDB vs sqlite-vec vs chromadb

- corpus: 2000 synthetic topic-clustered texts; 50 paraphrase queries at k=5
- host run: 2026-09-07 13:27, single process, warm OS cache
- honesty: each system measured with its own storage/embedding stack; recall@k
  is computed against each system's own vector space (see per-system notes).

| metric | omnidb | sqlite-vec | chromadb |
|---|---|---|---|
| cold_start_s | 0.0121 | 0.0022 | 0.2813 |
| insert_single_ms | 0.336 | 0.048 | 16.627 |
| insert_batch_per_sec | 6756.1 | n/a | n/a |
| recall_p50_ms | 0.686 | 1.947 | 212.289 |
| recall_p95_ms | 0.835 | 2.27 | 240.407 |
| recall_at_k | 0.416 | 0.988 | 1.0 |
| reopen_s | 0.0067 | 0.0015 | 0.0578 |

## Notes

- **sqlite-vec**: vector-only (no lexical leg, no rerank); vectors from same hashing embedder
- **chromadb**: default onnx MiniLM embedder; recall vs chroma's own vector space

## Negative finding (ours): hybrid fusion degrades under a weak embedder

Control experiment, same corpus/truth: OmniDB's **dense leg alone scores recall@5 = 0.968**,
but the **full hybrid scores 0.416**. RRF's equal-weight fusion lets the lexical leg — built
over the same weak hashing features — outvote correct dense evidence. With a real embedder
(fastembed) the lexical leg is true BM25 over text and this artifact largely disappears;
but it is the default offline experience and should be weighted/config-aware in a future release.
sqlite-vec and chromadb are unaffected because they run vector-only.
