# Offline LLM consolidation (llama.cpp / llamafile)

`consolidate` accepts any OpenAI-compatible `/chat/completions` endpoint. Pointing it at
a local llama.cpp server (or a single-file llamafile) keeps the entire pipeline —
episodes, distilled facts, embeddings, vectors — on your machine. Nothing ever leaves
the box: no telemetry exists in OmniDB at all.

## 1. Start the server

With [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`:

```bash
llama-server -m Qwen2.5-7B-Instruct-Q4_K_M.gguf --port 8080
```

or with a [llamafile](https://github.com/Mozilla-Ocho/llamafile) (server mode):

```bash
./Qwen2.5-7B-Instruct.Q4_K_M.llamafile --port 8080
```

Both expose an OpenAI-compatible API at `http://127.0.0.1:8080/v1`. Any model works —
consolidation only needs instruction following and JSON output; small 4-bit quantized
models are fine for fact extraction. Pick one that fits in RAM/GPU.

## 2. Configure OmniDB

Environment variables:

```bash
export OMNIDB_LLM_BASE_URL=http://127.0.0.1:8080/v1
export OMNIDB_LLM_MODEL=qwen2.5-7b-instruct   # must match what the server loaded
# export OMNIDB_LLM_API_KEY=sk-no-key-needed  # llama.cpp ignores this; set if your proxy requires it
```

or in `omnidb.toml` (field names mirror `Config`; env still wins over TOML):

```toml
llm_base_url = "http://127.0.0.1:8080/v1"
llm_model = "qwen2.5-7b-instruct"
llm_api_key = ""
```

Verify without touching the MCP server:

```bash
python -m omnidb_server consolidate --hours 24
```

## How consolidation behaves with an LLM configured

- The system prompt asks for a JSON array of `{"fact", "confidence"}` objects and marks
  the transcript as data, never instructions; episode text is sanitized before it reaches
  the prompt.
- A transcript budget (~24k tokens) caps each run; episodes beyond the budget are left
  unmarked for the next run.
- Facts are length-capped, confidence-clamped, and stored as semantic memories carrying
  `source_episodes`; only when at least one fact is created do source episodes get marked
  consolidated.

## Failures consume nothing

This is the important guarantee: **if the LLM call fails — endpoint down, timeout,
malformed JSON — no episode is consumed.** Episodes stay unmarked and a later run
retries them from scratch. OmniDB deliberately does *not* fall back to the offline stub
after a configured-LLM failure, because silently copying raw episodes into semantic
memories would permanently burn them as if they had been consolidated.

You get back a report like:

```json
{
  "episodes_scanned": 12,
  "facts_created": 0,
  "facts_dropped": 0,
  "llm_used": false,
  "episode_ids": [],
  "llm_error": "...",
  "code": "OMN-013",
  "took_ms": 60001.4
}
```

Fix the endpoint (restart `llama-server`, check the port) and run again — nothing was lost.

## Zero-cloud privacy story

- Storage is local SQLite + vector sidecar files under `OMNIDB_HOME`.
- Embeddings run in-process (`hashing` offline-safe by default, or `fastembed` with the
  model cached locally via `HF_HOME`) — no embedding API calls, ever.
- Consolidation talks to whatever you configure. With the setup above that is
  `127.0.0.1`: memories and their distilled facts never cross your network boundary.
- Unset `OMNIDB_LLM_*` entirely and even that last hop disappears: the stub runs fully
  offline (verbatim copies).
