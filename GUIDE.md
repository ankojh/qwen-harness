# Qwen Harness: guide to the code and the controls

This document is the tour: where to read, what happens during one Generate, what every control does, and what the app can and cannot show you.

The tool is deliberately small and flat. If a behavior is not visible in the UI and not in the event log, it is not happening.

---

## 1. Running and checking

```bash
./run.sh                 # uv sync + npm install + build + uvicorn on 127.0.0.1:8765
PORT=8010 ./run.sh       # if 8765 is busy
./run.sh dev             # backend + Vite dev server on 5173 (proxy /api -> 8765)
```

```bash
cd backend
.venv/bin/python -m pytest -q   # pure logic, no model
.venv/bin/python verify.py      # real model, prints PASS/FAIL per check
```

The frontend build lives in `frontend/dist`. `app.py` mounts it at `/` if it exists. Build before starting the production server (run.sh does), otherwise you get the placeholder page.

---

## 2. Code reading order

### Backend

1. `backend/config.py` — `AppConfig`, the single source of truth. Every field carries `group`, `purpose`, `tradeoff`, `applies`, and optional `library_default` in its JSON schema. `fields_schema()` is what the UI renders; `inactive_reasons()` explains settings that are valid but currently do nothing.
2. `backend/events.py` — bounded `EventLog` ring buffer, monotonically increasing `seq`, subscriber queues, `format_sse()`.
3. `backend/parsing.py` — two pure state machines: `StopMatcher` (hold back partial stop strings) and `ThinkingSplitter` (thinking/answer channels, markers kept only in raw).
4. `backend/prompt.py` — `build_prompt()`: session → messages → chat template → tokens, plus transformations, budget, checkpoint boundaries, and a dry-run cache prediction.
5. `backend/prompt_cache.py` — `PromptCacheStore`: prefix lookup, deep-copy on reuse, LRU/byte eviction, per-entry byte split, checkpoint dedupe, estimates, summary.
6. `backend/model.py` — `load_model()`, read-only model/tokenizer facts, `capabilities()`, memory info.
7. `backend/generation.py` — `run_generation()`: lookup, manual prefill with checkpoint snapshots, `generate_step` decode loop, parsing, finish reason, stats, cache insert and offset verification.
8. `backend/app.py` — FastAPI routes, the single-generation guard, idempotency map, SSE endpoint, static frontend.
9. `backend/verify.py` and `backend/tests/` — the checks that keep the claims above honest.

### Frontend

1. `frontend/src/api.js` — fetch helpers and the one `EventSource` subscription.
2. `frontend/src/App.jsx` — layout, session state, SSE handling (live tokens, reconnect), actions.
3. `frontend/src/Field.jsx` and `ConfigPanel.jsx` — generic controls generated from `/api/state`, presets, JSON editor, model facts, cache table.
4. `frontend/src/InputPanel.jsx` — messages/raw editor and prompt inspector.
5. `frontend/src/OutputPanel.jsx` — thinking/answer/raw/tokens/stats, append, finish and hash checks.
6. `frontend/src/EventLog.jsx` — compact event stream with expandable JSON.

---

## 3. The life of one Generate request

1. **Button.** `OutputPanel` calls `onGenerate`; `App.handleGenerate()` makes a `client_request_id` with `crypto.randomUUID()` and calls `POST /api/generate` through `api.startGeneration()`.
2. **Route and guards.** `app.py: generate()` checks that a model is loaded, returns the existing generation for a repeated `client_request_id`, and returns 409 if a different generation is active. It creates a `GenerationStore` record and emits `generation.started`.
3. **Worker thread.** The request body is submitted to `ThreadPoolExecutor(max_workers=1)` as `generation.run_generation()`. The FastAPI event loop never touches MLX arrays.
4. **Prompt.** `prompt.build_prompt()` collects enabled messages, inserts `config.system_prompt` if set, renders the chat template, tokenizes with `tokenizer.encode(text, add_special_tokens=False)`, compares that with `apply_chat_template(tokenize=True)`, checks the budget, computes checkpoint boundaries, and predicts cache reuse. The report is stored on the generation record.
5. **Budget refusal.** If the prompt does not fit (`ok=false`), the generation finishes with `context_overflow` and no inference runs.
6. **Sampler and processors.** `make_sampler()` and `make_logits_processors()` are built from the config. Because cache reuse means `generate_step` sees only the final prompt token, each processor is wrapped so it always receives the full prompt prefix plus the generated tokens. Penalties never silently lose context.
7. **Cache lookup.** `PromptCacheStore.lookup()` searches compatible entries (model, revision, adapter, KV settings), keeps the longest **usable** prefix, deep-copies the cache, and trims only if the model's cache supports trimming. Qwen3.5's does not.
8. **Manual prefill.** `generation.py` feeds `tokens[reused:-1]` to the model in `prefill_step_size` chunks inside the generation stream. It evaluates the cache after each chunk, emits `prefill.chunk` with real counts and timing, checks cancellation between chunks, and splits a chunk at every accepted checkpoint position so a snapshot lands on a clean token boundary.
9. **Snapshots.** At an accepted checkpoint, the cache is deep-copied and inserted into the store with `kind="checkpoint"`, `after_message`, and measured recurrent/KV bytes. If an identical entry already exists, `touch_exact()` marks it used instead of copying again. Each snapshot emits `cache.checkpoint`.
10. **Decode.** `generate_step()` receives the last prompt token and the prepared cache. For every yielded token the loop checks EOS, runs the streaming detokenizer, `StopMatcher`, then `ThinkingSplitter`, records id/piece/emitted text/logprob/top alternatives/channel, and emits `token`. Cancellation is checked after every token. The loop runs inside `wired_limit()`; that is logged, not hidden.
11. **Finish.** Finish reasons are `eos`, `stop_sequence`, `max_tokens`, `cancelled`, `context_overflow`, or `error`. The detokenizer is finalized and both parsers are flushed, so held-back text is released exactly once.
12. **Cache insert and verification.** The cache is keyed by `prompt + all yielded tokens` (including a yielded EOS, which is fed to the model before it is yielded). `cache.verify` compares every `KVCache.offset` with the key length; the entry is stored only if they match. `generation.finished` then refreshes the UI.

**Preview** uses the same `build_prompt()` and tokenizer but no model. It returns the exact text, token ids, budget, transformations, checkpoint plan, and a dry-run cache prediction (`peek()` does not change LRU order).

**Event flow.** Every step calls `EventLog.emit()`. `/api/events` streams from the ring buffer, replaying from `Last-Event-ID` and sending `resync` if that id fell out. The browser appends events to the log and builds the live output from `token` events. It fetches the full record (`GET /api/generations/{id}`) only when the run finishes, after a `resync`, or after the stream reconnects. There is deliberately no polling during generation: each fetch deep-copies every token record under the same lock the generation thread uses to append tokens, so polling would slow long generations down. None of these reads can start work.

---

## 4. Every control, group by group

The config panel is generated from `/api/state.fields`. Each field shows its effective value, default, applies badge, purpose, tradeoff, and any inactive/unsupported reason. Changing a control sends `PUT /api/config`; the server validates and returns the whole effective state. Presets, Import, Reset, and the JSON editor use the same path.

### Model (load-time)

- `model_path` — HF repo id or local folder for `mlx_lm.load`. A new repo downloads on Load.
- `revision` — branch/tag/commit of that repo.
- `adapter_path` — LoRA adapter folder applied on top of the base weights.
- `lazy_load` — materialize weights on first use; moves load time into the first generation.
- `trust_remote_code` — permanently disabled: it would execute Python from the repo and MLX Qwen models do not need it.

Weight quantization is **read-only model info**, not a control: bits, group size, mode from `config.json`, plus measured weight bytes. To change it, load a different repo.

### Prompt and template

- `system_prompt` — inserted as message 0 when non-empty; the inspector lists the insertion.
- `enable_thinking` — passed to the template as `enable_thinking`; disabled if the active template never reads it.
- `add_generation_prompt` — append the assistant turn header. Turning it off is allowed and usually produces odd continuations; the inspector shows the result.
- `template_kwargs` — extra Jinja template variables (JSON object). Keys the template never reads are reported as unused; reserved keys are dropped and reported.
- `chat_template_override` — Jinja text replacing the model's template. Empty means use the model's own. The model's template is viewable read-only.

Prompt mode (Messages vs Raw) belongs to the session, not the config. Raw mode bypasses the chat template and sends `tokenizer.encode(text, add_special_tokens=False)` as-is.

### Context and stopping

- `context_budget` — prompt tokens plus output reservation must fit. Larger budgets cost KV memory and prefill time; it must not exceed the model's 262144 positions.
- `overflow_policy` — `report` refuses and shows the numbers; `drop_oldest_messages` drops the oldest enabled non-system messages until it fits and lists each drop. Raw mode always reports.
- `max_tokens` — output reservation and hard generation cap.
- `stop_sequences` — text matched on the raw decoded stream (thinking included), with hold-back so a partial match never streams out and is retracted.
- `extra_eos_token_ids` — added to the tokenizer's EOS set; the tokenizer's own set is shown read-only.

### Sampling

- `temperature` — 0 is greedy argmax and makes the other sampling filters inactive.
- `top_p`, `top_k`, `min_p`, `min_tokens_to_keep` — filters; `top_p` only applies when `0 < p < 1`.
- `xtc_probability`, `xtc_threshold` — exclude-top-choices sampling; EOS and think markers are protected from removal.
- `repetition_penalty`, `repetition_context_size` — sign-aware multiplicative penalty over a recent window.
- `presence_penalty`, `presence_context_size` — additive, window-based (mlx-lm's definition, not OpenAI's whole-output one).
- `frequency_penalty`, `frequency_context_size` — additive per occurrence in the window.
- `logit_bias` — `{token_id: bias}`; ids are checked against the vocabulary when the prompt is built.
- `seed` — calls `mx.random.seed` before each generation. Cache reuse changes prefill shapes, which can still change floating-point results slightly.

Penalties are logits processors and still apply under greedy sampling. Presets are explicit buttons: Qwen thinking, Qwen non-thinking, Greedy.

### Runtime, KV cache, and prompt cache

- `prefill_step_size` — prompt tokens per forward pass; larger is faster but raises peak memory and makes Stop coarser.
- `kv_bits` — quantize the KV cache (None/2/3/4/5/6/8). Only the 8 full-attention layers change; the 24 recurrent layers stay `ArraysCache`. Changing any `kv_*` value clears the prompt cache.
- `kv_group_size`, `quantized_kv_start` — only used when `kv_bits` is set.
- `max_kv_size` — shown disabled for this model: it defines `make_cache`, so mlx-lm ignores the rotating-cache size.
- `cache_reuse` — look up and store prompt caches; off means every run is a cold prefill.
- `cache_checkpoints` — `off`, `last_message`, or `every_message` (see section 7).
- `cache_max_entries` — LRU entry limit.
- `cache_max_gb` — accounted-byte budget for stored caches.
- `mlx_cache_limit_gb` — MLX's allocator free-buffer limit (`mx.set_cache_limit`), not the prompt cache. Applied as soon as you change it; empty restores the MLX default recorded at startup. The effective limit is shown in the prompt-cache totals. Lower returns memory to the OS sooner at some speed cost.
- `top_logprobs` — alternatives recorded per token (0-20); a small top-k over the 248k vocabulary per token.

Speculative decoding is listed as unsupported: it needs a rewindable cache and this model's cache is not trimmable.

The **prompt cache** section shows every entry (kind, after message, tokens, recurrent MB, KV MB, total MB, age), totals versus the configured limit, utilization, evictions, live MLX memory, layer types, and the last lookup. Accounted bytes are summed `cache.nbytes`; MLX can share buffers between identical arrays, so real usage can be lower. Snapshots at different positions each hold their own recurrent state and KV buffers.

### Tools

Presets, **Export JSON**, **Import JSON**, **Reset to defaults**, and the **Advanced JSON editor**. The editor is validated by the server through the same PUT path; validation errors are shown inline.

---

## 5. Input panel and prompt inspector

- Messages mode: role, content, optional `reasoning_content`, enabled checkbox, move up/down, delete, add.
- Raw mode: free text; the template is bypassed.
- **Preview prompt** renders the exact prompt with special tokens and think markers highlighted. **auto-refresh** repeats it after a 600 ms debounce and only calls the tokenizer.
- The inspector shows: budget bar (input, output reservation, remaining), token count, checkpoint position, starts-in-thinking, prompt hash, submitted messages, dropped messages, transformation lines, template report (variables, passed/unused/reserved kwargs, tokenize mismatch), cache prediction, planned checkpoints with accepted/skipped reasons, and the estimated snapshot cost. The token table (id + piece) is opt-in.

The hash is the link between preview and submission: the output panel compares the preview hash with the generation record's `prompt_hash`.

---

## 6. Output panel and event log

- **Generate** / **Stop** / **Clear output** / **Clear conversation**.
- Views: **Thinking**, **Answer**, **Raw**, **Tokens**, **Stats**.
- Thinking is text generated between the think markers: ordinary sampled tokens, not activations or attention.
- Raw is the decoded stream with markers highlighted; thinking and answer are derived from the same ordered token list, so they stay synchronized.
- Tokens shows index, id, piece, emitted text, channel, logprob, and top alternatives.
- Stats shows measured prefill/decode timing and rates, memory counters, forward calls (one per prefill chunk, plus n + 1 decode calls for n tokens, because `generate_step` feeds each token to the model before yielding it), prompt-cache bytes, cache offset verification, prompt hash, checkpoints, and the full effective settings and prompt report.
- **Append to conversation** is explicit. The checkbox keeps the thinking text as `reasoning_content`; the chat template may drop it from history, and the inspector reports when it does.
- Finish badge: `eos`, `stop_sequence`, `max_tokens`, `cancelled`, `context_overflow`, or `error`.
- The event log is compact and chronological. Token events are hidden by default; click a row for JSON; filter by type or data; the SSE badge shows connection state.

---

## 7. Cache checkpoints on Qwen3.5, and their cost

Qwen3.5 has 24 linear-attention (`ArraysCache`) layers and 8 full-attention (`KVCache`) layers. `can_trim_prompt_cache()` is false, so a stored cache can only be reused when it is an **exact prefix** of the new prompt.

A checkpoint boundary is computed by rendering the first `i` submitted messages with the same template arguments and `add_generation_prompt=False`, tokenizing, and accepting it only if those ids are a real prefix of the full prompt and leave a token to feed. `prompt.py` reports every boundary:

```json
{"after_message": 3, "position": 21, "ok": true, "reason": null}
{"after_message": 2, "position": null, "ok": false, "reason": "not a token prefix ..."}
```

Two real Qwen3.5 cases are reported rather than hidden:

- A system-only prefix can make the template raise (“No user query found in messages”).
- A prefix ending at an assistant message can render differently from the full prompt (for example reasoning that is only kept at the end), so it is not a token prefix. In the real model run, assistant boundaries are commonly skipped and user boundaries accepted.

Modes:

- `off` — no checkpoints; only the after-generation entry is stored.
- `last_message` — the default: one checkpoint after the last submitted message, before the assistant header.
- `every_message` — try a snapshot after every message. An edit to message k can then reuse everything up to message k-1. Verified: editing message 4 of a 5-message conversation reuses exactly the checkpoint after message 3.

Measured cost per snapshot on this machine: the recurrent state of the 24 linear-attention layers is about **51.5 MB and constant**; the 8 attention layers add about **33.5 KB per token at fp16** (allocations are block-sized, so small snapshots show a rounded-up `KV` number). `every_message` therefore buys edit locality with one snapshot per message; long conversations hit `cache_max_gb` and evict.

The inspector's `checkpoint_estimate` is derived from a stored entry: recurrent bytes stay constant and KV scales with position. It is labelled as an estimate and is never produced by running the model.

Reuse can miss even when a checkpoint exists:

- the new prompt diverges before the checkpoint;
- the entry was built with different `kv_*` settings, model, revision, or adapter;
- the cache was cleared (model load/unload/reload, manual clear);
- `cache_reuse` is off, or `cache_checkpoints` is off and only an after-generation entry exists that is not an exact prefix.
When a longer entry shares a prefix but cannot be trimmed, the store falls back to a shorter usable checkpoint instead of refusing reuse.

Each insert dedupes an identical `(compat_key, tokens)` entry. The store evicts by LRU until both `cache_max_entries` and `cache_max_gb` are satisfied; every insert/evict is an event.

---

## 8. What is observable, and what is not

**Observable**

- The exact prompt text and token ids sent to the model, and the tokenizer check against `apply_chat_template(tokenize=True)`.
- Every generated token: id, raw piece, text emitted at that step, logprob, top alternatives, and channel.
- Why a generation stopped, at which token/sequence/offset.
- Prefill/decode timings and rates measured by the loop; memory counters; prompt-cache bytes; forward-call count.
- Cache lookup decisions, checkpoint positions, evictions, and offset verification.
- SSE events and reconnects that replay without starting work.

**Not observable here**

- Internal computation: activations, attention patterns, hidden states. The thinking pane is sampled text, not a window into the network.
- A full probability distribution: only the recorded top alternatives and the sampled token's logprob.
- Anything the app does not instrument. There is no hidden inference: Preview and auto-refresh only tokenize; the cache estimate uses measured bytes; Generate is the only model call.
- True wall-clock cost of other processes: hard CPU/GPU/memory limits imposed outside the app will slow inference and make timings worse. The app never changes power settings. Generation runs inside mlx-lm's `wired_limit` context (which temporarily raises the wired memory limit to the recommended working set, the same as `stream_generate`); that is logged as `runtime.wired_limit`.
- Real cache memory exactly: `cache.nbytes` is accounted per entry, but MLX shares identical buffers; the cache table and note state this.

---

## 9. Troubleshooting

- **Port 8765 busy** — another app owns it. Use `PORT=8010 ./run.sh` (and the Vite dev proxy follows `PORT`).
- **First Load downloads** — a repo not in `~/.cache/huggingface` downloads on Load; the event log records the path and size.
- **409 on Generate** — a generation is active. Stop it or wait; model/config changes are also refused while it runs.
- **Reuse seems not to happen** — read `cache.lookup.reason` and the inspector's checkpoint list. On this model only exact prefixes are reusable, and assistant-message boundaries are often skipped by the template.
- **Settings seem stale** — model fields (`model_path`, `revision`, `adapter_path`, `lazy_load`) apply on Reload; the model section shows a stale warning. `kv_*` changes invalidate the cache.
- **No answer text** — thinking may not have closed within `max_tokens`; the answer pane says so. Increase the limit or disable thinking to compare.
