#!/usr/bin/env python3
"""End-to-end verification against the real model (plan section 9).

The script starts its own uvicorn server on 127.0.0.1 (port 8011 unless
``VERIFY_PORT`` is set), drives the HTTP API with httpx, and prints PASS/FAIL
per check.  It uses greedy sampling and small token limits where determinism
matters, so a full run takes well under two minutes.

Run from ``backend/``:  ``.venv/bin/python verify.py``
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from mlx_lm.tokenizer_utils import load as load_tokenizer

import app as app_module

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    """Record and print one PASS/FAIL line."""
    RESULTS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    return bool(ok)


# --------------------------------------------------------------------------
# server + HTTP helpers
# --------------------------------------------------------------------------
def free_port() -> int:
    preferred = int(os.environ.get("VERIFY_PORT", "8011"))
    for port in (preferred, 0):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return int(sock.getsockname()[1])
            except OSError:
                continue
    raise RuntimeError("could not find a free local port")


def start_server(port: int) -> tuple[uvicorn.Server, threading.Thread]:
    config = uvicorn.Config(
        app_module.app, host="127.0.0.1", port=port, log_level="warning"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(400):
        if server.started:
            return server, thread
        if not thread.is_alive():
            raise RuntimeError("uvicorn exited during startup")
        time.sleep(0.05)
    raise RuntimeError("uvicorn did not start")


def wait_for_generation(client: httpx.Client, gen_id: str, timeout: float = 180) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        record = client.get(f"/api/generations/{gen_id}").json()
        if record["status"] == "finished":
            return record
        time.sleep(0.1)
    raise TimeoutError(f"generation {gen_id} did not finish")


def generate(client: httpx.Client, request_id: str, session: dict) -> dict:
    response = client.post(
        "/api/generate",
        json={"client_request_id": request_id, "session": session},
    )
    response.raise_for_status()
    return wait_for_generation(client, response.json()["generation_id"])


def msg(role: str, content: str) -> dict:
    return {"role": role, "content": content, "enabled": True}


def session_of(*messages: dict) -> dict:
    return {"mode": "messages", "messages": list(messages), "raw_text": ""}


def collect_sse(
    client: httpx.Client,
    url: str,
    last_id: int,
    stop_type: str,
    stop_key: str,
    stop_value: Any,
    max_frames: int = 5000,
) -> list[dict]:
    """Read SSE frames from ``last_id`` until the stop frame, then hang up."""
    frames: list[dict] = []
    with client.stream(
        "GET", url, headers={"Last-Event-ID": str(last_id)}
    ) as response:
        response.raise_for_status()
        seq = None
        etype = None
        data: Any = None
        for line in response.iter_lines():
            if line.startswith("id: "):
                seq = int(line[4:])
            elif line.startswith("event: "):
                etype = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
            elif line == "":
                if etype is not None:
                    frames.append({"seq": seq, "type": etype, "data": data or {}})
                    if (
                        etype == stop_type
                        and isinstance(data, dict)
                        and data.get(stop_key) == stop_value
                    ):
                        return frames
                    if len(frames) >= max_frames:
                        return frames
                seq, etype, data = None, None, None
    return frames


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------
def run_checks(client: httpx.Client, base: str) -> None:
    # ---------------------------------------------------------- load + config
    load = client.post("/api/model/load")
    check("model loads", load.status_code == 200, f"HTTP {load.status_code}")
    state = client.get("/api/state").json()
    check("model info is present", bool(state.get("model")), state["model"].get("architecture", ""))
    check(
        "hybrid layer mix reported",
        state["model"].get("layer_mix") == {"linear_attention": 24, "full_attention": 8},
        str(state["model"].get("layer_mix")),
    )
    check(
        "cache reported as not trimmable",
        state["model"].get("cache_trimmable") is False,
    )

    desired = {
        "temperature": 0.0,
        "top_p": 0.9,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
        "kv_bits": None,
        "kv_group_size": 64,
        "quantized_kv_start": 5000,
        "prefill_step_size": 2048,
        "max_tokens": 64,
        "context_budget": 32768,
        "enable_thinking": False,
        "cache_reuse": True,
        "cache_checkpoints": "last_message",
        "top_logprobs": 0,
    }
    applied = client.put("/api/config", json=desired)
    check("config accepted", applied.status_code == 200, f"HTTP {applied.status_code}")
    config = applied.json()["config"]

    turn1 = session_of(msg("user", "Reply with the single word: hello"))
    preview1 = client.post(
        "/api/prompt/preview", json={"session": turn1, "include_pieces": False}
    ).json()
    check("preview succeeds", preview1["ok"] is True, str(preview1.get("error")))
    rec1 = generate(client, "verify-turn1", turn1)

    # ------------------------------------------- 1. config == effective settings
    eff = rec1["effective_settings"]
    sampler_ok = (
        eff["sampler"]["temp"] == 0.0
        and eff["sampler"]["top_p"] == 0.9
        and eff["sampler"]["top_k"] == 20
        and eff["sampler"]["min_p"] == 0.0
        and eff["sampler"]["min_tokens_to_keep"] == 1
    )
    kv_ok = eff["kv"] == {
        "kv_bits": None,
        "kv_group_size": 64,
        "quantized_kv_start": 5000,
    }
    runtime_ok = (
        eff["prefill_step_size"] == 2048
        and eff["max_tokens"] == 64
        and eff["max_new_tokens"] == 64
        and eff["top_logprobs"] == 0
        and eff["seed"] is None
    )
    check(
        "effective sampler matches displayed config",
        sampler_ok,
        json.dumps(eff["sampler"]),
    )
    check(
        "effective logits processors match config",
        eff["logits_processors"] == ["presence_penalty"],
        str(eff["logits_processors"]),
    )
    check("effective KV settings match config", kv_ok, json.dumps(eff["kv"]))
    check("effective runtime settings match config", runtime_ok)
    check(
        "effective checkpoint mode matches config",
        eff.get("cache_checkpoints") == "last_message",
        str(eff.get("cache_checkpoints")),
    )
    check(
        "inactive fields are reported inactive",
        "top_p" in eff["inactive"] and "greedy" in eff["inactive"]["top_p"],
        str(list(eff["inactive"])[:6]),
    )

    # --------------------------------------- 2. inspector == submitted tokens
    check(
        "preview hash equals generation prompt hash",
        preview1["prompt_hash"] == rec1["prompt_hash"],
    )
    check(
        "preview token ids equal submitted token ids",
        preview1["token_ids"] == rec1["prompt"]["token_ids"],
    )
    resolved = state["model"]["resolved_path"]
    tokenizer = load_tokenizer(Path(resolved))
    template_ids = list(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "Reply with the single word: hello"}],
            tokenize=True,
            add_generation_prompt=config["add_generation_prompt"],
            enable_thinking=config["enable_thinking"],
        )
    )
    check(
        "submitted ids equal apply_chat_template(tokenize=True)",
        template_ids == rec1["prompt"]["token_ids"],
        f"{len(template_ids)} vs {len(rec1['prompt']['token_ids'])} tokens",
    )

    # ---------------------------------------- 3. raw / thinking / answer split
    raw_concat = "".join(token["text"] for token in rec1["tokens"])
    check(
        "raw text equals concatenated per-token text",
        rec1["channels"]["raw"] == raw_concat,
        repr(rec1["channels"]["raw"])[:60],
    )
    check(
        "thinking + answer reconstructs raw (non-thinking run)",
        rec1["channels"]["thinking"] + rec1["channels"]["answer"]
        == rec1["channels"]["raw"],
    )

    client.put("/api/config", json={"enable_thinking": True, "max_tokens": 16})
    think_session = session_of(msg("user", "Think briefly about the number 2."))
    rec_think = generate(client, "verify-thinking", think_session)
    raw_concat_think = "".join(token["text"] for token in rec_think["tokens"])
    check(
        "thinking run: raw equals concatenated token text",
        rec_think["channels"]["raw"] == raw_concat_think,
    )
    check(
        "thinking run: thinking + answer reconstructs raw",
        rec_think["channels"]["thinking"] + rec_think["channels"]["answer"]
        == rec_think["channels"]["raw"],
    )
    check(
        "thinking run: prompt starts inside <think>",
        rec_think["prompt"]["starts_in_thinking"] is True,
    )
    client.put("/api/config", json={"enable_thinking": False, "max_tokens": 64})

    # ------------------------------------- 4. cache reuse and offset verification
    state = client.get("/api/state").json()
    check(
        "checkpoint cache entry was stored",
        state["cache"]["entries_count"] >= 1,
        f"{state['cache']['entries_count']} entries",
    )
    checkpoint_at = rec1["prompt"]["checkpoint_at"]
    check("checkpoint position reported", checkpoint_at is not None, str(checkpoint_at))

    answer1 = rec1["channels"]["raw"].strip()
    turn2 = session_of(
        msg("user", "Reply with the single word: hello"),
        msg("assistant", answer1),
        # Long enough greedy output that "reuse == cold" compares many tokens.
        msg("user", "Count from one to fifteen in words, separated by commas."),
    )
    preview2 = client.post(
        "/api/prompt/preview", json={"session": turn2, "include_pieces": False}
    ).json()
    prediction = preview2["cache_prediction"]
    check(
        "preview predicts checkpoint reuse",
        prediction["reused_tokens"] >= checkpoint_at,
        json.dumps(prediction),
    )

    rec2 = generate(client, "verify-turn2", turn2)
    check(
        "turn 2 reuses at least the checkpoint prefix",
        rec2["stats"]["prefill"]["reused_tokens"] >= checkpoint_at,
        f"reused {rec2['stats']['prefill']['reused_tokens']} tokens",
    )
    check(
        "cache offset verification passes",
        bool(rec2.get("cache_verify", {}).get("ok")),
        json.dumps(rec2.get("cache_verify")),
    )

    client.put("/api/config", json={"cache_reuse": False})
    cold = generate(client, "verify-turn2-cold", turn2)
    warm_ids = [token["id"] for token in rec2["tokens"]]
    cold_ids = [token["id"] for token in cold["tokens"]]
    check(
        "greedy output with reuse equals a cold run",
        warm_ids == cold_ids and len(warm_ids) >= 16,
        f"{len(warm_ids)} vs {len(cold_ids)} tokens",
    )
    check(
        "the cold run really prefilled from scratch",
        cold["stats"]["prefill"]["reused_tokens"] == 0,
        f"reused {cold['stats']['prefill']['reused_tokens']}",
    )
    client.put("/api/config", json={"cache_reuse": True})

    # ------------------------- 4b. every_message checkpoints and their cost
    client.put(
        "/api/config",
        json={"cache_checkpoints": "every_message", "max_tokens": 8},
    )
    convo = session_of(
        msg("user", "u1"),
        msg("assistant", "a1"),
        msg("user", "u2"),
        msg("assistant", "a2"),
        msg("user", "u3"),
    )
    rec_many = generate(client, "verify-every", convo)
    boundaries = rec_many["prompt"]["checkpoints"]
    accepted = {
        entry["after_message"]: entry["position"]
        for entry in boundaries
        if entry["ok"]
    }
    check(
        "every_message reports every message boundary",
        [entry["after_message"] for entry in boundaries] == [1, 2, 3, 4, 5],
        str([entry["after_message"] for entry in boundaries]),
    )
    check(
        "checkpoint after message 3 is accepted",
        3 in accepted,
        json.dumps(accepted),
    )
    skipped = [entry for entry in boundaries if not entry["ok"]]
    check(
        "skipped boundaries carry a reason",
        all(entry["reason"] for entry in skipped),
        json.dumps([entry["reason"] for entry in skipped])[:200],
    )
    state = client.get("/api/state").json()
    checkpoint_rows = [
        entry for entry in state["cache"]["entries"] if entry["kind"] == "checkpoint"
    ]
    check(
        "every_message stored multiple checkpoints",
        len(checkpoint_rows) >= 3,
        f"{len(checkpoint_rows)} checkpoint entries",
    )
    check(
        "cache rows split recurrent state and KV bytes",
        all(
            entry["arrays_bytes"] > 0 and entry["kv_bytes"] > 0
            for entry in checkpoint_rows
        ),
        str(
            [
                (entry["after_message"], entry["arrays_bytes"], entry["kv_bytes"])
                for entry in checkpoint_rows
            ]
        ),
    )
    check(
        "cache summary reports limit, utilization and the sharing note",
        state["cache"]["limit_bytes"] == int(config["cache_max_gb"] * 1_000_000_000)
        and state["cache"]["utilization"] is not None
        and "three deepcopies" in state["cache"]["note"],
    )

    edited_messages = [dict(message) for message in convo["messages"]]
    edited_messages[3] = {**edited_messages[3], "content": "a2 edited"}
    edited = {
        "mode": "messages",
        "messages": edited_messages,
        "raw_text": "",
    }
    preview_edit = client.post(
        "/api/prompt/preview", json={"session": edited, "include_pieces": False}
    ).json()
    estimate = preview_edit["checkpoint_estimate"]
    check(
        "checkpoint estimate is scaled from a measured entry",
        estimate is not None
        and estimate["total_bytes"] > 0
        and estimate["derived_from"]["tokens"] > 0,
        json.dumps(estimate)[:240],
    )
    rec_edit = generate(client, "verify-every-edit", edited)
    check(
        "editing message 4 reuses the checkpoint after message 3",
        rec_edit["stats"]["prefill"]["reused_tokens"] == accepted.get(3),
        f"reused {rec_edit['stats']['prefill']['reused_tokens']}, expected {accepted.get(3)}",
    )

    client.put("/api/config", json={"system_prompt": "You are terse."})
    system_preview = client.post(
        "/api/prompt/preview", json={"session": convo, "include_pieces": False}
    ).json()
    system_entry = next(
        (
            entry
            for entry in system_preview["checkpoints"]
            if entry["after_message"] == 1
        ),
        None,
    )
    check(
        "system-only boundary is skipped with a reason",
        system_entry is not None
        and system_entry["ok"] is False
        and bool(system_entry["reason"]),
        json.dumps(system_entry),
    )
    client.put(
        "/api/config",
        json={
            "system_prompt": "",
            "cache_checkpoints": "last_message",
            "max_tokens": 64,
        },
    )

    # ------------------------------------------ 5. cache invalidation
    state = client.get("/api/state").json()
    check(
        "cache has entries before invalidation",
        state["cache"]["entries_count"] >= 1,
    )
    client.put("/api/config", json={"kv_bits": 4})
    state = client.get("/api/state").json()
    check(
        "changing kv_bits clears the prompt cache",
        state["cache"]["entries_count"] == 0,
    )
    client.put("/api/config", json={"kv_bits": None})

    generate(client, "verify-refill", turn1)
    state = client.get("/api/state").json()
    check("cache refilled before reload", state["cache"]["entries_count"] >= 1)
    reload = client.post("/api/model/reload")
    state = client.get("/api/state").json()
    check(
        "reloading the model clears the prompt cache",
        reload.status_code == 200 and state["cache"]["entries_count"] == 0,
        f"HTTP {reload.status_code}",
    )

    # ------------------------------------------------ 6. cancellation
    client.put(
        "/api/config",
        json={"enable_thinking": True, "max_tokens": 512, "temperature": 1.0},
    )
    long_session = session_of(
        msg("user", "Write a long, detailed essay about the history of computing.")
    )
    started = client.post(
        "/api/generate",
        json={"client_request_id": "verify-stop", "session": long_session},
    )
    gen_id = started.json()["generation_id"]
    tokens_at_stop = 0
    for _ in range(400):
        record = client.get(f"/api/generations/{gen_id}").json()
        tokens_at_stop = record["tokens_generated"]
        if tokens_at_stop >= 5 or record["status"] == "finished":
            break
        time.sleep(0.05)
    stop = client.post("/api/generate/stop")
    check("stop is accepted", stop.status_code == 200 and stop.json()["stopping"] is True)
    stopped = wait_for_generation(client, gen_id)
    calls_at_finish = client.get("/api/state").json()["counters"]["forward_calls_total"]
    time.sleep(1.0)
    calls_later = client.get("/api/state").json()["counters"]["forward_calls_total"]
    check(
        "stop finishes with reason cancelled",
        stopped["finish"]["reason"] == "cancelled",
        json.dumps(stopped["finish"]),
    )
    check(
        "cancellation stops computation",
        calls_later == calls_at_finish,
        f"{calls_at_finish} -> {calls_later}",
    )

    # ------------------------------- 7. idempotent generate + SSE replay
    client.put(
        "/api/config",
        json={"enable_thinking": False, "max_tokens": 8, "temperature": 0.0},
    )
    seq_before = client.get("/api/state").json()["events"]["seq"]
    generations_before = client.get("/api/state").json()["generations_total"]
    idem_session = session_of(msg("user", "Say yes."))
    first = client.post(
        "/api/generate",
        json={"client_request_id": "verify-idem", "session": idem_session},
    ).json()
    duplicate = client.post(
        "/api/generate",
        json={"client_request_id": "verify-idem", "session": idem_session},
    ).json()
    check(
        "duplicate client_request_id returns the same generation",
        duplicate["generation_id"] == first["generation_id"]
        and duplicate["deduplicated"] is True,
        json.dumps(duplicate),
    )
    idem_record = wait_for_generation(client, first["generation_id"])
    frames = collect_sse(
        client,
        f"{base}/api/events",
        seq_before,
        "generation.finished",
        "generation_id",
        first["generation_id"],
    )
    started_count = sum(
        1
        for frame in frames
        if frame["type"] == "generation.started"
        and frame["data"].get("generation_id") == first["generation_id"]
    )
    check(
        "SSE replay shows exactly one generation.started",
        started_count == 1,
        f"{started_count} started frames, {len(frames)} frames replayed",
    )
    check(
        "SSE replay from a recent id does not resync",
        all(frame["type"] != "resync" for frame in frames),
    )
    check(
        "duplicate request did not create a second generation",
        client.get("/api/state").json()["generations_total"] == generations_before + 1,
        f"{generations_before} -> {client.get('/api/state').json()['generations_total']}",
    )

    # ------------------------------------------------- 8. no inference while idle
    check("generation finished before idle check", idem_record["status"] == "finished")
    with client.stream("GET", f"{base}/api/events") as response:
        check("SSE connects for idle check", response.status_code == 200)
        before = client.get("/api/state").json()
        time.sleep(10.0)
        after = client.get("/api/state").json()
    check(
        "no inference while idle (forward_calls_total unchanged)",
        before["counters"]["forward_calls_total"]
        == after["counters"]["forward_calls_total"],
        f"{before['counters']['forward_calls_total']} -> {after['counters']['forward_calls_total']}",
    )
    check(
        "no generation is active while idle",
        after["active_generation"] is None,
    )


# --------------------------------------------------------------------------
def main() -> int:
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    server = None
    thread = None
    try:
        server, thread = start_server(port)
        print(f"verify: server on {base}\n")
        with httpx.Client(base_url=base, timeout=180.0) as client:
            run_checks(client, base)
    except Exception:  # noqa: BLE001 - a failed check must still print the summary
        traceback.print_exc()
        RESULTS.append(("verification run", False, "unexpected exception"))
    finally:
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=10)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = len(RESULTS) - passed
    print(f"\n{passed} passed, {failed} failed, {len(RESULTS)} checks")
    if failed:
        print("failed:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  - {name} {detail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
