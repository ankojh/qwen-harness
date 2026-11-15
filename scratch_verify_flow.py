#!/usr/bin/env python3
"""Drive the plan's Done-when flow against the running 8020 instance.

Alice → What is my name? → edit to Bob → relabel → reuse off, collecting the
real numbers for the report.  Read-only: only makes requests a user would.
"""
import json
import time
import urllib.request
import uuid

BASE = "http://127.0.0.1:8020"


def req(method, path, payload=None, raw=False):
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            body = resp.read().decode()
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"  !! {method} {path} -> {e.code}: {body[:200]}")
        return {"_error": e.code, "_detail": body[:200]}
    return json.loads(body) if body else {}


def state():
    return req("GET", "/api/state")


def wait_idle(timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = state()
        if not s.get("active_generation"):
            return s
        time.sleep(0.5)
    raise RuntimeError("generation did not finish")


def generate(messages):
    r = req(
        "POST",
        "/api/generate",
        {
            "client_request_id": str(uuid.uuid4()),
            "session": {
                "mode": "messages",
                "messages": [
                    {
                        "role": m["role"],
                        "content": m["content"],
                        "reasoning_content": None,
                        "enabled": m.get("enabled", True),
                    }
                    for m in messages
                ],
                "raw_text": "",
            },
        },
    )
    if r.get("_error"):
        return r
    gid = r.get("generation_id")
    wait_idle()
    gen = req("GET", f"/api/generations/{gid}")
    return gen


def prompt_of(messages):
    r = req(
        "POST",
        "/api/prompt/preview",
        {"session": {"mode": "messages", "messages": messages, "raw_text": ""}, "include_pieces": True},
    )
    return r


def main():
    s = state()
    print(f"model_loaded={s.get('model_loaded')} cache entries={len(s['cache']['entries'])}")

    # ---- turn 1: My name is Alice.
    msgs = [{"role": "user", "content": "My name is Alice.", "enabled": True}]
    p1 = prompt_of(msgs)
    print(f"\n[turn 1 preview] tokens={p1['budget']['input_tokens']} "
          f"reuse={p1['cache_prediction']['reused_tokens']} reason={p1['cache_prediction']['reason']}")
    g1 = generate(msgs)
    st1 = g1["stats"]["prefill"]
    answer1 = g1["channels"]["answer"].strip().replace("\n", " ")
    think1 = len([t for t in g1["tokens"] if t["channel"] == "thinking"])
    print(f"[turn 1 ran] answer={answer1!r} thinking_tokens={think1} "
          f"prefill.reused={st1['reused_tokens']} prefill.processed={st1['processed_tokens']} "
          f"generated={g1['stats']['decode']['generated_tokens']}")
    s1 = state()
    e1 = s1["cache"]["entries"]
    for e in e1:
        print(f"  entry #{e['id']} {e['kind']} tokens={e['tokens']} nbytes={e['nbytes']}")
    t1 = req("GET", "/api/cache/tokens")
    ids1 = {e["id"]: e["token_ids"] for e in t1["entries"]}
    print(f"  entry token lengths: " + ", ".join(f"#{k}:{len(v)}" for k, v in ids1.items()))

    # ---- turn 2 preview: What is my name?
    msgs2 = msgs + [
        {"role": "assistant", "content": answer1, "enabled": True},
        {"role": "user", "content": "What is my name?", "enabled": True},
    ]
    p2 = prompt_of(msgs2)
    cp = p2["cache_prediction"]
    print(f"\n[turn 2 preview] tokens={p2['budget']['input_tokens']} "
          f"prediction entry={cp['entry_id']} reuse={cp['reused_tokens']} reason={cp['reason']}")
    for c in p2["cache_entries"]:
        print(f"  compare #{c['id']} {c['kind']}: shared_prefix={c['shared_prefix']} "
              f"reusable={c['reusable_tokens']} reason={c['reason']}")
        win = c["token_window"]
        b = c["shared_prefix"]
        at = next((t for t in win if t["position"] == b), None)
        print(f"    lens @ break {b}: entry ⟨{at['piece'] if at else '?'}⟩")

    # ---- turn 2: send
    g2 = generate(msgs2)
    st2 = g2["stats"]["prefill"]
    answer2 = g2["channels"]["answer"].strip().replace("\n", " ")
    print(f"[turn 2 ran] answer={answer2!r} prefill.reused={st2['reused_tokens']} "
          f"lookup={g2['cache_lookup']['reused_tokens']} from #{g2['cache_lookup']['entry_id']}")
    s2 = state()
    e2 = s2["cache"]["entries"]
    print(f"  cache now: {len(e2)} entries")
    for e in e2:
        print(f"    #{e['id']} {e['kind']} tokens={e['tokens']}")

    # ---- edit Alice -> Bob (staged preview only: what-if)
    msgsB = [{"role": "user", "content": "My name is Bob.", "enabled": True}] + msgs2[1:]
    pB = prompt_of(msgsB)
    print(f"\n[edit Alice→Bob preview] tokens={pB['budget']['input_tokens']} "
          f"reuse={pB['cache_prediction']['reused_tokens']} reason={pB['cache_prediction']['reason']}")
    for c in pB["cache_entries"]:
        print(f"  compare #{c['id']}: shared_prefix={c['shared_prefix']} reusable={c['reusable_tokens']}")

    # ---- relabel the checkpoint: Alice -> Bob
    entry_to_relabel = None
    for e in e2:
        if e["kind"] == "checkpoint":
            entry_to_relabel = e
    print(f"\n[relabel] checkpoint entry #{entry_to_relabel['id']} has {entry_to_relabel['tokens']} tokens")
    detail = req("GET", f"/api/cache/entries/{entry_to_relabel['id']}")
    print(f"  detail: {len(detail['token_ids'])} tokens, spans={[(sp['role'], sp['start'], sp['end']) for sp in detail['spans']]}")
    # find the ' Alice' token position (id 28445 in the model vocab per earlier work)
    pos_alice = next((i for i, t in enumerate(detail["tokens"]) if t["text"] == " Alice"), None)
    print(f"  ' Alice' token at position {pos_alice} (id {detail['tokens'][pos_alice]['id'] if pos_alice is not None else '?'})")
    if pos_alice is not None:
        rr = req("POST", f"/api/cache/entries/{entry_to_relabel['id']}/relabel",
                 {"position": pos_alice, "text": " Bob"})
        print(f"  relabel -> {json.dumps(rr)[:200]}")
    pB2 = prompt_of(msgsB)
    print(f"[relabel then Bob preview] reuse={pB2['cache_prediction']['reused_tokens']} "
          f"relabeled={pB2['cache_prediction']['relabeled']} reason={pB2['cache_prediction']['reason']}")
    gB = generate(msgsB)
    print(f"[Bob with relabel ran] answer={gB['channels']['answer'].strip().replace(chr(10),' ')!r} "
          f"reused={gB['stats']['prefill']['reused_tokens']}")
    req("POST", f"/api/cache/entries/{entry_to_relabel['id']}/restore")
    print("[relabel] restored entry labels")

    # ---- reuse off
    req("PUT", "/api/config", {"cache_reuse": False})
    pOff = prompt_of(msgsB)
    print(f"\n[reuse off] reuse={pOff['cache_prediction']['reused_tokens']} "
          f"reason={pOff['cache_prediction']['reason']} "
          f"potential={pOff['cache_prediction'].get('potential_reused_tokens')}")
    req("PUT", "/api/config", {"cache_reuse": True})
    print("[reuse back on]")

    # ---- clear the cache for the long-context check to start clean-ish
    req("POST", "/api/cache/clear")


if __name__ == "__main__":
    main()
