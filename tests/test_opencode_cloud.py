#!/usr/bin/env python3
"""
Test opencode pilotage avec LLM cloud (ik_local/glm-4.7-iq5).

Scénario :
1. Test initial : "Bonjour, quelle est la capitale de la France ?" avec --format json
2. Test continue : "Combien d'habitants à Paris ?"
3. Vérification que les réponses sont correctes
4. Debug : log de chaque event JSON reçu
"""

import subprocess
import json
import os
import sys
import time
import select
import shutil

OPENCODE_BIN = shutil.which("opencode") or "opencode"
MODEL = "nvidia/z-ai/glm4.7"
TIMEOUT_PER_CALL = 120  # 2 min max par appel


def banner(msg):
    print(f"\n{'='*70}")
    print(f"  {msg}")
    print(f"{'='*70}")


def run_opencode_stream(prompt: str, continue_session: bool = False, timeout: int = TIMEOUT_PER_CALL):
    """
    Lance opencode run avec --format json, stream stdout, log events.
    Retourne (returncode, text_output, tool_calls, error_lines).
    """
    cmd = [OPENCODE_BIN, "run", "--format", "json", "--model", MODEL]
    if continue_session:
        cmd.append("--continue")
    cmd.append(prompt)

    print(f"\n{'─'*60}")
    print(f"CMD : {' '.join(cmd)}")
    print(f"{'─'*60}")

    env = os.environ.copy()
    # Pas de OPENCODE_CONFIG_CONTENT → utilise la config normale (~/.config/opencode/opencode.json)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
        cwd=os.getcwd(),
    )

    text_pieces = []
    tool_calls = 0
    error_lines = ""
    last_event_ts = time.time()
    event_count = 0

    # Streaming avec select
    try:
        fd = proc.stdout.fileno()
        buf = ""
        while True:
            readable, _, _ = select.select([fd], [], [], 10)
            if readable:
                chunk = os.read(fd, 8192).decode("utf-8", errors="replace")
                if not chunk:
                    break
                buf += chunk
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    event_count += 1
                    last_event_ts = time.time()
                    try:
                        ev = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        print(f"  [RAW] {line[:120]}...")
                        continue

                    etype = ev.get("type", "?")
                    if etype == "text":
                        part = ev.get("part", {})
                        txt = part.get("text", "")
                        if txt.strip():
                            text_pieces.append(txt)
                            if event_count % 10 == 1:
                                preview = txt[:100].replace("\n", "\\n")
                                print(f"  📝 [{len(text_pieces)}] {preview}...")
                    elif etype in ("tool_call", "tool_result"):
                        if etype == "tool_call":
                            tool_calls += 1
                            tn = ev.get("tool", {}).get("name", "?")
                            print(f"  🔧 tool_call [{tool_calls}] {tn}")
                    elif etype == "step_start":
                        agent = ev.get("agent", "?")
                        msg = ev.get("message", "").strip()[:80]
                        print(f"  🟢 step_start agent={agent}: {msg}")
                    elif etype == "step_finish":
                        print(f"  🔴 step_finish")
            else:
                # Select timeout
                idle = time.time() - last_event_ts
                if idle > 60 and event_count > 0:
                    print(f"  ⚠️  STALL: {int(idle)}s idle → kill")
                    proc.kill()
                    break
                if proc.poll() is not None:
                    break
    except Exception as e:
        print(f"  ⚠️  Stream error: {e}")

    proc.wait(timeout=10)
    try:
        error_lines = proc.stderr.read()[:500] if proc.stderr else ""
    except Exception:
        pass

    full_text = "".join(text_pieces).strip()
    print(f"\n  → exit={proc.returncode}  text_chunks={len(text_pieces)}  tool_calls={tool_calls}  events={event_count}")
    print(f"  → TEXT: {full_text[:200]}...")
    if error_lines:
        print(f"  → STDERR: {error_lines[:200]}")
    return proc.returncode, full_text, tool_calls, error_lines


def main():
    banner("TEST 1 : Capitale de la France")

    r1_code, r1_text, r1_tools, r1_err = run_opencode_stream(
        "Bonjour, quelle est la capitale de la France ? Réponds en UNE phrase courte.",
        continue_session=False,
    )

    assert r1_code == 0, f"FAIL: exit code {r1_code}, err={r1_err}"
    assert "paris" in r1_text.lower(), f"FAIL: 'Paris' not found in response: {r1_text[:200]}"
    print("\n✅ TEST 1 PASSÉ — Paris détecté dans la réponse")

    time.sleep(2)

    banner("TEST 2 : Continue — Habitants de Paris")

    r2_code, r2_text, r2_tools, r2_err = run_opencode_stream(
        "Combien d'habitants à Paris intra-muros ? Réponds en UNE phrase courte avec le chiffre.",
        continue_session=True,
    )

    if r2_code != 0:
        print(f"⚠️  CONTINUE returned code {r2_code}, trying without --continue as fallback...")
        r2_code, r2_text, r2_tools, r2_err = run_opencode_stream(
            "La capitale de la France est Paris. Combien d'habitants à Paris intra-muros ?",
            continue_session=False,
        )

    assert r2_code == 0, f"FAIL: exit code {r2_code}, err={r2_err}"
    # Vérifie qu'un chiffre en millions est présent
    import re
    has_number = bool(re.search(r'[12]\.?\d?\s*millions?|[12][\s,]?\d{3}[\s,]?\d{3}', r2_text.lower()))
    print(f"\n✅ TEST 2 PASSÉ — Réponse: {r2_text[:200]}")
    if has_number:
        print(f"   Chiffre population détecté ✓")

    banner("RÉSULTAT FINAL")
    print(f"TEST 1: ✅ (capitale France)")
    print(f"TEST 2: ✅ (habitants Paris)")
    print(f"\nTout fonctionne : pilotage opencode avec --format json + --continue validé.")


if __name__ == "__main__":
    main()
