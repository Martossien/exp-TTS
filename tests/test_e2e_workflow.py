#!/usr/bin/env python3
"""
Test end-to-end complet : multi-model STT → ZIP → arbitrage LLM.
Utilise les fichiers tests/test1.mp3, test2.mp3, test3.mp3.

Workflow :
1. VRAM check + kill LLM si nécessaire
2. run_multi_model.py sur chaque test file
3. Création ZIP
4. Relance LLM + attente
5. Arbitrage via opencode avec Qwen 27B vLLM
6. Vérification SRT final
"""

import os
import sys
import time
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# ── Configuration ────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = sys.executable
OPENCODE_BIN = os.environ.get("VOXTRAL_OPENCODE_BIN", shutil.which("opencode") or "opencode")
LLM_SCRIPT = os.environ.get("VOXTRAL_LAUNCH_SCRIPT", os.path.expanduser("~/launch_arbitrage.sh"))
LLM_PORT = 8080
ARB_MODEL = os.environ.get("VOXTRAL_ARB_MODEL", "arbitrage2/qwen36-27b-fp8")
TEST_FILES = [
    f"{PROJECT_DIR}/tests/test1.mp3",
    f"{PROJECT_DIR}/tests/test2.mp3",
    f"{PROJECT_DIR}/tests/test3.mp3",
]
OUTPUTS = f"{PROJECT_DIR}/outputs"
MODELS = ["large-v3", "cohere-transcribe-03-2026", "qwen3-asr-1.7b", "voxtral-mini-3b"]

sys.path.insert(0, PROJECT_DIR)


def run(cmd, timeout=600, check=True, cwd=None, env=None, log_prefix=""):
    """Lance une commande, log stdout/stderr en direct."""
    print(f"\n{'─'*60}")
    print(f"[CMD] {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    proc = subprocess.Popen(
        cmd if isinstance(cmd, list) else cmd,
        shell=isinstance(cmd, str),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=cwd, env=env,
    )
    out_lines = []
    start = time.time()
    for line in proc.stdout:
        elapsed = int(time.time() - start)
        line = line.rstrip()
        out_lines.append(line)
        if line.strip():
            print(f"  [{elapsed}s] {line[:200]}")
    proc.wait(timeout=timeout)
    rc = proc.returncode
    full = "\n".join(out_lines)
    if check and rc != 0:
        raise RuntimeError(f"Command failed (exit {rc}): {full[-500:]}")
    print(f"  → exit={rc}, {len(out_lines)} lignes, {int(time.time()-start)}s")
    return rc, full


def api_get(path, timeout=5):
    import requests
    return requests.get(f"http://localhost:{LLM_PORT}{path}", timeout=timeout)


def api_post(path, data, timeout=30):
    import requests
    return requests.post(f"http://localhost:{LLM_PORT}{path}", json=data, timeout=timeout)


def check_llm_ready():
    """Vérifie que la LLM est opérationnelle (health + completion test)."""
    for attempt in range(60):
        try:
            r = api_get("/health")
            if r.status_code == 200:
                break
        except Exception:
            pass
        if attempt % 10 == 0:
            print(f"  Attente LLM /health... {attempt*2}s")
        time.sleep(2)
    else:
        raise RuntimeError("LLM non disponible après 120s")

    for retry in range(3):
        try:
            r = api_post("/v1/chat/completions", {
                "model": "qwen36-27b-fp8",
                "messages": [{"role": "user", "content": "Dis OK"}],
                "max_tokens": 100, "temperature": 0,
            })
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"]
                if content and content.strip():
                    print(f"  ✅ LLM répond: {content.strip()[:80]}")
                    return True
        except Exception as e:
            print(f"  ⚠️ Vérif LLM échouée ({e}), retry {retry+1}/3")
        time.sleep(5)
    raise RuntimeError("LLM ne répond pas après 3 tentatives")


def kill_llm():
    """Tue les processus GPU connus (vllm, llama, ik_llama)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        killed = 0
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                pid, name = parts[0], parts[1].lower()
                if any(kw in name for kw in ("vllm", "llama", "ik_llama", "qwen")):
                    try:
                        os.kill(int(pid), 15)  # SIGTERM
                        print(f"  🔫 SIGTERM → PID {pid} ({name})")
                        killed += 1
                    except Exception:
                        pass
        if killed:
            print(f"  {killed} processus tués, attente 10s...")
            time.sleep(10)
        return killed
    except Exception as e:
        print(f"  ⚠️ Erreur kill: {e}")
        return 0


def run_multi_model(audio_path):
    """Lance run_multi_model.py sur un fichier audio, retourne (zip_path, status)."""
    basename = Path(audio_path).stem
    ts = datetime.now().strftime("%m%d%H%M%S")
    zip_path = os.path.join(OUTPUTS, f"{basename}-multi-{ts}.zip")
    # Supprimer les anciens TXT pour ce basename
    for old in Path(OUTPUTS).glob(f"{basename}-*-{ts}*.txt"):
        old.unlink(missing_ok=True)

    cmd = [
        PYTHON, f"{PROJECT_DIR}/run_multi_model.py",
        "--input", audio_path,
        "--models", ",".join(MODELS),
        "--language", "french",
        "--output-dir", OUTPUTS,
    ]
    run(cmd, timeout=3600, cwd=PROJECT_DIR)

    # Trouver les TXT produits (les plus récents)
    txt_files = sorted(
        [p for p in Path(OUTPUTS).glob(f"{basename}-*.txt")
         if "summary" not in p.name and "aligned" not in p.name],
        key=os.path.getmtime, reverse=True,
    )
    # Garder seulement ceux du dernier run
    if txt_files:
        latest_ts = os.path.getmtime(txt_files[0])
        txt_files = [p for p in txt_files if os.path.getmtime(p) >= latest_ts - 5]

    if len(txt_files) < 2:
        raise RuntimeError(f"Pas assez de TXT ({len(txt_files)}) — le run a échoué")

    print(f"  TXT produits: {len(txt_files)} fichiers")
    for tf in txt_files:
        print(f"    {tf.name} ({os.path.getsize(tf)} bytes)")

    # Créer le ZIP
    with __import__('zipfile').ZipFile(zip_path, "w") as zf:
        for tf in txt_files:
            zf.write(str(tf), tf.name)

    summary_files = sorted(Path(OUTPUTS).glob(f"{basename}-multi-summary-*.txt"),
                           key=os.path.getmtime, reverse=True)
    if summary_files:
        with __import__('zipfile').ZipFile(zip_path, "a") as zf:
            zf.write(str(summary_files[0]), summary_files[0].name)

    print(f"  ✅ ZIP créé: {zip_path} ({os.path.getsize(zip_path)} bytes, {len(txt_files)+1} fichiers)")

    # Vérifier aligné
    aligned = sorted(Path(OUTPUTS).glob(f"{basename}-aligned-*.txt"),
                     key=os.path.getmtime, reverse=True)
    if aligned:
        print(f"  Fichier aligné: {aligned[0].name} ({os.path.getsize(aligned[0])} bytes)")

    return zip_path, txt_files


def run_arbitrage(zip_path, lexicon_text=""):
    """Lance l'arbitrage via opencode sur un ZIP."""
    import tempfile, requests
    from app import App

    # Instance minimale
    app = App.__new__(App)
    app.args = type('Args', (), {'output_dir': OUTPUTS})()
    app.whisper_inf = type('WI', (), {'device': 'cuda'})()
    app._arb_config = {
        "opencode_bin": OPENCODE_BIN,
        "launch_script": LLM_SCRIPT,
        "model_id": ARB_MODEL,
        "api_port": LLM_PORT,
    }

    # Prompt minimal
    prompt_text = """# Arbitrage SRT — Test

Tu es un arbitre de transcription. Tu reçois des transcriptions alternatives pour chaque fenêtre.

ÉTAPE 0 : Découvre le répertoire de travail.
ÉTAPE 1 : Analyse les transcriptions.
ÉTAPE 2 : Choisis la meilleure par fenêtre.
ÉTAPE 3 : Écris le SRT final nommé NOM_AUDIO-arbitrage-TIMESTAMP.srt.

Fais juste une sélection rapide, pas besoin d'analyse détaillée pour ce test.
"""

    result = app.run_arbitration_for_web(
        zip_path,
        lexicon_text,
        prompt_text,
        LLM_SCRIPT,
        LLM_PORT,
        ARB_MODEL,
    )

    if isinstance(result, tuple) and len(result) == 2:
        status_text, download_update = result
        print(f"\n📊 STATUT ARBITRAGE :\n{status_text}")
        return status_text
    return str(result)


def main():
    print("="*70)
    print("  TEST END-TO-END : Multi-Model → ZIP → Arbitrage LLM")
    print(f"  LLM: {ARB_MODEL} (port {LLM_PORT})")
    print(f"  Fichiers: {len(TEST_FILES)} tests")
    print("="*70)

    results = {}

    for i, audio_file in enumerate(TEST_FILES):
        basename = Path(audio_file).stem
        print(f"\n{'#'*70}")
        print(f"# TEST {i+1}/{len(TEST_FILES)} : {basename}")
        print(f"{'#'*70}")

        # 1. Vérifier VRAM + kill LLM si nécessaire
        print("\n[1/5] VRAM check + kill LLM...")
        try:
            r = api_get("/health", timeout=2)
            if r.status_code == 200:
                print("  LLM détectée sur port 8080 → kill pour libérer VRAM")
                kill_llm()
        except Exception:
            print("  Pas de LLM sur port 8080")

        # 2. Attendre libération VRAM
        print("\n[2/5] Attente libération VRAM...")
        time.sleep(5)
        result_vram = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        print(f"  VRAM libre: {result_vram.stdout.strip()}")

        # 3. Multi-model
        print(f"\n[3/5] Multi-Model sur {basename}...")
        try:
            zip_path, txt_files = run_multi_model(audio_file)
        except Exception as e:
            print(f"  ❌ Multi-Model échoué: {e}")
            results[basename] = f"MULTI_MODEL_FAILED: {e}"
            continue

        # 4. Relancer LLM
        print(f"\n[4/5] Relance LLM ({LLM_SCRIPT})...")
        run(["bash", LLM_SCRIPT], timeout=300, check=False, cwd=PROJECT_DIR)
        time.sleep(5)

        # 5. Attendre LLM prête
        print("\n[5/5] Attente LLM prête + arbitrage...")
        try:
            check_llm_ready()
        except Exception as e:
            print(f"  ❌ LLM non prête: {e}")
            results[basename] = f"LLM_NOT_READY: {e}"
            continue

        # 6. Arbitrage
        try:
            status = run_arbitrage(zip_path, "ORG-ALPHA — Organisation\n")
            if "✅ SRT arbitré" in status:
                results[basename] = "OK"
                print(f"\n  ✅ {basename} : ARBITRAGE RÉUSSI")
            elif "Erreur" in status or "❌" in status:
                results[basename] = f"ARBITRAGE_ERROR: {status[-500:]}"
                print(f"\n  ❌ {basename} : ÉCHEC ARBITRAGE")
            else:
                results[basename] = "UNKNOWN"
        except Exception as e:
            results[basename] = f"EXCEPTION: {e}"
            print(f"\n  ❌ {basename} : EXCEPTION {e}")

    # Résumé final
    print("\n" + "="*70)
    print("  RÉSULTATS FINAUX")
    print("="*70)
    for name, result in results.items():
        emoji = "✅" if result == "OK" else "❌"
        print(f"  {emoji} {name}: {result}")


if __name__ == "__main__":
    main()
