#!/usr/bin/env python3
"""
Test isole du backend Voxtral Realtime via vLLM HTTP.

Utilise l'endpoint /v1/audio/transcriptions (approche recommandee, plus fiable
que le WebSocket Realtime qui necessite un protocole de streaming complexe).

Usage:
  python scripts/test_voxtral_realtime_vllm.py tests/test1.wav
  python scripts/test_voxtral_realtime_vllm.py tests/test1.wav --chunk-length 10

Pre-requis:
  - Serveur vLLM lance : bash scripts/start_voxtral_realtime_vllm.sh
  - librosa, soundfile installes : pip install librosa soundfile
"""

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from modules.whisper.data_classes import WhisperParams
from modules.whisper.voxtral_realtime_vllm_inference import VoxtralRealtimeVLLMInference


def main():
    parser = argparse.ArgumentParser(description="Test Voxtral Realtime vLLM")
    parser.add_argument("audio", help="Chemin du fichier audio (WAV, MP3, M4A...)")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--lang", default="fr", help="Code langue (fr, en, ...)")
    parser.add_argument("--chunk-length", type=int, default=30)
    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"ERREUR : fichier audio non trouve : {args.audio}")
        sys.exit(1)

    os.environ["VOXTRAL_VLLM_HOST"] = args.host
    os.environ["VOXTRAL_VLLM_PORT"] = str(args.port)

    from modules.whisper.voxtral_realtime_vllm_inference import _check_server_health
    if not _check_server_health():
        print(f"ERREUR health check sur : http://{args.host}:{args.port}/health")
        print("  Lancez : bash scripts/start_voxtral_realtime_vllm.sh")
        sys.exit(1)

    print(f"Health check OK")

    print(f"Inference Voxtral Realtime vLLM sur : {args.audio}")
    print(f"  langue={args.lang}, chunk_length={args.chunk_length}s")

    inf = VoxtralRealtimeVLLMInference()
    params = WhisperParams(
        model_size="voxtral-realtime-vllm",
        lang=args.lang,
        chunk_length=args.chunk_length,
        chunk_overlap=0,
    )

    t0 = time.time()
    segments, elapsed = inf.transcribe(args.audio, None, None, *params.to_list())

    print(f"Temps : {elapsed:.1f}s")
    print(f"Segments : {len(segments)}")
    print("-" * 60)
    for seg in segments:
        start = seg.start or 0
        end = seg.end or 0
        text = (seg.text or "").strip()
        print(f"  [{start:6.1f}s -> {end:6.1f}s] {text}")
    print("-" * 60)

    ok = any((s.text or "").strip() for s in segments)
    print("SUCCES" if ok else "ECHEC (texte vide)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()