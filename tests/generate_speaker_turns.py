#!/usr/bin/env python3
"""
Génère un fichier de référence des changements de locuteur
via pyannote community-1 UNIQUEMENT (aucune transcription STT).

Sortie : speaker_turns.txt avec format :
HH:MM:SS.mmm → HH:MM:SS.mmm  SPEAKER_XX

Usage: python generate_speaker_turns.py <audio_file> [output_path]
"""

import sys, os, time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from modules.diarize.diarize_pipeline import DiarizationPipeline, resolve_local_pyannote_checkpoint
from modules.diarize.audio_loader import load_audio, SAMPLE_RATE
from modules.utils.paths import DIARIZATION_MODELS_DIR


def fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <audio_file> [output_path]")
        sys.exit(1)

    audio_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not output_path:
        base = os.path.splitext(os.path.basename(audio_path))[0]
        output_path = os.path.join(os.path.dirname(audio_path) or ".", f"{base}-speaker-turns.txt")

    print(f"Audio : {audio_path}")
    print(f"Sortie: {output_path}")

    # ── Charger pyannote ─────────────────────────────────────────────
    print("Chargement pyannote community-1...")
    t0 = time.time()
    pipe = DiarizationPipeline(
        model_name="pyannote/speaker-diarization-community-1",
        cache_dir=DIARIZATION_MODELS_DIR,
        device=torch.device("cuda"),
    )
    print(f"  Modèle chargé en {time.time()-t0:.1f}s")

    # ── Diarization ──────────────────────────────────────────────────
    print("Diarization en cours...")
    t0 = time.time()
    diarize_df = pipe(audio_path)
    elapsed = time.time() - t0
    print(f"  Diarization terminée en {elapsed:.1f}s")

    if diarize_df.empty:
        print("Erreur: aucun segment trouvé")
        sys.exit(1)

    # ── Statistiques ─────────────────────────────────────────────────
    speakers = diarize_df["speaker"].unique()
    total_turns = len(diarize_df)
    total_duration = diarize_df["end"].max() - diarize_df["start"].min()

    # Fusionner les segments adjacents du même speaker
    merged = []
    current_speaker = None
    current_start = None
    current_end = None

    for _, row in diarize_df.iterrows():
        spk = row["speaker"]
        start = row["start"]
        end = row["end"]

        if spk == current_speaker and start - current_end < 0.5:
            current_end = max(current_end, end)
        else:
            if current_speaker is not None:
                merged.append((current_start, current_end, current_speaker))
            current_speaker = spk
            current_start = start
            current_end = end

    if current_speaker is not None:
        merged.append((current_start, current_end, current_speaker))

    # ── Écrire le fichier ────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# SPEAKER TURNS — Référence pyannote community-1\n")
        f.write(f"# Audio  : {os.path.basename(audio_path)}\n")
        f.write(f"# Durée  : {fmt_time(total_duration)}\n")
        f.write(f"# Locuteurs: {len(speakers)} ({', '.join(sorted(speakers))})\n")
        f.write(f"# Segments bruts: {total_turns}\n")
        f.write(f"# Segments fusionnés: {len(merged)}\n")
        f.write(f"#\n")
        f.write(f"# Format: DÉBUT → FIN  SPEAKER\n")
        f.write(f"#\n")

        for start, end, spk in merged:
            f.write(f"{fmt_time(start)} → {fmt_time(end)}  {spk}\n")

    print(f"\n✅ Fichier écrit: {output_path}")
    print(f"   Locuteurs: {len(speakers)}")
    print(f"   Segments fusionnés: {len(merged)}")
    print(f"   Durée totale: {fmt_time(total_duration)}")

    # Stats par speaker
    for spk in sorted(speakers):
        spk_dur = sum(e - s for s, e, sp in merged if sp == spk)
        spk_turns = sum(1 for _, _, sp in merged if sp == spk)
        print(f"   {spk}: {spk_turns} tours, {spk_dur:.1f}s ({spk_dur/total_duration*100:.0f}%)")


if __name__ == "__main__":
    main()
