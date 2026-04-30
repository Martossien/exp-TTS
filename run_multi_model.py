#!/usr/bin/env python3
"""
Multi-model ASR batch script.

Runs up to 4 ASR models sequentially on the same audio file.
Diarization is performed once and applied to all model outputs.

Each model produces a TXT file with inline timestamps and speaker labels:
    [HH:MM:SS → HH:MM:SS] SPEAKER_XX: transcribed text

Plus a summary file with timing and stats for each model.

Usage:
    python run_multi_model.py --input audio.m4a
    python run_multi_model.py --input audio.m4a --models large-v3,cohere-transcribe-03-2026
    python run_multi_model.py --input audio.m4a --language english --no-diarization
    python run_multi_model.py --input audio.m4a --output-dir /tmp/results

Available models:
    large-v3                    (Faster-Whisper, fastest, ~5 min/hr)
    cohere-transcribe-03-2026   (Cohere ASR, ~5 min/hr)
    qwen3-asr-1.7b              (Qwen3, ~20 min/hr)
    voxtral-mini-3b             (Voxtral/Mistral, ~14 min/hr)
"""

import argparse
import os
import sys
import time
import gc
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

# ── Project root on sys.path ──────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Offline mode — same as start.sh
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from modules.utils.paths import (
    FASTER_WHISPER_MODELS_DIR, DIARIZATION_MODELS_DIR, OUTPUT_DIR,
    VOXTRAL_MODELS_DIR, QWEN3_ASR_MODELS_DIR, COHERE_ASR_MODELS_DIR,
    VOXTRAL_REALTIME_MODELS_DIR, UVR_MODELS_DIR,
)
from modules.whisper.data_classes import (
    WhisperParams, VadParams, DiarizationParams, WhisperImpl, Segment,
)
from modules.whisper.whisper_factory import WhisperFactory
from modules.diarize.diarizer import Diarizer
from modules.utils.logger import get_logger

logger = get_logger()

# ── Constants ─────────────────────────────────────────────────────────────────

ALL_MODELS = [
    "large-v3",
    "cohere-transcribe-03-2026",
    "qwen3-asr-1.7b",
    "voxtral-mini-3b",
    "voxtral-realtime-vllm",
]

# Language name → ISO 639-1 code (Faster-Whisper requires ISO codes)
LANGUAGE_MAP = {
    "automatic detection": None, "auto": None,
    "afrikaans": "af", "arabic": "ar", "armenian": "hy",
    "azerbaijani": "az", "belarusian": "be", "bosnian": "bs",
    "bulgarian": "bg", "catalan": "ca", "chinese": "zh",
    "croatian": "hr", "czech": "cs", "danish": "da",
    "dutch": "nl", "english": "en", "estonian": "et",
    "finnish": "fi", "french": "fr", "galician": "gl",
    "german": "de", "greek": "el", "hebrew": "he",
    "hindi": "hi", "hungarian": "hu", "icelandic": "is",
    "indonesian": "id", "italian": "it", "japanese": "ja",
    "kannada": "kn", "kazakh": "kk", "korean": "ko",
    "latvian": "lv", "lithuanian": "lt", "macedonian": "mk",
    "malay": "ms", "marathi": "mr", "maori": "mi",
    "nepali": "ne", "norwegian": "no", "persian": "fa",
    "polish": "pl", "portuguese": "pt", "romanian": "ro",
    "russian": "ru", "serbian": "sr", "slovak": "sk",
    "slovenian": "sl", "spanish": "es", "swahili": "sw",
    "swedish": "sv", "tagalog": "tl", "tamil": "ta",
    "thai": "th", "turkish": "tr", "ukrainian": "uk",
    "urdu": "ur", "vietnamese": "vi", "welsh": "cy",
}


def normalize_language(lang: str) -> Optional[str]:
    """Convert full language name or ISO code to ISO 639-1 code."""
    if not lang:
        return None
    key = lang.strip().lower()
    if key in LANGUAGE_MAP:
        return LANGUAGE_MAP[key]
    # Already a 2-letter code
    if len(key) == 2:
        return key
    return key  # pass through, let the model handle it


MODEL_TO_WHISPER_TYPE = {
    "large-v3":                   WhisperImpl.FASTER_WHISPER.value,
    "cohere-transcribe-03-2026":  WhisperImpl.COHERE_ASR.value,
    "qwen3-asr-1.7b":             WhisperImpl.QWEN3_ASR.value,
    "voxtral-mini-3b":            WhisperImpl.VOXTRAL_MINI.value,
    "voxtral-realtime-vllm":      WhisperImpl.VOXTRAL_REALTIME_VLLM.value,
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_time(seconds: float, arrow: bool = False) -> str:
    """Format seconds as HH:MM:SS."""
    td = str(timedelta(seconds=int(seconds)))
    if td.startswith("0:"):
        td = "0" + td          # 0:01:23 → 00:01:23
    return td.zfill(8)         # ensure HH:MM:SS


def fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def segments_to_txt(segments: List[Segment]) -> str:
    """
    Convert segments to a plain-text format with inline timestamps.

    Format: [HH:MM:SS → HH:MM:SS] SPEAKER_XX: text
    If no speaker label, the speaker part is omitted.
    """
    lines = []
    for seg in segments:
        start = fmt_time(seg.start)
        end   = fmt_time(seg.end)
        text  = (seg.text or "").strip()
        if not text:
            continue

        # Speaker label is embedded as "SPEAKER_XX|text" by the diarizer
        if "|" in text:
            speaker, text = text.split("|", 1)
            speaker = speaker.strip()
            text    = text.strip()
            if speaker and speaker != "None":
                lines.append(f"[{start} → {end}] {speaker}: {text}")
                continue

        lines.append(f"[{start} → {end}] {text}")

    return "\n".join(lines)


def apply_diarization_to_segments(
    diarizer: Diarizer,
    audio_path: str,
    segments: List[Segment],
    diarization_result=None,
) -> Tuple[List[Segment], object]:
    """
    If diarization_result is None, run diarization on the audio and return it.
    Otherwise reuse the existing result to assign speakers to segments.
    Returns (diarized_segments, diarization_result).
    """
    from modules.diarize.diarize_pipeline import assign_word_speakers

    if diarization_result is None:
        from modules.diarize.audio_loader import load_audio
        logger.info("[MULTI] Running diarization (once)...")
        audio_np = load_audio(audio_path)
        diarizer.update_pipe(
            device=diarizer.device,
            use_auth_token=os.environ.get("HF_TOKEN", ""),
            model_name="pyannote/speaker-diarization-community-1",
        )
        diarization_result = diarizer.pipe(audio_np)
        logger.info("[MULTI] Diarization complete.")

    diarized = assign_word_speakers(
        diarization_result,
        {"segments": segments},
    )

    out_segments = []
    for seg in diarized["segments"]:
        if isinstance(seg, dict):
            speaker = seg.get("speaker", "None")
            out_segments.append(Segment(
                start=seg["start"],
                end=seg["end"],
                text=f"{speaker}|{seg['text'].strip()}",
            ))
        else:
            speaker = getattr(seg, "speaker", None) or "None"
            out_segments.append(Segment(
                start=seg.start,
                end=seg.end,
                text=f"{speaker}|{seg.text.strip()}",
            ))

    return out_segments, diarization_result


# ── Core ──────────────────────────────────────────────────────────────────────

def run_model(
    model_name: str,
    audio_path: str,
    language: str,
    compute_type: str,
    chunk_length: int,
    chunk_overlap: int,
) -> Tuple[List[Segment], float]:
    """
    Instantiate the right inference backend, run transcribe(), offload.
    Returns (segments, elapsed_seconds).
    """
    whisper_type = MODEL_TO_WHISPER_TYPE[model_name]
    logger.info("[MULTI] Creating inference for model=%s whisper_type=%s", model_name, whisper_type)

    # Chunk-based models (cohere, qwen3, voxtral) use stride = chunk_length - chunk_overlap.
    # With stride=25s and window=30s, LCM(25,30)=150s → bucket collision every 150s
    # (two consecutive chunks share the same 30s bucket, doubling its content).
    # Force chunk_overlap=0 → stride=chunk_length=window=30s → no collision possible.
    _CHUNK_MODELS = {WhisperImpl.COHERE_ASR.value, WhisperImpl.QWEN3_ASR.value, WhisperImpl.VOXTRAL_MINI.value, WhisperImpl.VOXTRAL_REALTIME_VLLM.value}
    if whisper_type in _CHUNK_MODELS:
        chunk_overlap = 0

    inf = WhisperFactory.create_whisper_inference(
        whisper_type=whisper_type,
        faster_whisper_model_dir=FASTER_WHISPER_MODELS_DIR,
        whisper_model_dir=FASTER_WHISPER_MODELS_DIR,
        insanely_fast_whisper_model_dir=FASTER_WHISPER_MODELS_DIR,
        voxtral_model_dir=VOXTRAL_MODELS_DIR,
        qwen3_asr_model_dir=QWEN3_ASR_MODELS_DIR,
        cohere_asr_model_dir=COHERE_ASR_MODELS_DIR,
        voxtral_realtime_model_dir=VOXTRAL_REALTIME_MODELS_DIR,
        diarization_model_dir=DIARIZATION_MODELS_DIR,
        uvr_model_dir=UVR_MODELS_DIR,
        output_dir=OUTPUT_DIR,
    )

    # Build WhisperParams for this model
    params = WhisperParams(
        model_size=model_name,
        lang=normalize_language(language),
        compute_type=compute_type,
        chunk_length=chunk_length,
        chunk_overlap=chunk_overlap,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        enable_offload=True,
    )

    import gradio as gr

    class _NoProgress:
        def __call__(self, *a, **kw): pass

    noop = _NoProgress()

    t0 = time.time()
    segments, _ = inf.transcribe(
        audio_path,
        noop,
        None,
        *params.to_list(),
    )
    elapsed = time.time() - t0

    inf.offload()
    gc.collect()

    return segments, elapsed


def normalize_segments_to_windows(
    segments: List[Segment], window: int = 30, use_start: bool = False
) -> List[Segment]:
    """
    Regroupe les segments en fenêtres non chevauchantes de `window` secondes.

    Méthode d'assignation :
    - use_start=False (modèles chunk) : midpoint du segment.
      Le midpoint place naturellement chaque chunk dans le bucket qui contient
      le plus de son contenu, et déduplique les 5s de chevauchement entre chunks.
    - use_start=True (large-v3 VAD) : début du segment.
      Les segments VAD sont non-chevauchants. Le start évite le décalage de
      phase causé par le midpoint sur les segments qui croisent une frontière
      (ex. [28-35s] : midpoint=31.5→bucket 30, start=28→bucket 0).

    Note : les modèles chunk utilisent chunk_overlap=0 (forcé dans run_model()) →
    stride=chunk_length=window=30s → aucune collision de bucket possible.
    """
    from collections import defaultdict

    buckets: dict = defaultdict(list)
    for seg in segments:
        if seg.start is None or seg.end is None:
            continue
        ref = seg.start if use_start else (seg.start + seg.end) / 2.0
        bucket_start = int(ref // window) * window
        buckets[bucket_start].append(seg)

    result = []
    for bucket_start in sorted(buckets.keys()):
        items = sorted(buckets[bucket_start], key=lambda s: s.start or 0)

        spk_weight: dict = defaultdict(int)
        text_parts = []
        for seg in items:
            raw = (seg.text or "").strip()
            if "|" in raw:
                spk, txt = raw.split("|", 1)
                spk = spk.strip()
                txt = txt.strip()
                if spk and spk != "None":
                    spk_weight[spk] += len(txt)
                if txt:
                    text_parts.append(txt)
            elif raw:
                text_parts.append(raw)

        combined = " ".join(text_parts)
        main_spk = max(spk_weight, key=spk_weight.get) if spk_weight else None

        result.append(Segment(
            start=float(bucket_start),
            end=float(bucket_start + window),
            text=f"{main_spk}|{combined}" if main_spk else combined,
        ))

    return result


def write_output(
    segments: List[Segment],
    output_path: str,
    normalize: bool = True,
    use_start: bool = False,
) -> int:
    """Write segments to file. Returns the final segment count (post-normalization if enabled)."""
    raw_count = len(segments)
    if normalize:
        segments = normalize_segments_to_windows(segments, use_start=use_start)
        logger.info("[MULTI] Written: %s (%d segments → %d fenêtres 30s)",
                    output_path, raw_count, len(segments))
    else:
        logger.info("[MULTI] Written: %s (%d segments natifs)", output_path, raw_count)
    content = segments_to_txt(segments)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return len(segments)


def write_summary(
    audio_path: str,
    audio_duration: float,
    language: str,
    model_results: list,
    output_path: str,
):
    lines = [
        "=" * 60,
        "MULTI-MODEL TRANSCRIPTION SUMMARY",
        "=" * 60,
        f"Source       : {os.path.basename(audio_path)}",
        f"Audio dur.   : {fmt_duration(audio_duration)}",
        f"Language     : {language}",
        f"Run at       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"{'Model':<35} {'Status':<8} {'Time':>8}  {'Fenêtres':>8}  {'Chars':>8}  Output file",
        "-" * 110,
    ]
    for r in model_results:
        status  = "✅ OK" if r["ok"] else "❌ ERR"
        time_s  = fmt_duration(r["elapsed"]) if r["ok"] else "-"
        segs    = str(r["segments"])          if r["ok"] else "-"
        chars   = str(r["chars"])             if r["ok"] else "-"
        outfile = os.path.basename(r["output"]) if r["ok"] else r.get("error", "")
        lines.append(f"{r['model']:<35} {status:<8} {time_s:>8}  {segs:>8}  {chars:>8}  {outfile}")

    lines += ["", "=" * 60]
    content = "\n".join(lines) + "\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(content)


# ── Aligned multi-source file ────────────────────────────────────────────────

# Short display names for the aligned file (canonical order = ALL_MODELS order)
MODEL_SHORT = {
    "large-v3":                  "large-v3",
    "cohere-transcribe-03-2026": "cohere",
    "qwen3-asr-1.7b":            "qwen3",
    "voxtral-mini-3b":           "voxtral",
}


def _parse_txt_to_windows(path: str) -> dict:
    """
    Parse a normalized TXT file into {window_key: display_text}.

    window_key  = "HH:MM:SS→HH:MM:SS"  (no spaces, for dict lookup)
    display_text = the full text part of the line (may include "SPEAKER_XX: …")
    """
    import re
    pat = re.compile(r'^\[(\d{2}:\d{2}:\d{2}) → (\d{2}:\d{2}:\d{2})\] (.+)$')
    windows: dict = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = pat.match(line.rstrip("\n"))
            if not m:
                continue
            key = f"{m.group(1)}→{m.group(2)}"
            text = m.group(3).strip()
            # After normalization each window appears once; guard just in case
            windows[key] = (windows[key] + " " + text) if key in windows else text
    return windows


def _speaker_to_text(display: str) -> str:
    """Strip speaker label: 'SPEAKER_00: text' → 'text'. Returns display unchanged otherwise."""
    if display.startswith("SPEAKER_") and ": " in display:
        return display.split(": ", 1)[1].strip()
    return display.strip()


def _normalize_text(text: str) -> str:
    """Lowercase + remove punctuation + collapse spaces (used for IDEM~ near-match)."""
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def create_aligned_file(
    model_files: dict,   # {full_model_name: abs_file_path}
    output_path: str,
    window_secs: int = 30,
) -> int:
    """
    Merge ≥2 normalized TXT transcriptions into one aligned file.

    Structure
    ---------
    • Header  : global stats + list of divergent window indices
    • Per window separator: ═════ FENÊTRE N : [start → end] [⚠ DIVERGENCE] ═════
    • Per source line:
        "  large-v3  : <full text with speaker>"
        "  cohere    : IDEM large-v3"      ← strict equality (text only, no speaker)
        "  qwen3     : IDEM~ cohere"       ← near-match (normalized, lowercase, no punct)
        "  voxtral   : [vide]"             ← window absent from this model

    DIVERGENCE flag = no strictly-identical pair exists in that window.

    Returns the total number of windows written.
    """
    # ── 1. Load files in canonical model order ────────────────────────────────
    ordered: list = []   # [(short_name, windows_dict)]
    for model_name in ALL_MODELS:
        path = model_files.get(model_name)
        if not path or not os.path.exists(path):
            continue
        short = MODEL_SHORT.get(model_name, model_name)
        ordered.append((short, _parse_txt_to_windows(path)))

    if len(ordered) < 2:
        logger.warning("[MULTI] create_aligned_file: need ≥2 models, got %d — skipped", len(ordered))
        return 0

    # ── 2. Collect all window keys in chronological order ─────────────────────
    all_keys: set = set()
    for _, wdict in ordered:
        all_keys.update(wdict.keys())

    def _key_to_seconds(k: str) -> int:
        h, m, s = k.split("→")[0].strip().split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)

    sorted_keys = sorted(all_keys, key=_key_to_seconds)
    total = len(sorted_keys)

    # ── 3. Identify divergent windows (no strictly-identical pair) ────────────
    divergent: set = set()
    for idx, key in enumerate(sorted_keys, 1):
        texts = [_speaker_to_text(wdict.get(key, "")) for _, wdict in ordered]
        texts = [t for t in texts if t]
        has_idem = any(
            texts[i] == texts[j]
            for i in range(len(texts))
            for j in range(i + 1, len(texts))
        )
        if not has_idem:
            divergent.add(idx)

    div_count  = len(divergent)
    cons_count = total - div_count

    # ── 4. Build file ─────────────────────────────────────────────────────────
    SEP   = "═" * 70
    COL_W = 10      # left-column width for model label
    out: list = []

    # Header block
    out.append(SEP)
    out.append(f"TRANSCRIPTION ALIGNÉE — {total} fenêtres de {window_secs}s")
    out.append(f"Sources    : {', '.join(s for s, _ in ordered)}")
    out.append("Note       : SPEAKER_?? = diarisation absente — ne pas utiliser pour l'attribution des locuteurs")
    out.append(
        f"Consensus  : {cons_count}/{total} fenêtres"
        f"  |  Divergence : {div_count}/{total} fenêtres"
    )
    if divergent:
        sorted_div = sorted(divergent)
        # Show at most 30 indices to keep the header scannable
        displayed = ", ".join(str(i) for i in sorted_div[:30])
        if div_count > 30:
            displayed += f", … ({div_count - 30} de plus)"
        out.append(f"Fenêtres en divergence : {displayed}")
    out.append(SEP)
    out.append("")

    # Window blocks
    for idx, key in enumerate(sorted_keys, 1):
        start_s, end_s = (p.strip() for p in key.split("→"))
        div_marker = "  ⚠ DIVERGENCE" if idx in divergent else ""

        out.append(f"{'═' * 5} FENÊTRE {idx} : [{start_s} → {end_s}]{div_marker} {'═' * 5}")
        out.append("")

        # Reference pools built incrementally (only unique texts enter the pool)
        shown_exact: dict = {}   # short → exact text_only
        shown_norm:  dict = {}   # short → normalized text_only

        for short, wdict in ordered:
            display   = wdict.get(key, "")
            text_only = _speaker_to_text(display)

            if not text_only:
                out.append(f"  {short:<{COL_W}}: [vide]")
                continue

            norm = _normalize_text(text_only)

            # Strict match against the pool of already-shown unique texts
            match_exact = next(
                (ref for ref, txt in shown_exact.items() if txt == text_only),
                None,
            )
            # Near match (only when no strict match found)
            match_near = None
            if match_exact is None:
                match_near = next(
                    (ref for ref, n in shown_norm.items() if n == norm),
                    None,
                )

            if match_exact:
                out.append(f"  {short:<{COL_W}}: IDEM {match_exact}")
            elif match_near:
                out.append(f"  {short:<{COL_W}}: IDEM~ {match_near}")
            else:
                # New unique text → show it and add to reference pool
                rendered = display if display.startswith("SPEAKER_") else f"SPEAKER_??: {display}"
                out.append(f"  {short:<{COL_W}}: {rendered}")
                shown_exact[short] = text_only
                shown_norm[short]  = norm

        out.append("")

    content = "\n".join(out)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(content)

    logger.info(
        "[MULTI] Aligned: %s — %d fenêtres, %d divergentes",
        os.path.basename(output_path), total, div_count,
    )
    return total


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run multiple ASR models on a single audio file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",       required=True, help="Path to audio/video file")
    parser.add_argument(
        "--models",
        default=",".join(ALL_MODELS),
        help=f"Comma-separated list of models (default: all). Choices: {', '.join(ALL_MODELS)}",
    )
    parser.add_argument("--language",      default="french",  help="Language (default: french)")
    parser.add_argument("--compute-type",  default="float16", help="Compute type (default: float16)")
    parser.add_argument("--chunk-length",  type=int, default=30,  help="Chunk length in seconds (default: 30)")
    parser.add_argument("--chunk-overlap", type=int, default=5,   help="Chunk overlap in seconds (default: 5)")
    parser.add_argument("--output-dir",    default=OUTPUT_DIR,    help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--no-diarization", action="store_true",  help="Skip diarization")
    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    audio_path = os.path.abspath(args.input)
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse and validate model list
    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    invalid = [m for m in selected_models if m not in MODEL_TO_WHISPER_TYPE]
    if invalid:
        print(f"ERROR: Unknown models: {invalid}\nValid: {list(MODEL_TO_WHISPER_TYPE.keys())}", file=sys.stderr)
        sys.exit(1)

    # Shared timestamp for all output filenames in this run
    run_ts = datetime.now().strftime("%m%d%H%M%S")
    basename = Path(audio_path).stem

    print(f"\n{'='*60}")
    print(f"MULTI-MODEL TRANSCRIPTION")
    print(f"{'='*60}")
    print(f"Input  : {audio_path}")
    print(f"Models : {', '.join(selected_models)}")
    print(f"Lang   : {args.language}")
    print(f"Diarize: {'no' if args.no_diarization else 'yes (once)'}")
    print(f"Output : {args.output_dir}")
    print(f"{'='*60}\n")

    # Get audio duration
    try:
        import librosa
        duration = librosa.get_duration(path=audio_path)
    except Exception:
        duration = 0.0

    # Diarizer (shared across models)
    diarizer = None
    diarization_result = None
    if not args.no_diarization:
        diarizer = Diarizer(model_dir=DIARIZATION_MODELS_DIR)

    model_results = []

    for i, model_name in enumerate(selected_models):
        print(f"\n[{i+1}/{len(selected_models)}] Running {model_name}...")

        try:
            segments, elapsed = run_model(
                model_name=model_name,
                audio_path=audio_path,
                language=args.language,
                compute_type=args.compute_type,
                chunk_length=args.chunk_length,
                chunk_overlap=args.chunk_overlap,
            )

            # Apply diarization (first call runs it, subsequent calls reuse result)
            if diarizer is not None:
                segments, diarization_result = apply_diarization_to_segments(
                    diarizer=diarizer,
                    audio_path=audio_path,
                    segments=segments,
                    diarization_result=diarization_result,
                )
                # Offload diarizer after first use to free VRAM for next model
                if i == 0:
                    diarizer.offload()

            # Write output file
            output_file = os.path.join(
                args.output_dir,
                f"{basename}-{model_name}-{run_ts}.txt",
            )
            final_count = write_output(segments, output_file, use_start=(model_name == "large-v3"))

            chars = sum(len((s.text or "").split("|")[-1]) for s in segments)
            model_results.append({
                "model":    model_name,
                "ok":       True,
                "elapsed":  elapsed,
                "segments": final_count,
                "chars":    chars,
                "output":   output_file,
            })
            print(f"    ✅ Done in {fmt_duration(elapsed)} — {final_count} segments, {chars} chars")
            print(f"    → {output_file}")

        except Exception as e:
            logger.exception("[MULTI] Model %s failed", model_name)
            model_results.append({
                "model":   model_name,
                "ok":      False,
                "elapsed": 0,
                "error":   str(e),
                "output":  "",
            })
            print(f"    ❌ FAILED: {e}")

    # Write summary
    summary_path = os.path.join(
        args.output_dir,
        f"{basename}-multi-summary-{run_ts}.txt",
    )
    write_summary(audio_path, duration, args.language, model_results, summary_path)
    print(f"\nSummary written to: {summary_path}")

    # Create aligned file (merged comparison, needed for LLM arbitration)
    ok_files = {r["model"]: r["output"] for r in model_results if r["ok"]}
    if len(ok_files) >= 2:
        aligned_path = os.path.join(
            args.output_dir,
            f"{basename}-aligned-{run_ts}.txt",
        )
        n_windows = create_aligned_file(ok_files, aligned_path)
        print(f"Aligned file written to: {aligned_path} ({n_windows} windows)")


if __name__ == "__main__":
    main()
