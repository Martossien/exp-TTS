import os
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import yaml

from modules.diarize.diarize_pipeline import DiarizationPipeline
from modules.diarize.audio_loader import load_audio, SAMPLE_RATE
from modules.utils.paths import DIARIZATION_MODELS_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)


@dataclass
class SpeakerTurn:
    speaker: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def format_timestamp(self, seconds: float) -> str:
        h = int(seconds) // 3600
        m = (int(seconds) % 3600) // 60
        s = int(seconds) % 60
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

    def __repr__(self):
        return f"[{self.format_timestamp(self.start)} → {self.format_timestamp(self.end)}] {self.speaker} ({self.duration:.1f}s)"


@dataclass
class SpeakerInfo:
    speaker_id: str
    name: str = ""
    function: str = ""
    role_meeting: str = ""
    notes: str = ""
    turns: List[SpeakerTurn] = field(default_factory=list)

    @property
    def total_speaking_time(self) -> float:
        return sum(t.duration for t in self.turns)


@dataclass
class DiarizationResult:
    turns: List[SpeakerTurn]
    speakers: Dict[str, SpeakerInfo]
    num_speakers_detected: int
    audio_duration: float
    elapsed_time: float

    def get_turns_text(self) -> str:
        lines = []
        for t in self.turns:
            h = int(t.start) // 3600
            m = (int(t.start) % 3600) // 60
            s = int(t.start) % 60
            h2 = int(t.end) // 3600
            m2 = (int(t.end) % 3600) // 60
            s2 = int(t.end) % 60
            lines.append(f"[{h:02d}:{m:02d}:{s:02d} → {h2:02d}:{m2:02d}:{s2:02d}] {t.speaker}  ({t.duration:.1f}s)")
        return "\n".join(lines)

    def get_speaking_time_text(self) -> str:
        lines = []
        total_time = sum(s.total_speaking_time for s in self.speakers.values())
        for sid, sinfo in sorted(self.speakers.items()):
            pct = (sinfo.total_speaking_time / total_time * 100) if total_time > 0 else 0
            minutes = int(sinfo.total_speaking_time) // 60
            seconds = int(sinfo.total_speaking_time) % 60
            lines.append(f"  {sid}:  {minutes:02d}m{seconds:02d}s  ({pct:.1f}%)  —  {sinfo.total_speaking_time:.1f}s / {total_time:.1f}s")
        lines.append(f"\n  Total speech: {total_time:.1f}s / {self.audio_duration:.1f}s  ({total_time/self.audio_duration*100:.1f}% of audio)")
        return "\n".join(lines)

    def get_turns_list(self) -> List[Dict]:
        return [{"speaker": t.speaker, "start": t.start, "end": t.end, "duration": t.duration} for t in self.turns]


def format_hms(seconds: float) -> str:
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def detect_speakers(
    audio_path: str,
    model_name: str = "pyannote/speaker-diarization-community-1",
    cache_dir: str = DIARIZATION_MODELS_DIR,
    use_auth_token: Optional[str] = None,
    device: str = "cuda",
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> DiarizationResult:
    """Run pyannote diarization on an audio file WITHOUT transcription.

    Returns a DiarizationResult with speaker turns, speaking time, and speaker info.
    """
    global _global_speakers
    logger.info("[SPEAKER-ID] Starting speaker detection on: %s", audio_path)
    logger.info("[SPEAKER-ID] Model: %s, device: %s, min_speakers=%s, max_speakers=%s",
                model_name, device, min_speakers, max_speakers)

    start_time = time.time()

    logger.info("[SPEAKER-ID] Loading audio...")
    audio = load_audio(audio_path)
    audio_duration = len(audio) / SAMPLE_RATE
    logger.info("[SPEAKER-ID] Audio loaded: duration=%.1fs (%d samples at %d Hz)",
                audio_duration, len(audio), SAMPLE_RATE)

    logger.info("[SPEAKER-ID] Creating diarization pipeline...")
    pipe = DiarizationPipeline(
        model_name=model_name,
        cache_dir=cache_dir,
        use_auth_token=use_auth_token,
        device=device,
    )

    if pipe.model is None:
        raise RuntimeError(
            f"Failed to load diarization model '{model_name}'. "
            f"Check your HuggingFace token and that you accepted the model terms at "
            f"https://huggingface.co/{model_name}"
        )

    try:
        logger.info("[SPEAKER-ID] Pipeline loaded. Running diarization...")
        diarize_df = pipe(audio, min_speakers=min_speakers, max_speakers=max_speakers)
    finally:
        logger.info("[SPEAKER-ID] Offloading pipeline after detection...")
        del pipe
        import torch as _torch, gc as _gc
        if device == "cuda" and _torch.cuda.is_available():
            _torch.cuda.empty_cache()
        _gc.collect()

    if diarize_df.empty:
        logger.warning("[SPEAKER-ID] Diarization returned empty result (no speech detected?)")
        return DiarizationResult(
            turns=[],
            speakers={},
            num_speakers_detected=0,
            audio_duration=audio_duration,
            elapsed_time=time.time() - start_time,
        )

    num_detected = diarize_df['speaker'].nunique()
    logger.info("[SPEAKER-ID] Diarization complete: %d turns, %d unique speakers detected",
                len(diarize_df), num_detected)

    turns: List[SpeakerTurn] = []
    speakers: Dict[str, SpeakerInfo] = {}

    for _, row in diarize_df.iterrows():
        turn = SpeakerTurn(
            speaker=row['speaker'],
            start=row['start'],
            end=row['end'],
        )
        turns.append(turn)

        if turn.speaker not in speakers:
            speakers[turn.speaker] = SpeakerInfo(
                speaker_id=turn.speaker,
                turns=[],
            )
        speakers[turn.speaker].turns.append(turn)

    elapsed = time.time() - start_time
    logger.info("[SPEAKER-ID] Detection finished in %.1fs", elapsed)

    result = DiarizationResult(
        turns=turns,
        speakers=speakers,
        num_speakers_detected=num_detected,
        audio_duration=audio_duration,
        elapsed_time=elapsed,
    )

    logger.info("[SPEAKER-ID] Speaking time breakdown:\n%s", result.get_speaking_time_text())

    return result


def extract_speaker_clips(
    audio_path: str,
    result: DiarizationResult,
    num_clips: int = 3,
    min_clip_duration: float = 5.0,
    max_clip_duration: float = 10.0,
    output_dir: Optional[str] = None,
) -> Dict[str, List[str]]:
    """Extract audio clips for each speaker.

    For each speaker, picks the `num_clips` longest segments where they speak alone
    (or nearly alone). Extracts clips of `max_clip_duration` seconds max.

    Returns a dict: {speaker_id: [path_to_clip1, path_to_clip2, ...]}
    """
    logger.info("[SPEAKER-ID] Extracting %d clips per speaker (min=%.1fs, max=%.1fs)",
                num_clips, min_clip_duration, max_clip_duration)

    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "speaker_clips")
    os.makedirs(output_dir, exist_ok=True)

    audio = load_audio(audio_path)
    clips: Dict[str, List[str]] = {}

    for speaker_id, sinfo in result.speakers.items():
        candidate_turns = sorted(sinfo.turns, key=lambda t: t.duration, reverse=True)

        selected_clips: List[str] = []
        turns_used = 0

        for turn in candidate_turns:
            if turns_used >= num_clips:
                break

            if turn.duration < min_clip_duration:
                logger.debug("[SPEAKER-ID] %s: skipping short turn %.1fs at %.1fs",
                            speaker_id, turn.duration, turn.start)
                continue

            clip_start = turn.start
            clip_duration = min(turn.duration, max_clip_duration)
            clip_end = clip_start + clip_duration

            start_sample = int(clip_start * SAMPLE_RATE)
            end_sample = min(int(clip_end * SAMPLE_RATE), len(audio))

            clip_audio = audio[start_sample:end_sample]

            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            clip_filename = f"{base_name}_{speaker_id}_clip{turns_used+1}.wav"
            clip_path = os.path.join(output_dir, clip_filename)

            import soundfile as sf
            sf.write(clip_path, clip_audio, SAMPLE_RATE)
            logger.info("[SPEAKER-ID] %s clip %d: %.1fs → %.1fs (%.1fs) → %s",
                        speaker_id, turns_used + 1, clip_start, clip_end, clip_duration, clip_path)

            selected_clips.append(clip_path)
            turns_used += 1

        clips[speaker_id] = selected_clips

        if len(selected_clips) < num_clips:
            logger.warning("[SPEAKER-ID] %s: only %d clips extracted (wanted %d) — not enough long segments",
                          speaker_id, len(selected_clips), num_clips)

    logger.info("[SPEAKER-ID] Clips extracted for %d speakers", len(clips))
    return clips


def export_speakers_yaml(
    result: DiarizationResult,
    speakers_info: Dict[str, Dict[str, str]],
    audio_path: str,
    output_path: Optional[str] = None,
    model_name: str = "unknown",
) -> str:
    """Export speaker identification data to a YAML file.

    Args:
        result: DiarizationResult from detect_speakers()
        speakers_info: Dict mapping speaker_id -> {nom, fonction, role_reunion, notes}
        audio_path: Path to the original audio file
        output_path: Optional output path. If None, auto-generated.
        model_name: Name of the diarization model used.

    Returns:
        Path to the written YAML file.
    """
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_speakers_{timestamp}.yaml")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else OUTPUT_DIR, exist_ok=True)

    total_speech = sum(s.total_speaking_time for s in result.speakers.values())

    speakers_data = {}
    for sid, sinfo in result.speakers.items():
        info = speakers_info.get(sid, {})
        speakers_data[sid] = {
            "nom": info.get("nom", ""),
            "fonction": info.get("fonction", ""),
            "role_reunion": info.get("role_reunion", ""),
            "notes": info.get("notes", ""),
        }

    turns_data = {}
    for sid, sinfo in result.speakers.items():
        turns_data[sid] = [
            [round(t.start, 2), round(t.end, 2)]
            for t in sinfo.turns
        ]

    pct_data = {}
    for sid, sinfo in result.speakers.items():
        pct = (sinfo.total_speaking_time / total_speech * 100) if total_speech > 0 else 0
        pct_data[sid] = f"{pct:.1f}%"

    yaml_data = {
        "audio": os.path.basename(audio_path),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "duree_audio": format_hms(result.audio_duration),
        "nombre_locuteurs_detectes": result.num_speakers_detected,
        "locuteurs": speakers_data,
        "tours_de_parole": turns_data,
        "temps_de_parole": pct_data,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Fichier locuteurs — Voxtral-WebUI\n")
        f.write(f"# Modele de diarization : {model_name}\n")
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    logger.info("[SPEAKER-ID] YAML exported to: %s", output_path)
    return output_path


def import_speakers_yaml(yaml_path: str) -> Tuple[Dict[str, Dict[str, str]], Dict]:
    """Import speaker identification data from a YAML file.

    Args:
        yaml_path: Path to the previously exported YAML file.

    Returns:
        Tuple of (speakers_info dict, raw yaml_data dict).
        speakers_info maps speaker_id -> {nom, fonction, role_reunion, notes}
    """
    logger.info("[SPEAKER-ID] Importing speaker YAML from: %s", yaml_path)

    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    speakers_info = {}
    locuteurs = yaml_data.get("locuteurs", {})
    for sid, info in locuteurs.items():
        speakers_info[sid] = {
            "nom": info.get("nom", ""),
            "fonction": info.get("fonction", ""),
            "role_reunion": info.get("role_reunion", ""),
            "notes": info.get("notes", ""),
        }

    logger.info("[SPEAKER-ID] Imported %d speakers from YAML", len(speakers_info))
    return speakers_info, yaml_data


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Speaker identification — detect speakers without transcription")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--model", default="pyannote/speaker-diarization-community-1", help="Diarization model")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token")
    parser.add_argument("--min-speakers", type=int, default=None, help="Min speakers")
    parser.add_argument("--max-speakers", type=int, default=None, help="Max speakers")
    parser.add_argument("--clips", type=int, default=3, help="Number of audio clips per speaker")
    parser.add_argument("--output-dir", default=None, help="Output directory")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SPEAKER IDENTIFICATION — CLI mode")
    logger.info("=" * 60)

    result = detect_speakers(
        audio_path=args.audio,
        model_name=args.model,
        use_auth_token=args.hf_token,
        device=args.device,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    print("\n" + "=" * 60)
    print(f"SPEAKERS DETECTED: {result.num_speakers_detected}")
    print(f"AUDIO DURATION: {format_hms(result.audio_duration)}")
    print(f"PROCESSING TIME: {result.elapsed_time:.1f}s")
    print("=" * 60)

    print("\nSPEAKING TIME BREAKDOWN:")
    print(result.get_speaking_time_text())

    print("\nSPEAKER TURNS:")
    print(result.get_turns_text())

    clips = extract_speaker_clips(
        audio_path=args.audio,
        result=result,
        num_clips=args.clips,
        output_dir=args.output_dir or os.path.join(OUTPUT_DIR, "speaker_clips"),
    )

    for sid, clip_list in clips.items():
        print(f"\n  {sid} clips:")
        for cp in clip_list:
            print(f"    {cp}")

    speakers_info = {sid: {"nom": "", "fonction": "", "role_reunion": "", "notes": ""} for sid in result.speakers}

    yaml_path = export_speakers_yaml(
        result=result,
        speakers_info=speakers_info,
        audio_path=args.audio,
    )
    print(f"\nYAML exported to: {yaml_path}")