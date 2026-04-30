import io
import os
import time
import tempfile
import logging
from typing import BinaryIO, Union, Tuple, List, Callable, Optional

import numpy as np
import requests
import gradio as gr

try:
    import librosa
    import soundfile as sf
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    librosa = None
    sf = None

from modules.utils.paths import (DIARIZATION_MODELS_DIR, UVR_MODELS_DIR, OUTPUT_DIR)
from modules.whisper.data_classes import Segment, WhisperParams
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline

logger = logging.getLogger("voxtral_realtime_vllm")

VLLM_HOST = os.environ.get("VOXTRAL_VLLM_HOST", "localhost")
VLLM_PORT = int(os.environ.get("VOXTRAL_VLLM_PORT", "8000"))
VLLM_MODEL = os.environ.get("VOXTRAL_VLLM_MODEL", "mistralai/Voxtral-Mini-4B-Realtime-2602")
VLLM_TRANSCRIBE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1/audio/transcriptions"
VLLM_HEALTH_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/health"
VLLM_TIMEOUT = int(os.environ.get("VOXTRAL_VLLM_TIMEOUT", "120"))


def _check_server_health() -> bool:
    try:
        resp = requests.get(VLLM_HEALTH_URL, timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def _transcribe_chunk_http(wav_bytes: bytes, model: str, language: str, timeout: int) -> str:
    files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
    data = {"model": model}
    if language:
        data["language"] = language

    resp = requests.post(VLLM_TRANSCRIBE_URL, files=files, data=data, timeout=timeout)

    if resp.status_code != 200:
        raise RuntimeError(
            f"[VOXTRAL-RT] HTTP {resp.status_code}: {resp.text[:500]}"
        )

    result = resp.json()
    return result.get("text", "")


def _pcm16_to_wav_bytes(pcm16_bytes: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16_bytes)
    return buf.getvalue()


class VoxtralRealtimeVLLMInference(BaseTranscriptionPipeline):

    available_models_info = {
        "voxtral-realtime-vllm": {
            "repo_id": "mistralai/Voxtral-Mini-4B-Realtime-2602",
            "languages": ["fr", "en", "es", "de", "it", "pt", "nl",
                          "ar", "zh", "ja", "ko", "ru", "hi"],
        },
    }

    def __init__(self,
                 model_dir: str = None,
                 diarization_model_dir: str = DIARIZATION_MODELS_DIR,
                 uvr_model_dir: str = UVR_MODELS_DIR,
                 output_dir: str = OUTPUT_DIR,
                 ):
        if model_dir is None:
            model_dir = os.path.join(tempfile.gettempdir(), "voxtral-realtime-vllm")
        os.makedirs(model_dir, exist_ok=True)

        super().__init__(model_dir=model_dir,
                         diarization_model_dir=diarization_model_dir,
                         uvr_model_dir=uvr_model_dir,
                         output_dir=output_dir)

        if not AUDIO_LIBS_AVAILABLE:
            raise ImportError(
                "[VOXTRAL-RT] librosa and soundfile are required. "
                "Install: pip install librosa soundfile"
            )

        self.model = None
        self.model_dir = model_dir
        self.current_model_size = None
        self.current_compute_type = "bfloat16"
        self.device = "cuda"
        self.available_models = list(self.available_models_info.keys())
        self.available_langs = self.available_models_info[
            "voxtral-realtime-vllm"]["languages"]
        self.available_compute_types = ["bfloat16"]

    def update_model(self,
                     model_size: str,
                     compute_type: str,
                     progress: gr.Progress = gr.Progress()):
        self.current_model_size = model_size or "voxtral-realtime-vllm"
        self.current_compute_type = compute_type or "bfloat16"
        logger.info("[VOXTRAL-RT] update_model (no-op: model is external vLLM process)")

    def transcribe(self,
                   audio: Union[str, BinaryIO, np.ndarray],
                   progress: gr.Progress = gr.Progress(),
                   progress_callback: Optional[Callable] = None,
                   *whisper_params,
                   ) -> Tuple[List[Segment], float]:
        t0 = time.time()

        params = WhisperParams.from_list(list(whisper_params))
        chunk_length = getattr(params, "chunk_length", 30) or 30
        lang = getattr(params, "lang", None)
        model_size = getattr(params, "model_size", "voxtral-realtime-vllm")

        language = None
        if lang and lang != "Automatic Detection" and lang != "auto":
            language = lang

        logger.info(
            "[VOXTRAL-RT] transcribe: model=%s lang=%s chunk_length=%ss",
            model_size, language, chunk_length,
        )

        audio_chunks = self._segment_audio(audio, chunk_length)

        segments = []
        total_chunks = len(audio_chunks)
        for idx, (chunk_start, chunk_end, chunk_bytes) in enumerate(audio_chunks):
            logger.info(
                "[VOXTRAL-RT] chunk %d/%d : [%.1fs -> %.1fs] %d bytes",
                idx, total_chunks, chunk_start, chunk_end, len(chunk_bytes),
            )

            wav_bytes = _pcm16_to_wav_bytes(chunk_bytes)

            try:
                text = _transcribe_chunk_http(wav_bytes, VLLM_MODEL, language, VLLM_TIMEOUT)
            except requests.exceptions.ConnectionError as e:
                raise ConnectionError(
                    f"[VOXTRAL-RT] server unavailable at {VLLM_TRANSCRIBE_URL}. "
                    f"Did you run start_voxtral_realtime_vllm.sh?"
                ) from e
            except requests.exceptions.Timeout as e:
                raise RuntimeError(
                    f"[VOXTRAL-RT] request timed out after {VLLM_TIMEOUT}s"
                ) from e

            text = (text or "").strip()
            segments.append(Segment(
                start=chunk_start,
                end=chunk_end,
                text=text,
            ))

        elapsed = time.time() - t0
        logger.info("[VOXTRAL-RT] done: %d segments in %.1fs", len(segments), elapsed)
        return segments, elapsed

    def _segment_audio(self, audio, chunk_length):
        if isinstance(audio, str) and os.path.isfile(audio):
            audio_data, sr = librosa.load(audio, sr=16000, mono=True)
        elif isinstance(audio, np.ndarray):
            audio_data = audio
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=0)
            sr = 16000
        else:
            raise ValueError(f"[VOXTRAL-RT] unsupported audio type: {type(audio)}")

        audio_data = audio_data.astype(np.float32)
        peak = np.max(np.abs(audio_data))
        if peak > 0:
            audio_data = audio_data / peak * 0.95

        pcm16 = (audio_data * 32767).astype(np.int16)
        total_samples = len(pcm16)
        total_duration = total_samples / sr

        chunk_samples = int(chunk_length * sr)
        if total_duration <= chunk_length:
            return [(0.0, round(total_duration, 1), pcm16.tobytes())]

        chunks = []
        for start_sample in range(0, total_samples, chunk_samples):
            end_sample = min(start_sample + chunk_samples, total_samples)
            chunk_pcm = pcm16[start_sample:end_sample]
            start_sec = round(start_sample / sr, 1)
            end_sec = round(end_sample / sr, 1)
            chunks.append((start_sec, end_sec, chunk_pcm.tobytes()))
        return chunks

    def offload(self):
        logger.info("[VOXTRAL-RT] offload (no-op: model is external vLLM process)")