import os
import tempfile
import time
import torch
from typing import BinaryIO, Union, Tuple, List, Callable, Optional
import gradio as gr
import numpy as np

from modules.utils.paths import (QWEN3_ASR_MODELS_DIR, DIARIZATION_MODELS_DIR, UVR_MODELS_DIR, OUTPUT_DIR)
from modules.whisper.data_classes import *
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline
from modules.utils.logger import get_logger

try:
    from qwen_asr import Qwen3ASRModel
    QWEN3_ASR_AVAILABLE = True
except ImportError:
    QWEN3_ASR_AVAILABLE = False
    Qwen3ASRModel = None
    print("Warning: qwen-asr not available. Please install: pip install -U qwen-asr")

logger = get_logger()


class Qwen3ASRInference(BaseTranscriptionPipeline):
    """
    Qwen3-ASR inference pipeline for multilingual speech recognition.
    Supports 52 languages and Chinese dialects.
    """
    available_models = {
        "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
        "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
    }
    _DEFAULT_CHUNK_LENGTH_SECONDS = 120
    _DEFAULT_CHUNK_OVERLAP_SECONDS = 2

    def __init__(
        self,
        model_dir: str = QWEN3_ASR_MODELS_DIR,
        diarization_model_dir: str = DIARIZATION_MODELS_DIR,
        uvr_model_dir: str = UVR_MODELS_DIR,
        output_dir: str = OUTPUT_DIR,
    ):
        super().__init__(model_dir=model_dir, diarization_model_dir=diarization_model_dir,
                        uvr_model_dir=uvr_model_dir, output_dir=output_dir)
        self.available_models = type(self).available_models

        if not QWEN3_ASR_AVAILABLE:
            raise ImportError("Qwen3-ASR support requires qwen-asr package. "
                          "Please install: pip install -U qwen-asr")

        self.device = self._select_device()
        self.model = None
        self.current_model_size = None
        self.current_compute_type = None

    def _select_device(self) -> str:
        """
        Choose the CUDA device with the most free VRAM.
        """
        if not torch.cuda.is_available():
            return "cpu"

        best_index = 0
        best_free_bytes = -1
        for index in range(torch.cuda.device_count()):
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                logger.info(
                    "[QWEN3-ASR] GPU %s (%s): %.1f GB free / %.1f GB total",
                    index,
                    torch.cuda.get_device_name(index),
                    free_bytes / 1024**3,
                    total_bytes / 1024**3,
                )
                if free_bytes > best_free_bytes:
                    best_free_bytes = free_bytes
                    best_index = index
            except Exception as e:
                logger.warning("[QWEN3-ASR] Could not inspect GPU %s: %s", index, e)

        device = f"cuda:{best_index}"
        logger.info("[QWEN3-ASR] Selected device: %s", device)
        return device

    @staticmethod
    def _empty_cuda_cache():
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                with torch.cuda.device(index):
                    torch.cuda.empty_cache()


    def _get_model_source(self, model_size: str) -> Tuple[str, bool]:
        """
        Return the model repo ID and whether it exists locally.
        """
        required_files = ("config.json", "preprocessor_config.json")

        local_candidates = [
            os.path.join(self.model_dir, model_size),
            os.path.join(self.model_dir, model_size.replace("/", "--")),
            os.path.join(self.model_dir, model_size.replace("-", "_")),
        ]
        for local_path in local_candidates:
            if os.path.isdir(local_path) and all(
                os.path.exists(os.path.join(local_path, file_name))
                for file_name in required_files
            ):
                return local_path, True

        if os.path.isdir(model_size) and all(
            os.path.exists(os.path.join(model_size, file_name))
            for file_name in required_files
        ):
            return model_size, True

        repo_id = type(self).available_models.get(model_size, type(self).available_models["qwen3-asr-1.7b"])
        return repo_id, False

    def _normalize_model_size(self, model_size: str) -> str:
        """
        Keep Qwen isolated from defaults saved by other whisper implementations.
        """
        if model_size in type(self).available_models or os.path.isdir(model_size):
            return model_size

        logger.warning(
            "[QWEN3-ASR] Unsupported model_size=%s. Falling back to qwen3-asr-1.7b",
            model_size,
        )
        return "qwen3-asr-1.7b"

    def _normalize_language(self, language: Optional[str]) -> Optional[str]:
        """
        Qwen3-ASR supports language detection when set to None.
        Otherwise expects language names like "English", "Chinese", etc.
        """
        if not language or language == "Automatic Detection":
            return None

        # Map Whisper language codes to Qwen language names
        language_map = {
            "en": "English",
            "zh": "Chinese",
            "fr": "French",
            "de": "German",
            "es": "Spanish",
            "it": "Italian",
            "pt": "Portuguese",
            "ja": "Japanese",
            "ko": "Korean",
            "ru": "Russian",
            "ar": "Arabic",
            "hi": "Hindi",
            "id": "Indonesian",
            "ms": "Malay",
            "nl": "Dutch",
            "sv": "Swedish",
            "da": "Danish",
            "fi": "Finnish",
            "pl": "Polish",
            "cs": "Czech",
            "tr": "Turkish",
            "vi": "Vietnamese",
            "th": "Thai",
            "el": "Greek",
            "hu": "Hungarian",
            "fa": "Persian",
            "ro": "Romanian",
            "yue": "Cantonese",
        }

        normalized = str(language).strip().lower()
        if len(normalized) == 2:
            return language_map.get(normalized, language)

        # Already a language name
        return language

    def update_model(self, model_size: str, compute_type: str, progress: gr.Progress = gr.Progress()):
        """
        Load or update the Qwen3-ASR model.
        """
        progress(0, desc="Loading Qwen3-ASR model...")

        if self.model is not None:
            del self.model
            self._empty_cuda_cache()

        model_size = self._normalize_model_size(model_size)
        model_source, is_local = self._get_model_source(model_size)

        try:
            progress(0.2, desc=f"Loading {model_size}...")

            # Map compute_type to torch dtype
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(compute_type, torch.bfloat16)

            self.model = Qwen3ASRModel.from_pretrained(
                model_source,
                dtype=dtype,
                device_map=self.device,
                max_inference_batch_size=1,
                max_new_tokens=512,
            )

            self.current_model_size = model_size
            self.current_compute_type = compute_type

            progress(1.0, desc=f"Model {model_size} loaded successfully")
            logger.info(f"[QWEN3-ASR] Model {model_size} loaded with {compute_type} on {self.device}")

        except Exception as e:
            logger.error(f"[QWEN3-ASR] Failed to load model: {e}")
            raise

    def _load_audio_array(self, audio: Union[str, BinaryIO, np.ndarray]) -> Tuple[np.ndarray, Optional[str]]:
        """
        Load every input form as mono 16 kHz float32 audio for deterministic chunking.
        """
        temp_audio_path = None

        if isinstance(audio, np.ndarray):
            audio_data = audio.astype(np.float32, copy=False)
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            return audio_data, None

        if not isinstance(audio, str):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio.read())
                temp_audio_path = tmp.name
            audio = temp_audio_path

        import librosa
        audio_data, _ = librosa.load(audio, sr=16000, mono=True)
        return audio_data.astype(np.float32, copy=False), temp_audio_path

    def _segment_audio(
        self,
        audio_data: np.ndarray,
        chunk_length: Optional[int],
        chunk_overlap: Optional[int],
    ) -> List[Tuple[float, float, np.ndarray]]:
        """
        Split long audio before qwen-asr so inference is sequential and memory bounded.
        """
        sample_rate = 16000
        duration = len(audio_data) / sample_rate
        chunk_length = int(chunk_length or self._DEFAULT_CHUNK_LENGTH_SECONDS)
        chunk_overlap = int(chunk_overlap if chunk_overlap is not None else self._DEFAULT_CHUNK_OVERLAP_SECONDS)

        if chunk_length <= 0:
            chunk_length = self._DEFAULT_CHUNK_LENGTH_SECONDS
        if chunk_overlap < 0:
            chunk_overlap = 0
        if chunk_overlap >= chunk_length:
            chunk_overlap = min(2, chunk_length - 1)

        if duration <= chunk_length:
            return [(0.0, duration, audio_data)]

        chunks = []
        start_sec = 0.0
        step_sec = chunk_length - chunk_overlap
        while start_sec < duration:
            end_sec = min(start_sec + chunk_length, duration)
            start_sample = int(start_sec * sample_rate)
            end_sample = int(end_sec * sample_rate)
            chunks.append((start_sec, end_sec, audio_data[start_sample:end_sample]))
            if end_sec >= duration:
                break
            start_sec += step_sec

        logger.info(
            "[QWEN3-ASR] Audio segmented: duration=%.2fs chunks=%s chunk_length=%ss overlap=%ss",
            duration,
            len(chunks),
            chunk_length,
            chunk_overlap,
        )
        return chunks

    def transcribe(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        progress: gr.Progress = gr.Progress(),
        progress_callback: Optional[Callable] = None,
        *whisper_params,
    ) -> Tuple[List[Segment], float]:
        """
        Transcribe audio using Qwen3-ASR.
        """
        start_time = time.time()
        temp_audio_path = None

        params = WhisperParams.from_list(list(whisper_params))

        model_size = self._normalize_model_size(params.model_size)
        compute_type = params.compute_type

        if (model_size != self.current_model_size or
                compute_type != self.current_compute_type or
                self.model is None):
            self.update_model(model_size, compute_type, progress)

        progress(0.3, desc="Preparing audio chunks...")
        audio_data, temp_audio_path = self._load_audio_array(audio)
        audio_chunks = self._segment_audio(audio_data, params.chunk_length, params.chunk_overlap)

        progress(0.5, desc="Transcribing with Qwen3-ASR...")

        try:
            language = self._normalize_language(params.lang)
            segments = []
            for index, (chunk_start, chunk_end, chunk_audio) in enumerate(audio_chunks, start=1):
                progress_value = 0.5 + (0.45 * (index - 1) / max(len(audio_chunks), 1))
                progress(
                    progress_value,
                    desc=f"Transcribing chunk {index}/{len(audio_chunks)} with Qwen3-ASR...",
                )
                if progress_callback is not None:
                    progress_callback(progress_value)

                results = self.model.transcribe(
                    audio=(chunk_audio, 16000),
                    language=language,
                    return_time_stamps=False,
                )
                result = results[0] if isinstance(results, list) else results
                text = result.text if hasattr(result, 'text') else str(result)
                segments.append(Segment(start=chunk_start, end=chunk_end, text=text))
                logger.info(
                    "[QWEN3-ASR] Chunk %s/%s decoded: %.2fs-%.2fs chars=%s",
                    index,
                    len(audio_chunks),
                    chunk_start,
                    chunk_end,
                    len(text),
                )
                self._empty_cuda_cache()

            progress(0.95, desc="Finalizing Qwen3-ASR result...")
            if progress_callback is not None:
                progress_callback(0.95)

            total_chars = sum(len(segment.text or "") for segment in segments)
            logger.info("[QWEN3-ASR] Transcribed %s chunks, %s characters", len(segments), total_chars)

        except Exception as e:
            logger.error(f"[QWEN3-ASR] Transcription failed: {e}")
            raise
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

        total_time = time.time() - start_time
        return segments, total_time
