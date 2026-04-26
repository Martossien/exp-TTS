import os
import time
import tempfile
import torch
from typing import BinaryIO, Union, Tuple, List, Callable, Optional
import gradio as gr
import numpy as np

from modules.utils.paths import (COHERE_ASR_MODELS_DIR, DIARIZATION_MODELS_DIR, UVR_MODELS_DIR, OUTPUT_DIR)
from modules.whisper.data_classes import *
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline
from modules.utils.logger import get_logger

logger = get_logger()

# Language map — Cohere does NOT support auto-detection, language must be explicit
# Supported: en, fr, de, it, es, pt, el, nl, pl, zh, ja, ko, vi, ar
_COHERE_SUPPORTED_LANGUAGES = {
    "english": "en", "en": "en",
    "french": "fr", "fr": "fr",
    "german": "de", "de": "de",
    "italian": "it", "it": "it",
    "spanish": "es", "es": "es",
    "portuguese": "pt", "pt": "pt",
    "greek": "el", "el": "el",
    "dutch": "nl", "nl": "nl",
    "polish": "pl", "pl": "pl",
    "chinese": "zh", "zh": "zh",
    "japanese": "ja", "ja": "ja",
    "korean": "ko", "ko": "ko",
    "vietnamese": "vi", "vi": "vi",
    "arabic": "ar", "ar": "ar",
}
_COHERE_DEFAULT_LANGUAGE = "fr"


class CohereASRInference(BaseTranscriptionPipeline):
    """
    Cohere Transcribe inference pipeline (CohereLabs/cohere-transcribe-03-2026).

    Specificities vs other models:
    - No automatic language detection — language MUST be explicit (default: fr)
    - No native timestamps — output is plain text per chunk
    - Built-in auto-chunking for long audio (handled by the model itself)
    - torch.compile() supported for speedup
    - 2B params, ~4-6 GB VRAM
    - Apache 2.0 license
    """

    available_models = {
        "cohere-transcribe-03-2026": "CohereLabs/cohere-transcribe-03-2026",
    }

    def __init__(
        self,
        model_dir: str = COHERE_ASR_MODELS_DIR,
        diarization_model_dir: str = DIARIZATION_MODELS_DIR,
        uvr_model_dir: str = UVR_MODELS_DIR,
        output_dir: str = OUTPUT_DIR,
    ):
        super().__init__(
            model_dir=model_dir,
            diarization_model_dir=diarization_model_dir,
            uvr_model_dir=uvr_model_dir,
            output_dir=output_dir,
        )
        self.model = None
        self.processor = None
        self.current_model_size = None
        self.current_compute_type = None
        self.available_models = type(self).available_models  # restore class dict overwritten by base __init__
        self.device = self._select_device()

    def _select_device(self) -> str:
        """Select the CUDA device with the most free VRAM."""
        if not torch.cuda.is_available():
            return "cpu"

        best_index, best_free = 0, -1
        for i in range(torch.cuda.device_count()):
            try:
                free, total = torch.cuda.mem_get_info(i)
                logger.info(
                    "[COHERE-ASR] GPU %s (%s): %.1f GB free / %.1f GB total",
                    i, torch.cuda.get_device_name(i),
                    free / 1024**3, total / 1024**3,
                )
                if free > best_free:
                    best_free, best_index = free, i
            except Exception as e:
                logger.warning("[COHERE-ASR] Could not inspect GPU %s: %s", i, e)

        device = f"cuda:{best_index}"
        logger.info("[COHERE-ASR] Selected device: %s", device)
        return device

    def offload(self):
        """Override base offload() to also release the processor and clear all CUDA caches.

        The base class checks ``self.device == 'cuda'`` which never matches when the device
        is ``'cuda:N'``, so empty_cache() would silently be skipped without this override.
        """
        if self.model is not None:
            del self.model
            self.model = None
            self.current_model_size = None
            self.current_compute_type = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._empty_cuda_cache()
        import gc
        gc.collect()
        logger.info("[COHERE-ASR] Model offloaded and VRAM released")

    @staticmethod
    def _empty_cuda_cache():
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()

    def _get_model_source(self, model_size: str) -> Tuple[str, bool]:
        """Return (path_or_repo_id, is_local)."""
        required_files = ("config.json",)
        local_dir = os.path.join(self.model_dir, model_size)
        if os.path.isdir(local_dir) and all(
            os.path.exists(os.path.join(local_dir, f)) for f in required_files
        ):
            return local_dir, True

        repo_id = self.available_models.get(model_size, self.available_models["cohere-transcribe-03-2026"])
        return repo_id, False

    def _normalize_model_size(self, model_size: str) -> str:
        if model_size in self.available_models or os.path.isdir(model_size):
            return model_size
        logger.warning(
            "[COHERE-ASR] Unsupported model_size=%s, falling back to cohere-transcribe-03-2026", model_size
        )
        return "cohere-transcribe-03-2026"

    def _normalize_language(self, language: Optional[str]) -> str:
        """
        Cohere requires an explicit language code (ISO 639-1).
        No auto-detection supported — defaults to 'fr' if None or unknown.
        """
        if not language or language.strip().lower() in ("", "automatic detection", "auto"):
            logger.info("[COHERE-ASR] No language specified, defaulting to '%s'", _COHERE_DEFAULT_LANGUAGE)
            return _COHERE_DEFAULT_LANGUAGE

        key = language.strip().lower()
        code = _COHERE_SUPPORTED_LANGUAGES.get(key)
        if code:
            return code

        logger.warning(
            "[COHERE-ASR] Language '%s' not supported by Cohere, defaulting to '%s'",
            language, _COHERE_DEFAULT_LANGUAGE,
        )
        return _COHERE_DEFAULT_LANGUAGE

    def update_model(
        self,
        model_size: str,
        compute_type: str,
        progress: gr.Progress = gr.Progress(),
    ):
        """Load or reload the Cohere ASR model."""
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

        progress(0, desc="Loading Cohere ASR model...")

        if self.model is not None:
            del self.model
            del self.processor
            self._empty_cuda_cache()

        model_size = self._normalize_model_size(model_size)
        source, is_local = self._get_model_source(model_size)

        # Cohere ASR uses -1e9 as attention-mask fill value, which overflows float16
        # (float16 max ≈ 65504).  Force bfloat16 which shares float32's exponent range.
        if compute_type == "float16":
            logger.warning(
                "[COHERE-ASR] float16 not supported (attention mask -1e9 overflows fp16). "
                "Forcing bfloat16."
            )
            compute_type = "bfloat16"

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(compute_type, torch.bfloat16)

        local_files_only = is_local
        logger.info(
            "[COHERE-ASR] Loading processor: source=%s local_files_only=%s", source, local_files_only
        )
        progress(0.2, desc="Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            source, local_files_only=local_files_only, trust_remote_code=True
        )

        logger.info(
            "[COHERE-ASR] Loading model: source=%s device=%s dtype=%s local_files_only=%s",
            source, self.device, dtype, local_files_only,
        )
        progress(0.4, desc=f"Loading {model_size}...")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            source,
            torch_dtype=dtype,
            device_map=self.device,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        self.model.eval()

        # torch.compile for speedup — disabled for Cohere due to custom generate() which uses
        # dynamic control flow and a non-standard attention mask workaround incompatible with
        # graph capture. Can be re-enabled if a future transformers version fixes the custom code.
        # try:
        #     self.model = torch.compile(self.model, mode="reduce-overhead")
        #     logger.info("[COHERE-ASR] Model compiled with torch.compile (mode=reduce-overhead)")
        # except Exception as e:
        #     logger.warning("[COHERE-ASR] torch.compile skipped: %s", e)

        self.current_model_size = model_size
        self.current_compute_type = compute_type
        progress(1.0, desc=f"Cohere ASR {model_size} ready")
        logger.info("[COHERE-ASR] Model %s loaded with %s on %s", model_size, compute_type, self.device)

    def transcribe(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        progress: gr.Progress = gr.Progress(),
        progress_callback: Optional[Callable] = None,
        *whisper_params,
    ) -> Tuple[List[Segment], float]:
        """
        Transcribe audio using Cohere ASR.

        Notes:
        - Language must be explicit (no auto-detection). Defaults to 'fr'.
        - Output is plain text per segment — no word-level timestamps.
        - Chunk-level timestamps are derived from chunk boundaries.
        - The model handles internal chunking for sequences > 35s automatically.
        """
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        import librosa

        start_time = time.time()
        params = WhisperParams.from_list(list(whisper_params))

        model_size = self._normalize_model_size(params.model_size)
        compute_type = params.compute_type

        if (
            model_size != self.current_model_size
            or compute_type != self.current_compute_type
            or self.model is None
        ):
            self.update_model(model_size, compute_type, progress)

        language = self._normalize_language(params.lang)
        logger.info(
            "[COHERE-ASR] Transcription requested: model=%s language=%s device=%s",
            model_size, language, self.device,
        )

        # Load audio as 16kHz mono float32
        progress(0.2, desc="Loading audio...")
        temp_path = None
        try:
            if isinstance(audio, np.ndarray):
                audio_data = audio.astype(np.float32, copy=False)
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)
            else:
                if not isinstance(audio, str):
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp.write(audio.read())
                        temp_path = tmp.name
                    audio = temp_path
                audio_data, _ = librosa.load(audio, sr=16000, mono=True)
                audio_data = audio_data.astype(np.float32, copy=False)

            duration = len(audio_data) / 16000
            chunk_length = int(params.chunk_length or 30)
            chunk_overlap = int(params.chunk_overlap if hasattr(params, "chunk_overlap") and params.chunk_overlap is not None else 2)

            # Split into chunks for progress reporting and VRAM control.
            # The Cohere processor handles internal sub-chunking automatically (max_duration=30s).
            if duration <= chunk_length:
                chunks = [(0.0, duration, audio_data)]
            else:
                chunks, step = [], chunk_length - chunk_overlap
                start = 0.0
                while start < duration:
                    end = min(start + chunk_length, duration)
                    s, e = int(start * 16000), int(end * 16000)
                    chunks.append((start, end, audio_data[s:e]))
                    if end >= duration:
                        break
                    start += step

                logger.info(
                    "[COHERE-ASR] Audio segmented: duration=%.2fs chunks=%s chunk_length=%ss overlap=%ss",
                    duration, len(chunks), chunk_length, chunk_overlap,
                )

            segments = []
            progress(0.3, desc="Transcribing with Cohere ASR...")

            for idx, (chunk_start, chunk_end, chunk_audio) in enumerate(chunks, start=1):
                progress_value = 0.3 + 0.65 * (idx - 1) / max(len(chunks), 1)
                progress(progress_value, desc=f"Transcribing chunk {idx}/{len(chunks)}...")
                if progress_callback:
                    progress_callback(progress_value)

                proc_out = self.processor(
                    audio=chunk_audio,
                    sampling_rate=16000,
                    return_tensors="pt",
                    language=language,
                )
                # 'audio_chunk_index' is used only by processor.decode for long-form
                audio_chunk_index = proc_out.pop("audio_chunk_index", None)
                # 'length' is the audio frame count — needed by generate() but must stay out of
                # the call when routing through the custom code (it re-adds it internally).
                # Keep it and pass explicitly to avoid decoder_attention_mask=None crash.
                length_tensor = proc_out.pop("length", None)

                # Cast float tensors to model dtype, keep integer tensors as-is
                gen_inputs = {}
                for k, v in proc_out.items():
                    if isinstance(v, torch.Tensor):
                        gen_inputs[k] = (
                            v.to(self.device, dtype=self.model.dtype)
                            if v.is_floating_point()
                            else v.to(self.device)
                        )

                # Pass length so the encoder can build its attention mask
                if length_tensor is not None:
                    gen_inputs["length"] = length_tensor.to(self.device)

                # Provide an explicit decoder_attention_mask (1×1 ones for the BOS token) to
                # prevent a crash in transformers 4.x generation utils when it is None.
                batch_size = gen_inputs["input_features"].shape[0]
                gen_inputs["decoder_attention_mask"] = torch.ones(
                    batch_size, 1, device=self.device, dtype=torch.long
                )

                with torch.no_grad():
                    output_ids = self.model.generate(
                        **gen_inputs,
                        max_new_tokens=params.max_new_tokens or 600,
                        repetition_penalty=params.repetition_penalty if params.repetition_penalty else 1.0,
                    )

                if audio_chunk_index is not None:
                    text = self.processor.decode(
                        output_ids, skip_special_tokens=True,
                        audio_chunk_index=audio_chunk_index, language=language
                    )[0].strip()
                else:
                    text = self.processor.decode(output_ids[0], skip_special_tokens=True).strip()

                segments.append(Segment(start=chunk_start, end=chunk_end, text=text))

                logger.info(
                    "[COHERE-ASR] Chunk %s/%s: %.2fs-%.2fs chars=%s",
                    idx, len(chunks), chunk_start, chunk_end, len(text),
                )
                self._empty_cuda_cache()

            total_chars = sum(len(s.text or "") for s in segments)
            logger.info("[COHERE-ASR] Transcribed %s chunks, %s characters", len(segments), total_chars)

            elapsed = time.time() - start_time
            return segments, elapsed

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
