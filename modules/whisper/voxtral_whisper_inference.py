import os
import time
import torch
import numpy as np

# Allow TF32 on Ampere+ (RTX 5090 = Blackwell SM 12.0): ~3× faster matmul with
# negligible precision loss for speech recognition tasks.
torch.set_float32_matmul_precision('high')
from typing import BinaryIO, Union, Tuple, List, Callable, Optional
import gradio as gr
import tempfile
import librosa

try:
    from transformers import VoxtralForConditionalGeneration, AutoProcessor
    VOXTRAL_AVAILABLE = True
except ImportError:
    print("Warning: VoxtralForConditionalGeneration not available. Please install latest transformers:")
    print("pip uninstall transformers -y")
    print("pip install git+https://github.com/huggingface/transformers.git")
    VOXTRAL_AVAILABLE = False
    VoxtralForConditionalGeneration = None
    AutoProcessor = None

from modules.utils.paths import (VOXTRAL_MODELS_DIR, DIARIZATION_MODELS_DIR, UVR_MODELS_DIR, OUTPUT_DIR)
from modules.whisper.data_classes import *
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline
from modules.utils.logger import get_logger

logger = get_logger()


class VoxtralWhisperInference(BaseTranscriptionPipeline):
    def __init__(self,
                 model_dir: str = VOXTRAL_MODELS_DIR,
                 diarization_model_dir: str = DIARIZATION_MODELS_DIR,
                 uvr_model_dir: str = UVR_MODELS_DIR,
                 output_dir: str = OUTPUT_DIR,
                 ):
        super().__init__(model_dir=model_dir, diarization_model_dir=diarization_model_dir,
                        uvr_model_dir=uvr_model_dir, output_dir=output_dir)

        if not VOXTRAL_AVAILABLE:
            raise ImportError("Voxtral support requires latest transformers. "
                            "Please install: pip install git+https://github.com/huggingface/transformers.git")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.current_model_size = None
        self.current_compute_type = None
        self.current_device_map = None   # track actual device_map used at load time

        # Voxtral repo ID from HuggingFace
        self.repo_id = "mistralai/Voxtral-Mini-3B-2507"
        self.local_model_path = os.path.join(self.model_dir, "voxtral-mini-3b")
        self.temp_dir = os.path.join(tempfile.gettempdir(), "voxtral-webui")
        os.makedirs(self.temp_dir, exist_ok=True)

    # Estimated VRAM requirement (GB, bfloat16) per model size.
    # Used by _select_device_map() to decide single-GPU vs "auto".
    _VRAM_REQUIREMENTS_GB: dict = {
        "voxtral-mini-3b": 10.0,
        # Add entries here when integrating larger models, e.g.:
        # "voxtral-7b": 15.0,
        # "voxtral-13b": 28.0,
        # "voxtral-30b": 62.0,
    }
    _DEFAULT_VRAM_GB: float = 16.0   # conservative estimate for unknown model sizes

    def _select_device_map(self, model_size: str) -> str:
        """
        Choose the best device_map for the given model.

        Strategy
        --------
        1. If no CUDA is available → "cpu"
        2. Find the GPU with the most free VRAM.
        3. If that GPU has enough room for the model (estimated need × 1.2 safety
           margin) → use that single GPU (e.g. "cuda:0").
           Rationale: a single-device model avoids the cross-device tensor error
           that HuggingFace generate() triggers when lm_head lives on a different
           device than input_ids, and also allows torch.compile().
        4. Otherwise → "auto" so Accelerate distributes layers across all GPUs.
           Note: very large models on "auto" may still hit generate() cross-device
           issues depending on the transformers version; this is a known upstream
           limitation.
        """
        if not torch.cuda.is_available():
            return "cpu"

        required_gb = self._VRAM_REQUIREMENTS_GB.get(model_size, self._DEFAULT_VRAM_GB)
        n_gpus = torch.cuda.device_count()

        best_gpu, max_free_gb = 0, 0.0
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            free_gb = (props.total_memory - torch.cuda.memory_allocated(i)) / 1024 ** 3
            logger.info(
                "[VOXTRAL] GPU %d (%s): %.1f GB free / %.1f GB total",
                i, props.name, free_gb, props.total_memory / 1024 ** 3,
            )
            if free_gb > max_free_gb:
                max_free_gb, best_gpu = free_gb, i

        if max_free_gb >= required_gb * 1.2:
            device_map = f"cuda:{best_gpu}"
            logger.info(
                "[VOXTRAL] device_map=%s (single GPU: %.1f GB free ≥ %.1f GB needed)",
                device_map, max_free_gb, required_gb * 1.2,
            )
        else:
            total_vram_gb = sum(
                torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
                for i in range(n_gpus)
            )
            device_map = "auto"
            logger.info(
                "[VOXTRAL] device_map=auto (largest GPU %.1f GB < %.1f GB needed; "
                "total across %d GPUs: %.1f GB)",
                max_free_gb, required_gb * 1.2, n_gpus, total_vram_gb,
            )

        return device_map

    def _get_generation_device(self) -> torch.device:
        """Return the device where generation input tensors must be placed."""
        if isinstance(self.current_device_map, str) and self.current_device_map.startswith("cuda:"):
            return torch.device(self.current_device_map)
        if self.current_device_map == "cpu":
            return torch.device("cpu")
        if self.model is not None:
            try:
                return next(self.model.parameters()).device
            except StopIteration:
                pass
        return torch.device(self.device)

    def _get_model_source(self) -> Tuple[str, bool]:
        """
        Return the preferred model source and whether it must be loaded offline.
        The deployed service sets HF_HOME to the project models directory, while
        the Voxtral snapshot is stored as a plain local folder under model_dir.
        Loading by repo_id can therefore block while resolving tokenizer files.
        """
        required_files = ("config.json", "preprocessor_config.json", "tekken.json")
        if os.path.isdir(self.local_model_path) and all(
            os.path.exists(os.path.join(self.local_model_path, file_name))
            for file_name in required_files
        ):
            return self.local_model_path, True
        return self.repo_id, False

    def _normalize_language(self, language: Optional[str]) -> str:
        """Voxtral expects ISO 639-1 alpha-2 language codes such as 'fr'."""
        if not language or language == "Automatic Detection":
            return "en"

        normalized = str(language).strip()
        if len(normalized) == 2:
            return normalized.lower()

        try:
            import whisper

            language_code = {
                value.lower(): key
                for key, value in whisper.tokenizer.LANGUAGES.items()
            }.get(normalized.lower())
            if language_code:
                return language_code
        except Exception as e:
            logger.warning("[VOXTRAL] Failed to map language with whisper tokenizer: %s", e)

        fallback_language_map = {
            "french": "fr",
            "francais": "fr",
            "français": "fr",
            "english": "en",
            "spanish": "es",
            "german": "de",
            "italian": "it",
            "portuguese": "pt",
        }
        language_code = fallback_language_map.get(normalized.lower())
        if language_code:
            return language_code

        logger.warning(
            "[VOXTRAL] Unknown language '%s'; falling back to English alpha-2 code.",
            language,
        )
        return "en"

    def _segment_audio(self, audio_path: str, chunk_length: int, chunk_overlap: int):
        """
        Segment audio file into chunks for processing long audio files.
        Returns list of (start_time, end_time, chunk_path) tuples.
        """
        import soundfile as sf

        if chunk_length <= 0:
            raise ValueError(f"chunk_length must be > 0, got {chunk_length}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
        if chunk_overlap >= chunk_length:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}s) must be smaller than chunk_length ({chunk_length}s)"
            )

        # Load audio
        audio_data, sr = librosa.load(audio_path, sr=16000)
        audio_duration = len(audio_data) / sr
        logger.info(
            "[VOXTRAL] Segmenting audio: path=%s duration=%.2fs sample_rate=%s "
            "chunk_length=%ss chunk_overlap=%ss temp_dir=%s",
            audio_path,
            audio_duration,
            sr,
            chunk_length,
            chunk_overlap,
            self.temp_dir,
        )

        chunks = []
        start = 0
        chunk_idx = 0
        step = chunk_length - chunk_overlap
        max_chunks = int(np.ceil(audio_duration / step)) + 1

        try:
            while start < audio_duration:
                if chunk_idx >= max_chunks:
                    raise RuntimeError(
                        f"Aborting audio segmentation after {chunk_idx} chunks. "
                        f"This indicates invalid chunk progress: start={start}, "
                        f"duration={audio_duration}, step={step}."
                    )

                end = min(start + chunk_length, audio_duration)

                # Extract chunk
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                chunk_data = audio_data[start_sample:end_sample]

                # Save chunk to temp file
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=f'_chunk_{chunk_idx}.wav',
                    dir=self.temp_dir,
                )
                chunk_path = temp_file.name
                temp_file.close()
                sf.write(chunk_path, chunk_data, sr)

                chunks.append((start, end, chunk_path))
                logger.info(
                    "[VOXTRAL] Created chunk %s: %.2fs-%.2fs path=%s size=%s",
                    chunk_idx + 1,
                    start,
                    end,
                    chunk_path,
                    os.path.getsize(chunk_path),
                )

                if end >= audio_duration:
                    break

                # Move to next chunk with overlap. The previous implementation
                # computed this value but never assigned it back to start, which
                # recreated chunk_0 forever and filled /tmp.
                start = end - chunk_overlap
                chunk_idx += 1

        except Exception:
            for _, _, chunk_path in chunks:
                if os.path.exists(chunk_path):
                    os.unlink(chunk_path)
            raise

        logger.info("[VOXTRAL] Audio segmentation completed: chunks=%s", len(chunks))
        return chunks

    def transcribe(self,
                   audio: Union[str, BinaryIO, np.ndarray],
                   progress: gr.Progress = gr.Progress(),
                   progress_callback: Optional[Callable] = None,
                   *whisper_params,
                   ) -> Tuple[List[Segment], float]:
        """
        Transcribe method for voxtral-mini.
        """
        import traceback
        start_time = time.time()
        
        params = WhisperParams.from_list(list(whisper_params))
        logger.info(
            "[VOXTRAL] Transcription requested: audio_type=%s model_size=%s "
            "compute_type=%s language=%s device=%s",
            type(audio).__name__,
            params.model_size,
            params.compute_type,
            params.lang,
            self.device,
        )

        if params.model_size != self.current_model_size or self.model is None:
            self.update_model(params.model_size, params.compute_type, progress)

        progress(0.1, desc="Processing audio...")

        # Convert audio to the format expected by voxtral
        audio_path = self._prepare_audio(audio)
        temp_files_to_cleanup = []

        try:
            # Get audio duration and determine if we need to segment
            audio_duration = self._get_audio_duration(audio_path)
            chunk_length = getattr(params, 'chunk_length', 30)   # Default 30 seconds (audio encoder max)
            chunk_overlap = getattr(params, 'chunk_overlap', 2)   # Default 2 seconds
            logger.info(
                "[VOXTRAL] Prepared audio: path=%s duration=%.2fs chunk_length=%ss "
                "chunk_overlap=%ss",
                audio_path,
                audio_duration,
                chunk_length,
                chunk_overlap,
            )

            segments_result = []

            # Check if audio is longer than chunk_length
            if audio_duration > chunk_length:
                progress(0.2, desc="Segmenting audio for long file processing...")

                # Segment the audio
                audio_chunks = self._segment_audio(audio_path, chunk_length, chunk_overlap)
                temp_files_to_cleanup.extend([chunk[2] for chunk in audio_chunks])

                total_chunks = len(audio_chunks)
                logger.info("[VOXTRAL] Processing segmented audio: total_chunks=%s", total_chunks)

                for i, (chunk_start, chunk_end, chunk_path) in enumerate(audio_chunks):
                    chunk_progress = 0.2 + (0.7 * i / total_chunks)
                    progress(chunk_progress, desc=f"Transcribing chunk {i+1}/{total_chunks} ({chunk_start:.0f}s - {chunk_end:.0f}s)")
                    logger.info(
                        "[VOXTRAL] Transcribing chunk %s/%s: %.2fs-%.2fs path=%s",
                        i + 1,
                        total_chunks,
                        chunk_start,
                        chunk_end,
                        chunk_path,
                    )

                    # Determine language for first chunk or use provided language
                    language = self._normalize_language(params.lang)
                    logger.info("[VOXTRAL] Chunk language normalized: input=%s output=%s", params.lang, language)

                    # Apply transcription request for this chunk
                    inputs = self.processor.apply_transcription_request(
                        language=language,
                        audio=chunk_path,
                        model_id=self.repo_id
                    )
                    inputs = inputs.to(self._get_generation_device(), dtype=torch.bfloat16)

                    # Generate transcription for this chunk
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=params.max_new_tokens or 32000,
                            temperature=params.temperature if params.temperature > 0 else 0.0,
                            do_sample=params.temperature > 0,
                            repetition_penalty=params.repetition_penalty if params.repetition_penalty else 1.0,
                            no_repeat_ngram_size=params.no_repeat_ngram_size if params.no_repeat_ngram_size else 0,
                        )

                    # Decode outputs
                    decoded_outputs = self.processor.batch_decode(
                        outputs[:, inputs.input_ids.shape[1]:],
                        skip_special_tokens=True
                    )

                    # Create segment from transcription with adjusted timestamps
                    transcription_text = decoded_outputs[0].strip() if decoded_outputs else ""
                    logger.info(
                        "[VOXTRAL] Chunk %s/%s decoded: chars=%s",
                        i + 1,
                        total_chunks,
                        len(transcription_text),
                    )

                    if transcription_text:  # Only add non-empty segments
                        segment = Segment(
                            id=i,
                            text=transcription_text,
                            start=chunk_start,
                            end=chunk_end,
                            seek=int(chunk_start),
                            tokens=None,
                            temperature=params.temperature,
                            avg_logprob=None,
                            compression_ratio=None,
                            no_speech_prob=None,
                            words=None
                        )
                        segments_result.append(segment)

                    # Free GPU memory between chunks on all available devices
                    if i < total_chunks - 1 and torch.cuda.is_available():
                        for dev_idx in range(torch.cuda.device_count()):
                            with torch.cuda.device(dev_idx):
                                torch.cuda.empty_cache()

            else:
                # Process as single segment for short audio
                progress(0.3, desc="Transcribing with Voxtral...")
                logger.info("[VOXTRAL] Processing audio as a single segment")

                language = self._normalize_language(params.lang)
                logger.info("[VOXTRAL] Language normalized: input=%s output=%s", params.lang, language)

                inputs = self.processor.apply_transcription_request(
                    language=language,
                    audio=audio_path,
                    model_id=self.repo_id
                )
                inputs = inputs.to(self._get_generation_device(), dtype=torch.bfloat16)

                progress(0.6, desc="Generating transcription...")

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=params.max_new_tokens or 32000,
                        temperature=params.temperature if params.temperature > 0 else 0.0,
                        do_sample=params.temperature > 0,
                        repetition_penalty=params.repetition_penalty if params.repetition_penalty else 1.0,
                        no_repeat_ngram_size=params.no_repeat_ngram_size if params.no_repeat_ngram_size else 0,
                    )

                progress(0.8, desc="Processing results...")

                decoded_outputs = self.processor.batch_decode(
                    outputs[:, inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )

                transcription_text = decoded_outputs[0].strip() if decoded_outputs else ""
                logger.info("[VOXTRAL] Single segment decoded: chars=%s", len(transcription_text))

                segments_result = [
                    Segment(
                        id=0,
                        text=transcription_text,
                        start=0.0,
                        end=audio_duration,
                        seek=0,
                        tokens=None,
                        temperature=params.temperature,
                        avg_logprob=None,
                        compression_ratio=None,
                        no_speech_prob=None,
                        words=None
                    )
                ]

            progress(1.0, desc="Transcription completed!")
            logger.info(
                "[VOXTRAL] Transcription completed: segments=%s elapsed=%.2fs",
                len(segments_result),
                time.time() - start_time,
            )

        except Exception as e:
            error_msg = f"Error during Voxtral transcription: {str(e)}"
            logger.error(f"[VOXTRAL ERROR] {error_msg}")
            logger.error(f"[VOXTRAL TRACEBACK] {traceback.format_exc()}")
            raise RuntimeError(error_msg) from e

        finally:
            # Clean up temporary files
            if not isinstance(audio, str) or audio_path != audio:
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
                    logger.info("[VOXTRAL] Removed prepared temporary audio: %s", audio_path)

            # Clean up chunk files
            removed_chunks = 0
            for temp_file in temp_files_to_cleanup:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    removed_chunks += 1
            if temp_files_to_cleanup:
                logger.info(
                    "[VOXTRAL] Removed temporary chunk files: removed=%s expected=%s",
                    removed_chunks,
                    len(temp_files_to_cleanup),
                )

        elapsed_time = time.time() - start_time
        return segments_result, elapsed_time

    def update_model(self,
                     model_size: str,
                     compute_type: str,
                     progress: gr.Progress = gr.Progress()
                     ):
        """
        Update current model setting for voxtral-mini
        """
        # Only handle voxtral models
        if model_size != "voxtral-mini-3b":
            raise ValueError(f"VoxtralWhisperInference only supports 'voxtral-mini-3b', not '{model_size}'. "
                           f"Please use a different whisper_type for model '{model_size}'.")

        progress(0, desc="Initializing Voxtral Model...")

        if self.model is not None and self.current_model_size == model_size:
            logger.info("[VOXTRAL] Model already loaded: model_size=%s", model_size)
            return

        model_source, local_files_only = self._get_model_source()

        # Load processor
        progress(0.3, desc="Loading processor...")
        logger.info(
            "[VOXTRAL] Loading processor: source=%s local_files_only=%s",
            model_source,
            local_files_only,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_source,
            local_files_only=local_files_only,
        )

        # Select the best device_map for this model size
        device_map = self._select_device_map(model_size)
        self.current_device_map = device_map
        if isinstance(device_map, str) and (device_map.startswith("cuda:") or device_map == "cpu"):
            self.device = device_map

        # Load model
        progress(0.6, desc="Loading model...")
        logger.info(
            "[VOXTRAL] Loading model: source=%s device_map=%s torch_dtype=%s local_files_only=%s",
            model_source, device_map, torch.bfloat16, local_files_only,
        )
        self.model = VoxtralForConditionalGeneration.from_pretrained(
            model_source,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            local_files_only=local_files_only,
            attn_implementation="sdpa",
        )

        # Log actual layer → device mapping (informational)
        if hasattr(self.model, "hf_device_map"):
            devices_used = sorted(set(str(d) for d in self.model.hf_device_map.values()))
            logger.info("[VOXTRAL] Active devices after placement: %s", devices_used)
            if len(devices_used) > 1:
                logger.info("[VOXTRAL] Layer distribution: %s", self.model.hf_device_map)

        self.current_model_size = model_size
        self.current_compute_type = compute_type

        # torch.compile() requires all model parameters on a single device.
        progress(0.9, desc="Compiling model for faster inference...")
        multi_device = (
            hasattr(self.model, "hf_device_map")
            and len(set(str(d) for d in self.model.hf_device_map.values())) > 1
        )
        if multi_device:
            logger.info("[VOXTRAL] Skipping torch.compile: model spans multiple devices")
        else:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("[VOXTRAL] Model compiled with torch.compile (mode=reduce-overhead)")
            except Exception as e:
                logger.warning("[VOXTRAL] torch.compile failed, using eager mode: %s", e)

        progress(1.0, desc="Model loaded successfully!")
        logger.info("[VOXTRAL] Model loaded successfully: model_size=%s", model_size)

    # Formats that soundfile (used internally by the processor's load_audio_as)
    # can read natively. Anything outside this set must be pre-converted to WAV.
    _SOUNDFILE_EXTENSIONS = {'.wav', '.flac', '.ogg', '.mp3', '.aif', '.aiff', '.au', '.snd'}

    def _prepare_audio(self, audio: Union[str, BinaryIO, np.ndarray]) -> str:
        """
        Prepare audio for voxtral processing.
        Voxtral's processor uses soundfile internally, which does NOT support AAC-based
        containers (.m4a, .aac, .wma, .mp4 audio-only, etc.).  When the extension is
        not in _SOUNDFILE_EXTENSIONS the file is decoded through librosa+ffmpeg and
        re-written as a 16 kHz mono WAV so the processor can read it safely.

        Returns
        -------
        str: Path to a WAV (or other soundfile-compatible) audio file
        """
        import soundfile as sf

        if isinstance(audio, str):
            ext = os.path.splitext(audio)[1].lower()
            if ext in self._SOUNDFILE_EXTENSIONS:
                logger.info("[VOXTRAL] Using existing audio path: %s", audio)
                return audio
            # Format not natively supported by soundfile (e.g. .m4a, .aac, .wma)
            # → decode via librosa+ffmpeg and write a temporary WAV
            logger.info(
                "[VOXTRAL] Converting %s → WAV (soundfile does not support %s)", audio, ext
            )
            audio_data, sr = librosa.load(audio, sr=16000, mono=True)
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix='.wav', dir=self.temp_dir
            )
            temp_path = temp_file.name
            temp_file.close()
            sf.write(temp_path, audio_data, sr)
            logger.info("[VOXTRAL] Converted to temporary WAV: %s (%.2fs)", temp_path, len(audio_data)/sr)
            return temp_path

        elif isinstance(audio, np.ndarray):
            # Convert numpy array to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=self.temp_dir)
            temp_path = temp_file.name
            temp_file.close()

            # Use librosa to save the numpy array as audio file
            import soundfile as sf
            sf.write(temp_path, audio, 16000)
            logger.info("[VOXTRAL] Wrote numpy audio to temporary file: %s", temp_path)
            return temp_path
        else:
            # Binary IO - save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=self.temp_dir)
            temp_path = temp_file.name

            audio.seek(0)
            temp_file.write(audio.read())
            temp_file.close()

            logger.info("[VOXTRAL] Wrote binary audio to temporary file: %s", temp_path)
            return temp_path

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds"""
        try:
            audio_data, sr = librosa.load(audio_path, sr=None)
            duration = len(audio_data) / sr
            logger.info("[VOXTRAL] Audio duration detected: path=%s duration=%.2fs sample_rate=%s", audio_path, duration, sr)
            return duration
        except Exception as e:
            # Fallback duration if we can't determine it
            logger.warning("[VOXTRAL] Failed to detect audio duration for %s: %s. Falling back to 30s.", audio_path, e)
            return 30.0

    def get_available_compute_type(self):
        """Return available compute types for voxtral"""
        if self.device == "cuda":
            return ["float16", "bfloat16"]
        else:
            return ["float32"]

    def get_compute_type(self):
        """Get default compute type for voxtral"""
        if self.device == "cuda":
            return "bfloat16"
        else:
            return "float32"
