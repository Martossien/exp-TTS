from typing import Optional
import os
import torch

from modules.utils.paths import (FASTER_WHISPER_MODELS_DIR, DIARIZATION_MODELS_DIR, OUTPUT_DIR,
                                 INSANELY_FAST_WHISPER_MODELS_DIR, WHISPER_MODELS_DIR, UVR_MODELS_DIR,
                                 VOXTRAL_MODELS_DIR, QWEN3_ASR_MODELS_DIR, COHERE_ASR_MODELS_DIR,
                                 VOXTRAL_REALTIME_MODELS_DIR)
from modules.whisper.faster_whisper_inference import FasterWhisperInference
from modules.whisper.whisper_Inference import WhisperInference
from modules.whisper.insanely_fast_whisper_inference import InsanelyFastWhisperInference
from modules.whisper.voxtral_whisper_inference import VoxtralWhisperInference
from modules.whisper.qwen3_asr_inference import Qwen3ASRInference
from modules.whisper.cohere_asr_inference import CohereASRInference
from modules.whisper.voxtral_realtime_vllm_inference import VoxtralRealtimeVLLMInference
from modules.whisper.base_transcription_pipeline import BaseTranscriptionPipeline
from modules.whisper.data_classes import *
from modules.utils.logger import get_logger


logger = get_logger()


class WhisperFactory:
    @staticmethod
    def _has_required_files(model_dir: str, required_files: tuple[str, ...]) -> bool:
        return os.path.isdir(model_dir) and all(
            os.path.exists(os.path.join(model_dir, file_name))
            for file_name in required_files
        )

    @staticmethod
    def _has_weight_file(model_dir: str) -> bool:
        if not os.path.isdir(model_dir):
            return False

        return any(
            file_name.endswith((".bin", ".safetensors", ".pt"))
            for file_name in os.listdir(model_dir)
        )

    @staticmethod
    def _hf_snapshot_dir(cache_dir: str, repo_id: str) -> Optional[str]:
        repo_cache_dir = os.path.join(cache_dir, f"models--{repo_id.replace('/', '--')}")
        refs_main = os.path.join(repo_cache_dir, "refs", "main")
        snapshots_dir = os.path.join(repo_cache_dir, "snapshots")

        if os.path.isfile(refs_main):
            with open(refs_main, "r", encoding="utf-8") as file:
                snapshot_id = file.read().strip()
            snapshot_dir = os.path.join(snapshots_dir, snapshot_id)
            if os.path.isdir(snapshot_dir):
                return snapshot_dir

        if os.path.isdir(snapshots_dir):
            for snapshot_id in sorted(os.listdir(snapshots_dir)):
                snapshot_dir = os.path.join(snapshots_dir, snapshot_id)
                if os.path.isdir(snapshot_dir):
                    return snapshot_dir

        return None

    @staticmethod
    def _installed_faster_whisper_models() -> list[str]:
        import faster_whisper

        installed_models = []
        required_files = ("config.json", "model.bin")
        ui_model_allowlist = {"large-v3", "large-v3-turbo"}

        for model_name in faster_whisper.available_models():
            if model_name not in ui_model_allowlist:
                continue

            local_dir = os.path.join(FASTER_WHISPER_MODELS_DIR, model_name)
            if WhisperFactory._has_required_files(local_dir, required_files):
                installed_models.append(model_name)
                continue

            repo_id = f"Systran/faster-whisper-{model_name}"
            snapshot_dir = WhisperFactory._hf_snapshot_dir(FASTER_WHISPER_MODELS_DIR, repo_id)
            if snapshot_dir and WhisperFactory._has_required_files(snapshot_dir, required_files):
                installed_models.append(model_name)

        return installed_models

    @staticmethod
    def _installed_voxtral_models() -> list[str]:
        model_name = "voxtral-mini-3b"
        model_dir = os.path.join(VOXTRAL_MODELS_DIR, model_name)
        if (
            WhisperFactory._has_required_files(model_dir, ("config.json", "preprocessor_config.json"))
            and WhisperFactory._has_weight_file(model_dir)
        ):
            return [model_name]
        repo_id = "mistralai/Voxtral-Mini-3B-2507"
        snapshot_dir = WhisperFactory._hf_snapshot_dir(
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
            repo_id,
        )
        if snapshot_dir and WhisperFactory._has_required_files(
            snapshot_dir, ("config.json", "preprocessor_config.json")
        ) and WhisperFactory._has_weight_file(snapshot_dir):
            return [model_name]
        snapshot_dir_local = WhisperFactory._hf_snapshot_dir(VOXTRAL_MODELS_DIR, repo_id)
        if snapshot_dir_local and WhisperFactory._has_required_files(
            snapshot_dir_local, ("config.json", "preprocessor_config.json")
        ) and WhisperFactory._has_weight_file(snapshot_dir_local):
            return [model_name]
        return []

    @staticmethod
    def _installed_qwen3_asr_models() -> list[str]:
        installed_models = []
        hf_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        for model_name in Qwen3ASRInference.available_models:
            model_dir = os.path.join(QWEN3_ASR_MODELS_DIR, model_name)
            if (
                WhisperFactory._has_required_files(model_dir, ("config.json", "preprocessor_config.json"))
                and WhisperFactory._has_weight_file(model_dir)
            ):
                installed_models.append(model_name)
                continue
            repo_id = Qwen3ASRInference.available_models[model_name]
            snapshot_dir = WhisperFactory._hf_snapshot_dir(hf_cache_dir, repo_id)
            if snapshot_dir and WhisperFactory._has_required_files(
                snapshot_dir, ("config.json", "preprocessor_config.json")
            ) and WhisperFactory._has_weight_file(snapshot_dir):
                installed_models.append(model_name)
                continue
            snapshot_dir_local = WhisperFactory._hf_snapshot_dir(QWEN3_ASR_MODELS_DIR, repo_id)
            if snapshot_dir_local and WhisperFactory._has_required_files(
                snapshot_dir_local, ("config.json", "preprocessor_config.json")
            ) and WhisperFactory._has_weight_file(snapshot_dir_local):
                installed_models.append(model_name)
        return installed_models

    @staticmethod
    def _installed_cohere_asr_models() -> list[str]:
        installed_models = []
        hf_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        for model_name in CohereASRInference.available_models:
            model_dir = os.path.join(COHERE_ASR_MODELS_DIR, model_name)
            if (
                WhisperFactory._has_required_files(model_dir, ("config.json",))
                and WhisperFactory._has_weight_file(model_dir)
            ):
                installed_models.append(model_name)
                continue
            repo_id = CohereASRInference.available_models[model_name]
            snapshot_dir = WhisperFactory._hf_snapshot_dir(hf_cache_dir, repo_id)
            if snapshot_dir and WhisperFactory._has_required_files(
                snapshot_dir, ("config.json",)
            ) and WhisperFactory._has_weight_file(snapshot_dir):
                installed_models.append(model_name)
                continue
            snapshot_dir_local = WhisperFactory._hf_snapshot_dir(COHERE_ASR_MODELS_DIR, repo_id)
            if snapshot_dir_local and WhisperFactory._has_required_files(
                snapshot_dir_local, ("config.json",)
            ) and WhisperFactory._has_weight_file(snapshot_dir_local):
                installed_models.append(model_name)
        return installed_models

    @staticmethod
    def _installed_voxtral_realtime_vllm_models() -> list[str]:
        host = os.environ.get("VOXTRAL_VLLM_HOST", "localhost")
        port = int(os.environ.get("VOXTRAL_VLLM_PORT", "8000"))
        try:
            import urllib.request
            urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2)
            return ["voxtral-realtime-vllm"]
        except Exception:
            return []

    @staticmethod
    def get_combined_available_models(whisper_type: Optional[str] = None):
        """
        Get installed models for a specific whisper implementation or all implementations.
        
        Parameters
        ----------
        whisper_type : Optional[str]
            The type of Whisper implementation to filter models for.
            If None, returns all available models from all implementations.
        
        Returns
        -------
        list
            List of available models for the specified implementation or all implementations
        """
        combined_models = []
        
        # If no specific type is requested, return all locally installed models.
        if whisper_type is None:
            combined_models.extend(WhisperFactory._installed_faster_whisper_models())
            combined_models.extend(WhisperFactory._installed_voxtral_models())
            combined_models.extend(WhisperFactory._installed_qwen3_asr_models())
            combined_models.extend(WhisperFactory._installed_cohere_asr_models())
            combined_models.extend(WhisperFactory._installed_voxtral_realtime_vllm_models())
            
            # Remove duplicates while preserving order
            seen = set()
            unique_models = []
            for model in combined_models:
                if model not in seen:
                    seen.add(model)
                    unique_models.append(model)
            
            return unique_models
        
        # If a specific type is requested, filter accordingly
        if whisper_type == WhisperImpl.VOXTRAL_MINI.value:
            return WhisperFactory._installed_voxtral_models()
        elif whisper_type == WhisperImpl.QWEN3_ASR.value:
            return WhisperFactory._installed_qwen3_asr_models()
        elif whisper_type == WhisperImpl.COHERE_ASR.value:
            return WhisperFactory._installed_cohere_asr_models()
        elif whisper_type == WhisperImpl.VOXTRAL_REALTIME_VLLM.value:
            return WhisperFactory._installed_voxtral_realtime_vllm_models()
        elif whisper_type in [WhisperImpl.FASTER_WHISPER.value, WhisperImpl.WHISPER.value, WhisperImpl.INSANELY_FAST_WHISPER.value]:
            return WhisperFactory._installed_faster_whisper_models()
        else:
            return WhisperFactory._installed_faster_whisper_models()
    
    @staticmethod
    def create_whisper_inference(
        whisper_type: str,
        whisper_model_dir: str = WHISPER_MODELS_DIR,
        faster_whisper_model_dir: str = FASTER_WHISPER_MODELS_DIR,
        insanely_fast_whisper_model_dir: str = INSANELY_FAST_WHISPER_MODELS_DIR,
        voxtral_model_dir: str = VOXTRAL_MODELS_DIR,
        qwen3_asr_model_dir: str = QWEN3_ASR_MODELS_DIR,
        cohere_asr_model_dir: str = COHERE_ASR_MODELS_DIR,
        voxtral_realtime_model_dir: str = VOXTRAL_REALTIME_MODELS_DIR,
        diarization_model_dir: str = DIARIZATION_MODELS_DIR,
        uvr_model_dir: str = UVR_MODELS_DIR,
        output_dir: str = OUTPUT_DIR,
    ) -> "BaseTranscriptionPipeline":
        """
        Create a whisper inference class based on the provided whisper_type.

        Parameters
        ----------
        whisper_type : str
            The type of Whisper implementation to use. Supported values (case-insensitive):
            - "faster-whisper": https://github.com/openai/whisper
            - "whisper": https://github.com/openai/whisper
            - "insanely-fast-whisper": https://github.com/Vaibhavs10/insanely-fast-whisper
            - "voxtral-mini": https://huggingface.co/mistralai/Voxtral-Mini-3B-2507
            - "qwen3-asr": https://github.com/QwenLM/Qwen3-ASR
            - "voxtral-realtime-vllm": requires external vLLM process on port 8000
        whisper_model_dir : str
            Directory path for the Whisper model.
        faster_whisper_model_dir : str
            Directory path for the Faster Whisper model.
        insanely_fast_whisper_model_dir : str
            Directory path for the Insanely Fast Whisper model.
        voxtral_model_dir : str
            Directory path for the Voxtral model.
        qwen3_asr_model_dir : str
            Directory path for the Qwen3-ASR model.
        diarization_model_dir : str
            Directory path for the diarization model.
        uvr_model_dir : str
            Directory path for the UVR model.
        output_dir : str
            Directory path where output files will be saved.

        Returns
        -------
        BaseTranscriptionPipeline
            An instance of the appropriate whisper inference class based on the whisper_type.
        """
        # Temporal fix of the bug : https://github.com/OlivierAlbertini/Whisper-WebUI/issues/144
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

        whisper_type = whisper_type.strip().lower()
        logger.info("[FACTORY] Creating inference for whisper_type='%s'", whisper_type)

        if whisper_type == WhisperImpl.FASTER_WHISPER.value:
            if torch.xpu.is_available():
                logger.warning(
                    "[FACTORY] FALLBACK: XPU detected but faster-whisper only supports CUDA. "
                    "Switching to insanely-fast-whisper."
                )
                return InsanelyFastWhisperInference(
                    model_dir=insanely_fast_whisper_model_dir,
                    output_dir=output_dir,
                    diarization_model_dir=diarization_model_dir,
                    uvr_model_dir=uvr_model_dir
                )
            logger.info("[FACTORY] Returning FasterWhisperInference (model_dir=%s)", faster_whisper_model_dir)
            return FasterWhisperInference(
                model_dir=faster_whisper_model_dir,
                output_dir=output_dir,
                diarization_model_dir=diarization_model_dir,
                uvr_model_dir=uvr_model_dir
            )
        elif whisper_type == WhisperImpl.WHISPER.value:
            logger.info("[FACTORY] Returning WhisperInference (model_dir=%s)", whisper_model_dir)
            return WhisperInference(
                model_dir=whisper_model_dir,
                output_dir=output_dir,
                diarization_model_dir=diarization_model_dir,
                uvr_model_dir=uvr_model_dir
            )
        elif whisper_type == WhisperImpl.INSANELY_FAST_WHISPER.value:
            logger.info("[FACTORY] Returning InsanelyFastWhisperInference")
            return InsanelyFastWhisperInference(
                model_dir=insanely_fast_whisper_model_dir,
                output_dir=output_dir,
                diarization_model_dir=diarization_model_dir,
                uvr_model_dir=uvr_model_dir
            )
        elif whisper_type == WhisperImpl.VOXTRAL_MINI.value:
            try:
                logger.info("[FACTORY] Returning VoxtralWhisperInference (model_dir=%s)", voxtral_model_dir)
                return VoxtralWhisperInference(
                    model_dir=voxtral_model_dir,
                    output_dir=output_dir,
                    diarization_model_dir=diarization_model_dir,
                    uvr_model_dir=uvr_model_dir
                )
            except ImportError as e:
                logger.warning(
                    "[FACTORY] FALLBACK: Voxtral not available (ImportError: %s). "
                    "Falling back to faster-whisper.", e
                )
                return FasterWhisperInference(
                    model_dir=faster_whisper_model_dir,
                    output_dir=output_dir,
                    diarization_model_dir=diarization_model_dir,
                    uvr_model_dir=uvr_model_dir
                )
        elif whisper_type == WhisperImpl.QWEN3_ASR.value:
            try:
                logger.info("[FACTORY] Returning Qwen3ASRInference (model_dir=%s)", qwen3_asr_model_dir)
                return Qwen3ASRInference(
                    model_dir=qwen3_asr_model_dir,
                    output_dir=output_dir,
                    diarization_model_dir=diarization_model_dir,
                    uvr_model_dir=uvr_model_dir
                )
            except ImportError as e:
                logger.warning(
                    "[FACTORY] FALLBACK: Qwen3-ASR not available (ImportError: %s). "
                    "Falling back to faster-whisper.", e
                )
                return FasterWhisperInference(
                    model_dir=faster_whisper_model_dir,
                    output_dir=output_dir,
                    diarization_model_dir=diarization_model_dir,
                    uvr_model_dir=uvr_model_dir
                )
        elif whisper_type == WhisperImpl.COHERE_ASR.value:
            try:
                logger.info("[FACTORY] Returning CohereASRInference (model_dir=%s)", cohere_asr_model_dir)
                return CohereASRInference(
                    model_dir=cohere_asr_model_dir,
                    output_dir=output_dir,
                    diarization_model_dir=diarization_model_dir,
                    uvr_model_dir=uvr_model_dir
                )
            except Exception as e:
                logger.warning(
                    "[FACTORY] FALLBACK: Cohere ASR not available (%s: %s). "
                    "Falling back to faster-whisper.", type(e).__name__, e
                )
                return FasterWhisperInference(
                    model_dir=faster_whisper_model_dir,
                    output_dir=output_dir,
                    diarization_model_dir=diarization_model_dir,
                    uvr_model_dir=uvr_model_dir
                )
        elif whisper_type == WhisperImpl.VOXTRAL_REALTIME_VLLM.value:
            logger.info("[FACTORY] Returning VoxtralRealtimeVLLMInference")
            return VoxtralRealtimeVLLMInference(
                model_dir=voxtral_realtime_model_dir,
                output_dir=output_dir,
                diarization_model_dir=diarization_model_dir,
                uvr_model_dir=uvr_model_dir,
            )
        else:
            logger.warning(
                "[FACTORY] FALLBACK: Unknown whisper_type='%s'. "
                "Falling back to faster-whisper.", whisper_type
            )
            return FasterWhisperInference(
                model_dir=faster_whisper_model_dir,
                output_dir=output_dir,
                diarization_model_dir=diarization_model_dir,
                uvr_model_dir=uvr_model_dir
            )
