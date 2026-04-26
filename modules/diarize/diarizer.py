import os
import torch
from typing import List, Union, BinaryIO, Optional, Tuple
import numpy as np
import time
import logging
import gc

from modules.utils.paths import DIARIZATION_MODELS_DIR
from modules.diarize.diarize_pipeline import DiarizationPipeline, assign_word_speakers
from modules.diarize.audio_loader import load_audio
from modules.whisper.data_classes import *


class Diarizer:
    def __init__(self,
                 model_dir: str = DIARIZATION_MODELS_DIR
                 ):
        self.device = self.get_device()
        self.available_device = self.get_available_device()
        self.compute_type = "float16"
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.pipe = None

    def run(self,
            audio: Union[str, BinaryIO, np.ndarray],
            transcribed_result: List[Segment],
            use_auth_token: str,
            device: Optional[str] = None,
            model_name: Optional[str] = "pyannote/speaker-diarization-community-1",
            min_speakers: Optional[int] = None,
            max_speakers: Optional[int] = None
            ) -> Tuple[List[Segment], float]:
        """
        Diarize transcribed result as a post-processing

        Parameters
        ----------
        audio: Union[str, BinaryIO, np.ndarray]
            Audio input. This can be file path or binary type.
        transcribed_result: List[Segment]
            transcribed result through whisper.
        use_auth_token: str
            Huggingface token with READ permission. This is only needed the first time you download the model.
            You must manually go to the website https://huggingface.co/pyannote/speaker-diarization-community-1 (or 3.1) and agree to their TOS to download the model.
        device: Optional[str]
            Device for diarization.
        model_name: Optional[str]
            Diarization model to use. Defaults to "pyannote/speaker-diarization-community-1".
            Options: "pyannote/speaker-diarization-community-1" or "pyannote/speaker-diarization-3.1"
        min_speakers: Optional[int]
            Minimum number of speakers to detect. None for automatic detection.
        max_speakers: Optional[int]
            Maximum number of speakers to detect. None for automatic detection.

        Returns
        ----------
        segments_result: List[Segment]
            list of Segment that includes start, end timestamps and transcribed text
        elapsed_time: float
            elapsed time for running
        """
        start_time = time.time()

        if device is None:
            device = self.device

        if device != self.device or self.pipe is None:
            self.update_pipe(
                device=device,
                use_auth_token=use_auth_token,
                model_name=model_name
            )

        audio = load_audio(audio)

        if self.pipe is not None and self.pipe.model is not None:
            diarization_segments = self.pipe(audio, min_speakers=min_speakers, max_speakers=max_speakers)
            diarized_result = assign_word_speakers(
                diarization_segments,
                {"segments": transcribed_result}
            )
        else:
            # If diarization is not available, return the original transcribed result
            diarized_result = {"segments": transcribed_result}

        segments_result = []
        for segment in diarized_result["segments"]:
            speaker = "None"
            # Handle both dictionary and Segment object cases
            if isinstance(segment, dict):
                # Dictionary case (from assign_word_speakers)
                if "speaker" in segment:
                    speaker = segment["speaker"]
                diarized_text = speaker + "|" + segment["text"].strip()
                segments_result.append(Segment(
                    start=segment["start"],
                    end=segment["end"],
                    text=diarized_text
                ))
            else:
                # Segment object case
                if hasattr(segment, "speaker") and segment.speaker is not None:
                    speaker = segment.speaker
                diarized_text = speaker + "|" + segment.text.strip()
                segments_result.append(Segment(
                    start=segment.start,
                    end=segment.end,
                    text=diarized_text
                ))

        elapsed_time = time.time() - start_time
        return segments_result, elapsed_time

    def update_pipe(self,
                    use_auth_token: Optional[str] = None,
                    device: Optional[str] = None,
                    model_name: Optional[str] = "pyannote/speaker-diarization-community-1",
                    ):
        """
        Set pipeline for diarization

        Parameters
        ----------
        use_auth_token: str
            Huggingface token with READ permission. This is only needed the first time you download the model.
            You must manually go to the website https://huggingface.co/pyannote/speaker-diarization-community-1 (or 3.1) and agree to their TOS to download the model.
        device: str
            Device for diarization.
        model_name: str
            Diarization model to use. Defaults to "pyannote/speaker-diarization-community-1".
        """
        if device is None:
            device = self.get_device()
        self.device = device

        os.makedirs(self.model_dir, exist_ok=True)

        # Determine which model URL to show in error message
        model_url = f"https://huggingface.co/{model_name}"

        # Always try to update the pipe if we have a token or if model files exist
        if use_auth_token or os.listdir(self.model_dir):
            pass  # Continue to update the pipe
        elif not os.listdir(self.model_dir):
            print(
                f"\nFailed to diarize. You need huggingface token and agree to their requirements to download the diarization model.\n"
                f"Go to \"{model_url}\" and follow their instructions to download the model.\n"
            )
            return

        logger = logging.getLogger("speechbrain.utils.train_logger")
        # Disable redundant torchvision warning message
        logger.disabled = True
        self.pipe = DiarizationPipeline(
            model_name=model_name,
            use_auth_token=use_auth_token,
            device=device,
            cache_dir=self.model_dir
        )
        logger.disabled = False

    def offload(self):
        """Offload the model and free up the memory"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
            # Use reset_peak_memory_stats instead of reset_max_memory_allocated
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
            else:
                torch.cuda.reset_max_memory_allocated()
        if self.device == "xpu":
            torch.xpu.empty_cache()
            torch.xpu.reset_accumulated_memory_stats()
            torch.xpu.reset_peak_memory_stats()
        gc.collect()

    def detect_speakers_only(
            self,
            audio: Union[str, BinaryIO, np.ndarray],
            use_auth_token: str,
            device: Optional[str] = None,
            model_name: Optional[str] = "pyannote/speaker-diarization-community-1",
            min_speakers: Optional[int] = None,
            max_speakers: Optional[int] = None,
    ) -> Tuple[List[dict], dict, float]:
        """Run diarization WITHOUT transcription — detect speaker turns only.

        Returns
        ----------
        turns : List[dict]
            List of dicts with keys: speaker, start, end, duration
        speakers_summary : dict
            Dict mapping speaker_id -> {total_time, speaking_pct, num_turns}
        elapsed_time : float
            Elapsed time in seconds
        """
        start_time = time.time()

        if device is None:
            device = self.device
        logger_diarizer = logging.getLogger(__name__)
        logger_diarizer.info("[DIARIZER] detect_speakers_only: audio=%s, model=%s, device=%s, min=%s, max=%s",
                             str(audio)[:80] if isinstance(audio, str) else "<audio>", model_name, device,
                             min_speakers, max_speakers)

        if device != self.device or self.pipe is None:
            self.update_pipe(
                device=device,
                use_auth_token=use_auth_token,
                model_name=model_name
            )

        audio_np = load_audio(audio)
        audio_duration = len(audio_np) / 16000
        logger_diarizer.info("[DIARIZER] Audio loaded: duration=%.1fs", audio_duration)

        if self.pipe is None or self.pipe.model is None:
            logger_diarizer.error("[DIARIZER] Pipeline not loaded — cannot run diarization")
            return [], {}, time.time() - start_time

        diarize_df = self.pipe(audio_np, min_speakers=min_speakers, max_speakers=max_speakers)

        if diarize_df.empty:
            logger_diarizer.warning("[DIARIZER] Diarization returned empty result")
            return [], {}, time.time() - start_time

        num_speakers = diarize_df['speaker'].nunique()
        logger_diarizer.info("[DIARIZER] Diarization complete: %d turns, %d speakers detected",
                             len(diarize_df), num_speakers)

        turns = []
        speakers_total: dict = {}

        for _, row in diarize_df.iterrows():
            spk = row['speaker']
            start = float(row['start'])
            end = float(row['end'])
            dur = end - start

            turns.append({
                'speaker': spk,
                'start': start,
                'end': end,
                'duration': dur,
            })

            if spk not in speakers_total:
                speakers_total[spk] = {'total_time': 0.0, 'num_turns': 0}
            speakers_total[spk]['total_time'] += dur
            speakers_total[spk]['num_turns'] += 1

        total_speech = sum(v['total_time'] for v in speakers_total.values())
        speakers_summary = {}
        for spk, data in speakers_total.items():
            pct = (data['total_time'] / total_speech * 100) if total_speech > 0 else 0
            speakers_summary[spk] = {
                'total_time': round(data['total_time'], 2),
                'num_turns': data['num_turns'],
                'speaking_pct': round(pct, 1),
            }

        elapsed = time.time() - start_time
        logger_diarizer.info("[DIARIZER] detect_speakers_only finished in %.1fs — %d speakers, %d turns",
                             elapsed, num_speakers, len(turns))
        return turns, speakers_summary, elapsed

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        if torch.xpu.is_available():
            return "xpu"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    @staticmethod
    def get_available_device():
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        if torch.xpu.is_available():
            devices.append("xpu")
        if torch.backends.mps.is_available():
            devices.append("mps")
        return devices