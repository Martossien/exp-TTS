# Adapted from https://github.com/m-bain/whisperX/blob/main/whisperx/diarize.py

import numpy as np
import pandas as pd
import os
import logging
from pathlib import Path
from pyannote.audio import Pipeline
from typing import Optional, Union
import torch

from modules.whisper.data_classes import *
from modules.utils.paths import DIARIZATION_MODELS_DIR
from modules.diarize.audio_loader import load_audio, SAMPLE_RATE


logger = logging.getLogger(__name__)


def resolve_local_pyannote_checkpoint(model_name: str, cache_dir: str) -> str:
    """Return a local pyannote snapshot path when the model is already cached."""
    checkpoint = Path(model_name)
    if checkpoint.exists():
        return str(checkpoint)

    model_cache_dir = Path(cache_dir) / f"models--{model_name.replace('/', '--')}"
    refs_main = model_cache_dir / "refs" / "main"
    snapshots_dir = model_cache_dir / "snapshots"

    snapshot = None
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        if revision:
            snapshot = snapshots_dir / revision

    if snapshot is None and snapshots_dir.exists():
        snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
        if snapshots:
            snapshot = max(snapshots, key=lambda path: path.stat().st_mtime)

    if snapshot is not None and (snapshot / "config.yaml").exists():
        logger.info("Using local pyannote checkpoint: %s", snapshot)
        return str(snapshot)

    return model_name


class DiarizationPipeline:
    def __init__(
        self,
        model_name="pyannote/speaker-diarization-community-1",
        cache_dir: str = DIARIZATION_MODELS_DIR,
        use_auth_token=None,
        device: Optional[Union[str, torch.device]] = "cpu",
    ):
        if isinstance(device, str):
            device = torch.device(device)
        local_checkpoint = resolve_local_pyannote_checkpoint(model_name, cache_dir)
        model = Pipeline.from_pretrained(
            local_checkpoint,
            token=use_auth_token,
            cache_dir=cache_dir
        )
        if model is not None:
            self.model = model.to(device)
        else:
            self.model = None

    def __call__(self, audio: Union[str, np.ndarray], min_speakers=None, max_speakers=None):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_data = {
            'waveform': torch.from_numpy(audio[None, :]),
            'sample_rate': SAMPLE_RATE
        }
        if self.model is not None:
            output = self.model(audio_data, min_speakers=min_speakers, max_speakers=max_speakers)
            # pyannote.audio 4.x returns a DiarizeOutput dataclass; the actual
            # Annotation (with itertracks) lives in .speaker_diarization.
            # Earlier versions returned the Annotation directly — handle both.
            if hasattr(output, 'speaker_diarization'):
                segments = output.speaker_diarization
            else:
                segments = output
            diarize_df = pd.DataFrame(segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
            diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
            diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)
            return diarize_df
        else:
            # Return empty DataFrame if model is not available
            return pd.DataFrame(columns=['segment', 'label', 'speaker', 'start', 'end'])


def assign_word_speakers(diarize_df, transcript_result, fill_nearest=False):
    transcript_segments = transcript_result["segments"]
    if transcript_segments and isinstance(transcript_segments[0], Segment):
        transcript_segments = [seg.model_dump() for seg in transcript_segments]
    for seg in transcript_segments:
        # assign speaker to segment (if any)
        diarize_df['intersection'] = np.minimum(diarize_df['end'], seg['end']) - np.maximum(diarize_df['start'],
                                                                                            seg['start'])
        diarize_df['union'] = np.maximum(diarize_df['end'], seg['end']) - np.minimum(diarize_df['start'], seg['start'])

        intersected = diarize_df[diarize_df["intersection"] > 0]

        speaker = None
        if len(intersected) > 0:
            # Choosing most strong intersection
            speaker = intersected.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
        elif fill_nearest:
            # Otherwise choosing closest
            speaker = diarize_df.sort_values(by=["intersection"], ascending=False)["speaker"].values[0]

        if speaker is not None:
            seg["speaker"] = speaker

        # assign speaker to words
        if 'words' in seg and seg['words'] is not None:
            for word in seg['words']:
                if 'start' in word:
                    diarize_df['intersection'] = np.minimum(diarize_df['end'], word['end']) - np.maximum(
                        diarize_df['start'], word['start'])
                    diarize_df['union'] = np.maximum(diarize_df['end'], word['end']) - np.minimum(diarize_df['start'],
                                                                                                  word['start'])

                    intersected = diarize_df[diarize_df["intersection"] > 0]

                    word_speaker = None
                    if len(intersected) > 0:
                        # Choosing most strong intersection
                        word_speaker = \
                            intersected.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
                    elif fill_nearest:
                        # Otherwise choosing closest
                        word_speaker = diarize_df.sort_values(by=["intersection"], ascending=False)["speaker"].values[0]

                    if word_speaker is not None:
                        word["speaker"] = word_speaker

    return {"segments": transcript_segments}


class DiarizationSegment:
    def __init__(self, start, end, speaker=None):
        self.start = start
        self.end = end
        self.speaker = speaker
