import os
import argparse
import shutil
import gradio as gr
import yaml
import torch

from modules.utils.paths import (FASTER_WHISPER_MODELS_DIR, DIARIZATION_MODELS_DIR, OUTPUT_DIR, WHISPER_MODELS_DIR,
                                 INSANELY_FAST_WHISPER_MODELS_DIR, VOXTRAL_MODELS_DIR, NLLB_MODELS_DIR, DEFAULT_PARAMETERS_CONFIG_PATH,
                                 QWEN3_ASR_MODELS_DIR, COHERE_ASR_MODELS_DIR, UVR_MODELS_DIR)
from modules.utils.files_manager import load_yaml, MEDIA_EXTENSION
from modules.whisper.whisper_factory import WhisperFactory
from modules.translation.nllb_inference import NLLBInference
from modules.ui.htmls import *
from modules.utils.cli_manager import str2bool
from modules.utils.youtube_manager import get_ytmetas
from modules.translation.deepl_api import DeepLAPI
from modules.whisper.data_classes import *
from modules.utils.logger import get_logger
from run_multi_model import (
    run_model as multi_run_model,
    apply_diarization_to_segments,
    write_output as multi_write_output,
    write_summary as multi_write_summary,
    create_aligned_file as multi_create_aligned,
    fmt_duration,
    ALL_MODELS,
)


logger = get_logger()


class App:
    ARBITRAGE_PROMPT_PATH  = os.path.abspath(os.path.join(os.path.dirname(__file__), "configs", "arbitrage_prompt.txt"))
    ARBITRAGE_LEXICON_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "configs", "lexique_metier.txt"))
    ARBITRAGE_CONFIG_PATH  = os.path.abspath(os.path.join(os.path.dirname(__file__), "configs", "arbitrage_config.json"))
    REFINE_PROMPT_PATH     = os.path.abspath(os.path.join(os.path.dirname(__file__), "configs", "prompt_refine_diarization.txt"))

    def __init__(self, args):
        import json as _json
        self.args = args
        self._gradio_major = int(gr.__version__.split('.')[0])
        if self._gradio_major >= 6:
            self.app = gr.Blocks(delete_cache=(3600, 86400))
        else:
            self.app = gr.Blocks(css=CSS, theme=self.args.theme, delete_cache=(3600, 86400))
        self.whisper_infs = {}
        self.whisper_inf = self.get_whisper_inference(self.args.whisper_type)
        self.nllb_inf = NLLBInference(
            model_dir=self.args.nllb_model_dir,
            output_dir=os.path.join(self.args.output_dir, "translations")
        )
        self.deepl_api = DeepLAPI(
            output_dir=os.path.join(self.args.output_dir, "translations")
        )
        self.default_params = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH)

        self._arb_config = self._load_arbitrage_config()
        self._arb_prompt_text = self._load_arbitrage_prompt()
        self._arb_lexicon_text = self._load_arbitrage_lexicon()

        self._last_speaker_result = None
        self._last_speaker_audio = None

        logger.info(f"Use \"{self.args.whisper_type}\" implementation\n"
                    f"Device \"{self.whisper_inf.device}\" is detected")

    # ── Helper VRAM partagé ───────────────────────────────────────────────────

    @staticmethod
    def _list_gpu_processes():
        import subprocess
        procs = []
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        try:
                            procs.append((int(parts[0]), parts[1], int(parts[2])))
                        except (ValueError, IndexError):
                            pass
        except Exception:
            pass
        return procs

    @staticmethod
    def _get_free_vram_mb():
        import subprocess
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                free_mb = [int(x.strip()) for x in result.stdout.strip().splitlines() if x.strip()]
                return sum(free_mb), free_mb
        except Exception:
            pass
        return None, []

    @staticmethod
    def _ensure_vram(min_vram_mb=24_000, aggressive_mb=36_000, status_list=None, log_fn=None):
        """Vérifie la VRAM, tue les processus GPU si besoin. Retourne (ok:bool, msg:str)."""
        import subprocess, time, os as _os, signal as _sig
        total_free, free_per_gpu = App._get_free_vram_mb()
        if total_free is None:
            return True, "pas de données VRAM — ignoré"

        free_gb = total_free // 1024
        per_gpu = " + ".join(str(m // 1024) for m in free_per_gpu)
        msg = f"VRAM libre : {free_gb} Go ({per_gpu} Go par GPU)"

        if total_free >= min_vram_mb:
            return True, msg

        if total_free < aggressive_mb:
            detail = f"{msg}\nVRAM < {aggressive_mb // 1024} Go — tentative de libération..."
            if log_fn: log_fn(detail)

            # ── Stratégie de cleanup multi-niveaux ─────────────────────────
            killed_any = False

            # 1. Lister les processus GPU
            indentified = App._list_gpu_processes()
            if indentified:
                if log_fn: log_fn(f"  Processus GPU détectés : {len(indentified)}")
                for pid, name, mem in indentified:
                    if log_fn: log_fn(f"    • PID {pid} — {name} ({mem} Mo)")

            # 2. systemctl stop pour les services LLM connus
            llm_services = [
                "launch_qwen36_27b_vllm", "launch_arbitrage_q8",
                "launch_arbitrage2", "launch_llm",
            ]
            for svc in llm_services:
                try:
                    r = subprocess.run(
                        ["systemctl", "stop", f"{svc}.service"],
                        capture_output=True, text=True, timeout=20,
                    )
                    if r.returncode == 0:
                        if log_fn: log_fn(f"    systemctl stop {svc}.service → OK")
                        killed_any = True
                except Exception:
                    pass

            # 3. fuser -k sur les ports LLM connus (vLLM/llama.cpp)
            for port in [8080, 8081]:
                try:
                    subprocess.run(
                        ["fuser", "-k", f"{port}/tcp"],
                        capture_output=True, text=True, timeout=10,
                    )
                except Exception:
                    pass

            if killed_any:
                if log_fn: log_fn("  ⏳ Attente 5s après systemctl stop...")
                time.sleep(5)

            # 4. Kill spécifique des workers vLLM (survivent au SIGTERM)
            try:
                r = subprocess.run(
                    ["pgrep", "-f", "VLLM::(EngineCore|Worker_TP)"],
                    capture_output=True, text=True, timeout=5,
                )
                if r.returncode == 0 and r.stdout.strip():
                    pids = r.stdout.strip().split()
                    if log_fn: log_fn(f"  Workers vLLM survivants: {pids}")
                    for pid in pids:
                        try:
                            _os.kill(int(pid), _sig.SIGTERM)
                            if log_fn: log_fn(f"    • SIGTERM → PID {pid} (worker vLLM)")
                        except Exception:
                            pass
                    time.sleep(3)
            except Exception:
                pass

            # 5. Kill des processus GPU restants
            indentified = App._list_gpu_processes()
            killed_now = 0
            for pid, name, mem in indentified:
                kill = False
                nl = name.lower()
                if any(kw in nl for kw in ("llama", "vllm", "ik_llama", "whisper", "voxtral", "asr", "stt", "qwen", "cohere", "pyannote")):
                    kill = True
                    reason = f"consommateur VRAM ({name}, {mem} Mo)"
                elif mem > 50_000:
                    kill = True
                    reason = f"gros consommateur VRAM ({name}, {mem // 1024} Go)"
                if kill:
                    try:
                        _os.kill(pid, _sig.SIGTERM)
                        killed_now += 1
                        if log_fn: log_fn(f"    🔫 SIGTERM → PID {pid} ({reason})")
                    except Exception as e:
                        if log_fn: log_fn(f"    ⚠️ Impossible de tuer PID {pid} : {e}")

            if killed_now > 0:
                time.sleep(5)
                # SIGKILL pour les survivants
                indentified2 = App._list_gpu_processes()
                for pid, name, mem in indentified2:
                    if any(kw in name.lower() for kw in ("llama", "vllm", "ik_llama")):
                        try:
                            _os.kill(pid, _sig.SIGKILL)
                            if log_fn: log_fn(f"    ☠️ SIGKILL → PID {pid} ({name})")
                        except Exception:
                            pass
                time.sleep(5)

            # 6. Re-vérifier VRAM
            total_free, free_per_gpu = App._get_free_vram_mb()
            if total_free is not None:
                per_gpu = " + ".join(str(m // 1024) for m in free_per_gpu)
                if log_fn: log_fn(f"  VRAM après libération : {total_free // 1024} Go ({per_gpu} Go par GPU)")
                msg = f"VRAM après libération : {total_free // 1024} Go ({per_gpu} Go par GPU)"

        if total_free is not None and total_free < min_vram_mb:
            return False, f"{msg}\n❌ VRAM insuffisante ({total_free // 1024} Go libres, besoin de {min_vram_mb // 1024} Go min)"
        return True, msg

    def get_whisper_inference(self, whisper_type: str):
        whisper_type = whisper_type.strip().lower()
        if whisper_type not in self.whisper_infs:
            logger.info("[WEBUI] Instantiating new inference object for whisper_type='%s'", whisper_type)
            self.whisper_infs[whisper_type] = WhisperFactory.create_whisper_inference(
                whisper_type=whisper_type,
                whisper_model_dir=self.args.whisper_model_dir,
                faster_whisper_model_dir=self.args.faster_whisper_model_dir,
                insanely_fast_whisper_model_dir=self.args.insanely_fast_whisper_model_dir,
                voxtral_model_dir=self.args.voxtral_model_dir,
                qwen3_asr_model_dir=self.args.qwen3_asr_model_dir,
                cohere_asr_model_dir=self.args.cohere_asr_model_dir,
                uvr_model_dir=self.args.uvr_model_dir,
                output_dir=self.args.output_dir,
            )
        return self.whisper_infs[whisper_type]

    @staticmethod
    def infer_whisper_type_from_model(model_size: str) -> str:
        model_size = str(model_size or "").strip()
        if model_size == "voxtral-mini-3b":
            return WhisperImpl.VOXTRAL_MINI.value
        if model_size.startswith("qwen3-asr") or model_size.startswith("Qwen/Qwen3-ASR"):
            return WhisperImpl.QWEN3_ASR.value
        if model_size.startswith("cohere") or model_size.startswith("CohereLabs"):
            return WhisperImpl.COHERE_ASR.value
        if model_size == "voxtral-realtime-vllm":
            return WhisperImpl.VOXTRAL_REALTIME_VLLM.value
        if model_size in ("large-v3", "large-v3-turbo"):
            return WhisperImpl.FASTER_WHISPER.value
        logger.warning(
            "[WEBUI] FALLBACK: Unknown model_size='%s', routing to faster-whisper. "
            "If this is unexpected, check the model name.", model_size
        )
        return WhisperImpl.FASTER_WHISPER.value

    def get_whisper_inference_for_pipeline(self, pipeline_params):
        model_size = pipeline_params[0] if pipeline_params else self.default_params["whisper"]["model_size"]
        whisper_type = self.infer_whisper_type_from_model(model_size)
        whisper_inf = self.get_whisper_inference(whisper_type)
        logger.info("[WEBUI] Routing model=%s to whisper_type=%s", model_size, whisper_type)
        return whisper_inf

    def create_pipeline_inputs(self):
        whisper_params = self.default_params["whisper"]
        vad_params = self.default_params["vad"]
        diarization_params = self.default_params["diarization"]
        uvr_params = self.default_params["bgm_separation"]

        with gr.Row():
            # Expose all supported model families in a single UI. The request
            # is routed to the correct backend from the selected model name.
            available_models = WhisperFactory.get_combined_available_models()
            selected_model = whisper_params["model_size"]
            if available_models and selected_model not in available_models:
                selected_model = available_models[0]
                logger.info(
                    "Configured model is not available in multi-model UI. Using %s instead.",
                    selected_model,
                )
            dd_model = gr.Dropdown(choices=available_models, value=selected_model,
                                   label="Model", allow_custom_value=False)
            dd_lang = gr.Dropdown(choices=self.whisper_inf.available_langs + ["Automatic Detection"],
                                  value="Automatic Detection" if whisper_params["lang"] == "Automatic Detection"
                                  else whisper_params["lang"], label="Language")
            dd_file_format = gr.Dropdown(choices=["SRT", "WebVTT", "txt", "LRC"], value=whisper_params["file_format"], label="File Format")
        with gr.Row():
            cb_translate = gr.Checkbox(value=whisper_params["is_translate"], label="Translate to English?",
                                       interactive=True)
        with gr.Row():
            cb_timestamp = gr.Checkbox(value=whisper_params["add_timestamp"],
                                       label="Add a timestamp to the end of the filename",
                                       interactive=True)

        with gr.Accordion("Advanced Parameters", open=False):
            whisper_inputs = WhisperParams.to_gradio_inputs(defaults=whisper_params, only_advanced=True,
                                                            whisper_type=WhisperImpl.FASTER_WHISPER.value,
                                                            available_compute_types=self.whisper_inf.available_compute_types,
                                                            compute_type=self.whisper_inf.current_compute_type)

        with gr.Accordion("Background Music Remover Filter", open=False):
            uvr_inputs = BGMSeparationParams.to_gradio_input(defaults=uvr_params,
                                                             available_models=self.whisper_inf.music_separator.available_models,
                                                             available_devices=self.whisper_inf.music_separator.available_devices,
                                                             device=self.whisper_inf.music_separator.device)

        with gr.Accordion("Voice Detection Filter", open=False):
            vad_inputs = VadParams.to_gradio_inputs(defaults=vad_params)

        with gr.Accordion("Diarization", open=False):
            diarization_inputs = DiarizationParams.to_gradio_inputs(defaults=diarization_params,
                                                                    available_devices=self.whisper_inf.diarizer.available_device,
                                                                    device=self.whisper_inf.diarizer.device)

        pipeline_inputs = [dd_model, dd_lang, cb_translate] + whisper_inputs + vad_inputs + diarization_inputs + uvr_inputs

        return (
            pipeline_inputs,
            dd_file_format,
            cb_timestamp
        )

    def launch(self):
        args = self.args
        translation_params = self.default_params["translation"]
        deepl_params = translation_params["deepl"]
        nllb_params = translation_params["nllb"]
        uvr_params = self.default_params["bgm_separation"]

        with self.app:
            lang = gr.Radio(choices=["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"],
                            label="Language", interactive=True,
                            visible=False,  # Set it by development purpose.
                            )
            with gr.Row():
                with gr.Column():
                    gr.Markdown(MARKDOWN, elem_id="md_project")
            with gr.Tabs():
                with gr.TabItem("File"):  # tab1
                    with gr.Column():
                        input_file = gr.Files(type="filepath", label="Upload File here", file_types=MEDIA_EXTENSION)
                        tb_input_folder = gr.Textbox(label="Input Folder Path (Optional)",
                                                     info="Optional: Specify the folder path where the input files are located, if you prefer to use local files instead of uploading them."
                                                          " Leave this field empty if you do not wish to use a local path.",
                                                     visible=self.args.colab,
                                                     value="")
                        cb_include_subdirectory = gr.Checkbox(label="Include Subdirectory Files",
                                                              info="When using Input Folder Path above, whether to include all files in the subdirectory or not.",
                                                              visible=self.args.colab,
                                                              value=False)
                        cb_save_same_dir = gr.Checkbox(label="Save outputs at same directory",
                                                       info="When using Input Folder Path above, whether to save output in the same directory as inputs or not, in addition to the original"
                                                            " output directory.",
                                                       visible=self.args.colab,
                                                       value=True)
                    pipeline_params, dd_file_format, cb_timestamp = self.create_pipeline_inputs()

                    with gr.Row():
                        btn_run = gr.Button("GENERATE SUBTITLE FILE", variant="primary")
                    with gr.Row():
                        tb_indicator = gr.Textbox(label="Output", scale=5)
                        files_subtitles = gr.Textbox(label="Output file path", scale=3, interactive=False)
                        btn_download = gr.DownloadButton(label="📥 Download", visible=False, scale=1)

                    params = [input_file, tb_input_folder, cb_include_subdirectory, cb_save_same_dir,
                              dd_file_format, cb_timestamp]
                    params = params + pipeline_params
                    btn_run.click(fn=self.transcribe_file_for_web,
                                  inputs=params,
                                  outputs=[tb_indicator, files_subtitles, btn_download],
                                  queue=False)

                with gr.TabItem("Youtube"):  # tab2
                    with gr.Row():
                        tb_youtubelink = gr.Textbox(label="Youtube Link")
                    with gr.Row(equal_height=True):
                        with gr.Column():
                            img_thumbnail = gr.Image(label="Youtube Thumbnail")
                        with gr.Column():
                            tb_title = gr.Label(label="Youtube Title")
                            tb_description = gr.Textbox(label="Youtube Description", max_lines=15)

                    pipeline_params, dd_file_format, cb_timestamp = self.create_pipeline_inputs()

                    with gr.Row():
                        btn_run = gr.Button("GENERATE SUBTITLE FILE", variant="primary")
                    with gr.Row():
                        tb_indicator = gr.Textbox(label="Output", scale=5)
                        files_subtitles = gr.Textbox(label="Output file path", scale=3, interactive=False)
                        btn_download = gr.DownloadButton(label="📥 Download", visible=False, scale=1)

                    params = [tb_youtubelink, dd_file_format, cb_timestamp]

                    btn_run.click(fn=self.transcribe_youtube_for_web,
                                  inputs=params + pipeline_params,
                                  outputs=[tb_indicator, files_subtitles, btn_download],
                                  queue=False)
                    tb_youtubelink.change(get_ytmetas, inputs=[tb_youtubelink],
                                          outputs=[img_thumbnail, tb_title, tb_description])

                with gr.TabItem("Mic"):  # tab3
                    with gr.Row():
                        _mic_kwargs = dict(label="Record with Mic", type="filepath", interactive=True)
                        if self._gradio_major < 6:
                            _mic_kwargs["show_download_button"] = True
                        mic_input = gr.Microphone(**_mic_kwargs)

                    pipeline_params, dd_file_format, cb_timestamp = self.create_pipeline_inputs()

                    with gr.Row():
                        btn_run = gr.Button("GENERATE SUBTITLE FILE", variant="primary")
                    with gr.Row():
                        tb_indicator = gr.Textbox(label="Output", scale=5)
                        files_subtitles = gr.Textbox(label="Output file path", scale=3, interactive=False)
                        btn_download = gr.DownloadButton(label="📥 Download", visible=False, scale=1)

                    params = [mic_input, dd_file_format, cb_timestamp]

                    btn_run.click(fn=self.transcribe_mic_for_web,
                                  inputs=params + pipeline_params,
                                  outputs=[tb_indicator, files_subtitles, btn_download],
                                  queue=False)

                with gr.TabItem("T2T Translation"):  # tab 4
                    with gr.Row():
                        file_subs = gr.Files(type="filepath", label="Upload Subtitle Files to translate here")

                    with gr.TabItem("DeepL API"):  # sub tab1
                        with gr.Row():
                            tb_api_key = gr.Textbox(label="Your Auth Key (API KEY)",
                                                    value=deepl_params["api_key"])
                        with gr.Row():
                            dd_source_lang = gr.Dropdown(label="Source Language",
                                                         value="Automatic Detection" if deepl_params["source_lang"] == "Automatic Detection"
                                                         else deepl_params["source_lang"],
                                                         choices=list(self.deepl_api.available_source_langs.keys()))
                            dd_target_lang = gr.Dropdown(label="Target Language",
                                                         value=deepl_params["target_lang"],
                                                         choices=list(self.deepl_api.available_target_langs.keys()))
                        with gr.Row():
                            cb_is_pro = gr.Checkbox(label="Pro User?", value=deepl_params["is_pro"])
                        with gr.Row():
                            cb_timestamp = gr.Checkbox(value=translation_params["add_timestamp"],
                                                       label="Add a timestamp to the end of the filename",
                                                       interactive=True)
                        with gr.Row():
                            btn_run = gr.Button("TRANSLATE SUBTITLE FILE", variant="primary")
                        with gr.Row():
                            tb_indicator = gr.Textbox(label="Output", scale=5)
                            files_subtitles = gr.Textbox(label="Output file path", scale=3, interactive=False)
                            btn_download = gr.DownloadButton(label="📥 Download", visible=False, scale=1)

                        btn_run.click(fn=self.translate_deepl_for_web,
                                      inputs=[tb_api_key, file_subs, dd_source_lang, dd_target_lang,
                                              cb_is_pro, cb_timestamp],
                                      outputs=[tb_indicator, files_subtitles, btn_download],
                                      queue=True)

                    with gr.TabItem("NLLB"):  # sub tab2
                        with gr.Row():
                            dd_model_size = gr.Dropdown(label="Model", value=nllb_params["model_size"],
                                                        choices=self.nllb_inf.available_models)
                            dd_source_lang = gr.Dropdown(label="Source Language",
                                                         value=nllb_params["source_lang"],
                                                         choices=self.nllb_inf.available_source_langs)
                            dd_target_lang = gr.Dropdown(label="Target Language",
                                                         value=nllb_params["target_lang"],
                                                         choices=self.nllb_inf.available_target_langs)
                        with gr.Row():
                            nb_max_length = gr.Number(label="Max Length Per Line", value=nllb_params["max_length"],
                                                      precision=0)
                        with gr.Row():
                            cb_timestamp = gr.Checkbox(value=translation_params["add_timestamp"],
                                                       label="Add a timestamp to the end of the filename",
                                                       interactive=True)
                        with gr.Row():
                            btn_run = gr.Button("TRANSLATE SUBTITLE FILE", variant="primary")
                        with gr.Row():
                            tb_indicator = gr.Textbox(label="Output", scale=5)
                            files_subtitles = gr.Textbox(label="Output file path", scale=3, interactive=False)
                            btn_download = gr.DownloadButton(label="📥 Download", visible=False, scale=1)
                        with gr.Column():
                            md_vram_table = gr.HTML(NLLB_VRAM_TABLE, elem_id="md_nllb_vram_table")

                        btn_run.click(fn=self.translate_nllb_for_web,
                                      inputs=[file_subs, dd_model_size, dd_source_lang, dd_target_lang,
                                              nb_max_length, cb_timestamp],
                                      outputs=[tb_indicator, files_subtitles, btn_download],
                                      queue=True)

                with gr.TabItem("Multi-Model"):  # tab5
                    with gr.Column():
                        mm_input_file = gr.File(type="filepath", label="Upload Audio/Video File",
                                                file_types=MEDIA_EXTENSION)
                    with gr.Row():
                        mm_models = gr.CheckboxGroup(
                            choices=ALL_MODELS,
                            value=ALL_MODELS,
                            label="Models to run (in order)",
                        )
                    with gr.Row():
                        mm_language = gr.Dropdown(
                            choices=["french", "english", "german", "spanish", "italian",
                                     "portuguese", "dutch", "polish", "russian", "chinese",
                                     "japanese", "korean", "arabic", "automatic detection"],
                            value="french",
                            label="Language",
                        )
                        mm_diarize = gr.Checkbox(value=True, label="Speaker diarization (once for all models)")
                        mm_align = gr.Checkbox(value=True, label="Aligner sur fenêtres 30s (recommandé pour l'arbitrage LLM)")
                    with gr.Row():
                        btn_mm_run = gr.Button("RUN MULTI-MODEL TRANSCRIPTION", variant="primary")
                    with gr.Row():
                        mm_indicator = gr.Textbox(label="Status", scale=5, lines=8)
                        mm_paths = gr.Textbox(label="Output file paths", scale=3, interactive=False, lines=8)
                        mm_download = gr.DownloadButton(label="📥 Télécharger tout (ZIP)", visible=False, scale=1)

                    btn_mm_run.click(
                        fn=self.run_multi_model_for_web,
                        inputs=[mm_input_file, mm_models, mm_language, mm_diarize, mm_align],
                        outputs=[mm_indicator, mm_paths, mm_download],
                        queue=False,
                    )

                with gr.TabItem("Arbitrage LLM"):  # tab6
                    gr.Markdown("### Arbitrage LLM — Reconstruction SRT depuis les 4 transcriptions")
                    with gr.Row():
                        arb_zip = gr.File(
                            type="filepath",
                            label="ZIP multi-modèle (depuis outputs/)",
                            file_types=[".zip"],
                        )
                    with gr.Row():
                        with gr.Column():
                            arb_prompt = gr.Textbox(
                                label="Prompt système (configs/arbitrage_prompt.txt)",
                                lines=20,
                                value=self._arb_prompt_text,
                            )
                            with gr.Row():
                                btn_load_prompt  = gr.Button("📂 Charger", scale=1)
                                btn_save_prompt  = gr.Button("💾 Sauvegarder", scale=1)
                        with gr.Column():
                            arb_lexicon = gr.Textbox(
                                label="Lexique métier (configs/lexique_metier.txt)",
                                lines=14,
                                value=self._arb_lexicon_text,
                            )
                            with gr.Row():
                                btn_load_lexicon = gr.Button("📂 Charger", scale=1)
                                btn_save_lexicon = gr.Button("💾 Sauvegarder", scale=1)
                    with gr.Row():
                        arb_script = gr.Textbox(
                            label="Script de lancement llama-server (.sh)",
                            value=self._arb_config.get("launch_script", ""),
                            scale=4,
                        )
                        arb_port = gr.Number(
                            label="Port API",
                            value=self._arb_config.get("api_port", 8080),
                            precision=0,
                            scale=1,
                        )
                        arb_model = gr.Textbox(
                            label="Modèle OpenCode (provider/model-id)",
                            value=self._arb_config.get("model_id", ""),
                            scale=2,
                        )
                    with gr.Row():
                        btn_load_arb_cfg = gr.Button("📂 Charger config", scale=1)
                        btn_save_arb_cfg = gr.Button("💾 Sauvegarder config", scale=1)
                    with gr.Row():
                        btn_arb_run = gr.Button("🚀 LANCER L'ARBITRAGE", variant="primary")
                    with gr.Row():
                        arb_indicator = gr.Textbox(label="Statut", lines=8, scale=5, interactive=False)
                        arb_download = gr.DownloadButton(label="📥 Télécharger SRT final", visible=False, scale=1)

                    btn_load_prompt.click(
                        fn=self._load_arbitrage_prompt,
                        inputs=[],
                        outputs=[arb_prompt],
                        queue=False,
                    )
                    btn_save_prompt.click(
                        fn=self.save_arbitrage_prompt,
                        inputs=[arb_prompt],
                        outputs=[arb_indicator],
                        queue=False,
                    )
                    btn_load_lexicon.click(
                        fn=self._load_arbitrage_lexicon,
                        inputs=[],
                        outputs=[arb_lexicon],
                        queue=False,
                    )
                    btn_save_lexicon.click(
                        fn=self.save_arbitrage_lexicon,
                        inputs=[arb_lexicon],
                        outputs=[arb_indicator],
                        queue=False,
                    )
                    btn_load_arb_cfg.click(
                        fn=self._load_arbitrage_config_for_ui,
                        inputs=[],
                        outputs=[arb_script, arb_model, arb_port],
                        queue=False,
                    )
                    btn_save_arb_cfg.click(
                        fn=self.save_arbitrage_config,
                        inputs=[arb_script, arb_model, arb_port],
                        outputs=[arb_script, arb_model, arb_port, arb_indicator],
                        queue=False,
                    )
                    btn_arb_run.click(
                        fn=self.run_arbitration_for_web,
                        inputs=[arb_zip, arb_lexicon, arb_prompt, arb_script, arb_port, arb_model],
                        outputs=[arb_indicator, arb_download],
                        queue=False,
                    )

                    # ── Raffinement Diarization ─────────────────────────────
                    gr.Markdown("---\n### 🎙️ Raffinement Diarization (split speakers)")
                    gr.Markdown("Prend un SRT arbitré + un fichier speaker-turns pyannote, "
                                "et produit un SRT avec des timestamps précis et des attributions "
                                "de locuteur corrigées.")
                    with gr.Row():
                        rd_srt = gr.File(
                            type="filepath", label="Fichier SRT arbitré (.srt)",
                            file_types=[".srt"], scale=2,
                        )
                        rd_turns = gr.File(
                            type="filepath", label="Fichier speaker turns (.txt)",
                            file_types=[".txt"], scale=2,
                        )
                    with gr.Row():
                        rd_prompt = gr.Textbox(
                            label="Prompt raffinement (configs/prompt_refine_diarization.txt)",
                            lines=10,
                            value=self._load_refine_prompt,
                        )
                        with gr.Column(scale=1):
                            btn_load_rd_prompt = gr.Button("📂 Charger")
                            btn_save_rd_prompt = gr.Button("💾 Sauvegarder")
                    with gr.Row():
                        btn_rd_run = gr.Button("🔊 RAFFINER LA DIARIZATION", variant="primary")
                    with gr.Row():
                        rd_indicator = gr.Textbox(label="Statut", lines=6, scale=5, interactive=False)
                        rd_download = gr.DownloadButton(label="📥 Télécharger SRT raffiné", visible=False, scale=1)

                    btn_load_rd_prompt.click(
                        fn=self._load_refine_prompt,
                        inputs=[], outputs=[rd_prompt], queue=False,
                    )
                    btn_save_rd_prompt.click(
                        fn=self.save_refine_prompt,
                        inputs=[rd_prompt], outputs=[rd_indicator], queue=False,
                    )
                    btn_rd_run.click(
                        fn=self.run_diarization_refine_for_web,
                        inputs=[rd_srt, rd_turns, rd_prompt, arb_script, arb_port, arb_model],
                        outputs=[rd_indicator, rd_download],
                        queue=False,
                    )

                with gr.TabItem("Identification des locuteurs"):
                    sp_audio = gr.File(type="filepath", label="Fichier audio/vid\u00e9o",
                                       file_types=MEDIA_EXTENSION)
                    with gr.Row():
                        sp_model = gr.Dropdown(
                            label="Mod\u00e8le de diarization",
                            choices=[
                                "pyannote/speaker-diarization-community-1",
                                "pyannote/speaker-diarization-3.1",
                            ],
                            value="pyannote/speaker-diarization-community-1",
                            info="community-1 : meilleure pr\u00e9cision | 3.1 : version legacy",
                        )
                        sp_device = gr.Dropdown(
                            label="Device",
                            choices=["cpu", "cuda", "xpu"] if torch.cuda.is_available() else ["cpu"],
                            value="cuda" if torch.cuda.is_available() else "cpu",
                        )
                        sp_hf_token = gr.Textbox(
                            label="HuggingFace Token",
                            value=self.default_params.get("diarization", {}).get("hf_token", ""),
                            info="N\u00e9cessaire uniquement au premier t\u00e9l\u00e9chargement du mod\u00e8le",
                            scale=1,
                        )
                    with gr.Row():
                        sp_min_speakers = gr.Number(label="Min locuteurs", value=1, precision=0, minimum=1, maximum=20)
                        sp_max_speakers = gr.Number(label="Max locuteurs", value=5, precision=0, minimum=1, maximum=20)
                    with gr.Row():
                        btn_sp_detect = gr.Button("\U0001f50d Analyser les locuteurs", variant="primary")

                    sp_indicator = gr.Textbox(label="Statut", lines=3, interactive=False)

                    with gr.Row():
                        sp_num_detected = gr.Textbox(label="Locuteurs d\u00e9tect\u00e9s", interactive=False, scale=1)
                        sp_audio_duration = gr.Textbox(label="Dur\u00e9e audio", interactive=False, scale=1)
                        sp_processing_time = gr.Textbox(label="Temps de traitement", interactive=False, scale=1)

                    sp_speaking_time = gr.Textbox(label="Temps de parole", lines=6, interactive=False)
                    sp_turns_text = gr.Textbox(label="Tours de parole (derniers 50)", lines=12, interactive=False)

                    gr.Markdown("### Confirmer le nombre de locuteurs")
                    gr.Markdown("Si le nombre d\u00e9tect\u00e9 ne semble pas correct, indiquez le bon nombre et relancez l'analyse.")
                    with gr.Row():
                        sp_confirmed_count = gr.Number(label="Nombre de locuteurs (confirmer/corriger)", precision=0, value=None)
                        btn_sp_redetect = gr.Button("\U0001f504 Relancer avec ce nombre", variant="secondary")

                    sp_redetect_indicator = gr.Textbox(label="Statut relance", lines=2, interactive=False)

                    MAX_SPEAKERS_UI = 10

                    gr.Markdown("### \U0001f3a4 Extraits audio par locuteur")
                    gr.Markdown("\u00c9coutez les extraits pour identifier chaque voix, puis remplissez les champs ci-dessous.")
                    sp_clip_audios = []
                    sp_clip_labels = []
                    for i in range(MAX_SPEAKERS_UI):
                        sp_clip_labels.append(gr.Textbox(label=f"Locuteur {i}", interactive=False, visible=False))
                        with gr.Row():
                            sp_clip_audios.append(gr.Audio(label=f"  Extrait 1", type="filepath", visible=False))
                            sp_clip_audios.append(gr.Audio(label=f"  Extrait 2", type="filepath", visible=False))
                            sp_clip_audios.append(gr.Audio(label=f"  Extrait 3", type="filepath", visible=False))

                    gr.Markdown("### \u270d\ufe0f Nommer les locuteurs")
                    sp_name_fields = []
                    sp_function_fields = []
                    sp_role_fields = []
                    sp_notes_fields = []
                    for i in range(MAX_SPEAKERS_UI):
                        with gr.Row():
                            sp_name_fields.append(gr.Textbox(label=f"Locuteur {i} \u2014 Nom", scale=3, visible=False))
                            sp_function_fields.append(gr.Textbox(label="Fonction", scale=3, visible=False))
                            sp_role_fields.append(gr.Textbox(label="R\u00f4le r\u00e9union", scale=3, visible=False))
                            sp_notes_fields.append(gr.Textbox(label="Notes", scale=3, visible=False))

                    with gr.Row():
                        btn_sp_export = gr.Button("\U0001f4be Exporter YAML", variant="primary")
                        btn_sp_import = gr.Button("\U0001f4c2 Importer YAML")
                    sp_yaml_path = gr.Textbox(label="Chemin du fichier YAML", interactive=False)
                    sp_download = gr.DownloadButton(label="\U0001f4e5 T\u00e9l\u00e9charger YAML", visible=False)

                    btn_sp_detect.click(
                        fn=self.run_speaker_detect_for_web,
                        inputs=[sp_audio, sp_model, sp_device, sp_hf_token,
                                sp_min_speakers, sp_max_speakers],
                        outputs=[sp_indicator, sp_num_detected, sp_audio_duration,
                                 sp_processing_time, sp_speaking_time, sp_turns_text]
                        + sp_clip_labels
                        + sp_clip_audios
                        + sp_name_fields
                        + sp_function_fields
                        + sp_role_fields
                        + sp_notes_fields,
                        queue=False,
                    )
                    btn_sp_redetect.click(
                        fn=self.run_speaker_redetect_for_web,
                        inputs=[sp_audio, sp_model, sp_device, sp_hf_token, sp_confirmed_count],
                        outputs=[sp_redetect_indicator, sp_num_detected, sp_audio_duration,
                                 sp_processing_time, sp_speaking_time, sp_turns_text]
                        + sp_clip_labels
                        + sp_clip_audios
                        + sp_name_fields
                        + sp_function_fields
                        + sp_role_fields
                        + sp_notes_fields,
                        queue=False,
                    )
                    btn_sp_export.click(
                        fn=self.run_speaker_export_for_web,
                        inputs=[sp_audio]
                        + [f for f in sp_name_fields]
                        + [f for f in sp_function_fields]
                        + [f for f in sp_role_fields]
                        + [f for f in sp_notes_fields],
                        outputs=[sp_yaml_path, sp_download],
                        queue=False,
                    )

                with gr.TabItem("🎬 Relecture SRT"):
                    _editor_url = os.environ.get("SRT_EDITOR_URL", "http://localhost:7861")
                    gr.HTML(f"""
                    <div style="text-align:center;padding:40px 20px;background:#0f1117;border-radius:12px;margin:10px 0">
                      <div style="font-size:48px;margin-bottom:12px">🎬</div>
                      <h2>Relecture & Correction SRT</h2>
                      <p style="color:#8b949e;margin:8px 0 20px 0;line-height:1.6">
                        Glissez un fichier audio + SRT pour lire, synchroniser et corriger.<br>
                        <b>Raccourcis :</b> Espace = Play/Pause | ← → = ±5s | ↑ ↓ = Segments | E = Éditer
                      </p>
                      <a href="{_editor_url}" target="_blank"
                         style="display:inline-block;padding:14px 32px;background:#3fb950;color:#000;
                                border-radius:8px;text-decoration:none;font-weight:700;font-size:16px">
                        🚀 Ouvrir l'ÉDITEUR SRT
                      </a>
                      <p style="color:#8b949e;font-size:11px;margin-top:10px">{_editor_url}</p>
                    </div>
                    """)

                with gr.TabItem("BGM Separation"):
                    files_audio = gr.Files(type="filepath", label="Upload Audio Files to separate background music")
                    dd_uvr_device = gr.Dropdown(label="Device", value=self.whisper_inf.music_separator.device,
                                                choices=self.whisper_inf.music_separator.available_devices)
                    dd_uvr_model_size = gr.Dropdown(label="Model", value=uvr_params["uvr_model_size"],
                                                    choices=self.whisper_inf.music_separator.available_models)
                    nb_uvr_segment_size = gr.Number(label="Segment Size", value=uvr_params["segment_size"],
                                                    precision=0)
                    cb_uvr_save_file = gr.Checkbox(label="Save separated files to output",
                                                   value=True, visible=False)
                    btn_run = gr.Button("SEPARATE BACKGROUND MUSIC", variant="primary")
                    with gr.Column():
                        with gr.Row():
                            ad_instrumental = gr.Audio(label="Instrumental", scale=8)
                            btn_open_instrumental_folder = gr.Button('📂', scale=1)
                        with gr.Row():
                            ad_vocals = gr.Audio(label="Vocals", scale=8)
                            btn_open_vocals_folder = gr.Button('📂', scale=1)

                    btn_run.click(fn=self.whisper_inf.music_separator.separate_files,
                                  inputs=[files_audio, dd_uvr_model_size, dd_uvr_device, nb_uvr_segment_size,
                                          cb_uvr_save_file],
                                  outputs=[ad_instrumental, ad_vocals])
                    btn_open_instrumental_folder.click(inputs=None,
                                                       outputs=None,
                                                       fn=lambda: self.open_folder(os.path.join(
                                                           self.args.output_dir, "UVR", "instrumental"
                                                       )))
                    btn_open_vocals_folder.click(inputs=None,
                                                 outputs=None,
                                                 fn=lambda: self.open_folder(os.path.join(
                                                     self.args.output_dir, "UVR", "vocals"
                                                 )))

        # Launch the app with optional gradio settings
        allowed_paths = eval(args.allowed_paths) if args.allowed_paths else []
        for path in [
            os.path.abspath(args.output_dir),
            os.path.abspath("outputs"),
            "/tmp/gradio",
            "/tmp/voxtral-webui",
        ]:
            if path not in allowed_paths:
                allowed_paths.append(path)
        logger.info("Gradio allowed paths: %s", allowed_paths)
        self.app.queue()

        _launch_kwargs = dict(
            share=args.share,
            server_name=args.server_name,
            server_port=args.server_port,
            auth=(args.username, args.password) if args.username and args.password else None,
            root_path=args.root_path,
            inbrowser=args.inbrowser,
            ssl_verify=args.ssl_verify,
            allowed_paths=allowed_paths,
            show_error=True,
        )
        if self._gradio_major >= 6:
            _launch_kwargs["css"] = CSS
            _launch_kwargs["theme"] = args.theme
        self.app.launch(**_launch_kwargs)

    @staticmethod
    def open_folder(folder_path: str):
        if os.path.exists(folder_path):
            # Cross-platform folder opening
            import platform
            system = platform.system()
            if system == "Windows":
                os.system(f'start "" "{folder_path}"')
            elif system == "Darwin":  # macOS
                os.system(f'open "{folder_path}"')
            else:  # Linux
                os.system(f'xdg-open "{folder_path}"')
        else:
            os.makedirs(folder_path, exist_ok=True)
            logger.info(f"The directory path {folder_path} has newly created.")

    def transcribe_file_for_web(self,
                                files=None,
                                input_folder_path=None,
                                include_subdirectory=None,
                                save_same_dir=None,
                                file_format="SRT",
                                add_timestamp=True,
                                progress=gr.Progress(),
                                *pipeline_params):
        whisper_inf = self.get_whisper_inference_for_pipeline(pipeline_params)
        result_text, output_files = whisper_inf.transcribe_file(
            files,
            input_folder_path,
            include_subdirectory,
            save_same_dir,
            file_format,
            add_timestamp,
            progress,
            *pipeline_params,
        )
        if isinstance(output_files, (list, tuple)):
            output_path_text = "\n".join(str(path) for path in output_files)
            first_file = str(output_files[0]) if output_files else None
        else:
            output_path_text = str(output_files or "")
            first_file = output_path_text or None
        logger.info("[WEBUI] File tab output paths returned as text: %s", output_path_text)
        return result_text, output_path_text, gr.update(value=first_file, visible=bool(first_file))

    def transcribe_youtube_for_web(self, youtubelink, file_format="SRT", add_timestamp=True,
                                   progress=gr.Progress(), *pipeline_params):
        whisper_inf = self.get_whisper_inference_for_pipeline(pipeline_params)
        result = whisper_inf.transcribe_youtube(
            youtubelink, file_format, add_timestamp, progress, *pipeline_params
        )
        result_text, output_files = result if isinstance(result, (list, tuple)) else (result, "")
        if isinstance(output_files, (list, tuple)):
            output_path_text = "\n".join(str(p) for p in output_files)
            first_file = str(output_files[0]) if output_files else None
        else:
            output_path_text = str(output_files or "")
            first_file = output_path_text or None
        logger.info("[WEBUI] Youtube tab output paths returned as text: %s", output_path_text)
        return result_text, output_path_text, gr.update(value=first_file, visible=bool(first_file))

    def transcribe_mic_for_web(self, mic_audio, file_format="SRT", add_timestamp=True,
                               progress=gr.Progress(), *pipeline_params):
        whisper_inf = self.get_whisper_inference_for_pipeline(pipeline_params)
        result = whisper_inf.transcribe_mic(
            mic_audio, file_format, add_timestamp, progress, *pipeline_params
        )
        result_text, output_files = result if isinstance(result, (list, tuple)) else (result, "")
        if isinstance(output_files, (list, tuple)):
            output_path_text = "\n".join(str(p) for p in output_files)
            first_file = str(output_files[0]) if output_files else None
        else:
            output_path_text = str(output_files or "")
            first_file = output_path_text or None
        logger.info("[WEBUI] Mic tab output paths returned as text: %s", output_path_text)
        return result_text, output_path_text, gr.update(value=first_file, visible=bool(first_file))

    def translate_deepl_for_web(self, api_key, file_subs, source_lang, target_lang, is_pro, add_timestamp,
                                progress=gr.Progress()):
        result = self.deepl_api.translate_deepl(
            api_key, file_subs, source_lang, target_lang, is_pro, add_timestamp, progress
        )
        result_text, output_files = result if isinstance(result, (list, tuple)) else (result, "")
        if isinstance(output_files, (list, tuple)):
            output_path_text = "\n".join(str(p) for p in output_files)
            first_file = str(output_files[0]) if output_files else None
        else:
            output_path_text = str(output_files or "")
            first_file = output_path_text or None
        logger.info("[WEBUI] DeepL tab output paths returned as text: %s", output_path_text)
        return result_text, output_path_text, gr.update(value=first_file, visible=bool(first_file))

    def translate_nllb_for_web(self, file_subs, model_size, source_lang, target_lang, max_length, add_timestamp,
                               progress=gr.Progress()):
        result = self.nllb_inf.translate_file(
            file_subs, model_size, source_lang, target_lang, max_length, add_timestamp, progress
        )
        result_text, output_files = result if isinstance(result, (list, tuple)) else (result, "")
        if isinstance(output_files, (list, tuple)):
            output_path_text = "\n".join(str(p) for p in output_files)
            first_file = str(output_files[0]) if output_files else None
        else:
            output_path_text = str(output_files or "")
            first_file = output_path_text or None
        logger.info("[WEBUI] NLLB tab output paths returned as text: %s", output_path_text)
        return result_text, output_path_text, gr.update(value=first_file, visible=bool(first_file))

    # ── Onglet Identification des locuteurs ──────────────────────────────

    MAX_SPEAKERS_UI = 10

    def _speaker_detect(self, audio_path, model_name, device, hf_token, min_speakers, max_speakers):
        from modules.diarize.speaker_identifier import detect_speakers, extract_speaker_clips, format_hms, DiarizationResult
        import gc, torch

        logger.info("[SPEAKER-ID-WEBUI] Starting detection: audio=%s, model=%s, device=%s, min=%s, max=%s",
                     audio_path, model_name, device, min_speakers, max_speakers)

        try:
            result = detect_speakers(
                audio_path=audio_path,
                model_name=model_name,
                cache_dir=self.args.diarization_model_dir,
                use_auth_token=hf_token if hf_token else None,
                device=device,
                min_speakers=int(min_speakers) if min_speakers else None,
                max_speakers=int(max_speakers) if max_speakers else None,
            )
        except Exception as e:
            logger.error("[SPEAKER-ID-WEBUI] Detection failed: %s", e, exc_info=True)
            raise
        finally:
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        self._last_speaker_result = result
        self._last_speaker_audio = audio_path
        self._last_speaker_model = model_name
        logger.info("[SPEAKER-ID-WEBUI] Detection done: %d speakers, %.1fs",
                     result.num_speakers_detected, result.elapsed_time)
        return result

    def _build_speaker_outputs(self, result):
        """Build the list of Gradio output values for clips, labels, and fields.

        Returns a flat list matching the output order:
        label × N, clip_audio × (3*N), name × N, function × N, role × N, notes × N
        Each component gets gr.update(value=..., visible=True/False).
        """
        from modules.diarize.speaker_identifier import extract_speaker_clips

        audio_path = self._last_speaker_audio
        n = self.MAX_SPEAKERS_UI

        label_updates = [gr.update(visible=False)] * n
        clip_updates = [gr.update(visible=False, value=None)] * (3 * n)
        name_updates = [gr.update(visible=False, value="")] * n
        function_updates = [gr.update(visible=False, value="")] * n
        role_updates = [gr.update(visible=False, value="")] * n
        notes_updates = [gr.update(visible=False, value="")] * n

        if result is None or not result.speakers:
            return (label_updates + clip_updates + name_updates
                    + function_updates + role_updates + notes_updates)

        audio_path = audio_path if isinstance(audio_path, str) else ""

        logger.info("[SPEAKER-ID-WEBUI] Extracting clips for %d speakers...", len(result.speakers))
        try:
            clips = extract_speaker_clips(
                audio_path=audio_path,
                result=result,
                num_clips=3,
                min_clip_duration=3.0,
                max_clip_duration=10.0,
            )
        except Exception as e:
            logger.error("[SPEAKER-ID-WEBUI] Clip extraction failed: %s", e, exc_info=True)
            clips = {}

        logger.info("[SPEAKER-ID-WEBUI] Clips extracted for %d speakers", len(clips))

        sorted_speakers = sorted(result.speakers.keys())
        for i, sid in enumerate(sorted_speakers[:n]):
            sinfo = result.speakers[sid]
            total_speech = sum(s.total_speaking_time for s in result.speakers.values())
            pct = (sinfo.total_speaking_time / total_speech * 100) if total_speech > 0 else 0
            minutes = int(sinfo.total_speaking_time) // 60
            secs = int(sinfo.total_speaking_time) % 60

            label_updates[i] = gr.update(
                visible=True,
                value=f"{sid}  ({minutes:02d}m{secs:02d}s \u2014 {pct:.1f}%)"
            )
            name_updates[i] = gr.update(visible=True, value="")
            function_updates[i] = gr.update(visible=True, value="")
            role_updates[i] = gr.update(visible=True, value="")
            notes_updates[i] = gr.update(visible=True, value="")

            speaker_clips = clips.get(sid, [])
            for clip_idx, clip_path in enumerate(speaker_clips[:3]):
                clip_updates[i * 3 + clip_idx] = gr.update(visible=True, value=clip_path)
            for clip_idx in range(len(speaker_clips), 3):
                clip_updates[i * 3 + clip_idx] = gr.update(visible=True, value=None)

        return (label_updates + clip_updates + name_updates
                + function_updates + role_updates + notes_updates)

    def run_speaker_detect_for_web(self, audio_file, model_name, device, hf_token, min_speakers, max_speakers):
        from modules.diarize.speaker_identifier import format_hms

        empty_outputs = [""] * 6
        empty_speaker = self._build_speaker_outputs(None)

        if not audio_file:
            return tuple(empty_outputs + empty_speaker)

        audio_path = audio_file if isinstance(audio_file, str) else audio_file

        try:
            result = self._speaker_detect(audio_path, model_name, device, hf_token, min_speakers, max_speakers)
        except Exception as e:
            logger.error("[SPEAKER-ID-WEBUI] run_speaker_detect_for_web failed: %s", e, exc_info=True)
            return tuple([f"Erreur lors de l'analyse : {e}", "", "", "", "", ""] + empty_speaker)

        num_detected = str(result.num_speakers_detected)
        duration_str = format_hms(result.audio_duration)
        time_str = f"{result.elapsed_time:.1f}s"

        speaking_time = result.get_speaking_time_text()

        total_turns = len(result.turns)
        last_n = min(50, total_turns)
        recent_turns = result.turns[-last_n:] if total_turns > 50 else result.turns
        turns_lines = [f"(Affichage des {last_n} derniers tours sur {total_turns} au total)\n"]
        for t in recent_turns:
            turns_lines.append(str(t))
        turns_text = "\n".join(turns_lines)

        indicator = f"D\u00e9tection termin\u00e9e : {num_detected} locuteur(s) d\u00e9tect\u00e9(s) sur {duration_str} d'audio"

        speaker_outputs = self._build_speaker_outputs(result)

        return tuple([indicator, num_detected, duration_str, time_str, speaking_time, turns_text]
                      + list(speaker_outputs))

    def run_speaker_redetect_for_web(self, audio_file, model_name, device, hf_token, confirmed_count):
        from modules.diarize.speaker_identifier import format_hms

        empty_speaker = self._build_speaker_outputs(None)

        if not audio_file:
            return tuple(["Aucun fichier audio.", "", "", "", "", ""] + list(empty_speaker))

        if confirmed_count is None or confirmed_count <= 0:
            return tuple(["Entrez un nombre de locuteurs valide.", "", "", "", "", ""] + list(empty_speaker))

        audio_path = audio_file if isinstance(audio_file, str) else audio_file

        try:
            result = self._speaker_detect(
                audio_path, model_name, device, hf_token,
                min_speakers=int(confirmed_count),
                max_speakers=int(confirmed_count),
            )
        except Exception as e:
            logger.error("[SPEAKER-ID-WEBUI] redetect failed: %s", e, exc_info=True)
            return tuple([f"Erreur lors de la relance : {e}", "", "", "", "", ""] + list(empty_speaker))

        num_detected = str(result.num_speakers_detected)
        duration_str = format_hms(result.audio_duration)
        time_str = f"{result.elapsed_time:.1f}s"
        speaking_time = result.get_speaking_time_text()

        total_turns = len(result.turns)
        last_n = min(50, total_turns)
        recent_turns = result.turns[-last_n:]
        turns_lines = [f"(Affichage des {last_n} derniers tours sur {total_turns} au total)\n"]
        for t in recent_turns:
            turns_lines.append(str(t))
        turns_text = "\n".join(turns_lines)

        indicator = f"Relance termin\u00e9e : {num_detected} locuteur(s) d\u00e9tect\u00e9(s)"

        speaker_outputs = self._build_speaker_outputs(result)

        return tuple([indicator, num_detected, duration_str, time_str, speaking_time, turns_text]
                      + list(speaker_outputs))

    def run_speaker_export_for_web(self, audio_file, *speaker_fields):
        import os
        from modules.diarize.speaker_identifier import export_speakers_yaml

        if not hasattr(self, '_last_speaker_result') or self._last_speaker_result is None:
            return ("Aucune analyse de locuteurs disponible. Lancez d'abord l'analyse.", gr.update(visible=False))

        result = self._last_speaker_result
        audio_path = audio_file if isinstance(audio_file, str) else audio_file
        model_name = getattr(self, '_last_speaker_model', 'unknown')

        n = self.MAX_SPEAKERS_UI
        speakers_info = {}
        sorted_speakers = sorted(result.speakers.keys())

        for i, sid in enumerate(sorted_speakers[:n]):
            base_idx = i * 4
            name_val = speaker_fields[base_idx] if len(speaker_fields) > base_idx else ""
            func_val = speaker_fields[base_idx + 1] if len(speaker_fields) > base_idx + 1 else ""
            role_val = speaker_fields[base_idx + 2] if len(speaker_fields) > base_idx + 2 else ""
            notes_val = speaker_fields[base_idx + 3] if len(speaker_fields) > base_idx + 3 else ""
            speakers_info[sid] = {
                "nom": name_val or "",
                "fonction": func_val or "",
                "role_reunion": role_val or "",
                "notes": notes_val or "",
            }

        yaml_path = export_speakers_yaml(
            result=result,
            speakers_info=speakers_info,
            audio_path=audio_path,
            model_name=model_name,
        )

        logger.info("[SPEAKER-ID-WEBUI] YAML exported to: %s", yaml_path)
        return (yaml_path, gr.update(value=yaml_path, visible=True))

    def run_multi_model_for_web(self, input_files, selected_models, language, enable_diarization, align_windows=True):
        import gc
        from datetime import datetime
        from pathlib import Path
        import subprocess, json as _json, time as _time, os as _os, signal as _sig

        if not input_files:
            return "No input file selected.", "", gr.update(visible=False)
        if not selected_models:
            return "No models selected.", "", gr.update(visible=False)

        audio_path = input_files if isinstance(input_files, str) else input_files[0]
        run_ts = datetime.now().strftime("%m%d%H%M%S")
        suffix = f"-{run_ts}"
        basename = Path(audio_path).stem
        output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        status_lines = [
            f"Audio : {os.path.basename(audio_path)}",
            f"Models: {', '.join(selected_models)}",
            f"Lang  : {language}",
            f"Diarz : {'yes' if enable_diarization else 'no'}",
            "-" * 50,
        ]

        # ── VRAM check ────────────────────────────────────────────────────────
        vram_ok, vram_msg = self._ensure_vram(
            min_vram_mb=24_000, aggressive_mb=36_000,
            status_list=status_lines,
            log_fn=lambda m: status_lines.append(m),
        )
        status_lines.append(vram_msg)
        if not vram_ok:
            return "\n".join(status_lines), "", gr.update(visible=False)
        model_results = []
        output_paths = []
        diarizer = None
        diarization_result = None

        if enable_diarization:
            from modules.diarize.diarizer import Diarizer
            diarizer = Diarizer(model_dir=self.args.diarization_model_dir)

        for i, model_name in enumerate(selected_models):
            status_lines.append(f"[{i+1}/{len(selected_models)}] {model_name} — en cours...")
            logger.info("[WEBUI-MULTI] Starting model %d/%d: %s", i + 1, len(selected_models), model_name)
            try:
                segments, elapsed = multi_run_model(
                    model_name=model_name,
                    audio_path=audio_path,
                    language=language,
                    compute_type="float16",
                    chunk_length=30,
                    chunk_overlap=5,
                )

                output_file = os.path.join(output_dir, f"{basename}-{model_name}{suffix}.txt")
                use_start = (model_name == "large-v3")   # VAD natif → start-based bucketing
                # Écrire l'ASR brut immédiatement — préservé même si la diarisation échoue
                final_count = multi_write_output(segments, output_file, normalize=align_windows, use_start=use_start)

                if diarizer is not None:
                    diar_was_none = diarization_result is None
                    try:
                        segments, diarization_result = apply_diarization_to_segments(
                            diarizer=diarizer,
                            audio_path=audio_path,
                            segments=segments,
                            diarization_result=diarization_result,
                        )
                        # Réécrire avec les speakers
                        final_count = multi_write_output(segments, output_file, normalize=align_windows, use_start=use_start)
                        # Offload dès que la diarisation a tourné pour la 1re fois
                        if diar_was_none:
                            diarizer.offload()
                    except Exception as diar_err:
                        logger.warning("[WEBUI-MULTI] Diarization failed for %s (ASR output kept): %s", model_name, diar_err)
                        status_lines[-1] += " (diarisation échouée)"

                output_paths.append(output_file)
                chars = sum(len((s.text or "").split("|")[-1]) for s in segments)
                seg_label = f"{final_count} fenêtres 30s" if align_windows else f"{final_count} segments"
                model_results.append({
                    "model": model_name, "ok": True, "elapsed": elapsed,
                    "segments": final_count, "chars": chars, "output": output_file,
                })
                status_lines[-1] = f"[{i+1}/{len(selected_models)}] {model_name} — ✅ {fmt_duration(elapsed)}, {seg_label}"
                logger.info("[WEBUI-MULTI] Done: %s in %s", model_name, fmt_duration(elapsed))

            except Exception as e:
                logger.exception("[WEBUI-MULTI] Model %s failed", model_name)
                model_results.append({
                    "model": model_name, "ok": False, "elapsed": 0, "error": str(e), "output": "",
                })
                status_lines[-1] = f"[{i+1}/{len(selected_models)}] {model_name} — ❌ {e}"
            gc.collect()

        # Fichier aligné (seulement si fenêtres normalisées et ≥2 modèles OK)
        if align_windows:
            ok_files = {r["model"]: r["output"] for r in model_results if r["ok"]}
            if len(ok_files) >= 2:
                aligned_path = os.path.join(output_dir, f"{basename}-aligned{suffix}.txt")
                try:
                    n_windows = multi_create_aligned(ok_files, aligned_path)
                    output_paths.append(aligned_path)
                    status_lines.append(f"Fichier aligné : {os.path.basename(aligned_path)} ({n_windows} fenêtres)")
                except Exception as aligned_err:
                    logger.warning("[WEBUI-MULTI] Aligned file failed: %s", aligned_err)

        # Summary file
        try:
            import librosa
            duration = librosa.get_duration(path=audio_path)
        except Exception:
            duration = 0.0

        summary_path = os.path.join(output_dir, f"{basename}-multi-summary{suffix}.txt")
        multi_write_summary(audio_path, duration, language, model_results, summary_path)
        output_paths.append(summary_path)

        ok_count = sum(1 for r in model_results if r["ok"])
        status_lines.append("-" * 50)
        status_lines.append(f"Terminé : {ok_count}/{len(selected_models)} modèles OK")

        # Créer un ZIP avec tous les fichiers de sortie
        import zipfile
        zip_path = os.path.join(output_dir, f"{basename}-multi{suffix}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in output_paths:
                if os.path.exists(path):
                    zf.write(path, os.path.basename(path))
        status_lines.append(f"Archive : {os.path.basename(zip_path)} ({len(output_paths)} fichiers)")

        status_text = "\n".join(status_lines)
        paths_text = "\n".join(output_paths)
        logger.info("[WEBUI-MULTI] All done. ZIP: %s", zip_path)
        return status_text, paths_text, gr.update(value=zip_path, visible=True)

    # ── Arbitrage LLM ─────────────────────────────────────────────────────────

    def _load_arbitrage_config(self) -> dict:
        import json as _json
        config_path = getattr(self, "ARBITRAGE_CONFIG_PATH", os.path.abspath("configs/arbitrage_config.json"))
        default_cfg = {
            "launch_script": os.environ.get("VOXTRAL_LAUNCH_SCRIPT", "~/launch_arbitrage.sh"),
            "model_id": os.environ.get("VOXTRAL_ARB_MODEL", "local/qwen3-35b-arbitrage"),
            "api_port": 8080,
            "opencode_bin": os.environ.get("VOXTRAL_OPENCODE_BIN", shutil.which("opencode") or "opencode"),
        }
        try:
            if os.path.exists(config_path):
                with open(config_path, encoding="utf-8") as f:
                    saved = _json.load(f)
                default_cfg.update(saved)
        except Exception:
            pass
        return default_cfg

    def _load_arbitrage_config_for_ui(self):
        self._arb_config = self._load_arbitrage_config()
        return (
            self._arb_config.get("launch_script", ""),
            self._arb_config.get("model_id", ""),
            self._arb_config.get("api_port", 8080),
        )

    def save_arbitrage_config(self, script_text: str, model_text: str, port_val: float):
        import json as _json
        cfg = {
            "launch_script": script_text.strip(),
            "model_id": model_text.strip(),
            "api_port": int(port_val),
        }
        os.makedirs(os.path.dirname(self.ARBITRAGE_CONFIG_PATH), exist_ok=True)
        with open(self.ARBITRAGE_CONFIG_PATH, "w", encoding="utf-8") as f:
            _json.dump(cfg, f, indent=2, ensure_ascii=False)
        if hasattr(self, "_arb_config"):
            self._arb_config.update(cfg)
        else:
            self._arb_config = cfg
        logger.info("[WEBUI-ARB] Configuration sauvegardée : %s", self.ARBITRAGE_CONFIG_PATH)
        return cfg["launch_script"], cfg["model_id"], cfg["api_port"], "✅ Configuration sauvegardée (script, modèle, port)"

    def _load_arbitrage_prompt(self) -> str:
        if os.path.exists(self.ARBITRAGE_PROMPT_PATH):
            return open(self.ARBITRAGE_PROMPT_PATH, encoding="utf-8").read()
        return ""

    def _load_arbitrage_lexicon(self) -> str:
        if os.path.exists(self.ARBITRAGE_LEXICON_PATH):
            return open(self.ARBITRAGE_LEXICON_PATH, encoding="utf-8").read()
        return "# Lexique métier vide — ajoutez vos termes ici au format :\n# TERME — Définition\n"

    def save_arbitrage_prompt(self, prompt_text: str):
        with open(self.ARBITRAGE_PROMPT_PATH, "w", encoding="utf-8") as f:
            f.write(prompt_text)
        return "✅ Prompt sauvegardé dans configs/arbitrage_prompt.txt"

    def save_arbitrage_lexicon(self, lexicon_text: str):
        with open(self.ARBITRAGE_LEXICON_PATH, "w", encoding="utf-8") as f:
            f.write(lexicon_text)
        return "✅ Lexique sauvegardé dans configs/lexique_metier.txt"

    # ── Raffinement Diarization ──────────────────────────────────────────────

    def _load_refine_prompt(self) -> str:
        if os.path.exists(self.REFINE_PROMPT_PATH):
            return open(self.REFINE_PROMPT_PATH, encoding="utf-8").read()
        return ""

    def save_refine_prompt(self, prompt_text: str):
        with open(self.REFINE_PROMPT_PATH, "w", encoding="utf-8") as f:
            f.write(prompt_text)
        return "✅ Prompt sauvegardé dans configs/prompt_refine_diarization.txt"

    def run_diarization_refine_for_web(self, srt_file, turns_file, prompt_text,
                                        script_path, api_port, arb_model):
        import subprocess, time, json, tempfile, os as _os, select
        from pathlib import Path
        from datetime import datetime

        api_port = int(api_port)
        api_base = f"http://localhost:{api_port}"
        opencode_bin = getattr(self, "_arb_config", {}).get(
            "opencode_bin", _os.environ.get("VOXTRAL_OPENCODE_BIN", "opencode")
        )
        status = []
        prompt_tmp = None

        def log(msg):
            status.append(msg)
            logger.info("[WEBUI-REFINE] %s", msg)

        if not srt_file or not turns_file:
            return "❌ Fichier SRT ou speaker-turns manquant.", gr.update(visible=False)

        srt_path = srt_file if isinstance(srt_file, str) else srt_file[0]
        turns_path = turns_file if isinstance(turns_file, str) else turns_file[0]

        # Créer un répertoire de travail isolé
        run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        stem = Path(srt_path).stem.replace("-arbitrage", "").replace("-refined", "")
        if not stem:
            stem = "refine"
        work_dir = _os.path.join(self.args.output_dir, f"_refine_{stem}-{run_ts}")
        _os.makedirs(work_dir, exist_ok=False)

        # Copier les fichiers dans le répertoire de travail
        import shutil
        shutil.copy2(srt_path, _os.path.join(work_dir, _os.path.basename(srt_path)))
        shutil.copy2(turns_path, _os.path.join(work_dir, _os.path.basename(turns_path)))
        log(f"Fichiers copiés dans {work_dir}")

        # Résoudre provider/model
        if "/" in arb_model:
            provider_name, model_id = arb_model.split("/", 1)
        else:
            provider_name, model_id = "local", arb_model

        # Vérifier LLM
        try:
            r = __import__('requests').get(f"{api_base}/health", timeout=3)
            if r.status_code == 200:
                log(f"✅ LLM détectée sur le port {api_port}")
            else:
                return "\n".join(status) + f"\n❌ LLM non disponible (health={r.status_code})", gr.update(visible=False)
        except Exception:
            return "\n".join(status) + f"\n❌ Aucune LLM sur le port {api_port}. Lancez d'abord une LLM.", gr.update(visible=False)

        # Auto-détecter le model_id
        try:
            models_r = __import__('requests').get(f"{api_base}/v1/models", timeout=5)
            if models_r.status_code == 200:
                server_models = [m["id"] for m in models_r.json().get("data", [])]
                if model_id not in server_models:
                    for sm in server_models:
                        if model_id in sm or sm in model_id:
                            model_id = sm
                            log(f"Model auto-détecté : {model_id}")
                            break
        except Exception:
            pass

        # Config opencode
        oc_cfg = {
            "provider": {
                provider_name: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": f"Local ({provider_name})",
                    "options": {"baseURL": f"http://localhost:{api_port}/v1", "apiKey": "sk-no-key-required", "timeout": 9999999},
                    "models": {model_id: {"name": model_id, "limit": {"context": 263144, "output": 131072}}},
                }
            },
            "permission": {
                "edit": "allow", "bash": "allow", "read": "allow", "write": "allow",
                "glob": "allow", "grep": "allow", "webfetch": "allow", "task": "allow",
                "skill": {"*": "allow"}, "question": "allow", "websearch": "deny",
                "external_directory": {"/tmp/**": "allow", "/var/tmp/**": "allow"},
            },
        }
        oc_env = _os.environ.copy()
        oc_env["OPENCODE_CONFIG_CONTENT"] = json.dumps(oc_cfg)

        # Écrire le prompt
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", prefix="refine_", delete=False, encoding="utf-8") as fh:
            fh.write(prompt_text)
            prompt_tmp = fh.name

        instruction = (
            f"Tu es dans le répertoire de travail {_os.path.abspath(work_dir)}. "
            f"Il contient un fichier SRT arbitré et un fichier speaker-turns de référence. "
            f"Suis scrupuleusement le prompt système joint. "
            f"Produis le fichier SRT raffiné dans ce même répertoire."
        )

        cmd = [opencode_bin, "run", "--format", "json", "--model", arb_model, instruction, "-f", prompt_tmp]
        log(f"opencode run --model {arb_model} | cwd={work_dir}")

        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, bufsize=1, env=oc_env, cwd=work_dir)
            fd = proc.stdout.fileno()
            buf = ""
            total_text, total_tools = 0, 0
            last_event = time.time()

            while True:
                readable, _, _ = select.select([fd], [], [], 10)
                if readable:
                    chunk = _os.read(fd, 8192).decode("utf-8", errors="replace")
                    if not chunk: break
                    buf += chunk
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if not line: continue
                        last_event = time.time()
                        try: ev = json.loads(line)
                        except: continue
                        etype = ev.get("type", "?")
                        if etype == "text":
                            txt = ev.get("part", {}).get("text", "")
                            if txt.strip():
                                total_text += 1
                                if total_text % 20 == 1:
                                    log(f"  📝 [{total_text}] {txt[:100].replace(chr(10),' ')}")
                        elif etype == "tool_call":
                            total_tools += 1
                            log(f"  🔧 [{total_tools}] {ev.get('tool',{}).get('name','?')}")
                        elif etype == "step_start":
                            log("  🟢 step_start")
                        elif etype == "step_finish":
                            log("  🔴 step_finish")
                else:
                    if proc.poll() is not None: break
                    idle = time.time() - last_event
                    if idle > 300:
                        log(f"⏰ STALL {int(idle)}s → kill")
                        proc.kill()
                        break

            proc.wait(timeout=10)
            log(f"opencode exit {proc.returncode} — {total_text} textes, {total_tools} tools")

            if proc.returncode != 0:
                err = (proc.stderr.read() if proc.stderr else "")[:600]
                raise RuntimeError(f"opencode exit {proc.returncode}: {err}")

            # Découvrir le fichier raffiné
            refined = sorted(Path(work_dir).glob("*-refined.srt"))
            if not refined:
                raise RuntimeError("Aucun fichier *-refined.srt produit")

            refined_path = str(refined[-1])
            sz = _os.path.getsize(refined_path)
            log(f"✅ SRT raffiné brut : {_os.path.basename(refined_path)} ({sz:,} bytes)")

            # Nettoyage post-traitement : fusionner segments adjacents même speaker,
            # supprimer interjections vides, segments < 0.5s, formater SRT standard
            clean_path = refined_path.replace(".srt", "-clean.srt")
            try:
                from tests.clean_srt import parse_srt as _parse, clean_srt as _clean, write_srt as _write
                entries = _parse(refined_path)
                log(f"  Entrées brutes : {len(entries)}")
                cleaned = _clean(entries, min_duration=0.5, merge_same_speaker=True)
                log(f"  Entrées après nettoyage : {len(cleaned)} (retirées/fusionnées : {len(entries)-len(cleaned)})")
                header = f"SRT nettoyé — prêt pour lecteur vidéo\nSource : {_os.path.basename(refined_path)}\n"
                _write(cleaned, clean_path, header)
                refined_path = clean_path
                sz = _os.path.getsize(refined_path)
                log(f"✅ SRT nettoyé : {_os.path.basename(refined_path)} ({sz:,} bytes)")
            except Exception as e:
                log(f"⚠️ Nettoyage SRT échoué, utilisation du brut : {e}")

        except Exception as e:
            log(f"⚠️ Erreur : {e}")
            for p in [prompt_tmp]:
                try:
                    if p and _os.path.exists(p): _os.unlink(p)
                except: pass
            return "\n".join(status) + f"\n❌ Erreur : {e}", gr.update(visible=False)

        for p in [prompt_tmp]:
            try:
                if p and _os.path.exists(p): _os.unlink(p)
                log(f"Nettoyage temp : {_os.path.basename(p)} supprimé")
            except: pass

        return "\n".join(status), gr.update(value=refined_path, visible=True)

    @staticmethod
    def _extract_srt_from_opencode_json(stdout_text: str) -> str:
        """Parse NDJSON from 'opencode run --format json'.
        Actual format: one JSON per line, type=='text' with part.text holding the content.
        """
        import json as _json
        pieces = []
        for raw_line in stdout_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = _json.loads(line)
            except Exception:
                continue
            if event.get("type") == "text":
                part = event.get("part", {})
                if part.get("type") == "text":
                    text = part.get("text", "")
                    if text.strip():
                        pieces.append(text)
        return "\n".join(pieces).strip()

    def run_arbitration_for_web(self, zip_file, lexicon_text, prompt_text, script_path, api_port, arb_model):
        import zipfile as zflib
        import subprocess
        import time
        import json
        import tempfile
        import requests
        import select
        from datetime import datetime
        from pathlib import Path

        api_port = int(api_port)
        api_base = f"http://localhost:{api_port}"
        opencode_bin = getattr(self, "_arb_config", {}).get(
            "opencode_bin", os.environ.get("VOXTRAL_OPENCODE_BIN", "opencode")
        )
        status = []
        prompt_tmp = None
        llm_proc = None

        def log(msg):
            status.append(msg)
            logger.info("[WEBUI-ARB] %s", msg)

        # ── 0. Vérifications préalables ────────────────────────────────────────
        if not os.path.exists(opencode_bin):
            return f"❌ opencode introuvable : {opencode_bin}", gr.update(visible=False)

        # ── 1. Vérifier VRAM et libérer si nécessaire ────────────────────────
        vram_ok, vram_msg = self._ensure_vram(
            min_vram_mb=24_000, aggressive_mb=36_000,
            log_fn=log,
        )
        log(vram_msg)
        if not vram_ok:
            return "\n".join(status) + f"\n❌ VRAM insuffisante — voir détails ci-dessus", gr.update(visible=False)

        # ── 2. Extraire les TXT du ZIP ─────────────────────────────────────────
        if not zip_file:
            return "❌ Aucun ZIP sélectionné.", gr.update(visible=False)

        zip_path = zip_file if isinstance(zip_file, str) else zip_file[0]
        run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        zip_stem = Path(zip_path).stem.replace("-multi", "")
        extract_dir = os.path.join(self.args.output_dir, f"_arb_{zip_stem}-{run_ts}")
        os.makedirs(extract_dir, exist_ok=False)

        log(f"Décompression ZIP → {extract_dir}")
        try:
            with zflib.ZipFile(zip_path, "r") as zf:
                txt_files = [n for n in zf.namelist() if n.endswith(".txt") and "summary" not in n]
                zf.extractall(extract_dir)
            log(f"ZIP extrait : {len(txt_files)} fichiers TXT ({sum(os.path.getsize(os.path.join(extract_dir, n)) for n in txt_files):,} bytes totaux)")
        except Exception as e:
            return f"❌ Erreur lecture ZIP : {e}", gr.update(visible=False)

        if len(txt_files) < 2:
            return "❌ ZIP invalide : moins de 2 fichiers TXT (hors summary).", gr.update(visible=False)

        txt_paths = [os.path.join(extract_dir, n) for n in txt_files]
        for name, path in zip(txt_files, txt_paths):
            with open(path, encoding="utf-8") as fh:
                chars = len(fh.read())
            log(f"  • {name} ({chars} chars)")

        # ── 3. Résoudre provider_name depuis arb_model ─────────────────────
        # Le model_id réel sera détecté depuis le serveur LLM après lancement.
        if "/" in arb_model:
            provider_name, _requested_model = arb_model.split("/", 1)
        else:
            provider_name, _requested_model = "local", arb_model

        # ── 3b. Détecter si le serveur LLM est déjà en marche ─────────────────
        llm_already_running = False
        try:
            pre_check = requests.get(f"{api_base}/health", timeout=3)
            if pre_check.status_code == 200:
                llm_already_running = True
                log(f"✅ Serveur LLM déjà en écoute sur le port {api_port} (health=200)")
        except Exception:
            log(f"ℹ️  Aucun serveur LLM détecté sur le port {api_port} — lancement requis")

        # ── 4. Lancer le serveur LLM (si pas déjà actif) ──────────────────────
        if llm_already_running:
            log("Serveur LLM déjà actif — pas de relancement")
        else:
            if not script_path or not os.path.exists(script_path):
                return "\n".join(status) + f"\n❌ Script introuvable : {script_path}", gr.update(visible=False)

            log(f"Lancement serveur LLM : {script_path}")
            try:
                llm_proc = subprocess.Popen(
                    ["bash", script_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except Exception as e:
                return "\n".join(status) + f"\n❌ Impossible de lancer le script : {e}", gr.update(visible=False)

        # ── 5. Attendre que le serveur soit prêt ──────────────────────────────
        # Polling /health uniquement si on vient de lancer le serveur
        if not llm_already_running:
            log("Attente du serveur LLM (polling /health toutes les 2s, max 300s)...")
            ready = False
            last_status = None
            for attempt in range(150):  # 300s max
                time.sleep(2)
                elapsed = (attempt + 1) * 2
                try:
                    r = requests.get(f"{api_base}/health", timeout=2)
                    if r.status_code == 200:
                        ready = True
                        log(f"✅ Serveur LLM prêt (modèle chargé en {elapsed}s)")
                        break
                    if r.status_code == 503 and elapsed % 30 == 0:
                        try:
                            srv_status = r.json().get("status", "loading")
                        except Exception:
                            srv_status = "loading"
                        log(f"  … chargement modèle en cours ({elapsed}s) — serveur: {srv_status}")
                        last_status = srv_status
                except requests.exceptions.ConnectionError:
                    if elapsed % 30 == 0:
                        log(f"  … en attente du démarrage du serveur ({elapsed}s)")
                except Exception:
                    pass

            if not ready:
                import signal as _signal
                if llm_proc:
                    try:
                        os.killpg(os.getpgid(llm_proc.pid), _signal.SIGTERM)
                    except Exception:
                        llm_proc.terminate()
                return "\n".join(status) + "\n❌ Serveur LLM non disponible après 300s.", gr.update(visible=False)

        # ── 5b. Délai de stabilisation + vérification que le modèle répond ─────
        if llm_already_running:
            log("ℹ️  LLM déjà active — pas de délai de stabilisation")
        else:
            log("⏳ Pause de 10s pour stabilisation du serveur LLM...")
            time.sleep(10)
        verified = False
        for retry in range(3):
            try:
                verify_r = requests.post(
                    f"{api_base}/v1/chat/completions",
                    json={
                        "model": _requested_model,
                        "messages": [{"role": "user", "content": "Dis juste OK."}],
                        "max_tokens": 500,
                        "temperature": 0,
                    },
                    timeout=30,
                )
                if verify_r.status_code == 200 and verify_r.json().get("choices"):
                    choice_text = verify_r.json()["choices"][0].get("message", {}).get("content", "").strip()
                    log(f"✅ Modèle opérationnel (réponse test : \"{choice_text}\")")
                    verified = True
                    break
                else:
                    log(f"⚠️  Vérification échouée (HTTP {verify_r.status_code}), retry {retry + 1}/3")
            except Exception as e:
                log(f"⚠️  Vérification échouée ({e}), retry {retry + 1}/3")
            time.sleep(5)

        if not verified:
            if llm_proc:
                import signal as _signal
                try:
                    os.killpg(os.getpgid(llm_proc.pid), _signal.SIGTERM)
                except Exception:
                    llm_proc.terminate()
                log("Serveur LLM arrêté (échec vérification modèle)")
            return "\n".join(status) + "\n❌ Modèle LLM non opérationnel après 3 tentatives.", gr.update(visible=False)

        # ── 5c. Auto-détecter le model_id depuis le serveur LLM ──────────────
        model_id = _requested_model
        try:
            models_r = requests.get(f"{api_base}/v1/models", timeout=5)
            if models_r.status_code == 200:
                server_models = [m["id"] for m in models_r.json().get("data", [])]
                if server_models:
                    if model_id not in server_models:
                        log(f"Model demandé '{model_id}' absent du serveur. Modèles disponibles : {server_models}")
                        # Chercher le meilleur match
                        for sm in server_models:
                            if model_id in sm or sm in model_id:
                                model_id = sm
                                log(f"  → Match automatique : '{model_id}'")
                                break
                        else:
                            model_id = server_models[0]
                            log(f"  → Utilisation du premier modèle disponible : '{model_id}'")
                    else:
                        log(f"Model '{model_id}' confirmé sur le serveur")
        except Exception as e:
            log(f"⚠️ Impossible d'interroger /v1/models : {e} — utilisation du model_id fourni '{model_id}'")

        # ── 5d. Construire la config opencode avec le model_id réel ─────────
        oc_cfg = {
            "provider": {
                provider_name: {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": f"Local llama.cpp ({provider_name})",
                    "options": {
                        "baseURL": f"http://localhost:{api_port}/v1",
                        "apiKey": "sk-no-key-required",
                        "timeout": 9999999,
                    },
                    "models": {
                        model_id: {
                            "name": model_id,
                            "limit": {"context": 263144, "output": 131072},
                        },
                    },
                }
            },
            "permission": {
                "edit": "allow",
                "bash": "allow",
                "read": "allow",
                "write": "allow",
                "glob": "allow",
                "grep": "allow",
                "websearch": "deny",
                "webfetch": "allow",
                "task": "allow",
                "skill": {"*": "allow"},
                "question": "allow",
                "external_directory": {
                    "/tmp/**": "allow",
                    "/var/tmp/**": "allow",
                    (os.path.abspath(os.path.join(os.path.dirname(__file__), "configs")) + "/**"): "allow",
                },
            },
        }
        oc_env = os.environ.copy()
        oc_env["OPENCODE_CONFIG_CONTENT"] = json.dumps(oc_cfg)
        log(f"Config opencode : provider={provider_name}, model={model_id}, port={api_port}")

        # ── 6. Appel opencode (streaming) — la LLM écrit les fichiers elle-même ──
        # opencode tourne avec cwd=extract_dir. La LLM découvre les fichiers source
        # et produit les sorties selon le prompt système (v2.5).
        # Monitoring : --format json → NDJSON stream → on log chaque event,
        # on détecte les stalls (>5 min sans événement) et on relance avec --continue.
        lexique_config_path = os.path.abspath("configs/lexique_metier.txt")
        srt_out = None
        reasoning_out = None
        oc_attempt = 0
        max_attempts = 4  # 1 initial + 3 continue

        while oc_attempt < max_attempts:
            oc_attempt += 1

            try:
                # ── 6a. Persister le lexique métier dans le fichier config ──────
                if lexicon_text and lexicon_text.strip():
                    lex_lines = [l for l in lexicon_text.strip().splitlines() if l.strip()]
                    os.makedirs(os.path.dirname(lexique_config_path), exist_ok=True)
                    with open(lexique_config_path, "w", encoding="utf-8") as lf:
                        lf.write(lexicon_text.strip() + "\n")
                    if oc_attempt == 1:
                        log(f"Lexique métier : {len(lex_lines)} entrées écrites dans {lexique_config_path}")

                # ── 6b. Écrire le prompt système dans un temp file ──────────────
                if oc_attempt == 1:
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".txt", prefix="arb_prompt_", delete=False, encoding="utf-8"
                    ) as fh:
                        fh.write(prompt_text)
                        prompt_tmp = fh.name

                if oc_attempt == 1:
                    instruction = (
                        f"Tu es dans le répertoire de travail {extract_dir}. "
                        f"Il contient {len(txt_paths)} transcriptions multi-modèles pré-alignées d'une réunion. "
                        f"Le lexique métier est dans {lexique_config_path}. "
                        f"Suis scrupuleusement le prompt système joint (étape 0 à étape 4). "
                        f"Tous les fichiers de sortie doivent être écrits dans le répertoire de travail."
                    )
                else:
                    instruction = (
                        "CONTINUE — tu as été interrompu. Reprends EXACTEMENT là où tu t'étais arrêté. "
                        "Ne recommence pas depuis le début. Vérifie les fichiers déjà écrits dans le "
                        "répertoire de travail avant de continuer."
                    )

                cmd = [
                    opencode_bin, "run",
                    "--format", "json",
                    "--model", arb_model,
                    instruction,
                    "-f", prompt_tmp,
                ]

                txt_size = sum(os.path.getsize(p) for p in txt_paths if os.path.exists(p))
                log(f"[TENTATIVE {oc_attempt}/{max_attempts}] opencode run --format json --model {arb_model} | cwd={extract_dir} | {len(txt_paths)} fichiers ASR | ~{txt_size // 4} tokens estimés")
                log(f"Instruction : {instruction}")

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    env=oc_env,
                    cwd=extract_dir,
                )

                # Streaming NDJSON via select — si indisponible (ex: tests mock),
                # fallback vers la lecture bloquante classique.
                try:
                    fd = proc.stdout.fileno()
                    if not isinstance(fd, int) or fd < 0:
                        raise OSError("stdout.fileno() returned non-integer")
                except Exception:
                    fd = None

                from io import StringIO

                last_event_ts = time.time()
                total_text_lines = 0
                total_tool_calls = 0
                stall_seconds = 300
                proc_done = False
                stdout_buf = ""

                try:
                    if fd is not None:
                        # Streaming mode — select + os.read
                        while True:
                            readable, _, _ = select.select([fd], [], [], 10)
                            if readable:
                                chunk = os.read(fd, 8192).decode("utf-8", errors="replace")
                                if not chunk:
                                    break
                                stdout_buf += chunk
                            else:
                                if proc.poll() is not None:
                                    proc_done = True
                                    break
                                idle = time.time() - last_event_ts
                                if idle > stall_seconds:
                                    log(f"⏰ STALL détecté : {int(idle)}s sans événement — kill + relance")
                                    proc.kill()
                                    proc_done = True
                                    break
                                continue

                            while "\n" in stdout_buf:
                                line, stdout_buf = stdout_buf.split("\n", 1)
                                line = line.strip()
                                if not line:
                                    continue
                                last_event_ts = time.time()
                                try:
                                    event = json.loads(line)
                                except (json.JSONDecodeError, ValueError):
                                    continue

                                event_type = event.get("type", "?")

                                if event_type == "text":
                                    part = event.get("part", {})
                                    text = part.get("text", "")
                                    if text.strip():
                                        total_text_lines += 1
                                        if total_text_lines % 50 == 1:
                                            preview = text[:120].replace("\n", "\\n")
                                            log(f"  📝 [{total_text_lines}] {preview}...")

                                elif event_type in ("tool_call", "tool_result"):
                                    total_tool_calls += 1
                                    tool_name = event.get("tool", {}).get("name", "?")
                                    if event_type == "tool_call":
                                        try:
                                            tool_input = json.dumps(event.get("tool", {}).get("input", {}), ensure_ascii=False)[:100]
                                        except Exception:
                                            tool_input = "?"
                                        log(f"  🔧 tool_call [{total_tool_calls}] {tool_name}({tool_input}...)")
                                    else:
                                        log(f"  📦 tool_result [{total_tool_calls}] {tool_name}")

                                elif event_type == "step_start":
                                    agent = event.get("agent", "?")
                                    step_msg = event.get("message", "").strip()[:80]
                                    log(f"  🟢 step_start agent={agent}: {step_msg}")

                                elif event_type == "step_finish":
                                    log("  🔴 step_finish")
                    else:
                        # Fallback: lecture bloquante ligne par ligne (sans stall detection)
                        log("ℹ️  Mode fallback stdout (pas de select disponible)")
                        for line in proc.stdout:
                            line = line.strip()
                            if not line:
                                continue
                            last_event_ts = time.time()
                            try:
                                event = json.loads(line)
                            except (json.JSONDecodeError, ValueError):
                                continue

                            event_type = event.get("type", "?")
                            if event_type == "text":
                                part = event.get("part", {})
                                text = part.get("text", "")
                                if text.strip():
                                    total_text_lines += 1
                                    if total_text_lines % 50 == 1:
                                        preview = text[:120].replace("\n", "\\n")
                                        log(f"  📝 [{total_text_lines}] {preview}...")
                            elif event_type in ("tool_call", "tool_result"):
                                total_tool_calls += 1
                                tool_name = event.get("tool", {}).get("name", "?")
                                if event_type == "tool_call":
                                    log(f"  🔧 tool_call [{total_tool_calls}] {tool_name}")
                                else:
                                    log(f"  📦 tool_result [{total_tool_calls}] {tool_name}")
                            elif event_type == "step_start":
                                agent = event.get("agent", "?")
                                step_msg = event.get("message", "").strip()[:80]
                                log(f"  🟢 step_start agent={agent}: {step_msg}")
                            elif event_type == "step_finish":
                                log("  🔴 step_finish")

                    if not proc_done:
                        proc_done = True

                except Exception as stream_err:
                    log(f"⚠️  Erreur flux stdout : {stream_err}")

                # Attendre la fin du process
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    log("⚠️  opencode ne termine pas après stdout fermé — kill")
                    proc.kill()
                    proc.wait()

                stderr_text = ""
                try:
                    stderr_text = proc.stderr.read()[:600]
                except Exception:
                    pass

                if proc.returncode != 0:
                    err = stderr_text or "pas de stderr"
                    log(f"⚠️  opencode exit {proc.returncode}: {err[:300]}")
                    if oc_attempt < max_attempts:
                        log(f"Tentative {oc_attempt+1}/{max_attempts} avec --continue...")
                        time.sleep(3)
                        continue
                    else:
                        raise RuntimeError(f"opencode exit {proc.returncode} après {max_attempts} tentatives: {err}")

                log(f"opencode terminé (exit 0) — {total_text_lines} textes, {total_tool_calls} tools — découverte du SRT")

            except Exception as e:
                log(f"⚠️  Exception opencode tentative {oc_attempt} : {e}")
                if oc_attempt < max_attempts:
                    log(f"Relance tentative {oc_attempt+1}/{max_attempts}...")
                    time.sleep(3)
                    continue
                else:
                    if llm_proc and not llm_already_running:
                        import signal as _signal
                        try:
                            os.killpg(os.getpgid(llm_proc.pid), _signal.SIGTERM)
                        except Exception:
                            llm_proc.terminate()
                        log("Serveur LLM arrêté (erreur)")
                    for p in [prompt_tmp]:
                        try:
                            if p and os.path.exists(p):
                                os.unlink(p)
                        except Exception:
                            pass
                    return "\n".join(status) + f"\n❌ Erreur opencode : {e}", gr.update(visible=False)

            # ── 6c. Découvrir le SRT produit ──────────────────────────────────
            srt_candidates = sorted(Path(extract_dir).glob("*-arbitrage-*.srt"))
            reasoning_candidates = sorted(Path(extract_dir).glob("*-arbitrage-reasoning-*.txt"))

            if srt_candidates:
                srt_out = str(srt_candidates[-1])
                srt_size = os.path.getsize(srt_out)
                log(f"✅ SRT arbitré : {os.path.basename(srt_out)} ({srt_size:,} bytes)")
            else:
                log(f"⚠️  Aucun *-arbitrage-*.srt trouvé (tentative {oc_attempt})")

            if reasoning_candidates:
                reasoning_out = str(reasoning_candidates[-1])
                log(f"✅ Raisonnement : {os.path.basename(reasoning_out)} ({os.path.getsize(reasoning_out):,} bytes)")
            elif not reasoning_out:
                log("⚠️  Fichier reasoning absent ou vide")

            # Si le SRT existe, on a terminé (même si reasoning manque)
            if srt_out:
                break

            # Pas de SRT : si on a encore des tentatives, relancer
            if oc_attempt < max_attempts:
                log(f"Pas de SRT après tentative {oc_attempt} — relance avec --continue...")
                time.sleep(3)
                continue

        # ── Vérification finale ──────────────────────────────────────────────
        if not srt_out:
            if llm_proc and not llm_already_running:
                import signal as _signal
                try:
                    os.killpg(os.getpgid(llm_proc.pid), _signal.SIGTERM)
                except Exception:
                    llm_proc.terminate()
                log("Serveur LLM arrêté (aucun SRT produit)")
            for p in [prompt_tmp]:
                try:
                    if p and os.path.exists(p):
                        os.unlink(p)
                except Exception:
                    pass
            return "\n".join(status) + (
                f"\n❌ Aucun fichier *-arbitrage-*.srt trouvé dans {extract_dir} "
                f"après {max_attempts} tentatives."
            ), gr.update(visible=False)

        # Nettoyage (succès)
        if llm_proc and not llm_already_running:
            import signal as _signal
            try:
                os.killpg(os.getpgid(llm_proc.pid), _signal.SIGTERM)
                log("Serveur LLM arrêté (SIGTERM)")
            except Exception:
                llm_proc.terminate()
                log("Serveur LLM terminé (fallback)")
        elif llm_already_running:
            log("Serveur LLM laissé actif (déjà présent avant arbitrage)")
        for p in [prompt_tmp]:
            try:
                if p and os.path.exists(p):
                    os.unlink(p)
                    log(f"Nettoyage temp : {os.path.basename(p)} supprimé")
            except Exception as ex:
                log(f"⚠️  Impossible de supprimer temp {p} : {ex}")

        return "\n".join(status), gr.update(value=srt_out, visible=True)


parser = argparse.ArgumentParser()
parser.add_argument('--whisper_type', type=str, default=WhisperImpl.VOXTRAL_MINI.value,
                    choices=[item.value for item in WhisperImpl],
                    help='A type of the whisper implementation (Github repo name)')
parser.add_argument('--share', type=str2bool, default=False, nargs='?', const=True, help='Gradio share value')
parser.add_argument('--server_name', type=str, default=None, help='Gradio server host')
parser.add_argument('--server_port', type=int, default=None, help='Gradio server port')
parser.add_argument('--root_path', type=str, default=None, help='Gradio root path')
parser.add_argument('--username', type=str, default=None, help='Gradio authentication username')
parser.add_argument('--password', type=str, default=None, help='Gradio authentication password')
parser.add_argument('--theme', type=str, default=None, help='Gradio Blocks theme')
parser.add_argument('--colab', type=str2bool, default=False, nargs='?', const=True, help='Is colab user or not')
parser.add_argument('--api_open', type=str2bool, default=False, nargs='?', const=True,
                    help='Enable api or not in Gradio')
parser.add_argument('--allowed_paths', type=str, default=None, help='Gradio allowed paths')
parser.add_argument('--inbrowser', type=str2bool, default=True, nargs='?', const=True,
                    help='Whether to automatically start Gradio app or not')
parser.add_argument('--ssl_verify', type=str2bool, default=True, nargs='?', const=True,
                    help='Whether to verify SSL or not')
parser.add_argument('--ssl_keyfile', type=str, default=None, help='SSL Key file location')
parser.add_argument('--ssl_keyfile_password', type=str, default=None, help='SSL Key file password')
parser.add_argument('--ssl_certfile', type=str, default=None, help='SSL cert file location')
parser.add_argument('--whisper_model_dir', type=str, default=WHISPER_MODELS_DIR,
                    help='Directory path of the whisper model')
parser.add_argument('--faster_whisper_model_dir', type=str, default=FASTER_WHISPER_MODELS_DIR,
                    help='Directory path of the faster-whisper model')
parser.add_argument('--insanely_fast_whisper_model_dir', type=str,
                    default=INSANELY_FAST_WHISPER_MODELS_DIR,
                    help='Directory path of the insanely-fast-whisper model')
parser.add_argument('--voxtral_model_dir', type=str, default=VOXTRAL_MODELS_DIR,
                    help='Directory path of the voxtral model')
parser.add_argument('--qwen3_asr_model_dir', type=str, default=QWEN3_ASR_MODELS_DIR,
                    help='Directory path of the Qwen3-ASR model')
parser.add_argument('--cohere_asr_model_dir', type=str, default=COHERE_ASR_MODELS_DIR,
                    help='Directory path of the Cohere ASR model')
parser.add_argument('--diarization_model_dir', type=str, default=DIARIZATION_MODELS_DIR,
                    help='Directory path of the diarization model')
parser.add_argument('--nllb_model_dir', type=str, default=NLLB_MODELS_DIR,
                    help='Directory path of the Facebook NLLB model')
parser.add_argument('--uvr_model_dir', type=str, default=UVR_MODELS_DIR,
                    help='Directory path of the UVR model')
parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Directory path of the outputs')
if __name__ == "__main__":
    _args = parser.parse_args()
    app = App(args=_args)
    app.launch()
