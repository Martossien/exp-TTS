import os
import sys
import re
import time
import shutil
import zipfile
import tempfile
import io
import unittest
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


SAMPLE_TXT_FRAGMENTS = {
    "audio-test-large-v3-20260424.txt": (
        "[00:00:00 -> 00:00:30] SPEAKER_00: Bonjour à tous, bienvenue dans cette réunion.\n"
        "[00:00:30 -> 00:01:00] SPEAKER_01: Merci, commençons par le point numéro un.\n"
    ),
    "audio-test-cohere-transcribe-03-2026-20260424.txt": (
        "[00:00:00 -> 00:00:30] SPEAKER_00: Bonjour à tous, bienvenue dans cette réunion.\n"
        "[00:00:30 -> 00:01:00] SPEAKER_01: Merci, commençons par le point numéro un.\n"
    ),
    "audio-test-qwen3-asr-1_7b-20260424.txt": (
        "[00:00:00 -> 00:00:30] SPEAKER_00: Bonjour à tous, bienvenue dans cette réunion.\n"
        "[00:00:30 -> 00:01:00] SPEAKER_01: Merci, commençons par le point numéro un.\n"
    ),
    "audio-test-voxtral-mini-3b-20260424.txt": (
        "[00:00:00 -> 00:00:30] SPEAKER_00: Bonjour à tous, bienvenue dans cette réunion.\n"
        "[00:00:30 -> 00:01:00] SPEAKER_01: Merci, commençons par le point numéro un.\n"
    ),
    "audio-test-aligned-20260424.txt": (
        "TRANSCRIPTION ALIGNÉE\n"
    ),
    "audio-test-multi-summary-20260424.txt": (
        "Fichier : audio-test.m4a\n"
    ),
}


def _make_zip(zip_path: str) -> str:
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, content in SAMPLE_TXT_FRAGMENTS.items():
            zf.writestr(name, content)
    return zip_path


def _fake_whisper_inf():
    mock = MagicMock()
    mock.device = "cuda"
    return mock


def _load_prompt():
    prompt_path = os.path.join(os.path.dirname(__file__), "..", "configs", "arbitrage_prompt.txt")
    with open(prompt_path, encoding="utf-8") as f:
        return f.read()


def _make_popen_mock(returncode=0, stderr_text="", stdout_lines=None, exit_delay=0.0,
                     side_effect_fn=None):
    """Crée un mock Popen réaliste avec stdout/stderr et fileno non entier → fallback."""
    mock = MagicMock()
    mock.returncode = returncode
    mock.poll.return_value = returncode
    mock.wait.return_value = None

    mock_stdout = MagicMock()
    mock_stdout.fileno.return_value = -1  # → fallback non-select
    mock_stdout.read.return_value = ""
    if stdout_lines:
        mock_stdout.__iter__.return_value = iter(stdout_lines)
    else:
        mock_stdout.__iter__.return_value = iter([])

    mock_stderr = MagicMock()
    mock_stderr.fileno.return_value = -1
    mock_stderr.read.return_value = stderr_text

    mock.stdout = mock_stdout
    mock.stderr = mock_stderr

    if exit_delay > 0:
        import time as _time
        original_sleep = _time.sleep

    if side_effect_fn:
        mock.side_effect_fn = side_effect_fn

    return mock


class TestArbitrationIsolation(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.outputs_dir = os.path.join(self.tmpdir, "outputs")
        os.makedirs(self.outputs_dir)

        self.zip_path = os.path.join(self.tmpdir, "audio-test-multi-20260424-113000.zip")
        _make_zip(self.zip_path)

        self.args = MagicMock()
        self.args.output_dir = self.outputs_dir

        from app import App
        self.app = App.__new__(App)
        self.app.args = self.args
        self.app.whisper_inf = _fake_whisper_inf()
        self.app._arb_config = {
            "opencode_bin": os.environ.get("VOXTRAL_OPENCODE_BIN", shutil.which("opencode") or "opencode"),
            "launch_script": os.environ.get("VOXTRAL_LAUNCH_SCRIPT", os.path.expanduser("~/launch_arbitrage.sh")),
            "model_id": "local/qwen3-35b-arbitrage",
            "api_port": 8080,
        }
        self.prompt_text = _load_prompt()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # TEST 1 : isolation + découverte SRT par glob
    # ------------------------------------------------------------------
    @patch("os.path.exists", return_value=True)
    @patch("subprocess.Popen")
    @patch("requests.post")
    @patch("requests.get")
    @patch("time.sleep")
    def test_directory_isolation_and_srt_discovery(
        self,
        mock_time_sleep,
        mock_requests_get,
        mock_requests_post,
        mock_subprocess_popen,
        mock_path_exists,
    ):
        mock_health_resp = MagicMock()
        mock_health_resp.status_code = 200
        mock_requests_get.return_value = mock_health_resp

        mock_verify_resp = MagicMock()
        mock_verify_resp.status_code = 200
        mock_verify_resp.json.return_value = {"choices": [{"message": {"content": "OK"}}]}
        mock_requests_post.return_value = mock_verify_resp

        def make_oc_proc(*args, **kwargs):
            cwd = kwargs.get("cwd", ".")
            # Écrit les fichiers SRT et reasoning comme side effect
            srt_name = "audio-test-arbitrage-20260424_113500.srt"
            reasoning_name = "audio-test-arbitrage-reasoning-20260424_113500.txt"
            with open(os.path.join(cwd, srt_name), "w") as f:
                f.write("1\n00:00:01,000 --> 00:00:30,000\nSPEAKER_00: test\n")
            with open(os.path.join(cwd, reasoning_name), "w") as f:
                f.write("Reasoning test\n")

            return _make_popen_mock(
                returncode=0,
                stdout_lines=[],  # aucun event JSON nécessaire
            )

        mock_subprocess_popen.side_effect = make_oc_proc

        lexicon = "ORG-ALPHA — Organisation Alpha\nORG-BETA — Organisation Beta\n"

        result = self.app.run_arbitration_for_web(
            self.zip_path, lexicon, self.prompt_text,
            self.app._arb_config["launch_script"], 8080, "local/qwen3-35b-arbitrage",
        )

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        status_text, download_update = result

        print(f"\n[TEST 1] STATUS OUTPUT:\n{status_text}\n")

        self.assertIn("ZIP extrait", status_text)

        extract_pattern = re.compile(r"_arb_.+?-\d{8}-\d{6}")
        extract_dirs = [
            d for d in os.listdir(self.outputs_dir)
            if re.match(r"_arb_.+?-\d{8}-\d{6}", d)
        ]
        self.assertEqual(
            len(extract_dirs), 1,
            f"Attendu 1 répertoire isolé, trouvé {len(extract_dirs)} : {extract_dirs}",
        )

        extract_dir = os.path.join(self.outputs_dir, extract_dirs[0])
        self.assertTrue(os.path.isdir(extract_dir))

        srt_files = sorted(Path(extract_dir).glob("*-arbitrage-*.srt"))
        self.assertEqual(len(srt_files), 1, f"Aucun SRT découvert dans {extract_dir}")

        reasoning_files = sorted(Path(extract_dir).glob("*-arbitrage-reasoning-*.txt"))
        self.assertEqual(len(reasoning_files), 1, f"Aucun reasoning découvert dans {extract_dir}")

        txt_files = list(Path(extract_dir).glob("*.txt"))
        self.assertGreaterEqual(len(txt_files), 6, f"Fichiers TXT sources insuffisants : {len(txt_files)}")

        self.assertIn("✅ SRT arbitré", status_text)

    # ------------------------------------------------------------------
    # TEST 2 : lexique écrit dans configs/lexique_metier.txt (chemin projet)
    # ------------------------------------------------------------------
    @patch("os.path.exists", return_value=True)
    @patch("subprocess.Popen")
    @patch("requests.post")
    @patch("requests.get")
    @patch("time.sleep")
    def test_lexicon_written_to_project_config(
        self,
        mock_time_sleep,
        mock_requests_get,
        mock_requests_post,
        mock_subprocess_popen,
        mock_path_exists,
    ):
        chdir_app = os.path.join(os.path.dirname(__file__), "..")
        os.chdir(chdir_app)

        mock_health_resp = MagicMock()
        mock_health_resp.status_code = 200
        mock_requests_get.return_value = mock_health_resp

        mock_verify_resp = MagicMock()
        mock_verify_resp.status_code = 200
        mock_verify_resp.json.return_value = {"choices": [{"message": {"content": "OK"}}]}
        mock_requests_post.return_value = mock_verify_resp

        lexique_config_path = os.path.abspath("configs/lexique_metier.txt")
        with open(lexique_config_path, encoding="utf-8") as f:
            original_lexicon = f.read()

        os.chdir(self.tmpdir)

        test_lexicon = (
            "ORG-ALPHA — Organisation Alpha\n"
            "ENO — Équipe Numérique Opérationnelle\n"
            "BGP — Brigade Gardes Portuaires\n"
        )

        def make_oc_proc(*args, **kwargs):
            cwd = kwargs.get("cwd", ".")
            srt_name = "audio-test-arbitrage-20260424_113700.srt"
            with open(os.path.join(cwd, srt_name), "w") as f:
                f.write("1\n00:00:01,000 --> 00:00:30,000\nSPEAKER_00: test.\n")
            return _make_popen_mock(returncode=0)

        mock_subprocess_popen.side_effect = make_oc_proc

        result = self.app.run_arbitration_for_web(
            self.zip_path, test_lexicon,
            "# Test prompt\n",
            self.app._arb_config["launch_script"], 8080, "local/qwen3-35b-arbitrage",
        )

        os.chdir(chdir_app)
        with open(lexique_config_path, encoding="utf-8") as f:
            restored = f.read()
        self.assertEqual(restored, original_lexicon,
                         "Le lexique du projet a été modifié")

        self.assertIsInstance(result, tuple)
        status_text, _ = result
        print(f"\n[TEST 2] STATUS OUTPUT:\n{status_text}\n")
        self.assertIn("ZIP extrait", status_text)

        extractions = [d for d in os.listdir(self.outputs_dir) if d.startswith("_arb_")]
        self.assertGreaterEqual(len(extractions), 1)

    # ------------------------------------------------------------------
    # TEST 3 : aucun SRT découvert → rerun auto-continue (max 4 tentatives)
    # ── puis échec final ────────────────────────────────────────────────
    # ------------------------------------------------------------------
    @patch("os.path.exists", return_value=True)
    @patch("subprocess.Popen")
    @patch("requests.post")
    @patch("requests.get")
    @patch("time.sleep")
    def test_no_srt_discovery_shows_error(
        self,
        mock_time_sleep,
        mock_requests_get,
        mock_requests_post,
        mock_subprocess_popen,
        mock_path_exists,
    ):
        mock_health_resp = MagicMock()
        mock_health_resp.status_code = 200
        mock_requests_get.return_value = mock_health_resp

        mock_verify_resp = MagicMock()
        mock_verify_resp.status_code = 200
        mock_verify_resp.json.return_value = {"choices": [{"message": {"content": "OK"}}]}
        mock_requests_post.return_value = mock_verify_resp

        # Chaque tentative renvoie exit 0 mais aucun SRT n'est écrit
        mock_subprocess_popen.return_value = _make_popen_mock(returncode=0)

        result = self.app.run_arbitration_for_web(
            self.zip_path,
            "ORG-ALPHA\n",
            "# Test prompt\n",
            self.app._arb_config["launch_script"], 8080, "local/qwen3-35b-arbitrage",
        )
        status_text, _ = result
        print(f"\n[TEST 3] STATUS OUTPUT:\n{status_text}\n")
        self.assertIn("aucun fichier", status_text.lower())

    # ------------------------------------------------------------------
    # TEST 4 : opencode exit != 0 → retry → puis échec final
    # ------------------------------------------------------------------
    @patch("os.path.exists", return_value=True)
    @patch("subprocess.Popen")
    @patch("requests.post")
    @patch("requests.get")
    @patch("time.sleep")
    def test_opencode_nonzero_exit_raises_error(
        self,
        mock_time_sleep,
        mock_requests_get,
        mock_requests_post,
        mock_subprocess_popen,
        mock_path_exists,
    ):
        mock_health_resp = MagicMock()
        mock_health_resp.status_code = 200
        mock_requests_get.return_value = mock_health_resp

        mock_verify_resp = MagicMock()
        mock_verify_resp.status_code = 200
        mock_verify_resp.json.return_value = {"choices": [{"message": {"content": "OK"}}]}
        mock_requests_post.return_value = mock_verify_resp

        # Toutes les tentatives exit 1
        mock_subprocess_popen.return_value = _make_popen_mock(
            returncode=1, stderr_text="Error: model not loaded — no GPU available"
        )

        result = self.app.run_arbitration_for_web(
            self.zip_path,
            "ORG-ALPHA\n",
            "# Test prompt\n",
            self.app._arb_config["launch_script"], 8080, "local/qwen3-35b-arbitrage",
        )
        status_text, _ = result
        print(f"\n[TEST 4] STATUS OUTPUT:\n{status_text}\n")
        self.assertIn("opencode exit 1", status_text)
        self.assertIn("model not loaded", status_text)


if __name__ == "__main__":
    unittest.main()
