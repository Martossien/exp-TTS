# Voxtral-WebUI — Technical Documentation

> Last updated: 2026-04-24 by Claude (DeepSeek v4)

---

## Architecture générale

Application Gradio de transcription audio basée sur plusieurs backends Whisper.
Chemin projet : `~/Voxtral-WebUI`
Virtualenv Python : `~/.pyenv/versions/voxtral-env/`
Service systemd ou script nohup, port **7860**

---

## 4 Modèles STT installés

| Modèle | Backend | WhisperImpl | Notes |
|--------|---------|-------------|-------|
| `large-v3` | Faster-Whisper (CTranslate2) | `faster-whisper` | Le plus rapide (~5 min/h audio) |
| `cohere-transcribe-03-2026` | CohereASRInference (2B, Apache 2.0) | `cohere-asr` | Excellent FR, nécessite bfloat16 |
| `qwen3-asr-1.7b` | Qwen3ASRInference (Alibaba) | `qwen3-asr` | Qualité correcte |
| `voxtral-mini-3b` | VoxtralWhisperInference (Mistral) | `voxtral-mini` | Le plus lent (~14 min/h audio) |

Tous les modèles sont stockés dans `models/` (offline, `HF_HUB_OFFLINE=1`).

---

## Contrainte critique : version transformers

**Ne jamais upgrader `transformers` sans retester les 4 modèles.**

- Version fixée : `transformers==4.57.6` (imposée par le package `qwen-asr`)
- transformers 5.x casse Cohere : `AutoModelForSpeechSeq2Seq` route vers l'implémentation native `parakeet` avec un layout de tenseurs incompatible → `mat1 and mat2 shapes cannot be multiplied (16x17664 and 4096x1280)`

---

## Corrections apportées (session 2026-04-17)

### 1. Cohere ASR — float16 overflow (CRITIQUE)

**Symptôme** : `RuntimeError: value cannot be converted to type c10::Half without overflow`

**Cause** : Le code Cohere utilise `-1e9` comme valeur de remplissage dans le masque d'attention. Float16 max ≈ 65504, donc -1e9 provoque un overflow.

**Fichier** : `modules/whisper/cohere_asr_inference.py` → méthode `update_model()`

**Correction** :
```python
if compute_type == "float16":
    logger.warning("[COHERE-ASR] float16 not supported (attention mask -1e9 overflows fp16). Forcing bfloat16.")
    compute_type = "bfloat16"
dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
dtype = dtype_map.get(compute_type, torch.bfloat16)
```
bfloat16 partage la plage d'exposant de float32 (jusqu'à ±3.4×10³⁸), donc -1e9 passe sans overflow.

**Cohere generate() — paramètres configurables** :
```python
output_ids = self.model.generate(
    **gen_inputs,
    max_new_tokens=params.max_new_tokens or 600,
    repetition_penalty=params.repetition_penalty if params.repetition_penalty else 1.0,
)
```

### 2. Hallucinations large-v3 (corrigé)

**Fichier** : `configs/default_parameters_faster_whisper.yaml`

**Paramètres anti-hallucination** :
```yaml
condition_on_previous_text: false
repetition_penalty: 1.3
no_repeat_ngram_size: 5
hallucination_silence_threshold: 2.0
vad_filter: true
```

### 3. Nom du modèle dans les fichiers de sortie

**Fichier** : `modules/whisper/base_transcription_pipeline.py`

**Avant** : `audio-2026041716.srt`
**Après** : `audio-large-v3-2026041716.srt`

**Code ajouté dans `transcribe_file()`** :
```python
file_name, file_ext = os.path.splitext(os.path.basename(file))
model_slug = params.whisper.model_size.replace("/", "-").replace(" ", "-")
output_file_name = f"{file_name}-{model_slug}"
```

**Code ajouté dans `transcribe_mic()`** :
```python
model_slug = params.whisper.model_size.replace("/", "-").replace(" ", "-")
file_name = f"Mic-{model_slug}"
```

### 4. Voxtral — paramètres de qualité

**Fichier** : `modules/whisper/voxtral_whisper_inference.py`

Ajouté dans les **deux** appels `generate()` (chemin chunked + chemin audio court) :
```python
outputs = self.model.generate(
    **inputs,
    max_new_tokens=params.max_new_tokens or 32000,
    temperature=params.temperature if params.temperature > 0 else 0.0,
    do_sample=params.temperature > 0,
    repetition_penalty=params.repetition_penalty if params.repetition_penalty else 1.0,
    no_repeat_ngram_size=params.no_repeat_ngram_size if params.no_repeat_ngram_size else 0,
)
```

### 5. Qwen3 — max_new_tokens augmenté

**Fichier** : `modules/whisper/qwen3_asr_inference.py`

`max_new_tokens=256` → `max_new_tokens=512` dans `Qwen3ASRModel.from_pretrained()`

### 6. chunk_overlap augmenté

**Fichier** : `configs/default_parameters.yaml`

`chunk_overlap: 2` → `chunk_overlap: 5`

Préserve mieux le contexte aux jonctions de chunks.

### 7. Gradio — queue et boutons (app.py)

**Problème** : conflit entre `queue=True` (SSE) et les longues transcriptions (~45 min).
SSE coupe/reconnecte et perd la réponse sur les tâches longues.

**Solution finale** :
- `self.app.queue()` ajouté avant `self.app.launch()` (nécessaire pour la traduction)
- Boutons de **transcription** : `queue=False` (HTTP direct, obligatoire pour les tâches longues)
- Boutons de **traduction** : `queue=True` (tâches courtes, peut afficher la progression)

**Lignes modifiées dans app.py** :
- L.187 : `transcribe_file_for_web` → `queue=False`
- L.213 : `transcribe_youtube_for_web` → `queue=False`
- L.236 : `transcribe_mic_for_web` → `queue=False`
- L.271 : `translate_deepl_for_web` → `queue=True`
- L.303 : `translate_nllb_for_web` → `queue=True`

**Conséquence acceptée** : pas de barre de progression pendant la transcription (affichage du temps écoulé uniquement).

---

## Nouveau script : run_multi_model.py

**Chemin** : `$PROJECT_DIR/run_multi_model.py`
**But** : Lancer jusqu'à 4 modèles ASR séquentiellement sur le même fichier audio.
La **diarisation est exécutée une seule fois** et réutilisée pour tous les modèles.

### Fonctionnement

1. Chargement du modèle → transcription → offload → GC
2. Après le 1er modèle : exécution de la diarisation (pyannote)
3. Diarisation offloadée après le 1er modèle pour libérer la VRAM
4. Pour chaque modèle suivant : `assign_word_speakers()` applique le résultat de diarisation aux segments ASR

### Format de sortie

```
[00:00:00 → 00:01:23] SPEAKER_00: Texte transcrit du premier locuteur.
[00:01:24 → 00:02:10] SPEAKER_01: Réponse du second locuteur.
```

Adapté pour être lu par un LLM (timestamps + locuteur + texte sans fragmentation SRT).

### Utilisation

```bash
# Batch complet avec diarisation (défaut)
python run_multi_model.py --input audio.m4a --language french

# Modèles spécifiques
python run_multi_model.py --input audio.m4a --models large-v3,cohere-transcribe-03-2026

# Sans diarisation (plus rapide)
python run_multi_model.py --input audio.m4a --no-diarization

# En arrière-plan avec log
nohup python run_multi_model.py --input audio.m4a --language french \
    >> logs/multi_model.log 2>&1 &
```

### Options CLI

| Option | Défaut | Description |
|--------|--------|-------------|
| `--input` | (requis) | Chemin vers le fichier audio/vidéo |
| `--models` | tous les 4 | Liste séparée par virgules |
| `--language` | `french` | Langue (nom complet ou code ISO) |
| `--compute-type` | `float16` | Type de calcul (Cohere force bfloat16 auto) |
| `--chunk-length` | `30` | Durée des chunks en secondes |
| `--chunk-overlap` | `5` | Chevauchement entre chunks en secondes |
| `--output-dir` | `outputs/` | Dossier de sortie |
| `--no-diarization` | off | Désactiver la diarisation |

### Fichiers produits

Pour un fichier `audio.m4a` avec le timestamp `0417161501` :
- `outputs/audio-large-v3-0417161501.txt`
- `outputs/audio-cohere-transcribe-03-2026-0417161501.txt`
- `outputs/audio-qwen3-asr-1.7b-0417161501.txt`
- `outputs/audio-voxtral-mini-3b-0417161501.txt`
- `outputs/audio-multi-summary-0417161501.txt` (tableau récapitulatif)

---

## Test en cours (2026-04-17 16:18)

**Commande lancée** :
```bash
nohup python (from venv) run_multi_model.py \
    --input /tmp/gradio/2632dedf99a03d56711d74426d059e8a27c7d272f5e8aab6bee644c3648b2533/audio1240087701.m4a \
    --models large-v3,cohere-transcribe-03-2026,qwen3-asr-1.7b,voxtral-mini-3b \
    --language french \
    >> logs/multi_model.log 2>&1 &
```

**PID** : 604312  
**Log** : `$PROJECT_DIR/logs/multi_model.log`  
**Fichier audio** : `audio1240087701.m4a` (65 min, 40 Mo, FR)  
**Lieu** : `/tmp/gradio/2632dedf99a03d56711d74426d059e8a27c7d272f5e8aab6bee644c3648b2533/audio1240087701.m4a`  

**Attention** : le fichier est dans `/tmp` — ne pas supprimer ou redémarrer le système pendant le test.

**Durée estimée** : ~1h30 total  
- large-v3 : ~15 min  
- cohere : ~20 min  
- qwen3 : ~15 min  
- voxtral : ~30 min  
- diarisation : ~5 min (une seule fois)

**Statut** : large-v3 en cours à 16:18 (détection langue FR confirmée à 100%)

**Pour suivre** :
```bash
tail -f $PROJECT_DIR/logs/multi_model.log
```

---

## Tests validés (2026-04-17)

| Modèle | Fichier test | Résultat |
|--------|-------------|---------|
| cohere-transcribe-03-2026 | audio1240087701.m4a (~60s) | ✅ OK (bfloat16 forcé) |
| qwen3-asr-1.7b | audio1240087701.m4a (~60s) | ✅ OK (GPU1 puis GPU0) |
| voxtral-mini-3b | audio1240087701.m4a (~60s) | ✅ OK |
| large-v3 (anti-hallucination) | audio1240087701.m4a complet | ⏳ en cours (batch multi-model) |
| run_multi_model.py (Option A) | tests/jfk.wav, 2 modèles, no-diarize | ✅ OK |
| run_multi_model.py (Option B) | audio1240087701.m4a, 4 modèles, diarize | ⏳ en cours |

---

## Après le batch

1. Vérifier les fichiers dans `outputs/` (4 TXT + 1 summary)
2. Redémarrer le service WebUI :
   ```bash
   ./start.sh --whisper_type cohere-asr --port 7860
   # ou selon la config souhaitée
   ```
3. Tester la transcription WebUI avec large-v3 pour confirmer le fix anti-hallucination

---

## Fichiers de configuration

### `configs/default_parameters.yaml` (Cohere / Voxtral / Qwen3)
Paramètres clés actuels :
```yaml
whisper:
  model_size: qwen3-asr-1.7b   # changé pendant les tests, remettre selon besoin
  lang: french
  chunk_overlap: 5              # augmenté de 2 → 5
  batch_size: 24
  enable_offload: true
diarization:
  is_diarize: true
  diarization_model: pyannote/speaker-diarization-community-1
```

### `configs/default_parameters_faster_whisper.yaml` (large-v3)
Paramètres anti-hallucination :
```yaml
whisper:
  condition_on_previous_text: false
  repetition_penalty: 1.3
  no_repeat_ngram_size: 5
  hallucination_silence_threshold: 2.0
vad:
  vad_filter: true
```

---

## Démarrage du service

```bash
# Démarrage standard (tous modèles, port 7860)
cd $PROJECT_DIR
./start.sh --whisper_type cohere-asr --port 7860

# Vérifier le statut
./status.sh

# Arrêt
./stop.sh

# Logs en direct
tail -f app.log
```

**Note importante** : `--whisper_type` doit être `cohere-asr` (pas `cohere`).
Valeurs valides : `faster-whisper`, `cohere-asr`, `qwen3-asr`, `voxtral-mini`

---

## Diarisation

**Modèle** : `pyannote/speaker-diarization-community-1` (téléchargé localement dans `models/`)
**Token HF** : configuré dans `configs/default_parameters.yaml` → `hf_token`

Pour vérifier que le modèle est présent :
```bash
ls models/diarization/
```

---

## Architecture des modules whisper

```
modules/whisper/
├── whisper_factory.py              # Factory — crée le bon backend selon whisper_type
├── base_transcription_pipeline.py  # Classe de base — gestion fichiers, output, nommage
├── faster_whisper_inference.py     # Backend large-v3
├── cohere_asr_inference.py         # Backend Cohere (bfloat16 forcé si float16)
├── qwen3_asr_inference.py          # Backend Qwen3
└── voxtral_whisper_inference.py    # Backend Voxtral/Mistral
```

---

## Prochaines étapes (backlog)

1. **Phase 2** : ~~onglet WebUI pour le multi-modèle~~ ✅ Implémenté (onglet "Multi-Model")
2. **Phase 3** : ~~Arbitrage LLM (opencode + Qwen 35B)~~ ✅ Implémenté (onglet "Arbitrage LLM")
3. **Phase 4** : génération de vidéo avec multi-sous-titres via ffmpeg (explicitement "deuxième temps")
4. **Phase 5** : exporter le SRT arbitré en VTT/ASS + téléchargement individuel par modèle
5. **Phase 6** : support de modèles LLM additionnels pour l'arbitrage (GLM-4.7 via `ik_local`)
6. **Phase 7** : historiques d'arbitrage, comparaison des versions de SRT

---

## Sauvegarde

- Backup complet : `~/old_backup/Voxtral-WebUI-backup-20250918.tar.gz`
- Procédure restauration :
  ```bash
  ./stop.sh
  tar -xzf ~/old_backup/Voxtral-WebUI-backup-20250918.tar.gz
  ./start.sh --whisper_type cohere-asr --port 7860
  ```

---

## Informations système

- Hardware : 2× NVIDIA GeForce RTX 5090 (32 GB chacune)
- Python : 3.11 (`voxtral-env`)
- OS : Linux 6.14.8-2-pve
- HuggingFace user : martossien

---

## Arbitrage LLM — Reconstruction SRT

### Concept

Après une transcription multi-modèle (4 TXT), un LLM de type Qwen-35B arbitre les différences et produit un **fichier SRT consolidé** avec la meilleure transcription pour chaque segment.

Le LLM tourne localement via **llama.cpp** (serveur HTTP sur port 8080) et est invoqué via **opencode** (`opencode run`) avec un prompt système en 4 étapes (`configs/arbitrage_prompt.txt`).

### Workflow complet

| Étape | Description | Détail |
|-------|-------------|--------|
| **1** | **Vérification VRAM** | `nvidia-smi` → si < 36 Go libres, `nvidia-smi --query-compute-apps` liste les processus GPU, tue ceux qui correspondent à `python/llama/vllm/ik_llama/whisper/voxtral/asr/stt` (SIGTERM) → attend 8s → re-vérifie. Abort si < 24 Go. |
| **2** | **Extraction ZIP** | Décompresse dans `outputs/_arb_{zip_stem}-{YYYYmmdd-HHMMSS}/` (répertoire isolé, `exist_ok=False`). Vérifie ≥ 2 TXT hors summary. |
| **3a** | **Préparation config opencode** | Injecte provider + permissions + `external_directory` via `OPENCODE_CONFIG_CONTENT` (pas de modification du fichier `opencode.json` global). |
| **3b** | **Détection LLM existante** | Teste `GET /health` sur le port cible. Si 200 → la LLM est déjà active, on ne la relance pas. Au cleanup, on ne la tue pas non plus. |
| **4** | **Lancement LLM** | Uniquement si pas déjà active. `subprocess.Popen(["bash", script_path], start_new_session=True)` → le `killpg` tuera tout le groupe (numactl + llama-server). |
| **5a** | **Polling /health** | Uniquement si on vient de lancer. Appelle `GET /health` toutes les 2s, max 300s. HTTP 503 = chargement en cours (normal). HTTP 200 = prêt. |
| **5b** | **Stabilisation + vérification** | Si on vient de lancer : pause 10s. Puis `POST /v1/chat/completions` avec un prompt test (`"Dis juste OK."`), 3 retries si échec. Abort si le modèle ne répond pas. |
| **6** | **Appel opencode** | `opencode run --format json --model <model> -f <prompt_tmp>` avec `cwd=extract_dir`. Mode streaming NDJSON. |
| **6b** | **Monitoring + stall detection** | `select.select` sur stdout avec timeout 10s. Chaque event JSON loggé (text, tool_call, step_start/stop). Si > 5 min sans événement → kill + relance `--continue` (max 4 tentatives). |
| **6c** | **Découverte SRT** | Après chaque tentative : glob `*-arbitrage-*.srt` et `*-arbitrage-reasoning-*.txt` dans `extract_dir`. Si SRT trouvé → succès. Sinon → relance. |

### Configuration persistante

Les paramètres Script / Modèle / Port sont sauvegardés dans `configs/arbitrage_config.json` :

```json
{
  "launch_script": "$VOXTRAL_LAUNCH_SCRIPT",
  "model_id": "local/qwen3-35b-arbitrage",
  "api_port": 8080
}
```

Chargé au démarrage du WebUI via `_load_arbitrage_config()`, sauvegardé via le bouton `💾 Sauvegarder config` dans l'onglet Arbitrage LLM.

Variables d'environnement supportées :
- `VOXTRAL_LAUNCH_SCRIPT` — chemin du script de lancement LLM (fallback: `~/launch_arbitrage.sh`)
- `VOXTRAL_ARB_MODEL` — modèle opencode (fallback: `local/qwen3-35b-arbitrage`)
- `VOXTRAL_OPENCODE_BIN` — binaire opencode (fallback: `opencode` depuis PATH)

### Configuration opencode injectée

`OPENCODE_CONFIG_CONTENT` contient :

```json
{
  "provider": {
    "local": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://localhost:8080/v1", "apiKey": "sk-no-key-required", "timeout": 9999999 },
      "models": { "qwen3-35b-arbitrage": { "limit": { "context": 263144, "output": 131072 } } }
    }
  },
  "permission": {
    "edit": "allow", "bash": "allow", "read": "allow", "write": "allow",
    "glob": "allow", "grep": "allow", "webfetch": "allow", "task": "allow",
    "skill": { "*": "allow" }, "question": "allow",
    "external_directory": {
      "/tmp/**": "allow",
      "/var/tmp/**": "allow",
      "$PROJECT_DIR/configs/**": "allow"
    }
  }
}
```

`--dangerously-skip-permissions` n'est **plus utilisé** — les permissions sont contrôlées via la config inline.

### Fichier de permissions global

`~/.config/opencode/opencode.json` contient également les permissions (pour les sessions interactives opencode hors WebUI) :

```json
"permission": {
  "external_directory": { "/tmp/**": "allow", "/var/tmp/**": "allow" },
  "edit": { "*": "allow", "*.env": "ask", "*.env.*": "ask" },
  "bash": "allow", "read": "allow", "write": "allow",
  "glob": "allow", "grep": "allow", "task": "allow"
}
```

### Monitoring en direct dans le WebUI

Pendant l'arbitrage, chaque événement opencode est loggé dans le champ Statut :

```
[TENTATIVE 1/4] opencode run --format json --model local/qwen3-35b-arbitrage | cwd=... | 5 fichiers ASR | ~9000 tokens estimés
  🟢 step_start agent=general: Étape 0 : je découvre le répertoire...
  🔧 tool_call [1] bash(ls *_arb_*)
  📦 tool_result [1] bash
  📝 [1] # Rapport d'arbitrage SRT
  📝 [51] 00:00:01,000 --> 00:00:30,000...
  🔧 tool_call [5] write(*-arbitrage.srt)
  📝 [101] 1\n00:00:01,000 -->...
  🔴 step_finish
opencode terminé (exit 0) — 234 textes, 12 tools — découverte du SRT
✅ SRT arbitré : audio-test-arbitrage-20260424_113500.srt (45,230 bytes)
```

En cas de stall (plus de 5 min sans événement) :

```
⏰ STALL détecté : 305s sans événement — kill + relance
Tentative 2/4 avec --continue...
```

### Limites connues

- Le LLM d'arbitrage (Qwen3.6-35B-A3B) occupe ~30 Go de VRAM. Les modèles STT doivent être offloadés avant l'arbitrage (géré automatiquement par le cleanup VRAM).
- Timeout opencode : 2h par tentative (pas de timeout global, 4 tentatives max).
- Le prompt système v2.5 (`configs/arbitrage_prompt.txt`) utilise des subagents (@explore, @general) — nécessite opencode ≥ 1.1.

---

## Test opencode cloud — NVIDIA GLM-4.7 (2026-04-24)

**Objectif** : Valider le pilotage d'opencode via `--format json` (streaming) et le mécanisme `--continue` avec une LLM cloud (NVIDIA NIM).

**Script** : `tests/test_opencode_cloud.py`

**Modèle** : `nvidia/z-ai/glm4.7` (NVIDIA NIM, gratuit via build.nvidia.com)

### Résultats

| Test | Commande | Prompt | Réponse | Statut |
|------|----------|--------|---------|--------|
| 1 | `opencode run --format json` | *Bonjour, quelle est la capitale de la France ?* | **La capitale de la France est Paris.** | ✅ |
| 2 | `opencode run --continue` | *Combien d'habitants à Paris intra-muros ?* | **Paris intra-muros compte environ 2,1 millions d'habitants.** | ✅ |

### Fonctionnement du test

1. Lance `opencode run --format json --model nvidia/z-ai/glm4.7 "..."` avec streaming NDJSON
2. Parse chaque event JSON (step_start, text, step_finish), log dans la console
3. Vérifie que la réponse contient "Paris"
4. Lance `opencode run --continue "Combien d'habitants..."` — opencode reprend la session précédente avec tout le contexte
5. Vérifie la réponse (contient un chiffre de population)
6. Seul le process opencode est arrêté (pas de LLM locale, pas de llama-server)

---

## Prochaines étapes (backlog)
