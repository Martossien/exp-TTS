# exp-TTS — Expérimental (fork de Voxtral-WebUI)

> **⚠️ Ce projet est EXPÉRIMENTAL. Il n'est PAS destiné à un usage en production.**
>
> Pour un usage stable, utilisez le projet original :
> **[OlivierAlbertini/Voxtral-WebUI](https://github.com/OlivierAlbertini/Voxtral-WebUI)**

---

## Contexte

Ce dépôt est un fork expérimental de [Voxtral-WebUI](https://github.com/OlivierAlbertini/Voxtral-WebUI),
lui-même fork de [Whisper-WebUI](https://github.com/jhj0517/Whisper-WebUI).

Il contient des modifications lourdes à usage personnel. Les ajouts ci-dessous
peuvent être instables, non documentés, et ne suivent pas nécessairement les
bonnes pratiques du projet d'origine.

## Nouveautés par rapport à l'original

### Modèles de transcription additionnels
- **[Cohere ASR](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)** — modèle 2B, 14 langues
- **[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)** — modèle 1.7B, 52 langues

### Transcription multi-modèles
- Onglet **Multi-Model** : lance 2 à 4 modèles ASR séquentiellement sur le même audio
- Fichier aligné multi-sources (comparaison fenêtre par fenêtre)
- Fichier résumé d'exécution

### Arbitrage LLM
- Onglet **Arbitrage LLM** : utilise une LLM locale pour arbitrer entre 4 transcriptions
  ASR et produire un SRT final
- Pilotage via [opencode](https://github.com/anomalyco/opencode)
- Prompt système configurable, lexique métier éditable
- Détection automatique du modèle sur le serveur LLM

### Raffinement de diarization
- Post-traitement SRT : split précis aux changements de locuteur (au lieu de fenêtres
  fixes de 30s)
- Utilise pyannote + LLM pour raffiner l'attribution des speakers
- Post-traitement automatique (fusion micro-segments, dédoublonnage)

### Identification des locuteurs
- Onglet dédié : détection des locuteurs par pyannote (sans transcription ASR)
- Extraction d'extraits audio par locuteur
- Export YAML des locuteurs identifiés
- Interface de saisie des noms, fonctions, rôles

### Éditeur SRT standalone (service séparé)
- Service Flask indépendant ([vtt-editor-pro](https://github.com/Martossien/vtt-editor-pro))
- Waveform interactive, édition inline, split/merge de segments
- Export SRT/VTT, auto-save, 5 thèmes

---

## Installation

```bash
git clone https://github.com/Martossien/exp-TTS.git
cd exp-TTS

# Créer un environnement virtuel et installer les dépendances
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Installer les dépendances spécifiques Voxtral (optionnel)
pip install git+https://github.com/huggingface/transformers.git
pip install mistral-common[audio]
```

### Variables d'environnement

| Variable | Rôle | Défaut |
|----------|------|--------|
| `HF_TOKEN` | Token HuggingFace (pyannote, modèles gated) | *(aucun)* |
| `VOXTRAL_LAUNCH_SCRIPT` | Script de lancement du serveur LLM | `~/launch_arbitrage.sh` |
| `VOXTRAL_OPENCODE_BIN` | Binaire opencode | `opencode` (depuis PATH) |
| `VOXTRAL_ARB_MODEL` | Modèle LLM pour l'arbitrage | `local/qwen3-35b-arbitrage` |
| `SRT_EDITOR_URL` | URL du service éditeur SRT | `http://localhost:7861` |

### Configuration

Les fichiers `configs/*.example` servent de modèles. Copiez-les sans `.example`
et adaptez-les à votre environnement :

```bash
cp configs/default_parameters.yaml.example configs/default_parameters.yaml
cp configs/default_parameters_faster_whisper.yaml.example configs/default_parameters_faster_whisper.yaml
cp configs/lexique_metier.txt.example configs/lexique_metier.txt
cp configs/arbitrage_config.json.example configs/arbitrage_config.json
```

## Lancement

```bash
# Mode Faster-Whisper (léger)
python app.py --whisper_type faster-whisper --server_port 7860

# Avec configuration personnalisée
python app.py --whisper_type faster-whisper \
  --server_port 7860 \
  --server_name 0.0.0.0 \
  --config configs/default_parameters.yaml
```

## Licence

Ce projet conserve la licence du projet original (MIT).
Voir [LICENSE](LICENSE).

---

**Rappel :** ce dépôt est un fork expérimental. Pour un usage stable,
préférez [OlivierAlbertini/Voxtral-WebUI](https://github.com/OlivierAlbertini/Voxtral-WebUI).
