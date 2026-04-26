import os
from pathlib import Path

# Constantes sans i18n au niveau module pour éviter l'erreur ContextVar
AUTOMATIC_DETECTION = "Automatic Detection"
GRADIO_NONE_NUMBER_MAX = 9999
FASTER_WHISPER_TYPE = "faster-whisper"
INSANELY_FAST_WHISPER_TYPE = "insanely-fast-whisper"
OPEN_AI_WHISPER_TYPE = "openai-whisper"

# Autres constantes
DEFAULT_MODEL_SIZE = "large-v3"
DEFAULT_DEVICE = "cuda"

# Chemins
BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / "cache"
LOGS_DIR = BASE_DIR / "logs"

# Créer les répertoires nécessaires
for directory in [MODELS_DIR, CACHE_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Fonction pour obtenir les traductions dans le contexte approprié
def get_translation(key):
    try:
        from gradio_i18n import gettext as _
        return _(key)
    except:
        return key  # Fallback au texte original si i18n échoue
