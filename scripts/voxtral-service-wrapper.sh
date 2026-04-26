#!/bin/bash

# Voxtral-WebUI Service Wrapper
# Checks VRAM before starting the service

# Configuration
MIN_VRAM_GB=16
WARNING_VRAM_GB=16
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/configs/default_parameters.yaml"
LOG_FILE="/var/log/voxtral-webui/service.log"
PID_FILE="/var/run/voxtral-webui.pid"

# Couleurs pour les messages
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Fonction de logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Fonction pour vérifier la VRAM disponible
check_vram() {
    if ! command -v nvidia-smi &> /dev/null; then
        log "${YELLOW}⚠️ WARNING: nvidia-smi non trouvé. Impossible de vérifier la VRAM.${NC}"
        return 1
    fi

    # Obtenir les infos VRAM pour tous les GPUs
    # Format: GPU, Total Memory (MiB), Used Memory (MiB), Free Memory (MiB)
    local gpu_info=$(nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null)
    
    if [ -z "$gpu_info" ]; then
        log "${YELLOW}⚠️ WARNING: Impossible d'obtenir les informations VRAM.${NC}"
        return 1
    fi

    log "Analyse VRAM par GPU:"
    
    # Variables pour stocker les valeurs maximales
    local max_free_mb=0
    local max_free_gpu=""
    local total_free_mb=0
    
    # Analyser chaque GPU
    while IFS=',' read -r gpu_id total_mb used_mb free_mb; do
        # Nettoyer les espaces
        gpu_id=$(echo "$gpu_id" | tr -d '[:space:]')
        total_mb=$(echo "$total_mb" | tr -d '[:space:]')
        used_mb=$(echo "$used_mb" | tr -d '[:space:]')
        free_mb=$(echo "$free_mb" | tr -d '[:space:]')
        
        local total_gb=$((total_mb / 1024))
        local used_gb=$((used_mb / 1024))
        local free_gb=$((free_mb / 1024))
        
        log "  GPU $gpu_id: ${total_gb}GB total, ${used_gb}GB utilisé, ${free_gb}GB libre"
        
        # Garder le GPU avec le plus de VRAM libre
        if [ "$free_mb" -gt "$max_free_mb" ]; then
            max_free_mb=$free_mb
            max_free_gpu=$gpu_id
        fi
        
        # Calculer le total libre (utile pour multi-GPU)
        total_free_mb=$((total_free_mb + free_mb))
    done <<< "$gpu_info"
    
    local max_free_gb=$((max_free_mb / 1024))
    local total_free_gb=$((total_free_mb / 1024))
    
    log "VRAM libre maximale (GPU $max_free_gpu): ${max_free_gb} GB"
    log "VRAM libre totale (tous GPUs): ${total_free_gb} GB"

    # Vérifier la VRAM disponible (info uniquement — le code gère le cleanup dynamiquement)
    if [ "$max_free_gb" -lt 10 ]; then
        log "${YELLOW}⚠️  VRAM limitée (${max_free_gb} GB libre). Le code gérera le cleanup GPU au besoin.${NC}"
    elif [ "$max_free_gb" -lt 16 ]; then
        log "ℹ️  VRAM: ${max_free_gb} GB libre."
    else
        log "${GREEN}✅ VRAM disponible: ${max_free_gb} GB${NC}"
    fi
    return 0
}

# Fonction pour vérifier si le service est déjà en cours
is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE" 2>/dev/null)
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# Fonction pour démarrer le service
start_service() {
    log "Démarrage du service Voxtral-WebUI..."

    # Vérifier si déjà en cours
    if is_running; then
        log "${YELLOW}⚠️ Le service est déjà en cours d'exécution (PID: $(cat $PID_FILE))${NC}"
        exit 1
    fi

    # Vérifier la VRAM
    if ! check_vram; then
        log "${RED}❌ Démarrage annulé: VRAM insuffisante.${NC}"
        exit 1
    fi

    # Créer le répertoire de logs si nécessaire
    mkdir -p "$(dirname "$LOG_FILE")"

    # Activer l'environnement virtuel Python
    if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
        source "$PROJECT_DIR/.venv/bin/activate"
    elif [ -f "$HOME/.pyenv/versions/voxtral-env/bin/activate" ]; then
        source "$HOME/.pyenv/versions/voxtral-env/bin/activate"
    fi

    cd "$PROJECT_DIR"

    # Default to offline Hugging Face cache usage. Models should be downloaded
    # explicitly; runtime inference must not depend on network metadata checks.
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    log "Mode Hugging Face offline: HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"

    # Démarrer l'application
    log "Lancement de Voxtral-WebUI..."
    python app.py \
        --whisper_type voxtral-mini \
        --server_port 7860 \
        --server_name 0.0.0.0 \
        >> "$LOG_FILE" 2>&1 &

    local pid=$!
    echo $pid > "$PID_FILE"

    # Attendre que le service démarre
    sleep 3

    if kill -0 "$pid" 2>/dev/null; then
        log "${GREEN}✅ Service démarré avec succès (PID: $pid)${NC}"
        log "${GREEN}   Accès: http://localhost:7860${NC}"
        return 0
    else
        log "${RED}❌ Échec du démarrage du service${NC}"
        rm -f "$PID_FILE"
        return 1
    fi
}

# Fonction pour afficher et nettoyer la VRAM après arrêt
cleanup_vram() {
    if ! command -v nvidia-smi &>/dev/null; then
        log "nvidia-smi non disponible — nettoyage VRAM ignoré"
        return
    fi

    # Rapport VRAM par GPU
    vram_report() {
        nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total \
            --format=csv,noheader,nounits \
        | while IFS=',' read -r idx name used free total; do
            used=$(echo "$used" | tr -d ' '); free=$(echo "$free" | tr -d ' ')
            total=$(echo "$total" | tr -d ' '); name=$(echo "$name" | xargs)
            log "  GPU $idx ($name): ${used}MiB utilisé / ${total}MiB total (${free}MiB libre)"
        done
    }

    # Processus encore accrochés aux device CUDA
    local gpu_pids
    gpu_pids=$(fuser /dev/nvidia* 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u)

    if [ -z "$gpu_pids" ]; then
        log "Aucun processus résiduel sur les GPUs."
    else
        log "${YELLOW}Processus GPU encore actifs après l'arrêt :${NC}"
        for pid in $gpu_pids; do
            local cmd; cmd=$(ps -p "$pid" -o comm= 2>/dev/null || echo "inconnu")
            log "  PID $pid ($cmd)"
        done

        log "Envoi SIGTERM aux processus GPU résiduels..."
        for pid in $gpu_pids; do
            kill "$pid" 2>/dev/null && log "  SIGTERM → PID $pid"
        done
        sleep 2

        local survivors
        survivors=$(fuser /dev/nvidia* 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u)
        if [ -n "$survivors" ]; then
            log "${RED}Forçage SIGKILL sur processus GPU persistants...${NC}"
            for pid in $survivors; do
                kill -9 "$pid" 2>/dev/null && log "  SIGKILL → PID $pid"
            done
            sleep 1
        fi
    fi

    log "État VRAM après nettoyage :"
    vram_report
}

# Fonction pour arrêter le service
stop_service() {
    log "Arrêt du service Voxtral-WebUI..."

    # Rapport VRAM avant arrêt
    if command -v nvidia-smi &>/dev/null; then
        log "État VRAM avant arrêt :"
        nvidia-smi --query-gpu=index,name,memory.used,memory.free \
            --format=csv,noheader,nounits \
        | while IFS=',' read -r idx name used free; do
            used=$(echo "$used" | tr -d ' '); free=$(echo "$free" | tr -d ' ')
            name=$(echo "$name" | xargs)
            log "  GPU $idx ($name): ${used}MiB utilisé, ${free}MiB libre"
        done
    fi

    if ! is_running; then
        log "${YELLOW}⚠️ Le service n'est pas en cours d'exécution${NC}"
        log "Vérification des processus GPU résiduels quand même..."
        cleanup_vram
        return 0
    fi

    local pid; pid=$(cat "$PID_FILE" 2>/dev/null)
    log "Envoi SIGTERM au processus principal (PID: $pid)..."
    kill "$pid" 2>/dev/null

    # Attente gracieuse jusqu'à 10 s
    for i in $(seq 1 10); do
        sleep 1
        if ! kill -0 "$pid" 2>/dev/null; then
            log "Processus terminé proprement après ${i}s."
            break
        fi
    done

    # Forcer si encore vivant
    if kill -0 "$pid" 2>/dev/null; then
        log "${YELLOW}Processus non terminé — forçage SIGKILL...${NC}"
        kill -9 "$pid" 2>/dev/null
        sleep 1
    fi

    rm -f "$PID_FILE"
    log "${GREEN}✅ Service arrêté (PID: $pid)${NC}"

    # Laisser l'OS libérer le contexte CUDA
    sleep 2

    log "Nettoyage VRAM..."
    cleanup_vram

    return 0
}

# Fonction pour redémarrer le service
restart_service() {
    stop_service
    sleep 2
    start_service
}

# Fonction pour afficher le statut
status_service() {
    if is_running; then
        local pid=$(cat "$PID_FILE" 2>/dev/null)
        log "${GREEN}✅ Service en cours d'exécution (PID: $pid)${NC}"

        # Afficher la VRAM utilisée
        if command -v nvidia-smi &> /dev/null; then
            local vram_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1)
            local vram_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
            log "VRAM utilisée: ${vram_used} MB / ${vram_total} MB"
        fi

        return 0
    else
        log "${RED}❌ Service arrêté${NC}"
        return 1
    fi
}

# Fonction pour afficher les logs
logs_service() {
    if [ -f "$LOG_FILE" ]; then
        tail -n 50 "$LOG_FILE"
    else
        log "Aucun fichier de log trouvé"
    fi
}

# Gestion des arguments
case "${1:-start}" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    status)
        status_service
        ;;
    logs)
        logs_service
        ;;
    check-vram)
        check_vram
        exit $?
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs|check-vram}"
        echo ""
        echo "Commandes:"
        echo "  start      - Démarrer le service (avec vérification VRAM)"
        echo "  stop       - Arrêter le service"
        echo "  restart    - Redémarrer le service"
        echo "  status     - Afficher le statut"
        echo "  logs       - Afficher les derniers logs"
        echo "  check-vram - Vérifier la VRAM sans démarrer"
        exit 1
        ;;
esac
