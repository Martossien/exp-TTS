#!/bin/bash

# Installation script for Voxtral-WebUI systemd service
# This service will be DISABLED by default (not started automatically)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_NAME="voxtral-webui"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
WRAPPER_SCRIPT="$SCRIPT_DIR/voxtral-service-wrapper.sh"

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE} Installation du service Voxtral-WebUI ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Vérifier si l'utilisateur est root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}❌ Ce script doit être exécuté en tant que root (sudo)${NC}"
    echo "   Usage: sudo $0"
    exit 1
fi

# Vérifier si systemd est disponible
if ! command -v systemctl &> /dev/null; then
    echo -e "${RED}❌ systemd n'est pas disponible sur ce système${NC}"
    exit 1
fi

# Créer les répertoires nécessaires
echo -e "${BLUE}📁 Création des répertoires...${NC}"
mkdir -p /var/log/voxtral-webui
mkdir -p /var/run
touch /var/log/voxtral-webui/service.log

# Copier le fichier de service
echo -e "${BLUE}📝 Installation du fichier de service...${NC}"
cp "$SCRIPT_DIR/voxtral-webui.service" "$SERVICE_FILE"

# Rendre les scripts exécutables
echo -e "${BLUE}🔧 Configuration des permissions...${NC}"
chmod +x "$WRAPPER_SCRIPT"

# Recharger systemd
echo -e "${BLUE}🔄 Rechargement de systemd...${NC}"
systemctl daemon-reload

# Désactiver le service par défaut (ne pas démarrer au boot)
echo -e "${BLUE}⚙️ Configuration du service (désactivé par défaut)...${NC}"
systemctl disable "$SERVICE_NAME"

# Vérifier la VRAM disponible
echo -e "${BLUE}🎮 Vérification de la VRAM...${NC}"
if command -v nvidia-smi &> /dev/null; then
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]')
    VRAM_GB=$((VRAM_MB / 1024))
    echo -e "   VRAM détectée: ${GREEN}${VRAM_GB} GB${NC}"

    if [ "$VRAM_GB" -lt 16 ]; then
        echo -e "   ${YELLOW}⚠️  WARNING: VRAM < 16 GB (${VRAM_GB} GB détectée)${NC}"
        echo -e "   ${YELLOW}   Le service démarrera avec un avertissement.${NC}"
    else
        echo -e "   ${GREEN}✅ VRAM suffisante pour Voxtral + Diarization${NC}"
    fi
else
    echo -e "   ${YELLOW}⚠️  nvidia-smi non trouvé, impossible de vérifier la VRAM${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Installation terminée avec succès! ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}⚠️  Le service est installé mais DÉSACTIVÉ par défaut.${NC}"
echo ""
echo -e "${BLUE}Commandes disponibles :${NC}"
echo ""
echo -e "  ${YELLOW}Activer au démarrage :${NC}"
echo -e "    ${GREEN}sudo systemctl enable voxtral-webui${NC}"
echo ""
echo -e "  ${YELLOW}Démarrer le service :${NC}"
echo -e "    ${GREEN}sudo systemctl start voxtral-webui${NC}"
echo "    (ou: sudo $WRAPPER_SCRIPT start)"
echo ""
echo -e "  ${YELLOW}Arrêter le service :${NC}"
echo -e "    ${GREEN}sudo systemctl stop voxtral-webui${NC}"
echo "    (ou: sudo $WRAPPER_SCRIPT stop)"
echo ""
echo -e "  ${YELLOW}Redémarrer le service :${NC}"
echo -e "    ${GREEN}sudo systemctl restart voxtral-webui${NC}"
echo "    (ou: sudo $WRAPPER_SCRIPT restart)"
echo ""
echo -e "  ${YELLOW}Voir le statut :${NC}"
echo -e "    ${GREEN}sudo systemctl status voxtral-webui${NC}"
echo -e "    (ou: sudo $WRAPPER_SCRIPT status)"
echo ""
echo -e "  ${YELLOW}Voir les logs :${NC}"
echo -e "    ${GREEN}sudo tail -f /var/log/voxtral-webui/service.log${NC}"
echo -e "    (ou: sudo $WRAPPER_SCRIPT logs)"
echo ""
echo -e "  ${YELLOW}Vérifier la VRAM :${NC}"
echo -e "    ${GREEN}sudo $WRAPPER_SCRIPT check-vram${NC}"
echo ""
echo -e "${BLUE}Configuration :${NC}"
echo -e "  - Fichier service: $SERVICE_FILE"
echo -e "  - Wrapper script: $WRAPPER_SCRIPT"
echo -e "  - Logs: /var/log/voxtral-webui/service.log"
echo -e "  - PID: /var/run/voxtral-webui.pid"
echo ""
echo -e "${BLUE}Fichiers du projet :${NC}"
echo -e "  - Répertoire: $PROJECT_DIR"
echo -e "  - Configuration: $PROJECT_DIR/configs/default_parameters.yaml"
echo ""
