#!/bin/bash

REPO_DIR="/workspace/YuE-Interface"
BACKUP_DIR="$REPO_DIR/transformers_bkp"
TARGET_DIR="/opt/conda/envs/pyenv/lib/python3.12/site-packages/transformers"

# Verifica se o backup existe antes de restaurar
if [ -d "$BACKUP_DIR" ]; then
    echo "Restoring backup..."
    
    # Copia os arquivos do backup para o diretório correto
    cp -rf "$BACKUP_DIR/generation" "$TARGET_DIR/"
    cp -rf "$BACKUP_DIR/models" "$TARGET_DIR/"

    # Remove o backup após a restauração
    rm -rf "$BACKUP_DIR"

    echo "Restore completed!"
else
    echo "Backup not found. No changes were made."
fi
