#!/bin/bash

REPO_DIR="/workspace/YuE-Interface"
BACKUP_DIR="$REPO_DIR/transformers_bkp"
TARGET_DIR="/opt/conda/envs/pyenv/lib/python3.12/site-packages/transformers"

if [ -d "$BACKUP_DIR" ]; then
    echo "Restoring backup..."

    cp -r "$BACKUP_DIR/generation" "$TARGET_DIR/"
    cp -r "$BACKUP_DIR/models" "$TARGET_DIR/"

    rm -rf "$BACKUP_DIR"

    echo "Restore completed!"
else
    echo "Backup not found. No changes were made."
fi
