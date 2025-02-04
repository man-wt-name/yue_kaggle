#!/bin/bash

REPO_DIR="/workspace/YuE-Interface"
BACKUP_DIR="$REPO_DIR/transformers_bkp"
TARGET_DIR="/opt/conda/envs/pyenv/lib/python3.12/site-packages/transformers"

echo "Creating backup..."

rm -rf "$BACKUP_DIR"

mkdir -p "$BACKUP_DIR"

cp -r "$TARGET_DIR/generation" "$BACKUP_DIR/"
cp -r "$TARGET_DIR/models" "$BACKUP_DIR/"

echo "Backup completed."

echo "Applying patch..."
cp -rf "$REPO_DIR/transformers" "$TARGET_DIR"

echo "Patch applied successfully!"
