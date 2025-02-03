#!/bin/bash

REPO_DIR="/workspace/YuE-Interface"
BACKUP_DIR="$REPO_DIR/transformers_bkp"

mkdir -p "$BACKUP_DIR/generation"
mkdir -p "$BACKUP_DIR/models"

cp -rf /opt/conda/envs/pyenv/lib/python3.12/site-packages/transformers/generation "$BACKUP_DIR/generation"
cp -rf /opt/conda/envs/pyenv/lib/python3.12/site-packages/transformers/models "$BACKUP_DIR/models"

cp -rf "$REPO_DIR/transformers" /opt/conda/envs/pyenv/lib/python3.12/site-packages
