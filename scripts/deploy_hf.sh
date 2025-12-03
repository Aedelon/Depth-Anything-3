#!/bin/bash
# Copyright (c) Delanoe Pirard / Aedelon
# Licensed under the Apache License, Version 2.0
#
# Deploy to HuggingFace Spaces with proper README front matter

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# HuggingFace Spaces YAML front matter
HF_YAML='---
title: Awesome Depth Anything 3
emoji: 🌊
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Metric 3D reconstruction from images/video
---

'

# Backup original README
cp "$PROJECT_ROOT/README.md" "$PROJECT_ROOT/README.md.bak"

# Prepend YAML front matter to README
echo "$HF_YAML$(cat "$PROJECT_ROOT/README.md")" > "$PROJECT_ROOT/README.md"

# Push to HuggingFace (force to override divergent history)
echo "Pushing to HuggingFace Spaces..."
git push huggingface main --force

# Restore original README
mv "$PROJECT_ROOT/README.md.bak" "$PROJECT_ROOT/README.md"

echo "Done! README restored to original state."