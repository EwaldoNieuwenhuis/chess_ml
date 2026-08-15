#!/usr/bin/env bash

# ==============================================================================
# Script: copy_md_to_clipboard.sh
# Purpose: Concatenates all Markdown (*.md) files in the repo (respecting .gitignore)
#          and copies the unified content to the system clipboard.
# Supported OS: Windows (Git Bash / WSL / MSYS), macOS (pbcopy), Linux (xclip/wl-copy)
# ==============================================================================

set -euo pipefail

# Ensure we run from the repository root
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

# Gather all tracked and untracked .md files, strictly excluding files ignored by .gitignore
MD_FILES=$(git ls-files --cached --others --exclude-standard "*.md")

if [ -z "$MD_FILES" ]; then
    echo "⚠️ No markdown (*.md) files found in the repository."
    exit 0
fi

# Detect available clipboard utility
CLIP_CMD=""
if command -v clip.exe &>/dev/null; then
    CLIP_CMD="clip.exe"
elif command -v clip &>/dev/null; then
    CLIP_CMD="clip"
elif command -v pbcopy &>/dev/null; then
    CLIP_CMD="pbcopy"
elif command -v wl-copy &>/dev/null; then
    CLIP_CMD="wl-copy"
elif command -v xclip &>/dev/null; then
    CLIP_CMD="xclip -selection clipboard"
elif command -v xsel &>/dev/null; then
    CLIP_CMD="xsel --clipboard --input"
elif command -v powershell.exe &>/dev/null; then
    CLIP_CMD="powershell.exe -NoProfile -Command Set-Clipboard"
fi

if [ -z "$CLIP_CMD" ]; then
    echo "❌ Error: No clipboard utility detected (tried: clip.exe, pbcopy, wl-copy, xclip, xsel, powershell.exe)."
    echo "Please install a clipboard tool or pipe output directly."
    exit 1
fi

echo "📋 Gathering Markdown files (excluding .gitignore)..."

# Process files and pipe directly to clipboard
TOTAL_FILES=0
{
    while IFS= read -r file; do
        if [ -f "$file" ]; then
            echo "================================================================================"
            echo "FILE: $file"
            echo "================================================================================"
            cat "$file"
            echo -e "\n"
        fi
    done <<< "$MD_FILES"
} | $CLIP_CMD

# Count files
TOTAL_FILES=$(echo "$MD_FILES" | wc -l | tr -d ' ')

echo "✅ Successfully copied $TOTAL_FILES Markdown file(s) to clipboard!"
echo "Files copied:"
echo "$MD_FILES" | sed 's/^/  - /'
