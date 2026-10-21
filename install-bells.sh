#!/bin/bash
# BELLS one-line installer for Linux/macOS
# Usage: curl -sSL https://raw.githubusercontent.com/DGuckert/llama.cpp-BELLS/bells-next/install-bells.sh | bash

set -e

echo ""
echo "  ____  _____ _     _     ____"
echo " | __ )| ____| |   | |   / ___|"
echo " |  _ \\|  _| | |   | |   \\___ \\"
echo " | |_) | |___| |___| |___ ___) |"
echo " |____/|_____|_____|_____|____/"
echo ""
echo " Per-layer VRAM expert cache for MoE models"
echo ""

# Check Python
PY=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PY="$cmd"
        break
    fi
done

if [ -z "$PY" ]; then
    echo "[BELLS] ERROR: Python 3.10+ required"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "  Fedora:        sudo dnf install python3 python3-pip"
    echo "  macOS:         brew install python@3.12"
    exit 1
fi

echo "[BELLS] Found $($PY --version)"

# Determine repo location
SCRIPT_DIR="$(cd "$(dirname "$0")" 2>/dev/null && pwd)" || SCRIPT_DIR=""
REPO_DIR="$SCRIPT_DIR"

if [ ! -f "$REPO_DIR/tools/bells-manager/install.py" ]; then
    REPO_DIR="$HOME/llama.cpp-BELLS"
    if [ ! -d "$REPO_DIR" ]; then
        echo "[BELLS] Cloning repository..."
        if ! git clone --branch bells-next https://github.com/DGuckert/llama.cpp-BELLS.git "$REPO_DIR"; then
            echo "[BELLS] ERROR: git clone failed"
            exit 1
        fi
    else
        echo "[BELLS] Updating repository at $REPO_DIR..."
        # Preserve local changes: check for dirty worktree before touching it.
        if ! git -C "$REPO_DIR" diff --quiet --ignore-submodules 2>/dev/null || ! git -C "$REPO_DIR" diff --cached --quiet --ignore-submodules 2>/dev/null; then
            echo "[BELLS] Local changes detected — skipping git update to preserve them."
            echo "[BELLS] Commit or stash your changes, then re-run the installer to update."
        else
            # Fetch and fast-forward only; fail loudly if it cannot be done.
            if git -C "$REPO_DIR" fetch origin bells-next; then
                if ! git -C "$REPO_DIR" checkout bells-next; then
                    echo "[BELLS] WARNING: could not checkout bells-next, keeping current branch"
                elif ! git -C "$REPO_DIR" merge --ff-only origin/bells-next; then
                    echo "[BELLS] WARNING: could not fast-forward — local commits or diverged history, keeping current state"
                    echo "[BELLS] Preserve your work and run: git -C \"$REPO_DIR\" merge --ff-only origin/bells-next"
                fi
            else
                echo "[BELLS] WARNING: git fetch failed — keeping current checkout"
            fi
        fi
    fi
fi

echo "[BELLS] Running installer..."
if ! "$PY" "$REPO_DIR/tools/bells-manager/install.py"; then
    echo "[BELLS] ERROR: installer failed"
    exit 1
fi

echo ""
echo "[BELLS] Done! Run 'bells' to start."
