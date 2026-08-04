#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# 🍄 Mushroom Classifier - Deployment Setup Script
# This script sets up the environment for Streamlit Community Cloud
# or any Linux-based deployment.
# ─────────────────────────────────────────────────────────────────────

set -e

echo "🍄 Mushroom Classifier - Setup Script"
echo "======================================"
echo ""

# ── Configuration ──────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_VERSION="3.10"
VENV_NAME="mushroom_env"

# ── Check Python Version ───────────────────────────────────────────
echo "📋 Checking Python version..."
if command -v python3 &>/dev/null; then
    PYTHON=$(command -v python3)
elif command -v python &>/dev/null; then
    PYTHON=$(command -v python)
else
    echo "❌ Python not found. Please install Python $PYTHON_VERSION or higher."
    exit 1
fi

PYTHON_VERSION_INSTALLED=$($PYTHON --version 2>&1 | awk '{print $2}')
echo "   Found Python $PYTHON_VERSION_INSTALLED"

# ── Create Virtual Environment ─────────────────────────────────────
echo ""
echo "🔧 Creating virtual environment..."
if [ -d "$PROJECT_DIR/$VENV_NAME" ]; then
    echo "   Virtual environment already exists. Skipping..."
else
    $PYTHON -m venv "$PROJECT_DIR/$VENV_NAME"
    echo "   Virtual environment created at $PROJECT_DIR/$VENV_NAME"
fi

# ── Activate Virtual Environment ───────────────────────────────────
echo ""
echo "🔌 Activating virtual environment..."
source "$PROJECT_DIR/$VENV_NAME/bin/activate" || source "$PROJECT_DIR/$VENV_NAME/Scripts/activate"

# ── Upgrade pip ────────────────────────────────────────────────────
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip

# ── Install Dependencies ───────────────────────────────────────────
echo ""
echo "📚 Installing dependencies..."
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    pip install -r "$PROJECT_DIR/requirements.txt"
    echo "   ✅ Dependencies installed successfully."
else
    echo "   ⚠️  requirements.txt not found. Skipping..."
fi

# ── Verify Models ──────────────────────────────────────────────────
echo ""
echo "🔍 Checking model files..."
MODELS_DIR="$PROJECT_DIR/models"
if [ -d "$MODELS_DIR" ]; then
    echo "   Models directory found."
    ls -la "$MODELS_DIR"/*.pkl 2>/dev/null && echo "   ✅ Pickle models found." || echo "   ⚠️  No pickle models found. Please add them."
    ls -la "$MODELS_DIR"/*.h5 2>/dev/null && echo "   ✅ H5 model found." || echo "   ⚠️  No H5 model found. Please add efficientnet_model.h5."
else
    echo "   ⚠️  Models directory not found. Creating..."
    mkdir -p "$MODELS_DIR"
fi

# ── Create Data Directory ──────────────────────────────────────────
echo ""
echo "📁 Checking data directory..."
DATA_DIR="$PROJECT_DIR/data"
if [ ! -d "$DATA_DIR" ]; then
    mkdir -p "$DATA_DIR"
    echo "   Data directory created."
else
    echo "   Data directory found."
fi

# ── Run Application ────────────────────────────────────────────────
echo ""
echo "🚀 Starting Mushroom Classifier..."
echo "   Run the following command to start the app:"
echo ""
echo "   cd $PROJECT_DIR"
echo "   source $VENV_NAME/bin/activate  # On Windows: $VENV_NAME\\Scripts\\activate"
echo "   streamlit run app.py"
echo ""
echo "   Or simply run: streamlit run app.py"
echo ""

# ── Create .streamlit/secrets.toml (if not exists) ─────────────────
if [ ! -f "$PROJECT_DIR/.streamlit/secrets.toml" ]; then
    echo "🔐 Creating secrets template..."
    cat > "$PROJECT_DIR/.streamlit/secrets.toml" <<EOF
# Streamlit Secrets for Mushroom Classifier
# Add any sensitive keys here (API keys, database URLs, etc.)
# These are kept separate from the code for security.

[app]
title = "Mushroom Classifier"
version = "2.0.0"
environment = "production"
EOF
    echo "   ✅ secrets.toml template created."
fi

echo ""
echo "✅ Setup complete! Happy classifying! 🍄"
