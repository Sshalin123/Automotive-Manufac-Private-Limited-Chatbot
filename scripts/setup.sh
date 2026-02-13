#!/usr/bin/env bash
# Initial setup script for AMPL Chatbot.
#
# Creates virtual environment, installs dependencies, and prepares .env.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== AMPL Chatbot Setup ==="

# 1. Python virtual environment
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# 2. Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Create .env from example if missing
if [ ! -f ".env" ]; then
  echo "Creating .env from .env.example..."
  cp .env.example .env
  echo ">>> Edit .env with your API keys before running the app <<<"
fi

# 4. Create data directory
mkdir -p data

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys (Pinecone, OpenAI/AWS)"
echo "  2. Ingest data:    ./scripts/ingest_data.sh defaults"
echo "  3. Start the API:  ./scripts/run_api.sh --dev"
echo "  4. Open widget:    open frontend/widget/index.html"
echo ""
