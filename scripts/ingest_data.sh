#!/usr/bin/env bash
# Run data ingestion for AMPL Chatbot.
#
# Usage:
#   ./scripts/ingest_data.sh                     # ingest all data from ./data
#   ./scripts/ingest_data.sh inventory data/inventory.csv
#   ./scripts/ingest_data.sh faq data/faqs.json
#   ./scripts/ingest_data.sh defaults            # ingest built-in FAQs

set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a; source .env; set +a
fi

SOURCE="${1:-all}"
FILE="${2:-}"

case "$SOURCE" in
  all)
    echo "Ingesting all data from ./data ..."
    python -m ingest_ampl.main --source all --dir ./data
    ;;
  inventory)
    echo "Ingesting inventory: $FILE"
    python -m ingest_ampl.main --source inventory --file "$FILE"
    ;;
  faq)
    echo "Ingesting FAQs: $FILE"
    python -m ingest_ampl.main --source faq --file "$FILE"
    ;;
  sales)
    echo "Ingesting sales docs from: ${FILE:-./data/sales}"
    python -m ingest_ampl.main --source sales --dir "${FILE:-./data/sales}"
    ;;
  insurance)
    echo "Ingesting insurance docs: $FILE"
    python -m ingest_ampl.main --source insurance --file "$FILE"
    ;;
  defaults)
    echo "Ingesting default FAQs..."
    python -m ingest_ampl.main --source defaults
    ;;
  *)
    echo "Usage: $0 {all|inventory|faq|sales|insurance|defaults} [file]"
    exit 1
    ;;
esac

echo "Ingestion complete."
