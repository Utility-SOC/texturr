#!/usr/bin/env bash
# Import the texturr workflows into the running n8n container and create the credentials that
# need no secret of yours: the texturr token (from .env) and a local Ollama connection.
# You still add your own key in the n8n UI if you choose a hosted chat model.
set -euo pipefail
cd "$(dirname "$0")/.."

token=$(grep -E '^TEXTURR_TOKEN=' .env | head -n1 | cut -d= -f2-)
if ! [[ "$token" =~ ^[A-Za-z0-9._~+-]{16,}$ ]]; then
  echo "TEXTURR_TOKEN in .env must be 16+ characters from A-Z a-z 0-9 . _ ~ + -" >&2
  exit 1
fi

creds=n8n/data/.credentials.json
trap 'rm -f "$creds"' EXIT        # the token only touches disk for the duration of the import
cat > "$creds" <<JSON
[
  {"id": "texturr-token", "name": "texturr token", "type": "httpHeaderAuth",
   "data": {"name": "Authorization", "value": "Bearer ${token}"}},
  {"id": "ollama-credential", "name": "Ollama (local)", "type": "ollamaApi",
   "data": {"baseUrl": "http://ollama:11434"}}
]
JSON

docker compose exec -T n8n n8n import:credentials --input=/data/.credentials.json
docker compose exec -T n8n n8n import:workflow --separate --input=/workflows
echo
echo "Done. Open http://localhost:5678, create the n8n owner account, then open the workflows."
echo "Pick a chat model that supports tool calling (e.g. llama3.1 in Ollama) in the 'Chat model' node."
