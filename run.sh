#!/usr/bin/env bash
# Open Epistemic Engine - run script. Single local LLM, no external web RAG.
#
# Provider resolution lives in oee/config.py: if OEE_PROVIDER is unset, the
# first local endpoint that answers is wired in - OpenAI-compatible
# (OEE_PROBE_OPENAI, default http://127.0.0.1:8082) then Ollama
# (OEE_PROBE_OLLAMA, default http://127.0.0.1:11434). If neither answers,
# startup fails: the engine has no offline stand-in and refuses to run
# without a real local model.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export OEE_DB="${OEE_DB:-data/oee.db}"
export OEE_PORT="${OEE_PORT:-8080}"

if [ "${1:-}" = "eval" ]; then
  shift
  exec python3 evals/harness.py "$@"
fi

python3 -c "from oee.config import load_settings; s=load_settings(); \
print('wiring:', s.wiring); print('provider:', s.provider, '| model:', s.model, \
'| base_url:', s.base_url, '| window:', s.context_window, '| embed_model:', s.embed_model); \
[print('config:', p) for p in s.validate()]"

exec python3 -m uvicorn oee.server:app --host "${OEE_HOST:-127.0.0.1}" --port "${OEE_PORT}" \
  --log-level "${OEE_LOG_LEVEL:-info}"
