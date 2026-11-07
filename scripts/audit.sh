#!/usr/bin/env bash
# Audit des vulnérabilités connues des dépendances du projet (réseau requis).
# Le projet est installé dans un venv jetable pour ne pas auditer le Python global.
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

"$PYTHON" -m venv "$work/venv"
if [[ -x "$work/venv/Scripts/python.exe" ]]; then
  venv_python="$work/venv/Scripts/python.exe"   # Windows (Git Bash)
else
  venv_python="$work/venv/bin/python"
fi

"$venv_python" -m pip install -q .
"$venv_python" -m pip freeze --exclude ia-cli > "$work/requirements-audit.txt"
"$venv_python" -m pip install -q pip-audit truststore
# truststore : utilise les certificats du système (utile derrière un proxy d'entreprise).
"$venv_python" -c "
import sys, truststore
truststore.inject_into_ssl()
from pip_audit._cli import audit
sys.argv = ['pip-audit', '-r', sys.argv[1], '--no-deps', '--disable-pip']
audit()
" "$work/requirements-audit.txt"
