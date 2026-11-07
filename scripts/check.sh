#!/usr/bin/env bash
# Vérifications locales, équivalent de la CI : lint, format, typage,
# contrats d'architecture, tests et couverture.
# Usage : scripts/check.sh [--audit]
#   --audit  ajoute l'audit des dépendances (réseau requis, voir scripts/audit.sh)
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"

step() { printf '\n==> %s\n' "$1"; }

step "ruff check"
"$PYTHON" -m ruff check .

step "ruff format --check"
"$PYTHON" -m ruff format --check .

step "mypy"
"$PYTHON" -m mypy

step "import-linter (contrats d'architecture)"
lint-imports

step "pytest (avec couverture)"
"$PYTHON" -m pytest --cov

if [[ "${1:-}" == "--audit" ]]; then
  step "audit des dépendances"
  scripts/audit.sh
fi

printf '\nToutes les vérifications sont passées.\n'
