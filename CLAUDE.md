# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Projet

IA-CLI (`iacli`) est une CLI Python (>= 3.11) pour discuter avec un modèle open source local via Ollama. Le README et les messages utilisateur sont en français ; garder le français pour les textes affichés et la documentation.

## Méthode de travail

- **TDD (Test Driven Development)** : le projet est développé en TDD. Écrire d'abord un test qui échoue, puis le code minimal pour le faire passer, puis refactorer. Toute nouvelle fonctionnalité ou correction de bug commence par un test.
- **Architecture hexagonale** : le projet vise à appliquer strictement ce style (ports et adaptateurs). Le domaine et les services ne dépendent jamais de l'infrastructure ni des adapters ; ils ne connaissent que les ports (`Protocol` dans `services/ports.py`). Les dépendances pointent toujours vers l'intérieur, et le câblage ne se fait que dans `cli.py`.

## Commandes

```bash
python -m pip install .            # installe la commande `iacli` (pas de mode editable documenté)
python -m pip install ".[dev]"     # pytest, pytest-cov, ruff, mypy, import-linter, pre-commit
scripts/check.sh                   # TOUTES les vérifications locales (ruff, format, mypy, lint-imports, pytest --cov)
scripts/check.sh --audit           # idem + audit des vulnérabilités des dépendances (réseau requis)
scripts/audit.sh                   # audit seul : venv jetable, pip-audit sur les seules dépendances du projet
python -m ruff check .             # lint (ajouter --fix pour les corrections sûres)
python -m ruff format .            # formatage
python -m mypy                     # typage (src) ; check.sh le lance aussi avec --platform linux et win32
lint-imports                       # contrats d'architecture hexagonale (import-linter)
python -m pytest --cov             # couverture (seuil 80 %)
pre-commit install --install-hooks && pre-commit install --hook-type pre-push   # hooks git (ruff + mypy au commit, pytest au push)
python -m pytest                   # toute la suite (pythonpath = src via pyproject.toml)
python -m pytest tests/test_chat.py::nom_du_test   # un seul test
iacli init                         # wizard : crée ~/.config/iacli/{config.toml,IACLI.md}, télécharge le modèle
iacli / iacli chat                 # conversation (chat est la commande par défaut)
```

Ruff (lint + format), mypy, import-linter et coverage sont configurés dans `pyproject.toml` ; il n'y a pas de build configuré. Les scripts `scripts/*.sh` se lancent avec Bash (Git Bash sous Windows). **Lancer `scripts/check.sh` avant de commiter.** `scripts/audit.sh` installe le projet dans un venv jetable pour auditer uniquement ses dépendances (auditer le Python global remonterait des paquets sans rapport). La CI GitHub (`.github/workflows/ci.yml`) est volontairement en déclenchement manuel (`workflow_dispatch`) car les minutes sont facturées ; Dependabot reste actif. Prérequis runtime : Git, ripgrep (`rg`) et Ollama sur `http://localhost:11434`.

## Vérification sous Linux (WSL)

Le projet est développé sous Windows mais doit rester portable (`killpg`, chemins, processus de `run_shell`). Pour lancer la suite sous Linux sans CI, utiliser une distro WSL (`wsl -l -q` pour les lister ; sur cette machine : `podman-machine-default`, Fedora 40 avec Python 3.12, sans pip ni git, utilisateur non root). Depuis Git Bash, à la racine du dépôt :

```bash
DISTRO=podman-machine-default

# 1. Copie propre du dépôt (état du dernier commit) dans le /tmp de la distro
git archive HEAD | wsl -d $DISTRO -- sh -c 'rm -rf /tmp/iacli && mkdir /tmp/iacli && tar -x -C /tmp/iacli'

# 2. Venv + dépendances (le Python système suffit, aucun droit root requis)
wsl -d $DISTRO -- sh -c 'python3 -m venv /tmp/iacli-venv && cd /tmp/iacli && /tmp/iacli-venv/bin/pip install -q ".[dev]"'

# 3. Toutes les vérifications, via un script passé sur l'entrée standard
cat <<'EOF' | wsl -d $DISTRO -- bash -s
cd /tmp/iacli
export PATH=/tmp/iacli-venv/bin:/usr/bin:/bin
PYTHON=python bash scripts/check.sh
EOF

# 4. Nettoyage
wsl -d $DISTRO -- sh -c 'rm -rf /tmp/iacli /tmp/iacli-venv'
```

Points à connaître :

- `git archive HEAD` ne copie que ce qui est **commité** : commiter (ou stasher) avant de tester. Cela évite aussi de polluer l'arbre Windows (`__pycache__`, `.mypy_cache`) et les problèmes de fins de ligne CRLF.
- Ne jamais écrire `PATH=...:$PATH` dans une commande `wsl -- sh -c '...'` lancée depuis Git Bash : `$PATH` est développé côté Windows (avec des parenthèses qui cassent le shell). D'où le script passé en entrée standard avec un heredoc entre quotes (`<<'EOF'`).
- Résultat attendu : tout passe, avec 1 test ignoré (`tests/test_workspace_tools.py`, page de code OEM propre à Windows) et une couverture d'environ 84 %.
- `scripts/check.sh` lance mypy avec `--platform linux` puis `--platform win32` : les `# type: ignore[attr-defined]` dans `workspace_tools.py` couvrent les API propres à une plateforme (`killpg`, `CREATE_NEW_PROCESS_GROUP`, `windll`).

## Architecture

Architecture en couches (ports et adaptateurs) sous `src/iacli/` :

- `domain/` : dataclasses de configuration et événements (`InitializationEvent`, `DownloadProgress`…).
- `services/` : cas d'usage (`initialize.py`, `chat.py`), contrats `Protocol` dans `ports.py`, et le graphe d'agent (`agent_graph.py`). Les services ne dépendent que des ports.
- `infrastructure/` : implémentations concrètes (client HTTP Ollama, config TOML, vérifications système, `workspace_tools.py` pour les outils lecture/liste/recherche/écriture/édition ciblée/shell).
- `adapters/` : sortie console (`cli_presenter.py`) et interaction de chat (`cli_chat.py`, confirmations, indicateur animé, minuteurs).
- `cli.py` : racine de composition ; seul endroit qui câble infrastructure, adapters et services (`OllamaClientFactory`, `handle_init`, `handle_chat`). `iacli/ollama.py` n'est qu'un shim de compatibilité vers `infrastructure/ollama.py`.

Flux du chat : `handle_chat` vérifie la config et que le modèle est installé, lit `IACLI.md` (instructions globales injectées dans le contexte), crée un `ChatOllama` (temperature 0) puis appelle `run_chat`. `services/agent_graph.build_control_graph` construit un graphe LangGraph : le modèle (injecté, compatible `bind_tools`) route vers l'exécution des outils, les `ToolMessage` sont réinjectés, et une réponse sans outil passe par un évaluateur avant la fin. Limite par défaut de 25 exécutions d'outils (`max_iterations`).

## Comportements à connaître

- Les chemins des outils de fichiers sont confinés au projet courant (`Path.cwd()`) ; les écritures et commandes shell exigent une confirmation utilisateur avec affichage du diff/de la commande. `run_shell` n'est pas isolé : seul son dossier de départ est le projet (délai max 600 s).
- `iacli init` s'arrête avant de créer la configuration si Python/Git/ripgrep manque ; les autres problèmes (Ollama injoignable, modèle introuvable, disque insuffisant) sont des avertissements. `iacli chat` refuse au contraire un `config.toml` illisible.
- `--config-dir` est accepté sur `iacli`, `iacli chat` et `iacli init` (défaut `~/.config/iacli`, y compris sous Windows).
- Le README est la documentation utilisateur de référence : le mettre à jour avec tout changement de comportement visible (outils, limites, messages, configuration).
- Un appel d'outil renvoyé en JSON dans le texte du modèle (`{"name":..., "arguments":...}`, éventuellement plusieurs objets) est reconnu pour les outils connus et converti en vrai tool call. Un nom de fichier seul est résolu s'il est unique dans le projet, sinon erreur d'ambiguïté.
- `edit_file(path, old_string, new_string)` remplace un bloc exact (une seule occurrence exigée, sinon erreur) dans un fichier existant, avec diff et confirmation ; `old_string` (en LF) est cherché dans le texte brut par `_find_block` (chaque `\n` vaut LF ou CRLF), donc le reste du fichier, même à fins de ligne mixtes, est conservé octet pour octet et seul le bloc remplacé prend la fin de ligne majoritaire. `write_file` crée ou réécrit un fichier entier (sur un fichier mixte, réécrire le même texte le normalise vers la fin de ligne majoritaire après confirmation ; à égalité ou sans fin de ligne, valeur par défaut) ; les deux partagent `confirm_and_save`, qui n'écrit pas si le fichier a changé pendant la confirmation (empreinte SHA-256 comparée avant l'écriture).
- La fin de ligne par défaut (nouveaux fichiers, égalité, fichier sans fin de ligne) vient de `[files] end_of_line` de `config.toml` (`auto` = `os.linesep`, `lf`, `crlf`) : résolue par `resolve_newline` (domaine) dans `cli.py`, puis injectée via `partial(create_workspace_tools, default_newline=...)` ; le port `ToolsFactory` des services ne change pas. Exception, `_default_newline_for` : les scripts shell (`.sh`, `.bash`, `.zsh`) et les `Makefile`/`GNUmakefile` ont toujours LF pour défaut, et les `.bat`/`.cmd` toujours CRLF, quel que soit le réglage (un fichier existant garde sa fin de ligne majoritaire).
- Lecture/recherche ignorent les fichiers > 1 000 000 octets ; sorties d'outils tronquées à 30 000 caractères ; recherche limitée à 100 correspondances (troncature signalée, pas une erreur).
- Les durées (modèle, outils, total incluant le temps de confirmation) sont affichées après chaque tour.
- `OLLAMA_MODELS` définit l'emplacement des modèles pour le calcul d'espace disque.

## Tests

Les tests utilisent des fakes (`tests/fakes.py`, `tests/conftest.py` avec `fake_urlopen` pour le HTTP) plutôt que la vraie CLI ou Ollama. Pour tester un service, injecter des fakes de ports au lieu de patcher l'infrastructure.

## Conventions de commit

Les messages de commit commencent par un gitmoji (ex. `:bug: Fix timer display`, `:wheelchair: Add timers on processes`, `:tada:`).
