# IA-CLI

IA-CLI (commande `iacli`) est une interface en ligne de commande pour discuter avec un modèle d'IA open source
**exécuté localement** via [Ollama](https://ollama.com/). Le modèle agit comme un assistant de programmation :
il peut lire, lister et rechercher les fichiers du projet courant, et proposer des modifications de fichiers ou des
commandes shell, qui ne sont exécutées qu'après votre confirmation.

Aucune donnée n'est envoyée à un service distant : seuls le téléchargement initial d'un modèle et la lecture de sa
taille interrogent le registre Ollama.

## Sommaire

- [Prérequis](#prérequis)
- [Installation](#installation)
- [Démarrage rapide](#démarrage-rapide)
- [Utiliser le chat](#utiliser-le-chat)
- [Configuration](#configuration)
- [Que fait `iacli init` ?](#que-fait-iacli-init-)
- [Limites et sécurité](#limites-et-sécurité)
- [Dépannage](#dépannage)
- [Développement](#développement)

## Prérequis

- Python 3.11 ou supérieur
- Git
- [ripgrep](https://github.com/BurntSushi/ripgrep) (`rg`)
- [Ollama](https://ollama.com/) installé et démarré (par défaut sur `http://localhost:11434`)
- Un modèle Ollama compatible avec les outils (*tool calling*) : `iacli chat` refuse de démarrer avec un modèle
  qui n'annonce pas cette capacité.

`iacli init` vérifie Python, Git et ripgrep et affiche une commande d'installation adaptée à votre système si
l'un d'eux manque.

## Installation

À la racine du dépôt, installez la commande dans l'environnement Python actif :

```bash
python -m pip install .
```

Les dépendances Python sont installées automatiquement :

- [`langgraph`](https://github.com/langchain-ai/langgraph) construit le graphe agent/outils ;
- [`langchain-core`](https://github.com/langchain-ai/langchain) fournit les messages et la définition des outils ;
- [`langchain-ollama`](https://github.com/langchain-ai/langchain) fournit `ChatOllama`, le client de chat qui
  parle à l'instance Ollama.

L'installation n'est pas faite en mode éditable : après une modification du code, relancez `python -m pip install .`.

## Démarrage rapide

```bash
iacli init     # une seule fois : crée la configuration et télécharge le modèle
cd mon-projet
iacli          # équivalent de `iacli chat`
```

`iacli init` peut être relancé sans risque : il ne remplace jamais les fichiers de configuration existants.

## Utiliser le chat

Lancez `iacli` (ou `iacli chat`) **depuis le dossier du projet** sur lequel vous voulez travailler : ce dossier est
le périmètre de travail de l'agent.

```text
IA-CLI — modèle qwen2.5-coder:7b — projet /chemin/vers/mon-projet
Entrez /exit ou /quit pour terminer.
vous> Explique le fichier main.py
```

### Quitter

- `/exit` ou `/quit` (insensible à la casse) ferme la conversation.
- `Ctrl+D` (fin de fichier) ou `Ctrl+C` **à l'invite `vous>`** ferme aussi la conversation.
- `Ctrl+C` **pendant le traitement d'une demande** annule seulement ce tour (et arrête la commande shell en cours) :
  la conversation reste ouverte.

L'historique de la conversation est conservé en mémoire pendant la session uniquement ; il n'est pas enregistré sur le
disque.

### Les outils de l'agent

| Outil          | Rôle                                                       | Confirmation |
|----------------|------------------------------------------------------------|--------------|
| `read_file`    | Lire un fichier texte (UTF-8) du projet                    | non          |
| `list_files`   | Lister les fichiers du projet ou d'un sous-dossier         | non          |
| `search_files` | Rechercher un texte, sans distinction de casse             | non          |
| `write_file`   | Créer ou remplacer un fichier                              | **oui** (diff affiché) |
| `run_shell`    | Exécuter une commande shell dans le dossier du projet      | **oui** (commande affichée) |

Les demandes de confirmation attendent `o`, `oui`, `y` ou `yes` ; toute autre réponse (y compris une réponse vide)
refuse l'action et le modèle en est informé.

Détails de fonctionnement :

- **Dossiers ignorés** : `list_files`, `search_files` et la résolution d'un nom de fichier ignorent `.git`,
  `.pytest_cache`, `.venv`, `venv`, `__pycache__`, `build`, `dist` et `node_modules`, ainsi que les liens symboliques.
- **Nom de fichier seul** : si `read_file` ne trouve pas le chemin demandé mais qu'un seul fichier du projet porte ce nom
  (sans tenir compte de la casse), il le lit et préfixe le résultat par
  `[Chemin demandé introuvable : … ; fichier lu : …]` pour que le modèle sache quel fichier a réellement été lu.
  Un nom ambigu produit une erreur demandant un chemin plus précis ; un chemin qui désigne un dossier est refusé
  (utiliser `list_files`).
- **Limites de taille** : `read_file`, `search_files` et `write_file` ignorent ou refusent les fichiers de plus de
  1 000 000 octets. Les sorties d'outils sont tronquées à 30 000 caractères. `list_files` s'arrête à 200 fichiers
  (`[liste limitée à 200 fichiers]`) et `search_files` à 100 correspondances
  (`[résultats limités à 100 correspondances]`) : ce sont des indications de troncature, pas des erreurs.
  `search_files` signale aussi le nombre de fichiers ignorés car trop volumineux ou non UTF-8.
- **Écriture** : un diff unifié est affiché avant confirmation. Les fins de ligne du fichier existant (LF ou CRLF)
  sont conservées ; si le fichier mélange les deux, un avertissement est affiché et tout est converti en CRLF.
  Une modification dont le diff dépasse 30 000 caractères est refusée : le modèle doit la découper.
- **Commande shell** : exécutée avec le shell du système (`cmd.exe` sous Windows) dans le dossier du projet, sans accès
  à l'entrée du chat, avec un délai maximal de 600 secondes (l'arbre de processus est alors arrêté). La réponse
  commence par `Code de sortie : N`. La sortie standard et la sortie d'erreur sont tronquées séparément
  (`[sortie tronquée]`) afin qu'une longue sortie n'efface pas l'erreur.
- **Appel d'outil écrit en JSON dans le texte** : certains petits modèles renvoient `{"name":"outil","arguments":{...}}`
  dans leur réponse (éventuellement plusieurs objets, ou dans un bloc ```` ```json ````) au lieu d'utiliser le mécanisme
  de *tool calling*. IA-CLI reconnaît ce format pour les outils connus et le traite comme un véritable appel d'outil.
- **Limite d'exécutions** : un tour s'arrête avec une erreur après 25 exécutions d'outils consécutives.

### Ce qui s'affiche pendant un tour

- Un indicateur animé avec la durée écoulée (uniquement dans un terminal interactif ; sur `stderr`), le nom des outils
  appelés et la réception de leurs résultats. L'affichage montre l'avancement sans révéler le raisonnement interne
  détaillé du modèle.
- Après chaque tour, le temps de chaque appel au modèle et de chaque outil, puis le temps total (qui inclut le temps
  passé à confirmer une action). Les durées sont affichées en ms, s, min ou h selon leur valeur.
- Quand Ollama renvoie le nombre de tokens consommés, l'occupation de la fenêtre de contexte, calculée sur le dernier
  appel au modèle : `contexte> 4 500 / 18 000 tokens (25 %)`. Une alerte s'ajoute à partir de 80 % : au-delà de la
  limite, Ollama peut tronquer les plus anciens messages de la conversation.

## Configuration

Les fichiers se trouvent par défaut dans `~/.config/iacli/` (aussi sous Windows : `C:\Users\<vous>\.config\iacli\`).
Ils sont créés au premier lancement de `iacli`, `iacli chat` ou `iacli init`, sans jamais remplacer un fichier existant.

Pour utiliser un autre dossier, ajoutez l'option `--config-dir` à n'importe quelle commande :

```bash
iacli --config-dir ./ma-config
iacli chat --config-dir ./ma-config
iacli init --config-dir ./ma-config
```

### `config.toml`

```toml
[ollama]
base_url = "http://localhost:11434"
registry_url = "https://registry.ollama.ai"
model = "qwen2.5-coder:7b"
num_ctx = 18000
```

| Clé            | Rôle                                                                                           |
|----------------|------------------------------------------------------------------------------------------------|
| `base_url`     | URL de l'instance Ollama.                                                                      |
| `registry_url` | URL du registre des modèles, utilisée uniquement par `iacli init` pour connaître la taille d'un modèle à télécharger. |
| `model`        | Modèle utilisé par le chat (doit gérer les outils).                                            |
| `num_ctx`      | Fenêtre de contexte demandée à Ollama, en tokens (entier strictement positif).                 |

- Les clés absentes prennent la valeur par défaut ci-dessus ; les nouvelles valeurs sont lues à chaque lancement.
- Une valeur plus grande de `num_ctx` consomme davantage de mémoire. Si elle dépasse le contexte maximal annoncé par
  Ollama pour le modèle, `iacli chat` affiche un avertissement et utilise ce maximum.
- Si le fichier est illisible ou invalide (TOML incorrect, valeur vide, mauvais type), `iacli init` avertit et
  utilise les valeurs par défaut, tandis que `iacli chat` s'arrête avec un message d'erreur : corrigez le fichier.
- La température du modèle est fixée à 0 (réponses les plus déterministes possible) et n'est pas configurable.

### `IACLI.md`

Fichier éditable qui contient vos instructions globales (langue, style, conventions…). Son contenu est ajouté à
l'invite système à chaque lancement d'une conversation ; il s'applique donc à tous vos projets. Il doit être encodé en
UTF-8 (un BOM, comme en ajoute le Bloc-notes, est accepté).

## Que fait `iacli init` ?

La commande déroule quatre étapes et affiche leur progression :

1. **Prérequis** : vérifie Python, Git et ripgrep. S'il en manque un, l'avertissement et le conseil d'installation sont
   affichés et la commande s'arrête **avant** de créer la configuration.
2. **Configuration** : crée `config.toml` et `IACLI.md` s'ils sont absents.
3. **Connexion** : vérifie qu'Ollama répond sur `base_url`. S'il est inaccessible, un avertissement est affiché, la
   suite de la vérification du modèle est ignorée, mais la configuration reste créée.
4. **Modèle** : si le modèle configuré est absent localement, vérifie qu'il existe dans le registre, lit la taille de ses
   couches dans son manifeste et la compare à l'espace libre de l'emplacement des modèles Ollama (variable
   d'environnement `OLLAMA_MODELS`, sinon `~/.ollama/models`). Il demande ensuite confirmation (`[O/n]`, oui par défaut)
   avant de le télécharger en affichant la progression.

Un modèle introuvable, un espace disque insuffisant, un échec de téléchargement ou Ollama inaccessible
produisent un avertissement, pas une erreur fatale.

## Limites et sécurité

- Les **chemins** des outils de fichiers (`read_file`, `list_files`, `search_files`, `write_file`) sont confinés au
  dossier courant : un chemin qui en sort, y compris via `..` ou un lien symbolique, est refusé.
- **`run_shell` n'est pas confiné** : la commande démarre dans le dossier du projet mais peut accéder à tout ce que votre
  compte peut atteindre. Lisez attentivement chaque commande avant de confirmer.
- Le modèle est local, mais ses réponses peuvent être fausses : relisez les diffs avant d'accepter une modification.
- Seuls les fichiers texte UTF-8 sont lus et modifiés.

## Dépannage

| Message                                                       | Cause et solution                                                                 |
|---------------------------------------------------------------|-----------------------------------------------------------------------------------|
| `Le modèle … n'est pas installé`                              | Lancez `iacli init`, ou installez le modèle avec `ollama pull <modèle>`.           |
| `Le modèle … ne gère pas les outils`                          | Choisissez un autre modèle dans `config.toml`, puis lancez `iacli init`.          |
| `Configuration illisible`                                     | Corrigez `config.toml` (voir le détail dans le message) ou supprimez-le.          |
| `… n'est pas encodé en UTF-8` (`IACLI.md`)                    | Réenregistrez le fichier en UTF-8.                                                |
| `Erreur : <urlopen error …>` (chat) ou `Ollama n'est pas accessible` (init) | Démarrez Ollama et vérifiez `base_url` dans `config.toml`. |
| `Limite de 25 exécutions d'outils atteinte.`                  | Le modèle boucle sur ses outils : reformulez ou précisez la demande.              |
| `Le modèle a renvoyé une réponse vide.`                       | Reformulez la demande ; essayez un modèle plus capable.                           |

## Développement

### Tests

```bash
python -m pip install ".[dev]"                   # pytest + ruff + mypy
python -m ruff check .                             # lint
python -m ruff format .                            # formatage
python -m mypy                                     # typage
python -m pytest                                   # toute la suite
python -m pytest tests/test_chat.py::nom_du_test   # un seul test
```

Les tests utilisent des fakes (`tests/fakes.py`, `tests/conftest.py`) plutôt que la vraie CLI ou Ollama. Le projet est
développé en TDD : tout comportement nouveau ou corrigé commence par un test qui échoue.

### Architecture

Le projet applique une architecture hexagonale (ports et adaptateurs) sous `src/iacli/`. Les dépendances pointent
toujours vers l'intérieur :

```text
cli.py (composition) ─► adapters ───────┐
                    └─► infrastructure ─┴─► services ─► domain
                                            (ports = Protocol)
```

`adapters` et `infrastructure` implémentent les ports définis par `services` ; `services` et `domain` ne les importent jamais.

- `domain/` contient les objets de configuration (`OllamaSettings`, `ConfigurationState`), les informations de modèle
  (`ModelInfo`) et les événements métier (`InitializationEvent`, `DownloadProgress`).
- `services/` contient les cas d'usage (`initialize.py`, `chat.py`), le graphe d'agent (`agent_graph.py`) et leurs
  contrats (`ports.py`, des `Protocol`). Un service ne connaît que les ports, ce qui permet de tester ses décisions
  avec des fakes sans exécuter la CLI ni dépendre des textes affichés.
- `infrastructure/` fournit les accès au système de fichiers (`configuration.py`), aux ressources système
  (`system.py`), à Ollama (`ollama.py`) et les outils du projet (`workspace_tools.py`).
- `adapters/` traduit les événements applicatifs en sortie console (`cli_presenter.py`) et porte l'interaction de chat
  (`cli_chat.py` : invite, confirmations, indicateur animé, durées).
- `cli.py` est la racine de composition : c'est le **seul** endroit qui câble infrastructure, adapters et services
  (`handle_init`, `handle_chat`). `iacli/ollama.py` n'est qu'un shim de compatibilité vers `infrastructure/ollama.py`.

`tests/test_architecture.py` vérifie qu'aucun module de `services/` n'importe `adapters` ou `infrastructure`.

### Déroulé d'un `iacli chat`

1. `handle_chat` charge la configuration, vérifie que le modèle est installé et gère les outils, plafonne `num_ctx`
   au contexte maximal du modèle et lit `IACLI.md`.
2. Il crée un `ChatOllama` (température 0) et appelle `run_chat`.
3. `run_chat` construit l'invite système (description des outils + instructions globales), instancie les outils et le
   graphe, puis boucle sur l'invite `vous>` ; chaque demande est un « tour » exécuté par le graphe.

### Graphe agent/outils

`iacli.services.agent_graph.build_control_graph` construit un graphe LangGraph dont le modèle (compatible
`bind_tools`) et l'évaluateur sont injectés par l'appelant :

```text
START ─► agent ──(appel d'outil)──► action ─┐
          ▲                                  │
          └──────────────────────────────────┘
          │
          └─(réponse sans outil, ou limite atteinte)─► evaluator ─► END
```

L'agent route les demandes d'outils vers leur exécution, puis réinjecte les `ToolMessage` au modèle. Une réponse sans
outil passe par l'évaluateur (`evaluate_response` dans `chat.py` : refuse une réponse vide ou qui garde des appels
d'outils en attente) avant la fin du graphe. La limite par défaut est de 25 exécutions d'outils et peut être ajustée avec
`max_iterations` ; au-delà, l'évaluateur lève `MaxToolIterationsError`.

Quand un tour échoue (erreur, `Ctrl+C`), `run_chat` retire de l'historique la fin incomplète du tour (appel d'outil sans
résultat, réponse vide) pour que la conversation reste valide et puisse continuer.
