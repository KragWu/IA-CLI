# IA-CLI

IA-CLI est une interface en ligne de commande pour utiliser des modèles d'IA open source en local avec Ollama.

## Prérequis

- Python 3.11 ou supérieur
- Git
- [ripgrep](https://github.com/BurntSushi/ripgrep) (`rg`)
- [Ollama](https://ollama.com/) installé et démarré sur `http://localhost:11434`

## Installation depuis le dépôt

À la racine du dépôt, installez la commande dans l'environnement Python actif :

```bash
python -m pip install .
```

Puis lancez le wizard :

```bash
iacli init
```

La commande vérifie Python, Git, ripgrep et la connexion à Ollama. Elle crée la configuration globale dans `~/.config/iacli/`, sans remplacer les fichiers déjà présents :

- `config.toml` : source des paramètres utilisés par le wizard : URL de l'instance Ollama, URL du registre des modèles et modèle par défaut (`qwen2.5-coder:7b`). Modifiez ce fichier pour cibler une autre instance ou choisir un autre modèle ; les nouvelles valeurs sont utilisées aux prochains lancements.
- `IACLI.md` : fichier éditable pour conserver les instructions globales propres à votre usage et à vos projets. `iacli init` le crée comme point de départ et ne remplace jamais vos modifications. Le wizard actuel le prépare, mais ne l'injecte pas encore dans une conversation IA.

Si le modèle est absent localement, le wizard vérifie qu'il existe dans le registre configuré et récupère la taille de ses couches depuis son manifeste. Il compare cette taille à l'espace disponible dans l'emplacement des modèles Ollama, puis demande confirmation avant le téléchargement. Définissez `OLLAMA_MODELS` si vos modèles sont stockés dans un emplacement personnalisé. En cas de modèle introuvable, de dépendance manquante, d'espace insuffisant ou d'Ollama inaccessible, un avertissement est affiché.

## Tests

```bash
python -m pip install pytest
python -m pytest
```

## Architecture

Le code est organisé par responsabilités :

- `domain/` contient les objets de configuration et les événements métier.
- `services/` contient le cas d'usage d'initialisation et ses contrats (`Protocol`).
- `infrastructure/` fournit les accès au système de fichiers, aux ressources système et à Ollama.
- `adapters/` traduit les événements applicatifs en sortie console et porte les interactions utilisateur.

Le cas d'usage dépend des contrats, ce qui permet de tester ses décisions métier avec des fakes sans exécuter
la CLI ni dépendre des textes affichés.

## Graphe agent/outils

`iacli.services.agent_graph.build_control_graph` construit un graphe LangGraph dont le modèle compatible
`bind_tools` et l'évaluateur sont injectés par l'appelant. L'agent route les demandes d'outils vers leur
exécution, puis réinjecte les `ToolMessage` au modèle. Une réponse sans outil passe par l'évaluateur avant
la fin du graphe. La limite par défaut est de 25 exécutions d'outils et peut être ajustée avec
`max_iterations`.
