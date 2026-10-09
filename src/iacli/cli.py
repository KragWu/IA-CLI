"""Point d'entrée de la commande iacli."""

import argparse
from pathlib import Path
import sys

from iacli.init import initialize


def handle_init(args: argparse.Namespace) -> None:
    """Gestionnaire de la sous-commande init."""
    initialize(config_dir=args.config_dir)


def build_parser() -> argparse.ArgumentParser:
    """Construit le parser d'arguments de la CLI."""
    parser = argparse.ArgumentParser(
        prog="iacli",
        description="CLI locale pour les IA open source"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    # Sous-commande : init
    init_parser = commands.add_parser(
        "init",
        help="Initialiser la configuration globale"
    )
    init_parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Chemin optionnel vers le dossier de configuration"
    )
    init_parser.set_defaults(func=handle_init)

    return parser


def main() -> int:
    """Exécute la CLI et retourne le code de statut système."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        args.func(args)
        return 0
    except KeyboardInterrupt:
        print("\n[iacli] Opération annulée par l'utilisateur.", file=sys.stderr)
        return 130
    except Exception as err:
        print(f"[iacli] Erreur : {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())