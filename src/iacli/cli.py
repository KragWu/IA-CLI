"""Point d'entrée de la commande iacli."""

import argparse

from iacli.init import initialize


def main() -> None:
    parser = argparse.ArgumentParser(prog="iacli", description="CLI locale pour les IA open source")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Initialiser la configuration globale")
    args = parser.parse_args()

    if args.command == "init":
        initialize()


if __name__ == "__main__":
    main()