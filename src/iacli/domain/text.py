"""Nettoyage du texte avant affichage dans un terminal."""

_KEPT = {"\n", "\t"}


def neutralize_control_characters(text: str) -> str:
    """Remplace les caractères de contrôle par une forme visible (`\\x1b`).

    Un texte venu du modèle, d'un fichier ou d'une commande ne doit pas pouvoir piloter le
    terminal (séquences d'échappement : titre de fenêtre, presse-papiers, retour chariot qui
    réécrit la ligne). Le saut de ligne et la tabulation sont conservés, et une fin de ligne
    CRLF (fichiers Windows) devient un simple saut de ligne ; seul un `\r` isolé est échappé."""
    text = text.replace(chr(13) + chr(10), chr(10))
    return "".join(
        f"\\x{ord(char):02x}" if _is_control(char) else char for char in text
    )


def _is_control(char: str) -> bool:
    code = ord(char)
    return char not in _KEPT and (code < 0x20 or 0x7F <= code <= 0x9F)
