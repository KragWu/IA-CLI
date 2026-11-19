"""Nettoyage du texte avant affichage dans un terminal."""

_KEPT = {"\n", "\t"}

# Caractères invisibles ou qui réordonnent l'affichage (« Trojan Source ») : un diff ou une
# commande pourrait se lire autrement que ce qui s'exécute. Les liants U+200C et U+200D restent
# permis : les emojis composés et le persan en ont besoin.
_DECEPTIVE = (
    {0x061C, 0x200B, 0x200E, 0x200F, 0x2028, 0x2029, 0x2060, 0xFEFF}
    | set(range(0x202A, 0x202F))
    | set(range(0x2066, 0x206A))
)


def neutralize_control_characters(text: str) -> str:
    """Remplace les caractères de contrôle par une forme visible (`\\x1b`, `\\u202e`).

    Un texte venu du modèle, d'un fichier ou d'une commande ne doit pas pouvoir piloter le
    terminal (séquences d'échappement : titre de fenêtre, presse-papiers, retour chariot qui
    réécrit la ligne) ni tromper la lecture (inversion du sens d'écriture, caractères de
    largeur nulle). Le saut de ligne et la tabulation sont conservés, et une fin de ligne
    CRLF (fichiers Windows) devient un simple saut de ligne ; seul un `\r` isolé est échappé."""
    text = text.replace(chr(13) + chr(10), chr(10))
    return "".join(_visible(char) for char in text)


def _visible(char: str) -> str:
    code = ord(char)
    if char in _KEPT:
        return char
    if code < 0x20 or 0x7F <= code <= 0x9F:
        return f"\\x{code:02x}"
    if code in _DECEPTIVE:
        return f"\\u{code:04x}"
    return char
