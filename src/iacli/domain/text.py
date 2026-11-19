"""Nettoyage du texte avant affichage dans un terminal."""

import unicodedata

_KEPT = {"\n", "\t"}

# Caractères invisibles ou qui réordonnent l'affichage (« Trojan Source », texte caché dans le
# bloc « tag ») : un diff ou une commande pourrait se lire autrement que ce qui s'exécute. On
# prend toute la catégorie Unicode « format » (Cf) et les séparateurs de ligne et de paragraphe,
# plus le joiner de graphèmes U+034F et les « fillers » hangul, qui s'affichent comme du blanc. Les liants U+200C et U+200D restent
# permis : les emojis composés et le persan en ont besoin.
_ALLOWED_FORMAT = {0x200C, 0x200D}
_FILLERS = {0x034F, 0x115F, 0x1160, 0x3164, 0xFFA0}


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
    if _is_deceptive(char):
        return f"\\u{code:04x}" if code <= 0xFFFF else f"\\U{code:08x}"
    return char


def _is_deceptive(char: str) -> bool:
    code = ord(char)
    if code in _FILLERS:
        return True
    category = unicodedata.category(char)
    return category in {"Zl", "Zp"} or (
        category == "Cf" and code not in _ALLOWED_FORMAT
    )
