import ast
from pathlib import Path

SERVICES = Path(__file__).parents[1] / "src" / "iacli" / "services"
FORBIDDEN = ("iacli.adapters", "iacli.infrastructure")


def _imports(path: Path) -> list[str]:
    names = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_services_do_not_depend_on_adapters_or_infrastructure():
    violations = [
        f"{path.name} importe {name}"
        for path in SERVICES.glob("*.py")
        for name in _imports(path)
        if name.startswith(FORBIDDEN)
    ]

    assert violations == []
