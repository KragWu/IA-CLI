"""Garde-fous de l'architecture hexagonale : les dépendances pointent vers l'intérieur.

    cli.py ─► adapters ────────┐
          └─► infrastructure ──┴─► services (ports) ─► domain

Règles vérifiées :
- `domain` n'utilise que la bibliothèque standard et lui-même ;
- `services` ne connaît ni les adapters, ni l'infrastructure, ni `cli` ;
- `infrastructure` et `adapters` ne dépendent ni l'un de l'autre, ni de `cli` ;
- seul `cli.py` (racine de composition) câble les adapters et l'infrastructure ;
  le shim `ollama.py` n'a le droit que de ré-exporter l'infrastructure.
"""

import ast
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parents[1] / "src" / "iacli"
LAYERS = ("domain", "services", "infrastructure", "adapters")
# Modules racine autorisés à câbler les couches externes.
COMPOSITION_ROOT = "cli.py"
COMPATIBILITY_SHIMS = {"ollama.py": {"iacli.infrastructure"}}


def imported_modules(source: str, module: str) -> set[str]:
    """Noms absolus des modules importés par `source`, imports relatifs résolus.

    `module` est le nom pointé du module analysé (ex. `iacli.services.chat`).
    Les imports dans les fonctions sont inclus. `from a import b` compte `a`
    et `a.b`, car `b` peut être un sous-module.
    """
    package = module.rsplit(".", 1)[0]
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1])
                target = f"{base}.{node.module}" if node.module else base
            else:
                target = node.module or ""
            names.add(target)
            names.update(f"{target}.{alias.name}" for alias in node.names)
    return names


def is_inside(name: str, package: str) -> bool:
    return name == package or name.startswith(package + ".")


def module_files(layer: str) -> list[tuple[Path, str]]:
    return [
        (path, ".".join(("iacli", layer, path.stem)))
        for path in sorted((PACKAGE / layer).glob("*.py"))
    ]


def imports_of(layer: str) -> list[tuple[str, str]]:
    """Paires (module analysé, module importé) pour toute une couche."""
    return [
        (module, name)
        for path, module in module_files(layer)
        for name in imported_modules(path.read_text(encoding="utf-8"), module)
    ]


def violations(layer: str, forbidden: tuple[str, ...]) -> list[str]:
    return [
        f"{module} importe {name}"
        for module, name in imports_of(layer)
        if any(is_inside(name, package) for package in forbidden)
    ]


# --- Le détecteur lui-même -------------------------------------------------


def test_detector_sees_absolute_relative_and_function_level_imports():
    source = (
        "import json\n"
        "from iacli.adapters import cli_chat\n"
        "from ..infrastructure.ollama import OllamaClient\n"
        "from . import ports\n"
        "def lazy():\n"
        "    import iacli.cli\n"
    )

    names = imported_modules(source, "iacli.services.chat")

    assert {
        "json",
        "iacli.adapters",
        "iacli.adapters.cli_chat",
        "iacli.infrastructure.ollama",
        "iacli.services",
        "iacli.services.ports",
        "iacli.cli",
    } <= names


def test_is_inside_matches_whole_packages_only():
    assert is_inside("iacli.adapters.cli_chat", "iacli.adapters")
    assert is_inside("iacli.adapters", "iacli.adapters")
    assert not is_inside("iacli.adapters_extra", "iacli.adapters")


def test_every_layer_has_modules_to_check():
    # Protège contre un renommage de dossier qui rendrait les règles vides.
    for layer in LAYERS:
        assert len(module_files(layer)) > 1, layer


# --- Les règles ------------------------------------------------------------


def test_domain_depends_only_on_standard_library_and_itself():
    allowed_roots = set(sys.stdlib_module_names)
    outside = [
        f"{module} importe {name}"
        for module, name in imports_of("domain")
        if not is_inside(name, "iacli.domain")
        and name.split(".")[0] not in allowed_roots
    ]

    assert outside == []


def test_services_do_not_depend_on_outer_layers():
    assert (
        violations("services", ("iacli.adapters", "iacli.infrastructure", "iacli.cli"))
        == []
    )


def test_infrastructure_does_not_depend_on_adapters_or_composition_root():
    assert violations("infrastructure", ("iacli.adapters", "iacli.cli")) == []


def test_adapters_do_not_depend_on_infrastructure_or_composition_root():
    assert violations("adapters", ("iacli.infrastructure", "iacli.cli")) == []


@pytest.mark.parametrize("layer", LAYERS)
def test_no_layer_depends_on_the_legacy_ollama_shim(layer):
    assert violations(layer, ("iacli.ollama",)) == []


def test_only_the_composition_root_wires_outer_layers():
    outer = ("iacli.adapters", "iacli.infrastructure")
    offenders = []
    for path in sorted(PACKAGE.glob("*.py")):
        if path.name in {COMPOSITION_ROOT, "__init__.py"}:
            continue
        module = f"iacli.{path.stem}"
        allowed = COMPATIBILITY_SHIMS.get(path.name, set())
        for name in imported_modules(path.read_text(encoding="utf-8"), module):
            if any(is_inside(name, package) for package in outer) and not any(
                is_inside(name, ok) for ok in allowed
            ):
                offenders.append(f"{module} importe {name}")

    assert offenders == []


def test_composition_root_is_the_only_module_importing_both_outer_layers():
    wiring = imported_modules(
        (PACKAGE / COMPOSITION_ROOT).read_text(encoding="utf-8"), "iacli.cli"
    )

    assert any(is_inside(name, "iacli.adapters") for name in wiring)
    assert any(is_inside(name, "iacli.infrastructure") for name in wiring)
    assert any(is_inside(name, "iacli.services") for name in wiring)
