from pathlib import Path
import subprocess

import pytest

from iacli.infrastructure.workspace_tools import create_workspace_tools


def create_tools(tmp_path, *, confirm_write=lambda _path, _diff: False, confirm_shell=lambda _command, _cwd: False):
    return {
        item.name: item
        for item in create_workspace_tools(tmp_path, confirm_write, confirm_shell)
    }


def test_workspace_read_list_and_search_are_limited_to_project(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('Hello')\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "hidden.txt").write_text("Hello", encoding="utf-8")
    tools = create_tools(tmp_path)

    assert tools["read_file"].invoke({"path": "src/app.py"}) == "print('Hello')\n"
    assert tools["list_files"].invoke({"path": "."}) == "src/app.py"
    assert "src/app.py:1: print('Hello')" in tools["search_files"].invoke(
        {"query": "hello"}
    )


def test_workspace_tools_reject_paths_outside_project(tmp_path):
    tools = create_tools(tmp_path)

    with pytest.raises(ValueError, match="doit rester dans le projet"):
        tools["read_file"].invoke({"path": "../outside.txt"})


def test_write_file_requires_confirmation_and_shows_diff(tmp_path):
    previews = []
    tools = create_tools(
        tmp_path,
        confirm_write=lambda path, diff: previews.append((path, diff)) or False,
    )

    result = tools["write_file"].invoke(
        {"path": "src/new.py", "content": "print('safe')\n"}
    )

    assert "refusée" in result
    assert not (tmp_path / "src" / "new.py").exists()
    assert previews[0][0] == "src/new.py"
    assert "+print('safe')" in previews[0][1]


def test_confirmed_write_creates_file_in_project(tmp_path):
    tools = create_tools(tmp_path, confirm_write=lambda _path, _diff: True)

    result = tools["write_file"].invoke(
        {"path": "src/new.py", "content": "print('approved')\n"}
    )

    assert result == "Fichier enregistré : src/new.py"
    assert (tmp_path / "src" / "new.py").read_text(encoding="utf-8") == (
        "print('approved')\n"
    )


def test_shell_command_requires_confirmation_before_execution(tmp_path, monkeypatch):
    executed = []
    monkeypatch.setattr(
        "iacli.infrastructure.workspace_tools.subprocess.run",
        lambda *args, **kwargs: executed.append((args, kwargs)),
    )
    tools = create_tools(tmp_path)

    result = tools["run_shell"].invoke({"command": "echo no"})

    assert result == "Exécution de la commande refusée par l'utilisateur."
    assert executed == []


def test_confirmed_shell_command_uses_project_directory(tmp_path, monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args[0], 0, "ok\n", "")

    monkeypatch.setattr("iacli.infrastructure.workspace_tools.subprocess.run", fake_run)
    tools = create_tools(tmp_path, confirm_shell=lambda _command, _cwd: True)

    result = tools["run_shell"].invoke({"command": "echo ok"})

    assert "Code de sortie : 0" in result
    assert calls[0][1]["cwd"] == Path(tmp_path).resolve()
    assert calls[0][1]["timeout"] == 120
