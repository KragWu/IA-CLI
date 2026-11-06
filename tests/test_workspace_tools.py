from pathlib import Path
import sys
import time

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


def test_workspace_tool_timing_callback_reports_tool_duration(tmp_path):
    (tmp_path / "notes.txt").write_text("contenu", encoding="utf-8")
    timings = []
    tools = {
        item.name: item
        for item in create_workspace_tools(
            tmp_path,
            lambda _path, _diff: False,
            lambda _command, _cwd: False,
            lambda name, elapsed: timings.append((name, elapsed)),
        )
    }

    tools["read_file"].invoke({"path": "notes.txt"})

    assert len(timings) == 1
    assert timings[0][0] == "read_file"
    assert timings[0][1] >= 0


def test_workspace_tools_reject_paths_outside_project(tmp_path):
    tools = create_tools(tmp_path)

    with pytest.raises(ValueError, match="doit rester dans le projet"):
        tools["read_file"].invoke({"path": "../outside.txt"})


def test_read_file_resolves_unique_filename_in_project(tmp_path):
    source = tmp_path / "src" / "iacli"
    source.mkdir(parents=True)
    (source / "cli.py").write_text("cli source", encoding="utf-8")
    tools = create_tools(tmp_path)

    assert tools["read_file"].invoke({"path": "cli.py"}) == "cli source"


def test_read_file_rejects_ambiguous_filename_in_project(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "cli.py").write_text("source", encoding="utf-8")
    (tmp_path / "tests" / "cli.py").write_text("test", encoding="utf-8")
    tools = create_tools(tmp_path)

    with pytest.raises(ValueError, match="est ambigu"):
        tools["read_file"].invoke({"path": "cli.py"})


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


def test_write_file_refuses_when_diff_is_too_large_to_be_shown_in_full(tmp_path):
    confirmations = []
    tools = create_tools(
        tmp_path,
        confirm_write=lambda path, diff: confirmations.append(path) or True,
    )

    result = tools["write_file"].invoke(
        {"path": "big.txt", "content": "ligne de contenu\n" * 5_000}
    )

    assert "trop volumineuse" in result
    assert confirmations == []
    assert not (tmp_path / "big.txt").exists()


def test_confirmed_write_creates_file_in_project(tmp_path):
    tools = create_tools(tmp_path, confirm_write=lambda _path, _diff: True)

    result = tools["write_file"].invoke(
        {"path": "src/new.py", "content": "print('approved')\n"}
    )

    assert result == "Fichier enregistré : src/new.py"
    assert (tmp_path / "src" / "new.py").read_text(encoding="utf-8") == (
        "print('approved')\n"
    )


def test_shell_command_requires_confirmation_before_execution(tmp_path):
    marker = tmp_path / "executed.txt"
    tools = create_tools(tmp_path)

    result = tools["run_shell"].invoke({"command": "echo no > executed.txt"})

    assert result == "Exécution de la commande refusée par l'utilisateur."
    assert not marker.exists()


def test_confirmed_shell_command_uses_project_directory(tmp_path):
    tools = create_tools(tmp_path, confirm_shell=lambda _command, _cwd: True)

    result = tools["run_shell"].invoke(
        {"command": f'"{sys.executable}" -c "import os; print(os.getcwd())"'}
    )

    assert "Code de sortie : 0" in result
    assert str(Path(tmp_path).resolve()) in result


def test_shell_timeout_kills_child_processes_and_returns(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "iacli.infrastructure.workspace_tools.MAX_TIME_FOR_SUBPROCESS", 1
    )
    script = tmp_path / "spawn.py"
    script.write_text(
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'])\n"
        "time.sleep(20)\n",
        encoding="utf-8",
    )
    tools = create_tools(tmp_path, confirm_shell=lambda _command, _cwd: True)

    started = time.perf_counter()
    result = tools["run_shell"].invoke({"command": f'"{sys.executable}" spawn.py'})
    elapsed = time.perf_counter() - started

    assert "interrompue après 1 secondes" in result
    assert elapsed < 10


def test_write_file_keeps_lf_line_endings_of_existing_file(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_bytes(b"un\ndeux\n")
    tools = create_tools(tmp_path, confirm_write=lambda _path, _diff: True)

    tools["write_file"].invoke({"path": "notes.txt", "content": "un\ndeux\ntrois\n"})

    assert target.read_bytes() == b"un\ndeux\ntrois\n"


def test_write_file_keeps_crlf_line_endings_of_existing_file(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_bytes(b"un\r\ndeux\r\n")
    diffs = []
    tools = create_tools(
        tmp_path,
        confirm_write=lambda _path, diff: diffs.append(diff) or True,
    )

    tools["write_file"].invoke({"path": "notes.txt", "content": "un\ndeux\ntrois\n"})

    assert target.read_bytes() == b"un\r\ndeux\r\ntrois\r\n"
    assert "-un" not in diffs[0]
    assert "+trois" in diffs[0]


def test_write_file_reports_no_modification_when_only_line_endings_differ(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_bytes(b"un\r\ndeux\r\n")
    tools = create_tools(tmp_path, confirm_write=lambda _path, _diff: True)

    result = tools["write_file"].invoke({"path": "notes.txt", "content": "un\ndeux\n"})

    assert result.startswith("Aucune modification")
    assert target.read_bytes() == b"un\r\ndeux\r\n"


def test_write_file_writes_new_file_without_converting_line_endings(tmp_path):
    tools = create_tools(tmp_path, confirm_write=lambda _path, _diff: True)

    tools["write_file"].invoke({"path": "new.txt", "content": "un\ndeux\n"})

    assert (tmp_path / "new.txt").read_bytes() == b"un\ndeux\n"


def test_write_file_rejects_existing_file_that_is_not_utf8(tmp_path):
    target = tmp_path / "latin.txt"
    target.write_bytes("caf\xe9\n".encode("latin-1"))
    tools = create_tools(tmp_path, confirm_write=lambda _path, _diff: True)

    with pytest.raises(ValueError, match="UTF-8"):
        tools["write_file"].invoke({"path": "latin.txt", "content": "autre\n"})

    assert target.read_bytes() == "caf\xe9\n".encode("latin-1")
