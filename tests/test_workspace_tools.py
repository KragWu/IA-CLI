import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from iacli.infrastructure import workspace_tools
from iacli.infrastructure.workspace_tools import create_workspace_tools


def create_tools(
    tmp_path,
    *,
    confirm_write=lambda _path, _diff: False,
    confirm_shell=lambda _command, _cwd: False,
):
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

    assert tools["read_file"].invoke({"path": "cli.py"}) == (
        "[Chemin demandé introuvable : cli.py ; fichier lu : src/iacli/cli.py]\n"
        "cli source"
    )


def test_read_file_names_the_file_actually_read_after_a_wrong_directory(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "config.py").write_text("test config", encoding="utf-8")
    tools = create_tools(tmp_path)

    result = tools["read_file"].invoke({"path": "src/missing/config.py"})

    assert result.startswith(
        "[Chemin demandé introuvable : src/missing/config.py ; "
        "fichier lu : tests/config.py]"
    )
    assert result.endswith("test config")


def test_read_file_rejects_a_directory_instead_of_reading_a_same_name_file(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "docs").write_text("not the directory", encoding="utf-8")
    tools = create_tools(tmp_path)

    with pytest.raises(IsADirectoryError, match="dossier"):
        tools["read_file"].invoke({"path": "docs"})


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


def test_workspace_walk_never_enters_ignored_directories(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('Hello')\n", encoding="utf-8")
    for ignored in (".git", "node_modules", ".venv"):
        (tmp_path / ignored / "deep").mkdir(parents=True)
        (tmp_path / ignored / "deep" / "lib.js").write_text("Hello", encoding="utf-8")
    scanned = []
    real_scandir = os.scandir

    def spy_scandir(path="."):
        scanned.append(os.fspath(path))
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", spy_scandir)
    tools = create_tools(tmp_path)

    listing = tools["list_files"].invoke({"path": "."})
    search = tools["search_files"].invoke({"query": "hello"})

    assert listing == "src/app.py"
    assert "src/app.py:1" in search
    assert not any(
        ignored in Path(path).parts
        for path in scanned
        for ignored in (".git", "node_modules", ".venv")
    )


def test_search_reports_skipped_files_even_when_results_are_truncated(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"\xff\xfe\x00binaire")
    (tmp_path / "b.txt").write_text("Hello\n" * 150, encoding="utf-8")
    tools = create_tools(tmp_path)

    result = tools["search_files"].invoke({"query": "hello"})

    assert "[résultats limités à 100 correspondances]" in result
    assert "[fichiers ignorés : 0 trop volumineux, 1 non UTF-8]" in result


def test_read_file_rejects_file_that_is_not_utf8(tmp_path):
    (tmp_path / "latin.txt").write_bytes("caf\xe9\n".encode("latin-1"))
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\xff")
    tools = create_tools(tmp_path)

    for name in ("latin.txt", "image.png"):
        with pytest.raises(ValueError, match="UTF-8") as error:
            tools["read_file"].invoke({"path": name})
        assert name in str(error.value)


def test_interrupted_shell_command_kills_its_process_tree(tmp_path, monkeypatch):
    command = f'"{sys.executable}" -c "import time; time.sleep(30)"'
    killed = []
    real_communicate = subprocess.Popen.communicate
    real_kill = workspace_tools._kill_process_tree

    def interrupted_communicate(self, *args, **kwargs):
        if self.args == command:
            raise KeyboardInterrupt
        return real_communicate(self, *args, **kwargs)

    def recording_kill(process):
        killed.append(process.args)
        real_kill(process)

    monkeypatch.setattr(subprocess.Popen, "communicate", interrupted_communicate)
    monkeypatch.setattr(workspace_tools, "_kill_process_tree", recording_kill)
    tools = create_tools(tmp_path, confirm_shell=lambda _command, _cwd: True)

    with pytest.raises(KeyboardInterrupt):
        tools["run_shell"].invoke({"command": command})

    assert killed == [command]


def test_write_file_creates_new_empty_file_after_confirmation(tmp_path):
    diffs = []
    tools = create_tools(
        tmp_path,
        confirm_write=lambda _path, diff: diffs.append(diff) or True,
    )

    result = tools["write_file"].invoke({"path": "pkg/__init__.py", "content": ""})

    assert result == "Fichier enregistré : pkg/__init__.py"
    assert (tmp_path / "pkg" / "__init__.py").read_bytes() == b""
    assert "pkg/__init__.py" in diffs[0]
    assert "vide" in diffs[0]


def test_write_file_does_not_create_empty_file_when_user_refuses(tmp_path):
    tools = create_tools(tmp_path)

    result = tools["write_file"].invoke({"path": "pkg/__init__.py", "content": ""})

    assert "refusée" in result
    assert not (tmp_path / "pkg").exists()


def test_write_file_reports_no_modification_for_existing_empty_file(tmp_path):
    (tmp_path / "empty.txt").write_bytes(b"")
    tools = create_tools(tmp_path, confirm_write=lambda _path, _diff: True)

    result = tools["write_file"].invoke({"path": "empty.txt", "content": ""})

    assert result.startswith("Aucune modification")


def test_ignored_directories_are_skipped_by_walks_but_explicit_files_stay_accessible(
    tmp_path,
):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret hello\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("hello\n", encoding="utf-8")
    tools = create_tools(tmp_path)

    # Parcours de dossiers : les dossiers ignorés sont filtrés.
    assert tools["list_files"].invoke({"path": ".git"}) == "Aucun fichier trouvé."
    assert (
        tools["search_files"].invoke({"query": "hello", "path": ".git"})
        == "Aucune correspondance."
    )
    assert tools["search_files"].invoke({"query": "hello"}) == "src/app.py:1: hello"

    # Fichier désigné explicitement : accessible, de la même façon pour lire et chercher.
    assert tools["read_file"].invoke({"path": ".git/config"}) == "secret hello\n"
    assert (
        tools["search_files"].invoke({"query": "hello", "path": ".git/config"})
        == ".git/config:1: secret hello"
    )


def test_write_file_warns_before_converting_a_mixed_line_ending_file(tmp_path):
    target = tmp_path / "mixte.txt"
    target.write_bytes(b"un\r\ndeux\ntrois\r\n")
    diffs = []
    tools = create_tools(
        tmp_path,
        confirm_write=lambda _path, diff: diffs.append(diff) or True,
    )

    tools["write_file"].invoke(
        {"path": "mixte.txt", "content": "un\ndeux\ntrois\nquatre\n"}
    )

    assert "fins de ligne mixtes" in diffs[0]
    assert "2 CRLF" in diffs[0]
    assert "1 LF" in diffs[0]
    assert "CRLF" in diffs[0].splitlines()[0]
    assert "+quatre" in diffs[0]
    assert target.read_bytes() == b"un\r\ndeux\r\ntrois\r\nquatre\r\n"


def test_write_file_does_not_warn_for_uniform_line_endings(tmp_path):
    (tmp_path / "lf.txt").write_bytes(b"un\ndeux\n")
    (tmp_path / "crlf.txt").write_bytes(b"un\r\ndeux\r\n")
    diffs = []
    tools = create_tools(
        tmp_path,
        confirm_write=lambda _path, diff: diffs.append(diff) or True,
    )

    tools["write_file"].invoke({"path": "lf.txt", "content": "un\ndeux\ntrois\n"})
    tools["write_file"].invoke({"path": "crlf.txt", "content": "un\ndeux\ntrois\n"})

    assert len(diffs) == 2
    assert all("mixtes" not in diff for diff in diffs)


def test_shell_output_decoding_prefers_utf8(monkeypatch):
    monkeypatch.setattr(workspace_tools, "_console_encoding", lambda: "cp850")

    assert workspace_tools._decode_output("café ✓".encode()) == "café ✓"


def test_shell_output_decoding_falls_back_to_the_console_code_page(monkeypatch):
    monkeypatch.setattr(workspace_tools, "_console_encoding", lambda: "cp850")

    assert workspace_tools._decode_output("répertoire".encode("cp850")) == "répertoire"


def test_shell_output_decoding_replaces_undecodable_bytes(monkeypatch):
    monkeypatch.setattr(workspace_tools, "_console_encoding", lambda: "ascii")

    assert workspace_tools._decode_output(b"ok \xff") == "ok �"


@pytest.mark.skipif(os.name != "nt", reason="page de code OEM propre à Windows")
def test_run_shell_decodes_output_written_in_the_console_code_page(tmp_path):
    encoding = workspace_tools._console_encoding()
    script = tmp_path / "emit.py"
    script.write_text(
        f"import sys\nsys.stdout.buffer.write('répertoire été'.encode('{encoding}'))\n",
        encoding="utf-8",
    )
    tools = create_tools(tmp_path, confirm_shell=lambda _command, _cwd: True)

    result = tools["run_shell"].invoke({"command": f'"{sys.executable}" emit.py'})

    assert "répertoire été" in result
    assert "�" not in result


@pytest.mark.parametrize(
    "content",
    ["a" * 1_000_001, "é" * 600_000],
    ids=["ascii-1000001-octets", "accents-1200000-octets"],
)
def test_write_file_rejects_oversized_content_before_computing_the_diff(
    tmp_path, monkeypatch, content
):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("le diff ne doit pas être calculé")

    monkeypatch.setattr(workspace_tools.difflib, "unified_diff", fail_if_called)
    confirmations = []
    tools = create_tools(
        tmp_path,
        confirm_write=lambda path, diff: confirmations.append(path) or True,
    )

    with pytest.raises(ValueError, match="trop volumineux"):
        tools["write_file"].invoke({"path": "enorme.txt", "content": content})

    assert confirmations == []
    assert not (tmp_path / "enorme.txt").exists()


def test_write_file_accepts_content_at_the_size_limit_up_to_the_diff_check(tmp_path):
    tools = create_tools(tmp_path, confirm_write=lambda _path, _diff: True)

    result = tools["write_file"].invoke(
        {"path": "limite.txt", "content": "a" * 1_000_000}
    )

    assert "trop volumineuse pour être affichée" in result
    assert not (tmp_path / "limite.txt").exists()


def test_run_shell_does_not_share_the_console_input_with_the_command(
    tmp_path, monkeypatch
):
    popen_options = []
    real_popen = subprocess.Popen

    def recording_popen(*args, **kwargs):
        popen_options.append(kwargs)
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    tools = create_tools(tmp_path, confirm_shell=lambda _command, _cwd: True)

    tools["run_shell"].invoke({"command": "echo bonjour"})

    assert popen_options[0]["stdin"] is subprocess.DEVNULL


def test_run_shell_command_reading_stdin_gets_end_of_input_immediately(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(workspace_tools, "MAX_TIME_FOR_SUBPROCESS", 10)
    command = f'"{sys.executable}" -c "import sys; print(repr(sys.stdin.read()))"'
    tools = create_tools(tmp_path, confirm_shell=lambda _command, _cwd: True)

    started = time.perf_counter()
    result = tools["run_shell"].invoke({"command": command})

    assert "Code de sortie : 0" in result
    assert "''" in result
    assert time.perf_counter() - started < 8


def _run_script(tmp_path, source, monkeypatch=None, timeout=None):
    if timeout is not None:
        monkeypatch.setattr(workspace_tools, "MAX_TIME_FOR_SUBPROCESS", timeout)
    (tmp_path / "emit.py").write_text(source, encoding="utf-8")
    tools = create_tools(tmp_path, confirm_shell=lambda _command, _cwd: True)
    return tools["run_shell"].invoke({"command": f'"{sys.executable}" emit.py'})


def test_run_shell_keeps_stderr_when_stdout_alone_exceeds_the_limit(tmp_path):
    result = _run_script(
        tmp_path,
        "import sys\n"
        "sys.stdout.write('x' * 40000)\n"
        "sys.stderr.write('Traceback: boom')\n"
        "sys.exit(1)\n",
    )

    assert result.startswith("Code de sortie : 1")
    assert "Traceback: boom" in result
    assert "[sortie tronquée]" in result
    assert len(result) <= workspace_tools.MAX_TOOL_OUTPUT_CHARS


def test_run_shell_shares_the_limit_between_stdout_and_stderr(tmp_path):
    result = _run_script(
        tmp_path,
        "import sys\nsys.stdout.write('o' * 40000)\nsys.stderr.write('e' * 40000)\n",
    )

    assert result.count("[sortie tronquée]") == 2
    assert result.count("o") > 10_000
    assert result.count("e") > 10_000
    assert len(result) <= workspace_tools.MAX_TOOL_OUTPUT_CHARS


def test_run_shell_timeout_keeps_stderr_and_marks_truncation(tmp_path, monkeypatch):
    result = _run_script(
        tmp_path,
        "import sys, time\n"
        "sys.stdout.write('x' * 40000)\n"
        "sys.stderr.write('Traceback: boom')\n"
        "sys.stdout.flush()\n"
        "sys.stderr.flush()\n"
        "time.sleep(30)\n",
        monkeypatch,
        timeout=2,
    )

    assert "interrompue après 2 secondes" in result
    assert "Traceback: boom" in result
    assert "[sortie tronquée]" in result
    assert len(result) <= workspace_tools.MAX_TOOL_OUTPUT_CHARS
