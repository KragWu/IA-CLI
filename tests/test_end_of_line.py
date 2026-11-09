"""Fin de ligne par défaut : réglage `[files] end_of_line`, résolution et usage par les outils."""

import pytest

from iacli.domain.config import EndOfLine, FileSettings, resolve_newline
from iacli.infrastructure.configuration import DEFAULT_CONFIG, FileConfigurationGateway
from iacli.infrastructure.workspace_tools import create_workspace_tools


@pytest.mark.parametrize(
    ("setting", "platform_newline", "expected"),
    [
        (EndOfLine.AUTO, "\r\n", "\r\n"),
        (EndOfLine.AUTO, "\n", "\n"),
        # Une valeur de plateforme inattendue (CR seul) retombe sur LF.
        (EndOfLine.AUTO, "\r", "\n"),
        (EndOfLine.LF, "\r\n", "\n"),
        (EndOfLine.CRLF, "\n", "\r\n"),
    ],
)
def test_resolve_newline_follows_the_setting_then_the_platform(
    setting, platform_newline, expected
):
    assert resolve_newline(setting, platform_newline) == expected


def load_configuration(tmp_path, toml):
    directory = tmp_path / "iacli"
    directory.mkdir()
    (directory / "config.toml").write_text(toml, encoding="utf-8")
    return FileConfigurationGateway().ensure_configuration(directory)


def test_files_settings_default_to_auto_when_the_section_is_missing(tmp_path):
    state = load_configuration(tmp_path, '[ollama]\nmodel = "custom:latest"\n')

    assert state.files == FileSettings(EndOfLine.AUTO)
    assert state.warnings == ()


def test_the_default_configuration_documents_end_of_line_and_loads_as_auto(tmp_path):
    assert "[files]" in DEFAULT_CONFIG
    assert "end_of_line" in DEFAULT_CONFIG

    state = FileConfigurationGateway().ensure_configuration(tmp_path / "iacli")

    assert state.files == FileSettings(EndOfLine.AUTO)
    assert state.warnings == ()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("auto", EndOfLine.AUTO),
        ("lf", EndOfLine.LF),
        ("crlf", EndOfLine.CRLF),
        ("CRLF", EndOfLine.CRLF),
    ],
)
def test_end_of_line_setting_is_read_from_config(tmp_path, value, expected):
    state = load_configuration(tmp_path, f'[files]\nend_of_line = "{value}"\n')

    assert state.files == FileSettings(expected)
    assert state.warnings == ()


@pytest.mark.parametrize(
    "toml",
    [
        '[files]\nend_of_line = "mac"\n',
        "[files]\nend_of_line = 3\n",
        'files = "texte"\n',
    ],
)
def test_invalid_end_of_line_setting_warns_and_falls_back_to_auto(tmp_path, toml):
    state = load_configuration(tmp_path, toml)

    assert state.files == FileSettings(EndOfLine.AUTO)
    assert len(state.warnings) == 1
    assert "files" in state.warnings[0]


def create_tools(tmp_path, default_newline):
    return {
        item.name: item
        for item in create_workspace_tools(
            tmp_path,
            lambda _path, _diff: True,
            lambda _command, _cwd: False,
            default_newline=default_newline,
        )
    }


def test_new_file_is_written_with_the_default_line_ending(tmp_path):
    tools = create_tools(tmp_path, "\r\n")

    tools["write_file"].invoke({"path": "new.txt", "content": "un\ndeux\n"})

    assert (tmp_path / "new.txt").read_bytes() == b"un\r\ndeux\r\n"


def test_new_file_stays_lf_with_an_lf_default(tmp_path):
    tools = create_tools(tmp_path, "\n")

    tools["write_file"].invoke({"path": "new.txt", "content": "un\ndeux\n"})

    assert (tmp_path / "new.txt").read_bytes() == b"un\ndeux\n"


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        (b"", b"x\r\ny\r\n"),  # fichier vide
        (b"abc", b"x\r\ny\r\n"),  # aucune fin de ligne
        (b"a\r\nb\n", b"x\r\ny\r\n"),  # égalité CRLF / LF
    ],
)
def test_existing_file_without_a_majority_uses_the_default_line_ending(
    tmp_path, original, expected
):
    target = tmp_path / "notes.txt"
    target.write_bytes(original)
    tools = create_tools(tmp_path, "\r\n")

    tools["write_file"].invoke({"path": "notes.txt", "content": "x\ny\n"})

    assert target.read_bytes() == expected


def test_the_majority_line_ending_wins_over_the_default(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_bytes(b"un\ndeux\ntrois\n")
    tools = create_tools(tmp_path, "\r\n")

    tools["write_file"].invoke({"path": "notes.txt", "content": "un\ndeux\ntrois\nx\n"})

    assert target.read_bytes() == b"un\ndeux\ntrois\nx\n"


def test_edit_file_uses_the_default_line_ending_on_a_file_without_any(tmp_path):
    target = tmp_path / "notes.txt"
    target.write_bytes(b"abc")
    tools = create_tools(tmp_path, "\r\n")

    tools["edit_file"].invoke(
        {"path": "notes.txt", "old_string": "abc", "new_string": "a\nb"}
    )

    assert target.read_bytes() == b"a\r\nb"


@pytest.mark.parametrize(
    "name",
    [
        "deploy.sh",
        "scripts/run.bash",
        "prompt.zsh",
        "Makefile",
        "sub/Makefile",
        "GNUmakefile",
        "INSTALL.SH",
    ],
)
def test_new_shell_scripts_and_makefiles_are_always_lf(tmp_path, name):
    tools = create_tools(tmp_path, "\r\n")

    tools["write_file"].invoke({"path": name, "content": "#!/bin/sh\necho ok\n"})

    assert (tmp_path / name).read_bytes() == b"#!/bin/sh\necho ok\n"


@pytest.mark.parametrize(
    "name", ["build.bat", "scripts/setup.cmd", "RUN.BAT", "sub/clean.CMD"]
)
def test_new_batch_files_are_always_crlf(tmp_path, name):
    tools = create_tools(tmp_path, "\n")

    tools["write_file"].invoke({"path": name, "content": "@echo off\necho ok\n"})

    assert (tmp_path / name).read_bytes() == b"@echo off\r\necho ok\r\n"


def test_existing_batch_file_without_a_majority_uses_crlf(tmp_path):
    target = tmp_path / "run.bat"
    target.write_bytes(b"echo hi")
    tools = create_tools(tmp_path, "\n")

    tools["write_file"].invoke({"path": "run.bat", "content": "echo a\necho b\n"})

    assert target.read_bytes() == b"echo a\r\necho b\r\n"


def test_edit_file_on_a_batch_file_without_line_ending_uses_crlf(tmp_path):
    target = tmp_path / "run.cmd"
    target.write_bytes(b"echo hi")
    tools = create_tools(tmp_path, "\n")

    tools["edit_file"].invoke(
        {"path": "run.cmd", "old_string": "echo hi", "new_string": "echo a\necho b"}
    )

    assert target.read_bytes() == b"echo a\r\necho b"


def test_an_existing_lf_batch_file_keeps_its_majority_line_ending(tmp_path):
    # L'exception ne vaut que pour la valeur par défaut : un fichier existant garde la sienne.
    target = tmp_path / "run.bat"
    target.write_bytes(b"echo a\necho b\n")
    tools = create_tools(tmp_path, "\r\n")

    tools["write_file"].invoke({"path": "run.bat", "content": "echo a\necho c\n"})

    assert target.read_bytes() == b"echo a\necho c\n"


@pytest.mark.parametrize("name", ["notes.txt", "script.sh.txt", "shell.md", "data.csv"])
def test_other_new_files_keep_the_default_line_ending(tmp_path, name):
    tools = create_tools(tmp_path, "\r\n")

    tools["write_file"].invoke({"path": name, "content": "un\ndeux\n"})

    assert (tmp_path / name).read_bytes() == b"un\r\ndeux\r\n"


def test_existing_shell_script_without_a_majority_uses_lf(tmp_path):
    target = tmp_path / "run.sh"
    target.write_bytes(b"echo hi")
    tools = create_tools(tmp_path, "\r\n")

    tools["write_file"].invoke({"path": "run.sh", "content": "echo a\necho b\n"})

    assert target.read_bytes() == b"echo a\necho b\n"


def test_edit_file_on_a_shell_script_without_line_ending_uses_lf(tmp_path):
    target = tmp_path / "run.sh"
    target.write_bytes(b"echo hi")
    tools = create_tools(tmp_path, "\r\n")

    tools["edit_file"].invoke(
        {"path": "run.sh", "old_string": "echo hi", "new_string": "echo a\necho b"}
    )

    assert target.read_bytes() == b"echo a\necho b"


def test_an_existing_crlf_shell_script_keeps_its_majority_line_ending(tmp_path):
    # L'exception ne vaut que pour la valeur par défaut : un fichier existant garde la sienne.
    target = tmp_path / "run.sh"
    target.write_bytes(b"echo a\r\necho b\r\n")
    tools = create_tools(tmp_path, "\n")

    tools["write_file"].invoke({"path": "run.sh", "content": "echo a\necho c\n"})

    assert target.read_bytes() == b"echo a\r\necho c\r\n"


@pytest.mark.parametrize("invalid", ["\r", "", "\n\n", "crlf"])
def test_default_line_ending_must_be_lf_or_crlf(tmp_path, invalid):
    with pytest.raises(ValueError, match="default_newline"):
        create_workspace_tools(
            tmp_path,
            lambda _path, _diff: True,
            lambda _command, _cwd: False,
            default_newline=invalid,
        )
