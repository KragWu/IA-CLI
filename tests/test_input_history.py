from prompt_toolkit.history import FileHistory

from iacli.domain.history import InputHistory, format_entry, parse_entries
from iacli.infrastructure.history import FileHistoryStore


def test_previous_goes_back_through_the_entries_then_stops_at_the_oldest():
    history = InputHistory(["un", "deux"])

    assert history.previous("brouillon") == "deux"
    assert history.previous("deux") == "un"
    assert history.previous("un") is None


def test_next_goes_forward_and_restores_the_draft():
    history = InputHistory(["un", "deux"])
    history.previous("brouillon")
    history.previous("deux")

    assert history.next() == "deux"
    assert history.next() == "brouillon"
    assert history.next() is None


def test_nothing_to_browse_when_the_history_is_empty():
    history = InputHistory()

    assert history.previous("x") is None
    assert history.next() is None


def test_add_ignores_blank_text_and_immediate_repeats():
    history = InputHistory(["un"])

    assert history.add("   ") is False
    assert history.add("un") is False
    assert history.add("deux") is True
    assert history.previous("") == "deux"
    assert history.previous("deux") == "un"


def test_adding_an_entry_restarts_the_browsing_from_the_end():
    history = InputHistory(["un"])
    history.previous("")

    history.add("deux")

    assert history.previous("") == "deux"


def test_entries_use_the_prompt_toolkit_file_format(tmp_path):
    path = tmp_path / "history"
    entries = ["simple", "deux\nlignes", "avec + plus"]
    content = "".join(
        format_entry(entry, "2026-01-01 10:00:00.000000") for entry in entries
    )
    path.write_bytes(content.encode("utf-8"))

    assert parse_entries(path.read_text(encoding="utf-8")) == entries
    # prompt_toolkit lit l'historique du chat console dans le même fichier.
    assert list(FileHistory(str(path)).load_history_strings()) == entries[::-1]


def test_the_store_appends_and_reloads_in_order(tmp_path):
    store = FileHistoryStore(tmp_path / "dossier" / "history")

    store.append("un")
    store.append("deux\nlignes")

    assert FileHistoryStore(tmp_path / "dossier" / "history").load() == [
        "un",
        "deux\nlignes",
    ]


def test_the_store_never_fails_on_an_unusable_path(tmp_path):
    blocker = tmp_path / "fichier"
    blocker.write_text("x", encoding="utf-8")
    store = FileHistoryStore(blocker / "history")

    store.append("un")

    assert store.load() == []


def test_exotic_line_separators_stay_inside_a_reloaded_entry():
    entries = ["a" + chr(0x2028) + "b", "c" + chr(12) + "d", "e" + chr(0x85) + "f"]
    content = "".join(format_entry(entry, "2026-01-01 10:00:00") for entry in entries)

    assert parse_entries(content) == entries
