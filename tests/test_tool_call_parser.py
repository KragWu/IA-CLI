import pytest
from langchain_core.messages import AIMessage

from iacli.services.tool_call_parser import (
    MalformedToolCall,
    normalize_text_tool_call,
)

TOOLS = {"read_file", "list_files"}


def _normalize(content, done_reason=None):
    metadata = {"done_reason": done_reason} if done_reason else {}
    return normalize_text_tool_call(
        AIMessage(content=content, response_metadata=metadata), TOOLS
    )


def test_fallback_xml_json_tool_call_parsing():
    result = _normalize(
        '<tool_call>{"name": "read_file", "arguments": {"path": "README.md"}}</tool_call>'
    )

    assert isinstance(result, AIMessage)
    assert result.content == ""
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call["name"] == "read_file"
    assert call["args"] == {"path": "README.md"}
    assert call["type"] == "tool_call"
    assert call["id"]


def test_fallback_xml_json_tool_call_parsing_handles_multiple_tags():
    result = _normalize(
        '<tool_call>{"name": "read_file", "arguments": {"path": "a"}}</tool_call>\n'
        '<tool_call>{"name": "list_files", "arguments": {}}</tool_call>'
    )

    assert [call["name"] for call in result.tool_calls] == ["read_file", "list_files"]
    assert result.tool_calls[0]["id"] != result.tool_calls[1]["id"]


def test_fallback_xml_json_tool_call_parsing_handles_multiline_json_and_prose():
    result = _normalize(
        "Je lis le fichier.\n<tool_call>\n"
        '{\n  "name": "read_file",\n  "arguments": {"path": "a"}\n}\n'
        "</tool_call>\nUn instant."
    )

    assert result.content == ""
    assert [call["args"] for call in result.tool_calls] == [{"path": "a"}]


def test_fallback_xml_json_tool_call_parsing_defaults_missing_arguments():
    result = _normalize('<tool_call>{"name": "list_files"}</tool_call>')

    assert result.tool_calls[0]["args"] == {}


def test_fallback_xml_tag_with_unknown_tool_still_becomes_a_tool_call():
    # ToolNode répondra lui-même par une erreur « outil inconnu » au modèle.
    result = _normalize('<tool_call>{"name": "inconnu", "arguments": {}}</tool_call>')

    assert [call["name"] for call in result.tool_calls] == ["inconnu"]


def test_fallback_keeps_usage_and_metadata_of_the_original_message():
    original = AIMessage(
        content='<tool_call>{"name": "list_files", "arguments": {}}</tool_call>',
        id="run-1",
        response_metadata={"model": "m"},
        usage_metadata={"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
    )

    result = normalize_text_tool_call(original, TOOLS)

    assert result.id == "run-1"
    assert result.response_metadata == {"model": "m"}
    assert result.usage_metadata == original.usage_metadata


def test_native_ollama_tool_call_parsing():
    native = AIMessage(
        content="<tool_call>ignoré</tool_call>",
        tool_calls=[
            {
                "name": "read_file",
                "args": {"path": "a"},
                "id": "native-1",
                "type": "tool_call",
            }
        ],
    )

    assert normalize_text_tool_call(native, TOOLS) is native


def test_plain_answer_is_left_untouched():
    answer = AIMessage(content="Bonjour, comment puis-je aider ?")

    assert normalize_text_tool_call(answer, TOOLS) is answer


def test_malformed_tool_call_handling_reports_invalid_json():
    result = _normalize('<tool_call>{"name": "read_file", </tool_call>')

    assert isinstance(result, MalformedToolCall)
    assert "JSON" in result.error
    assert "<tool_call>" in result.error


def test_malformed_tool_call_handling_reports_unclosed_tag():
    result = _normalize('<tool_call>{"name": "read_file", "arguments": {}}')

    assert isinstance(result, MalformedToolCall)


@pytest.mark.parametrize(
    "body",
    [
        '["read_file"]',
        '{"arguments": {}}',
        '{"name": 3, "arguments": {}}',
        '{"name": "read_file", "arguments": "a.txt"}',
    ],
)
def test_malformed_tool_call_handling_reports_wrong_structure(body):
    result = _normalize(f"<tool_call>{body}</tool_call>")

    assert isinstance(result, MalformedToolCall)
    assert '"name"' in result.error


def test_malformed_tag_among_valid_ones_is_reported():
    result = _normalize(
        '<tool_call>{"name": "list_files", "arguments": {}}</tool_call>'
        "<tool_call>{oups</tool_call>"
    )

    assert isinstance(result, MalformedToolCall)


def test_invalid_fenced_json_without_tag_stays_plain_text():
    answer = AIMessage(content='```json\n{"name": "read_file", \n```')

    assert normalize_text_tool_call(answer, TOOLS) is answer


@pytest.mark.parametrize(
    "prose",
    [
        "La balise <tool_call> sert à appeler un outil.",
        "Écris <tool_call> puis le JSON puis </tool_call> pour appeler un outil.",
        "Utilise `<tool_call>` et `</tool_call>` comme délimiteurs.",
    ],
)
def test_prose_mentioning_the_tag_is_not_a_malformed_call(prose):
    answer = AIMessage(content=prose)

    assert normalize_text_tool_call(answer, TOOLS) is answer


def test_tool_call_arguments_may_contain_the_closing_tag():
    result = _normalize(
        '<tool_call>{"name": "read_file", '
        '"arguments": {"path": "</tool_call> <tool_call>"}}</tool_call>'
    )

    assert isinstance(result, AIMessage)
    assert [call["args"] for call in result.tool_calls] == [
        {"path": "</tool_call> <tool_call>"}
    ]


def test_prose_mention_before_a_real_call_does_not_hide_the_call():
    result = _normalize(
        "Je n'utilise pas <tool_call> en prose.\n"
        '<tool_call>{"name": "list_files", "arguments": {}}</tool_call>'
    )

    assert isinstance(result, AIMessage)
    assert [call["name"] for call in result.tool_calls] == ["list_files"]


CUT_IN_JSON = '<tool_call>{"name": "write_file", "arguments": {"content": "abc'
CUT_BEFORE_CLOSING_TAG = '<tool_call>{"name": "list_files", "arguments": {}}'


@pytest.mark.parametrize("content", [CUT_IN_JSON, CUT_BEFORE_CLOSING_TAG])
def test_tool_call_cut_by_the_token_limit_is_reported_as_truncated(content):
    result = _normalize(content, done_reason="length")

    assert isinstance(result, MalformedToolCall)
    assert result.truncated
    assert "limite de tokens" in result.error
    assert "plus court" in result.error


@pytest.mark.parametrize("content", [CUT_IN_JSON, CUT_BEFORE_CLOSING_TAG])
def test_malformed_tool_call_without_truncation_keeps_the_generic_message(content):
    result = _normalize(content, done_reason="stop")

    assert isinstance(result, MalformedToolCall)
    assert not result.truncated
    assert "limite de tokens" not in result.error


def test_complete_tool_call_is_kept_even_when_generation_stopped_on_length():
    result = _normalize(
        '<tool_call>{"name": "list_files", "arguments": {}}</tool_call>',
        done_reason="length",
    )

    assert isinstance(result, AIMessage)
    assert [call["name"] for call in result.tool_calls] == ["list_files"]


def test_wrong_structure_is_not_blamed_on_truncation():
    result = _normalize(
        '<tool_call>{"arguments": {}}</tool_call>', done_reason="length"
    )

    assert isinstance(result, MalformedToolCall)
    assert not result.truncated
