from langchain_core.messages import AIMessage

from iacli.services.tool_call_parser import normalize_text_tool_call

TOOLS = {"read_file", "list_files"}


def _normalize(content):
    return normalize_text_tool_call(AIMessage(content=content), TOOLS)


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
