"""Shared test doubles."""

from collections.abc import Iterable, Sequence

from langchain_core.messages import AIMessage, AnyMessage


class FakeChatModel:
    def __init__(self, responses: Iterable[AIMessage] | None = None) -> None:
        self.responses = iter(responses) if responses is not None else None
        self.calls: list[Sequence[AnyMessage]] = []
        self.bound_tools = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages: Sequence[AnyMessage]) -> AIMessage:
        self.calls.append(messages)
        if self.responses is None:
            return AIMessage(content=f"Réponse {len(self.calls)}")
        return next(self.responses)
