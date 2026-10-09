"""Reusable HTTP fakes for adapter tests."""

import io
from collections import defaultdict

import pytest


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body
        self._stream = io.BytesIO(body)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self._body

    def readline(self) -> bytes:
        return self._stream.readline()


class FakeUrlopen:
    def __init__(self) -> None:
        self.requests = []
        self._responses = defaultdict(list)

    def add_response(self, url: str, response: bytes | BaseException) -> None:
        self._responses[url].append(response)

    def __call__(self, request, timeout: int):
        self.requests.append((request, timeout))
        queued = self._responses[request.full_url]
        if not queued:
            raise AssertionError(f"Unexpected HTTP request: {request.full_url}")
        response = queued.pop(0)
        if isinstance(response, BaseException):
            raise response
        return FakeResponse(response)


@pytest.fixture
def fake_urlopen() -> FakeUrlopen:
    return FakeUrlopen()
