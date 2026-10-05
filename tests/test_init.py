import io
import json
from urllib.error import URLError

from iacli.init import initialize


class FakeResponse:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


def test_init_creates_default_config_directory(tmp_path):
    config_dir = tmp_path / ".config" / "iacli"
    output = io.StringIO()

    def urlopen(_request, timeout):
        return FakeResponse(json.dumps({"models": []}).encode())

    initialize(
        config_dir=config_dir,
        input_fn=lambda _prompt: "n",
        output_fn=lambda message: print(message, file=output),
        urlopen=urlopen,
        which=lambda _name: "/usr/bin/tool",
        disk_free_bytes=10 * 1024**3,
    )

    assert (config_dir / "config.toml").is_file()
    assert (config_dir / "IACLI.md").is_file()
    assert "Initialisation réussie" in output.getvalue()


def test_init_checks_missing_prerequisites(tmp_path):
    output = io.StringIO()

    def urlopen(_request, timeout):
        return FakeResponse(json.dumps({"models": []}).encode())

    initialize(
        config_dir=tmp_path / "iacli",
        input_fn=lambda _prompt: "n",
        output_fn=lambda message: print(message, file=output),
        urlopen=urlopen,
        which=lambda name: None if name == "rg" else "/usr/bin/tool",
        disk_free_bytes=10 * 1024**3,
    )

    assert "ripgrep" in output.getvalue()
    assert "avertissement" in output.getvalue().lower()


def test_init_ollama_model_pull(tmp_path):
    config_dir = tmp_path / "iacli"
    output = io.StringIO()
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        if request.full_url.endswith("/api/tags"):
            return FakeResponse(b'{"models": []}')
        return FakeResponse(b'{"status":"success"}')

    initialize(
        config_dir=config_dir,
        input_fn=lambda _prompt: "y",
        output_fn=lambda message: print(message, file=output),
        urlopen=urlopen,
        which=lambda _name: "/usr/bin/tool",
        disk_free_bytes=10 * 1024**3,
    )

    pull_request = next(request for request in requests if request.full_url.endswith("/api/pull"))
    assert json.loads(pull_request.data) == {"name": "qwen2.5-coder:7b", "stream": False}
    assert "qwen2.5-coder:7b" in output.getvalue()


def test_init_skips_model_pull_when_disk_space_is_insufficient(tmp_path):
    output = io.StringIO()
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        return FakeResponse(b'{"models": []}')

    def unexpected_prompt(_prompt):
        raise AssertionError("Aucune confirmation ne doit être demandée sans espace suffisant.")

    initialize(
        config_dir=tmp_path / "iacli",
        input_fn=unexpected_prompt,
        output_fn=lambda message: print(message, file=output),
        urlopen=urlopen,
        which=lambda _name: "/usr/bin/tool",
        disk_free_bytes=1,
    )

    assert len(requests) == 1
    assert "espace disque insuffisant" in output.getvalue()


def test_init_warns_when_ollama_is_unreachable(tmp_path):
    output = io.StringIO()

    def urlopen(_request, timeout):
        raise URLError("connection refused")

    initialize(
        config_dir=tmp_path / "iacli",
        input_fn=lambda _prompt: "n",
        output_fn=lambda message: print(message, file=output),
        urlopen=urlopen,
        which=lambda _name: "/usr/bin/tool",
        disk_free_bytes=10 * 1024**3,
    )

    assert "Ollama" in output.getvalue()