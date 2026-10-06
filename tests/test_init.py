import io
import json
from urllib.error import HTTPError, URLError

from iacli.init import initialize


class FakeResponse:
    def __init__(self, body: bytes):
        self.body = io.BytesIO(body)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body.read()

    def readline(self):
        return self.body.readline()


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
    assert 'registry_url = "https://registry.ollama.ai"' in (config_dir / "config.toml").read_text()
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
        if request.full_url.startswith("https://registry.ollama.ai"):
            manifest = {"config": {"size": 1024}, "layers": [{"size": 1536 * 1024**2}]}
            return FakeResponse(json.dumps(manifest).encode())
        events = [
            {"status": "pulling manifest"},
            {"status": "downloading layer", "completed": 25 * 1024**2, "total": 100 * 1024**2},
            {"status": "downloading layer", "completed": 100 * 1024**2, "total": 100 * 1024**2},
            {"status": "success"},
        ]
        return FakeResponse("".join(json.dumps(event) + "\n" for event in events).encode())

    initialize(
        config_dir=config_dir,
        input_fn=lambda _prompt: "y",
        output_fn=lambda message: print(message, file=output),
        urlopen=urlopen,
        which=lambda _name: "/usr/bin/tool",
        disk_free_bytes=10 * 1024**3,
    )

    pull_request = next(request for request in requests if request.full_url.endswith("/api/pull"))
    assert json.loads(pull_request.data) == {"name": "qwen2.5-coder:7b", "stream": True}
    assert "Étape 1/4" in output.getvalue()
    assert "Étape 4/4" in output.getvalue()
    assert "1.50 Gio" in output.getvalue()
    assert "25% (25.0/100.0 Mio)" in output.getvalue()
    assert "100% (100.0/100.0 Mio)" in output.getvalue()
    assert "Modèle qwen2.5-coder:7b téléchargé" in output.getvalue()


def test_init_skips_model_pull_when_disk_space_is_insufficient(tmp_path):
    output = io.StringIO()
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        if request.full_url.endswith("/api/tags"):
            return FakeResponse(b'{"models": []}')
        return FakeResponse(json.dumps({"config": {"size": 0}, "layers": [{"size": 5 * 1024**3}]}).encode())

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

    assert len(requests) == 2
    assert "espace disque insuffisant" in output.getvalue()
    assert "5.00 Gio requis" in output.getvalue()


def test_init_warns_when_default_model_does_not_exist(tmp_path):
    output = io.StringIO()
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        if request.full_url.endswith("/api/tags"):
            return FakeResponse(b'{"models": []}')
        raise HTTPError(request.full_url, 404, "Not Found", {}, None)

    initialize(
        config_dir=tmp_path / "iacli",
        input_fn=lambda _prompt: (_ for _ in ()).throw(AssertionError("Aucun pull attendu pour un modèle absent.")),
        output_fn=lambda message: print(message, file=output),
        urlopen=urlopen,
        which=lambda _name: "/usr/bin/tool",
        disk_free_bytes=10 * 1024**3,
    )

    assert len(requests) == 2
    assert "n'existe pas dans le registre Ollama" in output.getvalue()
    assert not any(request.full_url.endswith("/api/pull") for request in requests)


def test_init_uses_existing_config_values(tmp_path):
    config_dir = tmp_path / "iacli"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        '[ollama]\nbase_url = "http://ollama.test:11434/"\n'
        'registry_url = "https://registry.test/"\nmodel = "custom-coder:latest"\n',
        encoding="utf-8",
    )
    output = io.StringIO()
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        if request.full_url == "http://ollama.test:11434/api/tags":
            return FakeResponse(b'{"models": []}')
        if request.full_url == "https://registry.test/v2/library/custom-coder/manifests/latest":
            manifest = {"config": {"size": 1}, "layers": [{"size": 1024}]}
            return FakeResponse(json.dumps(manifest).encode())
        if request.full_url == "http://ollama.test:11434/api/pull":
            return FakeResponse(b'{"status":"success"}\n')
        raise AssertionError(f"URL inattendue : {request.full_url}")

    initialize(
        config_dir=config_dir,
        input_fn=lambda _prompt: "y",
        output_fn=lambda message: print(message, file=output),
        urlopen=urlopen,
        which=lambda _name: "/usr/bin/tool",
        disk_free_bytes=10 * 1024**3,
    )

    assert (config_dir / "config.toml").read_text(encoding="utf-8").count("custom-coder") == 1
    assert "http://ollama.test:11434" in output.getvalue()
    assert "custom-coder:latest" in output.getvalue()
    pull_request = next(request for request in requests if request.full_url.endswith("/api/pull"))
    assert pull_request.full_url == "http://ollama.test:11434/api/pull"
    assert json.loads(pull_request.data) == {"name": "custom-coder:latest", "stream": True}
    assert len(requests) == 3


def test_init_uses_default_config_values_for_missing_keys(tmp_path):
    config_dir = tmp_path / "iacli"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        '[ollama]\nbase_url = "http://ollama.test:11434"\n',
        encoding="utf-8",
    )
    output = io.StringIO()
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        if request.full_url == "http://ollama.test:11434/api/tags":
            return FakeResponse(b'{"models": []}')
        if request.full_url == (
            "https://registry.ollama.ai/v2/library/qwen2.5-coder/manifests/7b"
        ):
            manifest = {"config": {"size": 1}, "layers": [{"size": 1024}]}
            return FakeResponse(json.dumps(manifest).encode())
        raise AssertionError(f"URL inattendue : {request.full_url}")

    initialize(
        config_dir=config_dir,
        input_fn=lambda _prompt: "n",
        output_fn=lambda message: print(message, file=output),
        urlopen=urlopen,
        which=lambda _name: "/usr/bin/tool",
        disk_free_bytes=10 * 1024**3,
    )

    assert len(requests) == 2
    assert "qwen2.5-coder:7b" in output.getvalue()
    assert "Modèle qwen2.5-coder:7b trouvé" in output.getvalue()


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
