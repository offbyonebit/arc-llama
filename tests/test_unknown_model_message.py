from click.testing import CliRunner

from arc_llama.cli import cli
from arc_llama.cli_support import unknown_model_message
from arc_llama.config import Config, ModelConfig


def _config():
    return Config(models=[ModelConfig(
        name="qwen", path="model.gguf", port=18080, gpu_pci_slot="gpu",
        aliases=["qwen-chat", "qwen-code", "qwen-mini", "qwen"],
    )])


def test_unknown_model_suggests_names_and_aliases_with_limit():
    message = unknown_model_message(_config(), "qwen-chta")
    assert message.startswith("No model named 'qwen-chta'. Did you mean ")
    assert "'qwen-chat'" in message
    assert "'qwen'" in message
    suggestions = message.split("Did you mean ")[1].split("?")[0].split(" or ")
    assert len(suggestions) == 3
    assert len(set(suggestions)) == 3
    assert message.endswith("Run arc-llama list to see your models.")


def test_unknown_model_without_close_match():
    assert unknown_model_message(_config(), "zzzzz") == (
        "No model named 'zzzzz'. Run arc-llama list to see your models."
    )


def test_unknown_model_with_empty_config():
    assert unknown_model_message(Config(), "qwen") == (
        "No models are registered yet. Add one with arc-llama add."
    )


def test_model_runtime_unknown_name_has_guidance(monkeypatch):
    monkeypatch.setattr("arc_llama.cli.load_config", lambda _: _config())
    result = CliRunner().invoke(cli, ["model-runtime", "qwen-chta"])
    assert result.exit_code == 1
    assert "No model named 'qwen-chta'." in result.output
    assert "Did you mean" in result.output and "'qwen-chat'" in result.output
    assert "Run arc-llama list to see your models." in result.output
