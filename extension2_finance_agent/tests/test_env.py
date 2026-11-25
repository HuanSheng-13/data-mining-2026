"""Local secret configuration: parsing, precedence and evaluator integration."""
from dataclasses import replace
import io
import json
import os
from types import SimpleNamespace

import pytest

from extension2_finance_agent.core import env
from extension2_finance_agent.core.client import ChatClient
from extension2_finance_agent.core.config import Config, write_json
from extension2_finance_agent.experiments.evaluate import evaluate


@pytest.fixture
def dotenv(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    monkeypatch.setattr(env, "DOTENV_FILE", path)
    monkeypatch.setattr(env, "user_environment_value", lambda name: None)
    for name in ("FINANCEQA_API_KEY", "FINANCEQA_CHAT_URL", "FINANCEQA_MODEL", "FINANCEQA_JUDGE_API_KEY",
                 "FINANCEQA_JUDGE_URL", "FINANCEQA_JUDGE_MODEL", "CUSTOM_FINANCEQA_KEY"):
        monkeypatch.delenv(name, raising=False)
    return path


def test_literal_dotenv_parsing(dotenv):
    dotenv.write_text('\ufeff# comment\nexport FINANCEQA_API_KEY = "key#with=characters" # comment\n'
                      "SINGLE='literal ${UNCHANGED}'\nUNQUOTED=abc#def # ignored\nEMPTY=\nBLANK=# ignored\n", encoding="utf-8")
    assert env.read_dotenv() == {"FINANCEQA_API_KEY": "key#with=characters", "SINGLE": "literal ${UNCHANGED}",
                                 "UNQUOTED": "abc#def", "EMPTY": "", "BLANK": ""}
    assert "FINANCEQA_API_KEY" not in os.environ


@pytest.mark.parametrize("text", ["secret-without-assignment", "BAD KEY=private-value", "KEY='unclosed", 'KEY="value" trailing-secret'])
def test_dotenv_errors_never_echo_values(dotenv, text):
    dotenv.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError) as error:
        env.read_dotenv()
    assert str(error.value) == "Invalid .env syntax on line 1; expected single-line KEY=VALUE"
    assert text not in str(error.value)


def test_precedence_and_file_reload(dotenv, monkeypatch):
    monkeypatch.setattr(env, "user_environment_value", lambda name: "user-value")
    assert env.environment_value("CUSTOM_FINANCEQA_KEY") == "user-value"
    dotenv.write_text("CUSTOM_FINANCEQA_KEY=file-value\n", encoding="utf-8")
    assert env.environment_value("CUSTOM_FINANCEQA_KEY") == "file-value"
    monkeypatch.setenv("CUSTOM_FINANCEQA_KEY", "process-value")
    assert env.environment_value("CUSTOM_FINANCEQA_KEY") == "process-value"
    monkeypatch.delenv("CUSTOM_FINANCEQA_KEY")
    dotenv.write_text("CUSTOM_FINANCEQA_KEY=updated-value\n", encoding="utf-8")
    assert env.environment_value("CUSTOM_FINANCEQA_KEY") == "updated-value"


def test_model_config_uses_fixed_dotenv_not_working_directory(dotenv, tmp_path, monkeypatch):
    dotenv.write_text("FINANCEQA_CHAT_URL=https://example.com/chat\nFINANCEQA_MODEL=from-file\n", encoding="utf-8")
    config_file = tmp_path / "config.json"
    write_json(config_file, {})
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / ".env").write_text("FINANCEQA_MODEL=wrong-file\n", encoding="utf-8")
    monkeypatch.chdir(elsewhere)
    config = Config.load(config_file)
    assert config.model == "from-file" and config.chat_url == "https://example.com/chat"
    monkeypatch.setenv("FINANCEQA_MODEL", "from-process")
    assert Config.load(config_file).model == "from-process"


def test_chat_reads_key_from_dotenv_without_putting_it_in_config(dotenv, monkeypatch):
    dotenv.write_text("CUSTOM_FINANCEQA_KEY=local-fixture-secret\n", encoding="utf-8")
    config = Config(api_key_env="CUSTOM_FINANCEQA_KEY")
    client = ChatClient(config)
    payload = {"choices": [{"message": {"role": "assistant", "content": "Fixture reply"}, "finish_reason": "stop"}]}

    def urlopen(request, timeout):
        assert request.get_header("Authorization") == "Bearer local-fixture-secret"
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    response = client.complete([], [])
    assert response["message"]["content"] == "Fixture reply"
    assert "local-fixture-secret" not in json.dumps(config.snapshot()) + json.dumps(response)
    assert "CUSTOM_FINANCEQA_KEY" not in os.environ


def test_evaluator_reads_dotenv_credentials_without_global_mutation(dotenv, tmp_path, monkeypatch):
    dotenv.write_text("FINANCEQA_JUDGE_API_KEY=judge-fixture-secret\nFINANCEQA_JUDGE_URL=https://example.com/judge\nFINANCEQA_JUDGE_MODEL=judge-fixture\n", encoding="utf-8")
    predictions = tmp_path / "predictions.json"
    write_json(predictions, [])
    captured = {}

    def run(command, **kwargs):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", run)
    assert evaluate(Config(), "train", predictions, tmp_path / "score.json") == 0
    assert captured["env"]["LLM_API_KEY"] == "judge-fixture-secret"
    assert "https://example.com/judge" in captured["command"] and "judge-fixture" in captured["command"]
    assert "judge-fixture-secret" not in " ".join(captured["command"])
    assert os.environ.get("LLM_API_KEY") != "judge-fixture-secret"
