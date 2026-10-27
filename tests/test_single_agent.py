import importlib

import pytest


pytest.importorskip("langgraph")
pytest.importorskip("dotenv")

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from urbanomy.methods.agent import SingleAgentBaseline
from urbanomy.methods.land_value_modeling import Evaluation, StrategicAlignmentScorer


class StructuredModel:
    def with_structured_output(self, schema):
        assert "score" in schema["properties"]
        return self

    def invoke(self, messages):
        return {"score": 0.8}


def test_single_agent_text_output_and_stream():
    baseline = SingleAgentBaseline(llm=FakeListChatModel(responses=['{"score": 0.6}']))
    state = baseline.invoke_state("Evaluate")
    assert state["output"] == '{"score": 0.6}'
    assert state["log"][0].content == "Evaluate"
    assert list(baseline.stream("Evaluate"))[0]["single_agent"]["output"] == state["output"]


def test_structured_baseline_connects_to_scorer():
    baseline = SingleAgentBaseline(llm=StructuredModel(), output_schema=Evaluation)
    scorer = StrategicAlignmentScorer(baseline=baseline, prompt="Evaluate")
    result = scorer.score_candidate(params_repaired={}, land_value_gain=100, investor_npv=50)
    assert result["score"] == 0.8


def test_agent_runs_with_a_local_chat_model(monkeypatch):
    pytest.importorskip("langchain")
    pytest.importorskip("langchain_openai")
    from urbanomy.methods.agent import config

    monkeypatch.setattr(config, "api_key", "test-key")
    monkeypatch.setattr(config, "base_url", None)
    llms = importlib.import_module("urbanomy.methods.agent.llms")
    monkeypatch.setattr(llms, "llm", FakeListChatModel(responses=["Scenario evaluated"]))
    from urbanomy.methods.agent import Agent

    agent = Agent(system_prompt="Evaluate scenarios")
    assert agent.invoke([{"role": "user", "content": "Evaluate this scenario"}]) == "Scenario evaluated"


def test_agent_accepts_an_explicit_model_without_default_clients():
    pytest.importorskip("langchain")
    pytest.importorskip("langchain_openai")
    from urbanomy.methods.agent import Agent

    agent = Agent(system_prompt="Evaluate", model=FakeListChatModel(responses=["done"]))
    assert agent.invoke([{"role": "user", "content": "Evaluate"}]) == "done"


def test_agent_reads_structured_output_without_messages(monkeypatch):
    pytest.importorskip("langchain")
    pytest.importorskip("langchain_openai")
    from types import SimpleNamespace

    module = importlib.import_module("urbanomy.methods.agent.agent")
    response = {"structured_response": Evaluation(score=0.7)}
    monkeypatch.setattr(module, "create_agent", lambda **kwargs: SimpleNamespace(invoke=lambda **kwargs: response))
    agent = module.Agent(system_prompt="Evaluate", model=object())
    assert agent.invoke([]).score == 0.7


@pytest.mark.parametrize("provider", ["llm", "baseline"])
@pytest.mark.parametrize("response_type", ["message", "schema", "mapping"])
def test_scorer_accepts_raw_and_structured_provider_responses(provider, response_type):
    from types import SimpleNamespace

    from langchain_core.messages import AIMessage

    response = {
        "message": AIMessage(content=[{"type": "text", "text": '{"score": 0.9}'}]),
        "schema": Evaluation(score=0.9),
        "mapping": {"score": 0.9},
    }[response_type]
    model = SimpleNamespace(invoke=lambda prompt: response)
    scorer = StrategicAlignmentScorer(**{provider: model}, prompt="Evaluate")
    assert scorer.score_candidate(params_repaired={}, land_value_gain=0, investor_npv=0)["score"] == 0.9
