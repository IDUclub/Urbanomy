import json
import subprocess
import sys
from datetime import datetime
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from urbanomy.utils import json_ready


def test_scoring_import_does_not_load_geospatial_or_llm_dependencies():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from urbanomy.methods.land_value_modeling import Evaluation, StrategicAlignmentScorer
assert Evaluation(score=0.5).score == 0.5
assert not {'geopandas', 'pymoo', 'catboost', 'langchain', 'langchain_openai'} & set(sys.modules)
""",
        ],
        check=True,
    )


def test_llm_clients_are_created_only_on_demand():
    pytest.importorskip("langchain_openai")
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
from unittest.mock import patch
with patch('langchain_openai.ChatOpenAI') as chat, patch('langchain_openai.OpenAIEmbeddings') as embeddings:
    from urbanomy.methods.agent import init_llm, Agent
    import urbanomy.methods.agent.llms as clients
    chat.assert_not_called()
    embeddings.assert_not_called()
    init_llm(model='test-model')
    chat.assert_called_once()
    assert clients.llm is clients.llm
    assert chat.call_count == 2
    embeddings.assert_not_called()
    assert clients.embedding is clients.embedding
    embeddings.assert_called_once()
""",
        ],
        check=True,
    )


def test_json_payloads_support_numpy_and_readonly_mappings():
    source = MappingProxyType(
        {
            1: np.int64(42),
            "values": (np.float32(0.5), float("nan"), float("inf")),
            "when": datetime(2026, 1, 1),
            "flag": np.bool_(True),
        }
    )
    payload = json_ready(source)
    assert payload == {"1": 42, "values": [0.5, None, None], "when": "2026-01-01 00:00:00", "flag": True}
    json.dumps(payload, allow_nan=False)


def test_pareto_table_uses_cached_baseline_and_keeps_third_objective():
    from urbanomy.methods.land_value_modeling import build_pareto_front_dataframe
    from urbanomy.methods.land_value_modeling.ga_mc_optimizer import Evaluation
    from urbanomy.methods.land_value_modeling.strategic_alignment import Evaluation as Score

    assert Score is Evaluation
    problem = SimpleNamespace(
        constraints={"fsi": {"min": 0, "max": 2}},
        target_id=86,
        baseline_land_value=lambda: 100.0,
        _repair_genome=lambda params: {"land_use": "residential", **params},
    )
    result = SimpleNamespace(X=np.array([[1.5]]), F=np.array([[-150.0, -20.0, -0.8]]))
    frame = build_pareto_front_dataframe(res=result, problem=problem)
    assert frame.loc[0, "land_value_gain"] == 50
    assert frame.loc[0, "llm score"] == 0.8
    assert frame.loc[0, "target_id"] == 86
    assert frame.loc[0, "params_repaired"]["fsi"] == 1.5


def test_scenario_plot_remains_available_at_original_import_path():
    from urbanomy.methods.land_value_modeling import plot_scenario_impact
    from urbanomy.methods.land_value_modeling.scenario_modification import plot_scenario_impact as original
    from urbanomy.methods.land_value_modeling.scenario_visualization import plot_scenario_impact as moved

    assert original is moved is plot_scenario_impact


def test_pareto_domain_and_client_settings_do_not_load_llm_frameworks():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from urbanomy.methods.land_value_modeling.pareto import build_pareto_front_dataframe
assert callable(build_pareto_front_dataframe)
from urbanomy.methods.agent import init_llm, config
assert not {'langchain', 'langgraph', 'langchain_openai', 'pymoo', 'catboost'} & set(sys.modules)
""",
        ],
        check=True,
    )


def test_consolidated_single_agent_and_input_keep_public_exports():
    pytest.importorskip("langgraph")
    from urbanomy.methods.agent import SingleAgentBaseline, build_single_agent_graph
    from urbanomy.methods.agent.single_agent import SingleAgentBaseline as Baseline
    from urbanomy.methods.agent.single_agent import build_single_agent_graph as build
    from urbanomy.methods.investment_potential import prepare_investment_input
    from urbanomy.methods.investment_potential.cashflows import InvestmentMetricsResult
    from urbanomy.methods.investment_potential.input import prepare_investment_input as prepare
    from urbanomy.methods.investment_potential.investment_metrics import InvestmentMetricsResult as Result

    assert Baseline is SingleAgentBaseline
    assert build is build_single_agent_graph
    assert prepare is prepare_investment_input
    assert Result is InvestmentMetricsResult


def test_shared_message_conversion_preserves_text_and_ignores_nontext_blocks():
    from urbanomy.utils import message_to_text

    blocks = [" first ", {"type": "text", "text": "second"}, {"type": "image_url", "image_url": "unused"}]
    assert message_to_text(SimpleNamespace(content=blocks)) == " first \nsecond"
    assert message_to_text(" unchanged ") == " unchanged "
