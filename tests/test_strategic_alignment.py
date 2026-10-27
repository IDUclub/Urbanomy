import copy
import json

import geopandas as gpd
import numpy as np
import pytest
from pydantic import ValidationError
from shapely.geometry import box

from urbanomy.methods.land_value_modeling import (
    DistrictProblem,
    StrategicAlignmentScorer,
    build_pareto_front_dataframe,
    run_nsga2_with_strategic_alignment,
)


class StubLLM:
    def __init__(self, score=0.75):
        self.score = score
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        return {"score": self.score}


def test_scorer_caches_and_keeps_llm_when_copied():
    llm = StubLLM()
    scorer = StrategicAlignmentScorer(llm=llm, prompt="Evaluate scenario")
    candidate = dict(params_repaired={"fsi": np.float64(1.5)}, land_value_gain=100, investor_npv=50)
    assert scorer.score_candidate(**candidate)["score"] == 0.75
    assert scorer.score_candidate(**candidate)["score"] == 0.75
    assert len(llm.calls) == 1
    assert json.loads(llm.calls[0].split("Сценарий:\n")[1])["params_repaired"]["fsi"] == 1.5
    copied = copy.deepcopy(scorer)
    assert copied.llm is llm
    assert copied.score_candidate(**candidate)["score"] == 0.75
    assert len(llm.calls) == 1


@pytest.mark.parametrize("score", [-0.1, 1.1])
def test_scorer_rejects_out_of_range_scores(score):
    scorer = StrategicAlignmentScorer(llm=StubLLM(score), prompt="Evaluate")
    with pytest.raises(ValidationError):
        scorer.score_candidate(params_repaired={}, land_value_gain=0, investor_npv=0)


def test_nsga2_logs_three_objectives_and_builds_pareto_table(monkeypatch, tmp_path):
    blocks = gpd.GeoDataFrame(
        {
            "id": [86],
            "site_area": [1000.0],
            "footprint_area": [100.0],
            "l": [2.0],
            "build_floor_area": [200.0],
            "living_area": [100.0],
            "non_living_area": [100.0],
        },
        geometry=[box(0, 0, 10, 100)],
        crs=3857,
    )
    monkeypatch.setattr(DistrictProblem, "_recompute_morphotype", lambda self, genome: "test")
    monkeypatch.setattr(
        DistrictProblem,
        "evaluate_catboost",
        lambda self, geonome, **kwargs: float(geonome["footprint_area"].sum() * 100),
    )
    monkeypatch.setattr(
        DistrictProblem,
        "evaluate_investment_potential",
        lambda self, geonome, **kwargs: float(50000 - geonome["footprint_area"].sum() * 10),
    )
    constraints = {
        "footprint_area": {"min": 0.0, "max": 100.0},
        "l": {"min": 1.0, "max": 10.0},
        "residential": {"min": 0.0, "max": 1.0},
        "business": {"min": 0.0, "max": 1.0},
    }
    log_path = tmp_path / "logs" / "optimization.jsonl"
    result, problem = run_nsga2_with_strategic_alignment(
        blocks=blocks,
        model=None,
        estimator_kwargs={"orig_features": [], "categorical_features": []},
        constraints=constraints,
        target_id=86,
        llm=StubLLM(),
        prompt="Evaluate",
        pop_size=4,
        n_gen=2,
        seed=42,
        verbose=False,
        optimization_log_path=log_path,
    )
    assert result.F.shape[1] == 3
    assert np.allclose(result.F[:, 2], -0.75)
    assert len(result.history) == 2
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(records) == 8
    assert all(row["target_id"] == 86 and row["llm score"] == 0.75 for row in records)
    frame = build_pareto_front_dataframe(res=result, problem=problem)
    assert len(frame) == len(result.F)
    assert np.allclose(frame["llm score"], 0.75)
    assert np.allclose(frame["land_value_gain"], -result.F[:, 0] - 10000)
    assert set(frame["target_id"]) == {86}
    assert all(np.isclose(row["residential"] + row["business"], 1) for row in frame["params_repaired"])
    history = build_pareto_front_dataframe(res=result, problem=problem, use_history=True)
    assert len(history) == 8


def test_plain_nsga2_problem_still_has_two_objectives():
    problem = DistrictProblem(
        blocks=gpd.GeoDataFrame(),
        model=None,
        estimator_kwargs={},
        constraints={"l": {"min": 1, "max": 10}},
        target_id=86,
    )
    assert problem.n_obj == 2
