import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import muse_research_plan as plan


def valid_response(package):
    ids = {n: package["experiments"][n]["evidence_id"] for n in ("003", "004")}
    ref3 = {"evidence_id": ids["003"], "observation_key": "control_workload"}
    ref4 = {"evidence_id": ids["004"], "observation_key": "partial_m_workload"}
    answer = {key: {"text": "The verified comparison is scoped to these workloads; runtime is unmeasured.",
                    "citations": [ref3, ref4]} for key in plan.FIELDS}
    answer["next_hypothesis"]["text"] = "A change in pipeline staging may alter compiled resource use at fixed workload."
    answer["proposed_experiment_005"] = {"workload": "FP16 GEMM on the recorded RTX GPU with the partial M workload.",
        "controlled_variables": "Hold source, inputs, tile sizes, warps, compiler and GPU fixed; vary only stage count.",
        "expected_observations": "Correctness and compiled resources may change; runtime remains unknown.",
        "confounders": "Compiler version and GPU state if timing is later authorized.", "citations": [ref4]}
    answer["decision_criteria"] = {"support": "Correctness passes and compiler resources change.",
        "weaken": "A code difference appears without the proposed resource change.",
        "reject": "Code and resources are identical or correctness fails.", "citations": [ref4]}
    return answer


@pytest.fixture
def package():
    return plan.evidence_package()


def test_verified_cross_experiment_package_and_baseline(package):
    a, b = (package["experiments"][n] for n in ("003", "004"))
    assert a["observations"]["source_identity"]["exact_generated_source_evaluated"] is False
    assert a["observations"]["control_workload"]["runtime_measured"] is False
    assert b["observations"]["partial_m_workload"]["runtime_measured"] is False
    assert b["observations"]["control_workload"]["sass_counts"]["baseline"]["predicate_set"] == 5
    assert b["observations"]["partial_m_workload"]["sass_counts"]["baseline"]["predicate_set"] == 16
    assert "not an optimization benefit" in b["observations"]["cross_workload_caution"]
    baseline = plan.baseline(package)
    assert "num_stages 3 versus 2" in baseline["experiment"]["controlled_variables"]
    assert baseline["citations"]


def test_missing_or_corrupt_records_fail_closed(tmp_path):
    with pytest.raises(ValueError): plan.evidence_package(tmp_path)
    original = Path.cwd() / "evaluations/experiment-004-partial-boundary-tiles"
    destination = tmp_path / "evaluations/experiment-004-partial-boundary-tiles"
    shutil.copytree(original, destination)
    (destination / "20261030T005455534566Z/result.json").write_text("{}")
    with pytest.raises(ValueError): plan.evidence_package(tmp_path)


def test_response_validation(package):
    answer = valid_response(package)
    assert plan.parse_response(json.dumps(answer), package) == answer
    broken = copy.deepcopy(answer)
    broken["evidence_synthesis"]["citations"][0]["observation_key"] = "invented"
    with pytest.raises(ValueError, match="citation"): plan.parse_response(json.dumps(broken), package)
    for text in ("The measured speedup was 2x.", "The compiler is always equivalent across all GPUs."):
        broken = copy.deepcopy(answer)
        broken["evidence_synthesis"]["text"] = text
        with pytest.raises(ValueError): plan.parse_response(json.dumps(broken), package)
    broken = copy.deepcopy(answer)
    broken["next_hypothesis"]["text"] = "Repeat mask hoisting."
    with pytest.raises(ValueError, match="mask-hoisting"): plan.parse_response(json.dumps(broken), package)
    broken = copy.deepcopy(answer)
    broken["proposed_experiment_005"]["controlled_variables"] = "Try many configurations."
    with pytest.raises(ValueError, match="uncontrolled"): plan.parse_response(json.dumps(broken), package)


def test_api_audit_idempotence_and_rejection(tmp_path, monkeypatch, package):
    monkeypatch.setattr(plan, "evidence_package", lambda root=plan.ROOT: package)
    answer = valid_response(package)
    def response(text, status="completed"):
        return SimpleNamespace(status=status, model=plan.MODEL, id="response-test",
            output=[{"type": "message", "status": status, "content": [{"type": "output_text", "text": text}]}])
    class Client:
        def __init__(self, result):
            self.result, self.calls = result, 0
            self.responses = self
        def create(self, **kwargs):
            self.calls += 1
            return self.result
    incomplete = Client(response(json.dumps(answer), "incomplete"))
    assert plan.request_once(incomplete, tmp_path) == "rejected"
    assert plan.load_accepted(tmp_path, package) is None
    assert len(list((tmp_path / plan.SECTION).glob("*/audit.json"))) == 1
    good = Client(response(json.dumps(answer)))
    assert plan.request_once(good, tmp_path) == "accepted"
    accepted = plan.load_accepted(tmp_path, package)
    assert accepted["comparison"]["judgment"].startswith("No automatic winner")
    assert plan.request_once(good, tmp_path) == "already_accepted"
    assert good.calls == 1
    assert len(list((tmp_path / plan.SECTION).glob("*/audit.json"))) == 2


def test_dashboard_missing_and_accepted(monkeypatch, package):
    from dashboard_app import create_app
    monkeypatch.setattr(plan, "load_accepted", lambda root, evidence: None)
    app = create_app(Path.cwd())
    client = app.test_client()
    page = client.get("/research")
    assert page.status_code == 200
    assert b"No accepted cross-experiment Muse proposal" in page.data
    saved = {"proposal": valid_response(package), "comparison": plan.comparison(valid_response(package), package)}
    monkeypatch.setattr(plan, "load_accepted", lambda root, evidence: saved)
    page = client.get("/research")
    assert page.status_code == 200
    assert b"What Muse learned" in page.data
    assert b"Approval:" in page.data
