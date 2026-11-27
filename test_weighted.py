import os, sys, tempfile, types
from unittest.mock import patch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from weighted_scoring import load_weights, select_subset, combo_score, gate, unit_rows, prune_pool

def test_weights_normalize():
    w = load_weights()
    assert abs(w["coherence"] + w["weakest_link"] + w["trust"] - 1) < 1e-12

def test_negative_weight_rejected():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write('{"coherence": -1, "weakest_link": 1, "trust": 1}')
    try:
        load_weights(f.name); raise AssertionError("expected ValueError")
    except ValueError:
        pass
    finally:
        os.unlink(f.name)

def test_identical_items_score_one():
    X = unit_rows(np.ones((4, 3))); S = X @ X.T
    sc, parts = combo_score(S, [1, 1, 1, 1], (0, 1, 2), load_weights())
    assert abs(sc - 1) < 1e-9 and abs(parts[1] - 1) < 1e-9

def test_gate_bands():
    assert gate(0.3, 0.3) == "ALLOW" and gate(0.28, 0.3) == "ESCALATE" and gate(0.1, 0.3) == "DENY"

def test_prefers_coherent_trusted_cluster():
    rng = np.random.default_rng(1); t = rng.normal(size=8)
    X = np.vstack([t + 0.05 * rng.normal(size=8) for _ in range(4)] + [rng.normal(size=8) for _ in range(6)])
    v = np.r_[np.ones(4), np.full(6, 0.2)]
    assert set(select_subset(X, v, 3, pool=10)["indices"]) <= {0, 1, 2, 3}

def test_noise_is_denied():
    X = np.random.default_rng(5).normal(size=(10, 32)); v = np.full(10, 0.1)
    assert select_subset(X, v, 3, pool=10)["verdict"] == "DENY"


def test_evidence_floor_validation():
    import json
    for value in (0, -0.1, 1.01, "high", None, True):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"evidence_floor": value}, f)
            path = f.name
        try:
            try:
                load_weights(path)
                raise AssertionError(f"expected ValueError for evidence_floor={value!r}")
            except ValueError as exc:
                assert "evidence_floor" in str(exc)
        finally:
            os.unlink(path)

def test_nonnumeric_scoring_weight_rejected():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write('{"coherence": "high"}')
        path = f.name
    try:
        try:
            load_weights(path)
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "coherence" in str(exc)
    finally:
        os.unlink(path)


def test_recall_expected_handles_empty_and_short_rows():
    from baseline import recall_at_k
    D = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5]])
    true_rows = [[], [1, 2]]
    approx_rows = [[], [1]]
    assert recall_at_k(true_rows, approx_rows, D, 2) == 0.25


def test_recall_returned_uses_returned_edge_denominator():
    from baseline import recall_at_k
    D = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5]])
    true_rows = [[1, 2], []]
    approx_rows = [[1], [0, 1]]
    assert abs(recall_at_k(
        true_rows, approx_rows, D, 2, normalize="returned"
    ) - (1.0 / 3.0)) < 1e-12


def test_prune_pool_reports_truncation():
    S = np.eye(4, dtype=float)
    kept, truncated = prune_pool(S, np.ones(4), pool=2)
    assert len(kept) == 2
    assert truncated is True
    kept_all, truncated_all = prune_pool(S, np.ones(4), pool=4)
    assert kept_all == [0, 1, 2, 3]
    assert truncated_all is False

def test_select_subset_exposes_pruned_out():
    X = np.random.default_rng(12).normal(size=(6, 4))
    result = select_subset(X, np.ones(6), size=2, pool=4)
    assert result["pool_size"] == 4
    assert result["pruned_out"] is True

def _assert_numpy_rng_state_equal(actual, expected):
    assert actual[0] == expected[0]
    assert np.array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]

def test_run_annchor_persistent_rng_state_restoration_on_success():
    import baseline

    class FakeAnnchor:
        def __init__(self, *args, **kwargs):
            self.neighbor_graph = (np.array([[1], [0]]), np.zeros((2, 1)))
        def fit(self):
            np.random.random()

    fake_module = types.SimpleNamespace(Annchor=FakeAnnchor)
    cfg = {"anchors": 1, "k": 1, "p_work": 0.1}
    np.random.seed(812)
    before = np.random.get_state()
    with patch.dict(sys.modules, {"annchor": fake_module}):
        baseline.run_annchor(np.ones((2, 2)), "euclidean", cfg, seed=7)
    _assert_numpy_rng_state_equal(np.random.get_state(), before)

def test_run_annchor_persistent_rng_state_restoration_on_exception():
    import baseline

    class FakeAnnchor:
        def __init__(self, *args, **kwargs):
            pass
        def fit(self):
            np.random.random()
            raise RuntimeError("deliberate test failure")

    fake_module = types.SimpleNamespace(Annchor=FakeAnnchor)
    cfg = {"anchors": 1, "k": 1, "p_work": 0.1}
    np.random.seed(913)
    before = np.random.get_state()
    with patch.dict(sys.modules, {"annchor": fake_module}):
        try:
            baseline.run_annchor(np.ones((2, 2)), "euclidean", cfg, seed=9)
            raise AssertionError("expected RuntimeError")
        except RuntimeError as exc:
            assert str(exc) == "deliberate test failure"
    _assert_numpy_rng_state_equal(np.random.get_state(), before)

if __name__ == "__main__":      # runs without pytest: python test_weighted.py
    failures = []
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("ok", name)
            except Exception as exc:
                failures.append((name, exc))
                print("FAIL", name, "--", exc)
    if failures:
        raise SystemExit(f"{len(failures)} test(s) failed")
