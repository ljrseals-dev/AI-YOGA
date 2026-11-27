import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from weighted_scoring import load_weights, select_subset, combo_score, gate, unit_rows

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
