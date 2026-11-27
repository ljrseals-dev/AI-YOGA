"""
Weighted subset selection with an evidence-floor gate.

Composite score for a combination C of m items (unit-normalized embeddings,
cosine similarity s_ij, per-item trust prior v_i in [0, 1]):

    coherence    mu(C)  = mean of s_ij over pairs i<j in C
    weakest link lam(C) = min  of s_ij over pairs i<j in C
    trust        tau(C) = mean of v_i over i in C

    S(C) = w_mu * mu(C) + w_lam * lam(C) + w_tau * tau(C),   w >= 0, sum w = 1

Gate (theta = evidence floor):  ALLOW if S >= theta;
ESCALATE if 0.9*theta <= S < theta; DENY otherwise.
"""
import heapq, itertools, json, math
import numpy as np

DEFAULT_WEIGHTS = {"coherence": 0.5, "weakest_link": 0.3, "trust": 0.2, "evidence_floor": 0.3}


def load_weights(path=None):
    w = dict(DEFAULT_WEIGHTS)
    if path:
        with open(path) as f:
            w.update({k: v for k, v in json.load(f).items() if not k.startswith("_")})
    parts = [w["coherence"], w["weakest_link"], w["trust"]]
    if any(p < 0 for p in parts) or sum(parts) <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    total = sum(parts)
    w["coherence"], w["weakest_link"], w["trust"] = (p / total for p in parts)
    return w


def unit_rows(X):
    X = np.asarray(X, dtype=float)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def combo_score(S, v, combo, w):
    sims = [S[a, b] for a, b in itertools.combinations(combo, 2)]
    mu, lam = float(np.mean(sims)), float(np.min(sims))
    tau = float(np.mean([v[i] for i in combo]))
    return w["coherence"] * mu + w["weakest_link"] * lam + w["trust"] * tau, (mu, lam, tau)


def gate(score, floor):
    if score >= floor:
        return "ALLOW"
    if score >= 0.9 * floor:
        return "ESCALATE"
    return "DENY"


def prune_pool(S, v, pool, k_nn=5):
    """Cheap pre-filter: keep the `pool` items with the best blend of trust and
    local density. Swap in an ANNchor k-NN graph here when the metric is slow."""
    n = len(v)
    if pool >= n:
        return list(range(n))
    S2 = S.copy(); np.fill_diagonal(S2, -np.inf)
    dens = np.sort(S2, axis=1)[:, -min(k_nn, n - 1):].mean(axis=1)
    rank = 0.5 * np.asarray(v) + 0.5 * dens
    return sorted(int(i) for i in np.argsort(-rank)[:pool])


def select_subset(X, priors, size, weights=None, pool=24, top_n=5, max_combos=2_000_000):
    """Return the best `size`-item subsets by S(C), plus an evidence-floor verdict."""
    w = weights or load_weights()
    X = unit_rows(X)
    v = np.asarray(priors, dtype=float)
    if len(v) != len(X):
        raise ValueError("priors must have one value per item")
    S = X @ X.T
    cand = prune_pool(S, v, pool)
    if math.comb(len(cand), size) > max_combos:
        raise ValueError("too many combinations; lower pool or size")
    heap = []
    for combo in itertools.combinations(cand, size):
        sc, parts = combo_score(S, v, combo, w)
        item = (sc, combo, parts)
        if len(heap) < top_n:
            heapq.heappush(heap, item)
        elif sc > heap[0][0]:
            heapq.heapreplace(heap, item)
    best = sorted(heap, reverse=True)
    top = best[0]
    return {
        "indices": list(top[1]), "score": top[0],
        "coherence": top[2][0], "weakest_link": top[2][1], "trust": top[2][2],
        "verdict": gate(top[0], w["evidence_floor"]),
        "runner_ups": [{"indices": list(b[1]), "score": b[0]} for b in best[1:]],
        "pool_size": len(cand),
    }
