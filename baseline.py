#!/usr/bin/env python3
"""
Baseline: exhaustive combination search (Brute-Genius-style) vs approximate
k-NN graph (ANNchor), with Critic-Harness-style verdicts and optional Gaffer
persistence.

Design notes
- Exact k-NN is computed in NumPy (independent of ANNchor), so recall@k can
  always be measured. ANNchor's own BruteForce / compare_neighbor_graphs are
  tried as a secondary cross-check.
- Exhaustive enumeration is streamed (running mean + top-N heap); it never
  stores every combination in memory.
- ANNchor / gafferpy calls are UNTESTED in the environment this was written in.
  Each is wrapped so a failure is reported, not hidden.
"""
import argparse, heapq, itertools, json, math, random, time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

try:
    from numba import njit
except ImportError:                      # numba ships with annchor; fallback
    def njit(f=None, **kw):
        return f if f is not None else (lambda g: g)

try:
    from gafferpy import gaffer as g, gaffer_connector
    GAFFER_AVAILABLE = True
except ImportError:
    GAFFER_AVAILABLE = False

# ---------------------------------------------------------------- metrics
@njit
def cosine_metric(a, b):
    na, nb = np.sqrt(np.sum(a * a)), np.sqrt(np.sum(b * b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    return 1.0 - np.sum(a * b) / (na * nb)

@njit
def euclidean_metric(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

PROFILES = {
    "default": dict(n_items=40, dim=8, comb=3, k=5, anchors=10, p_work=0.2,
                    floor=0.10, metric="cosine", seed=42),
    "stress": dict(n_items=80, dim=16, comb=4, k=10, anchors=25, p_work=0.1,
                   floor=0.05, metric="euclidean", seed=7),
    "wasserstein_demo": dict(n_items=60, dim=8, comb=3, k=8, anchors=20,
                             p_work=0.05, floor=0.10, metric="wasserstein", seed=99),
}

# ---------------------------------------------------------------- numpy ground truth
def pairwise_distance(X, metric):
    n = len(X)
    if metric == "cosine":
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        return 1.0 - Xn @ Xn.T
    if metric == "euclidean":
        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        return np.sqrt(d2)
    if metric == "wasserstein":          # 1-D histograms, ground cost |i-j|
        c = np.cumsum(X, axis=1)
        return np.abs(c[:, None, :] - c[None, :, :]).sum(-1)
    raise ValueError(metric)

def knn_from_dist(D, k):
    D = D.copy(); np.fill_diagonal(D, np.inf)
    idx = np.argsort(D, axis=1, kind="stable")[:, :k]
    return idx, np.take_along_axis(D, idx, axis=1)

def rows_without_self(indices, k):
    out = []
    for i, row in enumerate(np.asarray(indices)):
        r = [int(j) for j in row if int(j) >= 0 and int(j) != i][:k]
        out.append(r)
    return out

def recall_at_k(true_rows, approx_rows, D, k, tol=1e-9, normalize="expected"):
    """Fraction of true k-NN recovered, accepting distance ties at the k-th
    true-neighbour distance.

    "expected" uses fixed denominator k * number of true rows; missing edges
    count as misses. "returned" divides by the number of approximate edges
    actually evaluated (up to k per row), measuring correctness among returned
    edges rather than expected-neighbour recall.
    """
    if normalize not in ("expected", "returned"):
        raise ValueError("normalize must be 'expected' or 'returned'")
    hit = returned = 0
    for i, t in enumerate(true_rows):
        a = approx_rows[i] if i < len(approx_rows) else []
        edges = a[:k]
        returned += len(edges)
        if not t:
            continue
        kth = D[i, t[-1]]
        tset = set(t)
        for j in edges:
            hit += (j in tset) or (D[i, j] <= kth + tol)
    total = k * len(true_rows) if normalize == "expected" else returned
    return hit / total if total else 0.0

# ---------------------------------------------------------------- enumeration
def streamed_enumeration(S, size, top_n=200, max_combos=5_000_000, force=False):
    n = S.shape[0]
    total = math.comb(n, size)
    if total > max_combos and not force:
        raise SystemExit(f"{total:,} combinations > {max_combos:,}; use --force or shrink the profile.")
    pairs = list(itertools.combinations(range(size), 2))
    count, running, heap = 0, 0.0, []
    for combo in itertools.combinations(range(n), size):
        sc = sum(S[combo[a], combo[b]] for a, b in pairs) / len(pairs)
        count += 1; running += sc
        item = (sc, combo)
        if len(heap) < top_n: heapq.heappush(heap, item)
        elif sc > heap[0][0]: heapq.heapreplace(heap, item)
    return count, running / count, sorted(heap, reverse=True)

# ---------------------------------------------------------------- critic
@dataclass
class Verdict:
    decision: str; reason: str; floor: float; pipeline: str; run_id: str

def critic(mean_score, n, floor, pipeline, run_id):
    if n == 0:
        return Verdict("DENY", "empty result set", floor, pipeline, run_id)
    if mean_score >= floor:
        return Verdict("ALLOW", f"mean {mean_score:.4f} >= floor {floor:.4f}", floor, pipeline, run_id)
    if mean_score >= 0.9 * floor:
        return Verdict("ESCALATE", f"mean {mean_score:.4f} within 10% of floor {floor:.4f}", floor, pipeline, run_id)
    return Verdict("DENY", f"mean {mean_score:.4f} < floor {floor:.4f}", floor, pipeline, run_id)

# ---------------------------------------------------------------- annchor (UNTESTED)
def run_annchor(X, metric, cfg, seed):
    import annchor
    func = "wasserstein" if metric == "wasserstein" else (cosine_metric if metric == "cosine" else euclidean_metric)
    kw = {}
    if metric == "wasserstein":
        n = X.shape[1]
        kw["func_kwargs"] = {"cost_matrix": np.abs(np.subtract.outer(np.arange(n), np.arange(n))).astype(float)}
    saved_state = np.random.get_state()
    try:
        np.random.seed(seed)
        t0 = time.perf_counter()
        ann = annchor.Annchor(X, func, n_anchors=cfg["anchors"], n_neighbors=cfg["k"],
                              p_work=cfg["p_work"], **kw)
        ann.fit()
        el = time.perf_counter() - t0
        graph = ann.neighbor_graph
        cross = None
        try:
            bf = annchor.BruteForce(X, func, **kw); bf.fit()
            err = annchor.compare_neighbor_graphs(bf.neighbor_graph, graph, cfg["k"])
            cross = dict(incorrect=int(err), total=cfg["k"] * len(X))
        except Exception as e:
            cross = dict(error=str(e))
        return graph, el, cross
    finally:
        # Restores persistent process-wide state even for KeyboardInterrupt/SystemExit.
        # This is not thread isolation: concurrent code can observe the temporary seed.
        np.random.set_state(saved_state)

# ---------------------------------------------------------------- gaffer (UNTESTED)
def persist_gaffer(url, items, top, verdicts, run_id):
    gc = gaffer_connector.GafferConnector(url)
    gc.execute_get(g.GetSchema())
    els = [g.Entity(group="Item", vertex=i, properties={
        "feature_vector": json.dumps(v.tolist()), "source": "synthetic"}) for i, v in items]
    els += [g.Entity(group="Combination", vertex=f"{run_id}-bg-{n:05d}", properties={
        "item_ids": ",".join(ids), "score": float(sc), "method": "brute_genius",
        "objective_name": "mean_pairwise_cosine"}) for n, (sc, ids) in enumerate(top)]
    els += [g.Entity(group="CriticVerdict", vertex=f"{run_id}-{v.pipeline}", properties={
        "decision": v.decision, "reason": v.reason, "evidence_floor": v.floor,
        "pipeline": v.pipeline, "run_id": v.run_id}) for v in verdicts]
    for i in range(0, len(els), 500):
        gc.execute_operation(g.AddElements(input=els[i:i + 500]))

# ---------------------------------------------------------------- main
def run(profile, a):
    cfg = PROFILES[profile]; metric = cfg["metric"]
    run_id = f"{profile}-{int(time.time())}"
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
    if metric == "wasserstein":
        raw = np.abs(np.random.randn(cfg["n_items"], cfg["dim"]))
        X = raw / raw.sum(1, keepdims=True)
    else:
        X = np.random.randn(cfg["n_items"], cfg["dim"])
    ids = [f"item-{i:04d}" for i in range(len(X))]

    # objective: mean pairwise cosine similarity (higher = more similar)
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    S = Xn @ Xn.T

    t0 = time.perf_counter()
    n_comb, mean_all, top = streamed_enumeration(S, cfg["comb"], max_combos=a.max_combos, force=a.force)
    brute_t = time.perf_counter() - t0
    print(f"[brute] {n_comb:,} combinations, mean score {mean_all:.4f}, {brute_t:.2f}s")

    D = pairwise_distance(X, metric)
    t0 = time.perf_counter()
    true_idx, _ = knn_from_dist(D, cfg["k"]); true_t = time.perf_counter() - t0
    true_rows = [list(map(int, r)) for r in true_idx]
    mean_true_edges = float(np.mean([S[i, j] for i, r in enumerate(true_rows) for j in r]))
    rand_pairs = S[np.triu_indices(len(X), 1)]
    print(f"[exact-knn] numpy, {true_t:.3f}s; mean edge similarity {mean_true_edges:.4f} "
          f"(all-pairs mean {rand_pairs.mean():.4f})")

    out = dict(profile=profile, run_id=run_id, n_items=len(X), k=cfg["k"], metric=metric,
               n_combinations=n_comb, brute_mean=round(mean_all, 4), brute_s=round(brute_t, 3),
               exact_edge_mean=round(mean_true_edges, 4), annchor="skipped")
    verdicts = [critic(mean_all, n_comb, cfg["floor"], "brute_genius", run_id)]
    verdicts.append(critic(mean_true_edges, len(X) * cfg["k"], cfg["floor"], "exact_knn", run_id))

    if not a.no_annchor:
        try:
            graph, ann_t, cross = run_annchor(X, metric, cfg, cfg["seed"])
            rows = rows_without_self(graph[0], cfg["k"])
            rec = recall_at_k(true_rows, rows, D, cfg["k"])
            ann_mean = float(np.mean([S[i, j] for i, r in enumerate(rows) for j in r]))
            print(f"[annchor] {ann_t:.3f}s  recall@{cfg['k']} = {rec:.2%}  edge mean {ann_mean:.4f}  cross-check {cross}")
            verdicts.append(critic(ann_mean, sum(map(len, rows)), cfg["floor"], "annchor", run_id))
            out.update(annchor="ok", annchor_s=round(ann_t, 3), recall_at_k=round(rec, 4),
                       annchor_edge_mean=round(ann_mean, 4), annchor_crosscheck=cross)
        except Exception as e:
            print(f"[annchor] FAILED: {type(e).__name__}: {e}")
            out.update(annchor=f"failed: {e}")

    for v in verdicts:
        print(f"[critic] {v.pipeline:13s} {v.decision:8s} {v.reason}")
    out["verdicts"] = {v.pipeline: v.decision for v in verdicts}

    if not a.dry_run:
        if not GAFFER_AVAILABLE:
            print("[gaffer] gafferpy not installed; skipping")
        else:
            try:
                persist_gaffer(a.gaffer_url, list(zip(ids, X)),
                               [(sc, tuple(ids[i] for i in c)) for sc, c in top], verdicts, run_id)
                print("[gaffer] persisted items, top combinations, verdicts")
            except Exception as e:
                print(f"[gaffer] FAILED: {e}")
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="default", choices=PROFILES)
    p.add_argument("--dry-run", action="store_true", help="skip Gaffer")
    p.add_argument("--no-annchor", action="store_true")
    p.add_argument("--gaffer-url", default="http://localhost:8080/rest/latest")
    p.add_argument("--output", default="results.jsonl")
    p.add_argument("--max-combos", type=int, default=5_000_000)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    res = run(a.profile, a)
    with open(a.output, "a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
