# Exhaustive search vs ANNchor k-NN, with optional Gaffer storage

Public open-source libraries only (gchq/annchor, gchq/gafferpy). No affiliation with GCHQ.

## Files
- `baseline.py` – the experiment
- `elements.json`, `types.json` – Gaffer schema (entities Item, Combination, CriticVerdict)

## What it does
1. Generates synthetic items (random vectors; non-negative normalized histograms for the Wasserstein profile).
2. **Exhaustive enumeration** of all combinations, scored by mean pairwise cosine similarity. Streamed: only a running mean and the top 200 are kept.
3. **Exact k-NN in NumPy** (ground truth, independent of ANNchor).
4. **ANNchor approximate k-NN**, scored for recall@k against the exact graph (ANNchor's own `BruteForce` + `compare_neighbor_graphs` run as a cross-check).
5. **Evidence-floor verdicts** (ALLOW / ESCALATE / DENY) per result set.
6. Optional Gaffer persistence of items, top combinations and verdicts.

## Run
```bash
pip install numpy
pip install git+https://github.com/gchq/annchor.git      # optional
pip install git+https://github.com/gchq/gafferpy.git     # optional, archived project

python baseline.py --profile default --dry-run --no-annchor   # numpy only
python baseline.py --profile default --dry-run                # + ANNchor
python baseline.py --profile stress                           # + Gaffer (REST API up)
```
Each run appends a JSON line to `results.jsonl`.

## Tested vs untested
- **Tested here:** enumeration, NumPy exact k-NN, recall logic, critic, all three profiles with `--no-annchor`. The ANNchor code path was only exercised against a throwaway mock, which checks plumbing and says nothing about real ANNchor behavior.
- **Untested:** real `annchor` calls (`Annchor(...).fit()`, `.neighbor_graph`, `BruteForce.fit()`, `compare_neighbor_graphs`) and all Gaffer calls. Failures are caught and printed (`annchor: failed: ...` in the result line).
- Assumptions to check on first run: ANNchor accepts numba-jitted callables; `neighbor_graph[0]` is the index array (self may be included; the script strips it); constructor args used are only `n_anchors`, `n_neighbors`, `p_work`, `func_kwargs`.
- Gaffer schema: aggregate function class names in `types.json` and the schema-directory layout depend on your Gaffer version. Confirm against its docs.

## Reading the results
- On random Gaussian data the mean similarity over all combinations is about 0, so the exhaustive set is DENY against a 0.10 floor, while k-NN edges are nearest neighbors by construction and score ALLOW. That is a difference in what is being scored, not a ranking of the methods.
- The Wasserstein profile uses a 1-D ground cost |i-j| (an all-ones matrix would make every distance identical). Its histograms are all similar, so the exhaustive mean is high; the profile is for exercising the metric, not for comparing verdicts.
- Combination count grows fast: the script refuses more than 5M combinations unless `--force`.
- Critic and Brute Genius here are stubs standing in for reimagined-spoon and redesigned-fiesta.
