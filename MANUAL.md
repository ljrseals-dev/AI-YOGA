# Manual

## 1. Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/gchq/annchor.git        # optional
pip install git+https://github.com/gchq/gafferpy.git       # optional, archived
python tests/test_weighted.py
```
Python 3.9+. Only NumPy is required for the weighted formula.

## 2. Concepts for non-specialists
- **Embedding:** a list of numbers that represents a piece of content so that similar content has similar numbers.
- **Cosine similarity:** the angle-based closeness of two embeddings, 1 = same direction, 0 = unrelated, negative = opposed.
- **Combination:** a chosen group of items (for example 4 passages).
- **Exhaustive search:** score every possible group; slow but certain.
- **k-NN graph:** for each item, a list of its k closest neighbors.
- **Recall@k:** of the true k nearest neighbors, the fraction the fast method found.
- **Evidence floor:** the minimum score needed to proceed.

## 3. The formula, step by step
1. Normalize each embedding to length 1.
2. Compute the similarity matrix \(S = XX^\top\).
3. For a candidate group, take its pairwise similarities. \(\mu\) = their mean, \(\lambda\) = their minimum.
4. \(\tau\) = mean trust of the members.
5. \(S(C) = w_\mu\mu + w_\lambda\lambda + w_\tau\tau\) with weights normalized to sum to 1.
6. Compare to the floor \(\theta\): ALLOW, ESCALATE (within 10% below), or DENY.

Worked example (weights 0.5/0.3/0.2, floor 0.3): a group with \(\mu=0.90\), \(\lambda=0.87\), \(\tau=0.91\) scores
\(0.45 + 0.261 + 0.182 = 0.893\), which is ALLOW. A group with \(\mu=0.05\), \(\lambda=-0.2\), \(\tau=0.5\) scores
\(0.025 - 0.06 + 0.10 = 0.065\), which is DENY.

## 4. Integrating into an AI system

### 4.1 Retrieval-augmented generation (choose the context set)
```python
from weighted_scoring import load_weights, select_subset
res = select_subset(passage_embeddings, source_trust, size=4,
                    weights=load_weights("weights.json"), pool=24)
if res["verdict"] == "ALLOW":
    context = [passages[i] for i in res["indices"]]    # give to the model
elif res["verdict"] == "ESCALATE":
    ...                                                # human review or re-retrieve
else:
    ...                                                # answer "insufficient evidence"
```
`passage_embeddings` is an \(n \times d\) array from any embedding model; `source_trust` is one value in [0,1]
per passage (publisher reliability, recency, human rating, etc.).

### 4.2 Tool or agent selection
Embed tool descriptions and the user request; set trust to historical success rate; choose a small set of tools.
A DENY means no tool combination is a confident match, so ask the user a clarifying question.

### 4.3 Ensemble or model-output screening
Treat several model answers as items. High coherence means they agree; a low weakest link flags an outlier answer.
DENY or ESCALATE triggers a re-run or human check.

### 4.4 Scaling past a few dozen candidates
`select_subset` prunes to `pool` items (trust plus local density) before enumerating. If your similarity is expensive,
build an ANNchor k-NN graph (`baseline.py` shows the call) and use it to choose the pool. The enumeration cost is
\(\binom{pool}{m}\); the function refuses more than 2,000,000 combinations by default.

## 5. Running the experiment
`python baseline.py --profile {default|stress|wasserstein_demo} [--dry-run] [--no-annchor]`
Output: combination count, exact k-NN edge quality, ANNchor recall@k (if available), verdicts, and a JSON line in
`results.jsonl`. `--force` lifts the 5,000,000-combination guard.

## 6. Calibrating the weights and floor
1. Collect 50 to 200 cases where a human labeled the right set (or "no good set exists").
2. Run `select_subset` with a grid of weights (for example steps of 0.1 on the simplex).
3. Choose weights maximizing agreement with the labels; set \(\theta\) so that "no good set" cases are DENY at your
   tolerated false-allow rate.
4. Re-check on held-out cases. Do not tune on your test set.

## 7. Gaffer storage (optional, untested here)
`elements.json` and `types.json` define Item, Combination and CriticVerdict entities. `baseline.py` writes
items, top combinations and verdicts when run without `--dry-run`. Confirm the schema loading mechanism and the
aggregate function class names for your Gaffer version.

## 8. Troubleshooting
| Symptom | Likely cause |
|---|---|
| `annchor: failed: ...` | API mismatch or numba rejecting a callable; read the message, try a built-in metric string |
| Everything DENY | floor too high, or items genuinely unrelated; inspect `coherence` and `weakest_link` |
| Everything ALLOW | floor too low, or `trust` dominating; shift weight to coherence |
| "too many combinations" | lower `pool` or `size` |
| Gaffer errors | gafferpy is archived; confirm versions and schema |

## 9. Security and responsible use
Do not use the verdict as the only control for safety-critical decisions. Log verdicts, keep a human path for ESCALATE,
and review the trust inputs, since an attacker who controls them controls the outcome.
