# Weighted Combination Search with Governance Gate 🇺🇸 🇬🇧

**Built in the United States 🇺🇸 on open-source software published by the United Kingdom's GCHQ 🇬🇧**
([Gaffer](https://github.com/gchq/Gaffer), [ANNchor](https://github.com/gchq/annchor)).
This is an independent research project. It is **not** an official United States or United Kingdom
government project, and it is not affiliated with or endorsed by GCHQ.

---

## 1. The idea in plain English

Imagine you must pick the **best team of 4 people** from 80 candidates.

- The careful way is to try **every possible team** and score each. That is **exhaustive search**.
- The fast way is to look for nearby candidates using a **k-nearest-neighbor graph**. An approximate graph such as ANNchor's can be faster, but may miss neighbors.
- Before acting on a result, an **evidence-floor gate** classifies its score as **ALLOW**, **ESCALATE** (ask a human), or **DENY**.

This repository contains **two related but separate workflows**:

1. `baseline.py` compares streamed exhaustive combination enumeration with NumPy exact k-NN, and has optional ANNchor and Gaffer paths. Its exhaustive objective is mean pairwise cosine similarity.
2. `weighted_scoring.py` implements a separate weighted subset selector using coherence, weakest-link similarity, and supplied trust priors. `integrate_example.py` demonstrates that selector offline.

**The baseline experiment does not import or invoke `weighted_scoring.py`.** The two workflows are in the same repository, but they are not one integrated pipeline. ANNchor is not currently wired into `weighted_scoring.py`'s pool-pruning function.

For AI, the weighted example can be adapted to select documents, tools, or passages for a model. It is a reference implementation, not evidence that a production retrieval system is safe or accurate.

## 2. What's in the repo

| File | Purpose |
|---|---|
| `baseline.py` | Experiment: streamed exhaustive search vs NumPy exact k-NN, optional ANNchor, verdicts, optional Gaffer storage |
| `weighted_scoring.py` | Separate weighted subset selection and evidence-floor gate |
| `weights.json` | Default scoring weights and evidence floor |
| `integrate_example.py` | Offline example using the weighted selector for a RAG context set |
| `elements.json`, `types.json` | Gaffer schema definitions (not validated against a live Gaffer service) |
| `test_weighted.py` | Weighted-scoring unit tests; located at the repository root |
| `MANUAL.md` | Setup, formula, integration recipes, tuning, and troubleshooting |
| `BASELINE_README.md` | Documentation for the original exhaustive-search / ANNchor baseline |

Other tracked files include `LICENSE`, `requirements.txt`, `push.sh`,
`gaffer-annchor-baseline.zip`, `README (7).md`, and
`research/exhaustive-set/IMPORT-AUDIT.md`.
`README (7).md` is byte-identical to `README.md`; `requirements (1).txt`
duplicates `requirements.txt`, and `push (1).sh` duplicates `push.sh`.
The duplicate download-name files are retained in the current tree; they are
not required by the workflows described here.

## 3. The weighted formula (graduate-level statement)

Let items \(i = 1..n\) have unit-normalized embeddings \(x_i\), cosine similarities \(s_{ij} = x_i \cdot x_j\),
and trust priors \(v_i \in [0,1]\). For a combination \(C\) of size \(m\):

- Coherence: \( \mu(C) = \binom{m}{2}^{-1} \sum_{i<j \in C} s_{ij} \)
- Weakest link: \( \lambda(C) = \min_{i<j \in C} s_{ij} \)
- Trust: \( \tau(C) = \frac{1}{m} \sum_{i \in C} v_i \)

\[ S(C) = w_\mu\,\mu(C) + w_\lambda\,\lambda(C) + w_\tau\,\tau(C), \qquad w \ge 0,\ \textstyle\sum w = 1 \]

The gate with evidence floor \( \theta \):

\[ \text{verdict}(C) = \begin{cases} \text{ALLOW} & S(C) \ge \theta \\ \text{ESCALATE} & 0.9\theta \le S(C) < \theta \\ \text{DENY} & S(C) < 0.9\theta \end{cases} \]

The weighted selector searches combinations within a candidate pool \(P\) and returns the highest-scoring subset. Its current `prune_pool` implementation ranks candidates using supplied trust priors and local cosine density; it does not call ANNchor. The ANNchor path in `baseline.py` is a separate experiment.

**Layman translation:** \(\mu\) is "how alike the group is on average", \(\lambda\) is "how bad is the worst pairing"
(a chain is only as strong as its weakest link), and \(\tau\) is "how much we trust the sources." The weights are
three dials that add to 100%. \(\theta\) is the bar the final score must clear.

### Default weights (`weights.json`)

| Dial | Default | Raise it when |
|---|---|---|
| `coherence` | 0.50 | you want a tightly focused set |
| `weakest_link` | 0.30 | one off-topic item would be harmful (e.g. legal, medical, safety contexts) |
| `trust` | 0.20 | source reliability varies widely |
| `evidence_floor` | 0.30 | raise to be stricter, lower to be more permissive |

These defaults are starting points chosen for illustration, **not** values fitted to any dataset. Calibrate them (Manual, Section 6).

## 4. Quick start

```bash
pip install -r requirements.txt
python integrate_example.py                       # weighted selection demo
python test_weighted.py                            # weighted formula tests (root-level file)
python baseline.py --profile default --dry-run --no-annchor      # NumPy baseline only
python baseline.py --profile default --dry-run                   # + ANNchor (unverified)
```

## 5. What is verified and what is not

**Reported tested in development:** the weighted formula and gate (unit tests), the exhaustive enumerator, the NumPy exact k-NN,
recall@k logic, and the three experiment profiles with `--no-annchor`. These reported runs have not been independently reproduced as part of this documentation correction.

**Not verified:** real ANNchor calls and all Gaffer calls (no network in the build environment). The ANNchor code path
was exercised only against a mock. Run it yourself and read `results.jsonl` (`annchor: ok` or `failed: ...`).
Also unverified: the aggregate-function class names in `types.json`, which depend on your Gaffer version.

**Profile geometry matters:** the `stress` profile uses Euclidean distance for k-NN construction, while exhaustive
combinations are still scored by mean pairwise cosine similarity. The `wasserstein_demo` profile likewise uses
Wasserstein distance for k-NN while exhaustive combinations remain cosine-scored. The k-NN and exhaustive
measurements in those profiles therefore use different geometries and are not direct comparisons of the same objective.

**Honest limits of the method:**
1. Cosine similarity of embeddings is a proxy for relevance; it can be wrong.
2. Trust priors \(v_i\) are inputs you supply; the formula cannot make them accurate.
3. The 0.9 ESCALATE band and the default weights are design choices, not empirical findings.
4. On random data, exhaustive vs k-NN verdicts differ because they score different sets of pairs, not because one method is better.
5. "Weights" here means scoring weights. This repo does not ship trained neural-network weights.
6. gafferpy is archived upstream; pin a version you have tested.
7. No voice-cloning capability is demonstrated by this repository.

## 6. Licensing and attribution

This repository is licensed under the MIT License; see [LICENSE](LICENSE).
Gaffer and ANNchor are separate upstream projects under their own licenses; check
those licenses before redistribution.
