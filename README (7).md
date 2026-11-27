# Weighted Combination Search with Governance Gate 🇺🇸 🇬🇧

**Built in the United States 🇺🇸 on open-source software published by the United Kingdom's GCHQ 🇬🇧**
([Gaffer](https://github.com/gchq/Gaffer), [ANNchor](https://github.com/gchq/annchor)).
This is an independent research project. It is **not** an official United States or United Kingdom
government project, and it is not affiliated with or endorsed by GCHQ.

---

## 1. The idea in plain English

Imagine you must pick the **best team of 4 people** from 80 candidates.

- The careful way is to try **every possible team** and score each. That is always right, but the number of
  teams explodes as the pool grows. This is **exhaustive search**.
- The fast way is to look only at people who already look like good matches for each other and ignore the
  rest. It is quick but can miss a good team. This is **approximate nearest-neighbor search** (ANNchor).
- Before you act on the answer, a **referee** asks: "Is the evidence strong enough?" The referee says
  **ALLOW**, **ESCALATE** (ask a human), or **DENY**. This is the **evidence-floor gate**.

This repo does all three on the same data so you can measure what the fast way gives up, and it adds a
**weighted score** (Section 3) so you decide what "best" means: how well the members fit together, how weak the
weakest pair is, and how much you trust each member.

For AI: replace "people" with "documents, tools, or facts an AI could use." The method picks the best
set to hand to a model and refuses to proceed when the evidence is weak.

## 2. What's in the repo

| File | Purpose |
|---|---|
| `baseline.py` | Experiment: exhaustive search vs exact k-NN vs ANNchor, recall@k, verdicts, optional Gaffer storage |
| `weighted_scoring.py` | **Weighted** subset selection and the evidence-floor gate (the integration formula) |
| `weights.json` | The weights and the floor. Edit this file to change behavior |
| `integrate_example.py` | Offline example: choose passages for a RAG prompt |
| `elements.json`, `types.json` | Gaffer schema (Item, Combination, CriticVerdict) |
| `tests/test_weighted.py` | Unit tests for the formula |
| `docs/MANUAL.md` | Full manual: setup, formula derivation, AI integration recipes, tuning, troubleshooting |
| `docs/BASELINE_README.md` | The original baseline README, kept unchanged |

## 3. The formula (graduate-level statement)

Let items \(i = 1..n\) have unit-normalized embeddings \(x_i\), cosine similarities \(s_{ij} = x_i \cdot x_j\),
and trust priors \(v_i \in [0,1]\). For a combination \(C\) of size \(m\):

- Coherence: \( \mu(C) = \binom{m}{2}^{-1} \sum_{i<j \in C} s_{ij} \)
- Weakest link: \( \lambda(C) = \min_{i<j \in C} s_{ij} \)
- Trust: \( \tau(C) = \frac{1}{m} \sum_{i \in C} v_i \)

\[ S(C) = w_\mu\,\mu(C) + w_\lambda\,\lambda(C) + w_\tau\,\tau(C), \qquad w \ge 0,\ \textstyle\sum w = 1 \]

The gate with evidence floor \( \theta \):

\[ \text{verdict}(C) = \begin{cases} \text{ALLOW} & S(C) \ge \theta \\ \text{ESCALATE} & 0.9\theta \le S(C) < \theta \\ \text{DENY} & S(C) < 0.9\theta \end{cases} \]

Search: \( C^\* = \arg\max_{C \subseteq P,\ |C| = m} S(C) \), where \(P\) is a candidate pool pruned from the \(n\) items.
Cost is \( \binom{|P|}{m} \cdot \binom{m}{2} \) similarity lookups, which is why pruning \(P\) matters.
That is the role of ANNchor: build an approximate k-NN graph cheaply when the metric is slow, then enumerate exhaustively
inside the pruned pool.

**Layman translation:** \(\mu\) is "how alike the group is on average", \(\lambda\) is "how bad is the worst pairing"
(a chain is only as strong as its weakest link), \(\tau\) is "how much we trust the sources." The weights are
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
python tests/test_weighted.py                     # formula tests
python baseline.py --profile default --dry-run --no-annchor      # exhaustive vs exact k-NN
python baseline.py --profile default --dry-run                   # + ANNchor (needs annchor)
```

## 5. What is verified and what is not

**Verified in development:** the weighted formula and gate (unit tests), the exhaustive enumerator, the NumPy exact k-NN,
recall@k logic, the three experiment profiles with `--no-annchor`.

**Not verified:** real ANNchor calls and all Gaffer calls (no network in the build environment). The ANNchor code path
was exercised only against a mock. Run it yourself and read `results.jsonl` (`annchor: ok` or `failed: ...`).
Also unverified: the aggregate-function class names in `types.json`, which depend on your Gaffer version.

**Honest limits of the method:**
1. Cosine similarity of embeddings is a proxy for relevance; it can be wrong.
2. Trust priors \(v_i\) are inputs you supply; the formula cannot make them accurate.
3. The 0.9 ESCALATE band and the default weights are design choices, not empirical findings.
4. On random data, exhaustive vs k-NN verdicts differ because they score different sets of pairs, not because one method is better.
5. "Weights" here means scoring weights. This repo does not ship trained neural-network weights.
6. gafferpy is archived upstream; pin a version you have tested.

## 6. Licensing and attribution

No license file is included yet; add one before publishing (GitHub's "Add a license" flow lists the options).
Gaffer and ANNchor are separate projects under their own licenses; check them before redistribution.
