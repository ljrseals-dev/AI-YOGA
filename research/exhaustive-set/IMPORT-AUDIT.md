# Exhaustive Set — Import Audit

**Repository:** `ljrseals-dev/AI-YOGA`  
**Branch reviewed:** `main`  
**Audit date:** 2026-11-27  
**Scope:** Archive import history plus the current four-file baseline: `baseline.py`, `README.md`, `elements.json`, and `types.json`.

## 1. Current import status

The four named implementation files are present on `main`:

- `baseline.py`
- `README.md`
- `elements.json`
- `types.json`

The root README and this audit report are also present on `main`. The earlier statement that the audit branch contained only a report described the earlier PR state; it is not a current description of the four files now on `main`.

This audit reviews the contents currently visible on `main`. It does not claim that every file from the original ZIP has been imported.

## 2. Archive inventory and historical findings

The supplied archive inventory previously recorded these files:

- `weights.json`
- `weighted_scoring.py`
- `test_weighted.py`
- `requirements.txt`
- `push.sh`
- `integrate_example.py`
- `MANUAL.md`
- `BASELINE_README.md`
- `gaffer-annchor-baseline.zip`
- `types.json`
- `elements.json`
- `README (7).md`
- `baseline.py`

The earlier inventory also recorded byte-identical duplicate files `requirements (1).txt` / `requirements.txt` and `push (1).sh` / `push.sh`. The nested ZIP was reported to contain duplicate source, bytecode cache, and a `.git` directory without Git objects.

Those are historical archive-inventory findings, not a fresh byte-for-byte comparison of every archive entry against the current Git tree.

## 3. Current implementation review

### `baseline.py`

**What the source implements:**

- Three named profiles: `default`, `stress`, and `wasserstein_demo`.
- NumPy pairwise-distance calculation and exact k-NN construction.
- Streaming combination enumeration, retaining a running mean and a bounded top-results heap rather than storing every combination.
- A simple evidence-floor critic returning `ALLOW`, `ESCALATE`, or `DENY`.
- An optional ANNchor execution path and optional Gaffer persistence path, each wrapped in exception handling.
- JSONL result output.

**Important limitations and review findings:**

1. The exhaustive-search objective is mean pairwise cosine similarity. It does not implement the weighted coherence / weakest-link / trust formula described in the README.
2. The exact k-NN graph is calculated in NumPy. ANNchor's graph and its `BruteForce` / `compare_neighbor_graphs` functions are additional integration and cross-check paths, not the source of the NumPy ground truth.
3. ANNchor behavior is not verified by merely reading this source. The actual installed package must confirm callable metric support, constructor arguments, graph tuple layout, and the comparison function's signature and return semantics.
4. The Wasserstein NumPy distance uses cumulative histogram differences. The ANNchor path separately passes a string metric and a cost matrix through `func_kwargs`; that integration remains unverified until exercised against the real package.
5. `recall_at_k` assumes each true-neighbor row has at least one element and indexes its final element. The current profiles satisfy `k < n_items`, but the helper is not defensive for empty or undersized rows.
6. The reported critic verdicts are threshold classifications of the measured mean scores. They are not calibrated quality judgments and do not establish that one search method is superior.
7. The Gaffer persistence path uses class names and schema conventions that have not been validated against a live, compatible Gaffer installation.

### `elements.json` and `types.json`

The schema is split into two JSON files. `elements.json` defines Item, Combination, and CriticVerdict entities; `types.json` defines the referenced string and maximum-double types.

The JSON content is readable and structurally separated. This review does not establish that Gaffer accepts the schema, that the aggregate-function class names resolve in the target Gaffer version, or that the Python persistence calls successfully write and retrieve these entities.

### `README.md`

The README currently describes more files and functionality than the four-file implementation demonstrates. In particular, its repository inventory lists `weighted_scoring.py`, `weights.json`, `integrate_example.py`, `tests/test_weighted.py`, and `docs/` files, and its formula section presents a weighted selector as part of the project. The reviewed `baseline.py` does not import or invoke that weighted selector.

The README should be revised to distinguish:
- the current four-file baseline;
- any weighted-scoring modules that are actually present in the repository;
- historical archive contents that are not part of the current runnable baseline;
- verified NumPy behavior from unverified ANNchor and Gaffer integrations.

The README's GCHQ attribution wording should remain explicitly independent and non-affiliated. Use neutral attribution to the upstream open-source projects; do not imply government sponsorship, endorsement, or affiliation.

## 4. Interpretation of the reported test results

The supplied development notes report that all three profiles ran in NumPy-only mode, with the stress profile completing in approximately three seconds. These are reported results from the development run; this audit did not independently execute the profiles.

The reported default-profile outcomes were:
- exhaustive combination mean approximately 0.005 against a 0.10 floor: `DENY`;
- exact k-NN edge mean approximately 0.55 against the same floor: `ALLOW`.

The two pipelines score different populations of pairs. The exhaustive mean averages pairwise similarities across all combinations; the exact k-NN mean averages similarities among nearest-neighbor edges. Their different verdicts are therefore not a controlled algorithm-quality comparison.

The Wasserstein profile's reported `ALLOW` outcomes likewise describe the generated data and selected objective, not a general result about Wasserstein search quality.

## 5. Verification status

| Area | Status | What remains |
|---|---|---|
| Four named files on `main` | Present | Keep repository inventory synchronized |
| NumPy exact k-NN | Implemented in source; reported tested | Re-run in the target environment |
| Streamed exhaustive enumeration | Implemented in source; reported tested | Preserve stress-profile regression coverage |
| Three NumPy-only profiles | Reported passing | Independent reproducible run and retained logs |
| ANNchor | Unverified against real package | Install a pinned compatible version and validate API, graph shape, recall, and metrics |
| Gaffer schema and persistence | Unverified against real service | Validate schema loading, aggregate classes, writes, and reads |
| Weighted scoring integration | Not implemented in reviewed `baseline.py` | Integrate only after module presence, API, and tests are confirmed |
| README accuracy | Needs correction | Align file list and claims with actual tracked files |

## 6. Recommended actions

1. Correct `README.md` so its file inventory and feature claims match the current Git tree and the reviewed `baseline.py`.
2. Keep the unweighted baseline as a clearly named reference implementation.
3. Treat weighted scoring as a separate, not-yet-integrated component until its source files are present and its tests are run from the repository.
4. Add regression tests for empty/undersized neighbor rows, ties, zero vectors, invalid profile dimensions, and combination-count limits.
5. Validate ANNchor against a real, pinned installation; record the exact version and observed API behavior.
6. Validate Gaffer schema loading and persistence against a real compatible service; do not infer success from JSON validity or mocked calls.
7. Avoid using the bundled direct-to-`main` push script. Continue with branch-first changes and review before merging.

## 7. Audit conclusion

**Disposition: PARTIALLY VERIFIED — INTEGRATIONS AND DOCUMENTATION REQUIRE FOLLOW-UP.**

The four named files are present on `main`, and the source contains a self-contained NumPy baseline with streamed enumeration and exact k-NN calculation. The development notes report successful NumPy-only profile runs, but those results were not independently reproduced in this audit. ANNchor and Gaffer remain unverified against real installations. The README currently overstates the integration of weighted scoring and lists supporting files that must be checked against the actual repository tree.

No claim of GCHQ affiliation, official endorsement, or government involvement is made or supported by this audit.
