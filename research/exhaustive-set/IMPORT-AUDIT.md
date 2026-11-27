# Exhaustive Set — Import Audit

Source archive: `Exhaustive set.zip` (uploaded to ChatGPT, 52 KB)
Date: 2026-11-27

## Archive inventory

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

Duplicate files found and verified byte-identical:
- `requirements (1).txt` and `requirements.txt`
- `push (1).sh` and `push.sh`

The nested ZIP contains duplicate project source, bytecode cache, and a .git directory with no Git objects. It is not a meaningful history archive.

## Audit observations

- Six bundled weighted-scoring unit tests passed in the local review.
- Weighted selector and baseline experiment are not integrated: the baseline uses mean pairwise similarity, not the weighted coherence/weakest-link/trust formula.
- Trust priors and several selector inputs lack strict validation.
- Candidate pruning can discard the global best subset; results are exact only within the pruned pool.
- Default gate thresholds and weights are not calibrated against labeled evidence.
- Baseline and k-NN results use different statistical quantities and are not a fair method comparison.
- The bundled push script pushes directly to main; do not use as-is under branch-first/PR review policy.

## Import limitation

This PR contains this audit report only, not the extracted project source. The GitHub file-write connector available in this session supports UTF-8 text content but cannot transfer the uploaded ZIP's local files directly. The nested binary ZIP is also not uploaded. This limitation is stated explicitly to avoid falsely claiming that the archive was extracted into Git.

## Suggested next steps

1. Transfer the source files into this folder in a follow-up commit (omit bytecode and duplicate files; retain a single requirements file).
2. Harden input validation and edge-case handling.
3. Integrate weighted scoring into baseline experiments while retaining the unweighted baseline.
4. Add an exhaustive oracle comparison to quantify pruning loss.
5. Calibrate the gate with labeled, held-out examples.
6. Validate ANNchor and Gaffer integrations against real installations.
