#!/usr/bin/env python3
"""Example: choose which retrieved passages to hand to an LLM (RAG context set).
Replace `embeddings` / `trust` with your retriever's output. Runs offline."""
import numpy as np
from weighted_scoring import load_weights, select_subset

rng = np.random.default_rng(0)
topic = rng.normal(size=16)
embeddings = np.vstack([topic + 0.4 * rng.normal(size=16) for _ in range(8)] +
                       [rng.normal(size=16) for _ in range(12)])      # 8 on-topic, 12 noise
trust = np.r_[rng.uniform(0.6, 1.0, 8), rng.uniform(0.0, 0.7, 12)]    # e.g. source reliability

res = select_subset(embeddings, trust, size=4, weights=load_weights("weights.json"), pool=16)
print("chosen passages:", res["indices"])
print(f"S = {res['score']:.3f}  (coherence {res['coherence']:.3f}, "
      f"weakest link {res['weakest_link']:.3f}, trust {res['trust']:.3f})")
print("verdict:", res["verdict"])
if res["verdict"] == "ALLOW":
    print("-> pass these passages to the model")
elif res["verdict"] == "ESCALATE":
    print("-> send to human review or re-retrieve before use")
else:
    print("-> do not use; answer 'insufficient evidence'")
