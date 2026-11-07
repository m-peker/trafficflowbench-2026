"""One score scale for two formulations, so everything downstream stays the same.

The classifier outputs P(queued). The speed model outputs a predicted speed
ratio, where *low* means queued and the flag is `r <= 0.6`. Mapping the ratio to

    p = clip(0.5 + (0.6 - r_hat), 0, 1)

puts them on one monotone scale: p = 0.5 is exactly the 0.6 ratio cutoff, higher
p means more queue-like either way, and every existing threshold sweep,
aggregation and averaging path works on both without knowing which it has.
"""
import numpy as np

CUT = 0.6


def kind_of(bundle):
    """'reg' or 'clf', from the bundle if it says so, else from the output range."""
    k = bundle.get("kind")
    if k:
        return k
    return "clf"


def to_score(kind, raw):
    if kind == "reg":
        return np.clip(0.5 + (CUT - np.asarray(raw, float)), 0.0, 1.0)
    return np.asarray(raw, float)
