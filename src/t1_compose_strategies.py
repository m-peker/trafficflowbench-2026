"""Build several different Task 1 state files from the same cached net fields.

Five genuinely different ways of combining the same underlying predictions,
so one predict_grid pass per net (via t1_field_cache.py) serves all of them:

  swap        replace the weakest original member (m1) outright
  soft_add    add a new member at LESS than equal weight (every earlier "add"
              used equal weight and cost the board ~0.0005; a fractional voice
              has never been tried)
  khead_swap  replace m1 with a density-head net, blending its own flow head
              with its density-derived flow at its locally-best beta
  trimmed     drop the single highest and lowest prediction per cell across a
              wide pool of members, average the rest - robust to whichever
              member is having a bad cell, untried all session
  khead_pair  two density-head members share one beta (see t1_khead_sweep2.py)

Each field cache (`_field_<net>.csv`, from t1_field_cache.py) is written by the
same panel/split loop in the same order as every other, so rows line up
positionally - checked once per pair, not re-joined every time.

    python t1_compose_strategies.py swap
    python t1_compose_strategies.py all
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
NETW = 0.85
TREES = RUNS / "s15_onset-step6" / "out" / "state_submission.csv"


def load(name):
    f = RUNS / f"_field_{name}.csv"
    df = pd.read_csv(f, dtype={"link_id": str})
    return df


_KEY = None


def check_aligned(df):
    global _KEY
    key = df[["panel", "timestamp", "link_id"]]
    if _KEY is None:
        _KEY = key.reset_index(drop=True)
    else:
        assert (key.reset_index(drop=True) == _KEY).all().all(), "row order mismatch"


def arrays(name):
    df = load(name)
    check_aligned(df)
    v = df.speed_kmh.to_numpy(np.float64)
    q = df.flow_vph.to_numpy(np.float64)
    k = df.density.to_numpy(np.float64) if "density" in df.columns else None
    return v, q, k


def bounds():
    """vf/cap per row, from the trees file merged the same positional way."""
    tree = load_tree()
    return tree


def load_tree():
    df = pd.read_csv(TREES, dtype={"link_id": str, "station_id": str})
    check_aligned(df)
    return df


def write(tag, v, q, tree_df):
    v_final = NETW * v + (1 - NETW) * tree_df.speed_kmh.to_numpy(np.float64)
    q_final = NETW * q + (1 - NETW) * tree_df.flow_vph.to_numpy(np.float64)
    out = tree_df[["panel", "timestamp", "station_id", "link_id", "mask_regime"]].copy()
    out["speed_kmh"] = v_final
    out["flow_vph"] = q_final
    out_dir = RUNS / tag / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "state_submission.csv", index=False)
    print(f"wrote {tag}: {len(out):,} rows, mean speed {v_final.mean():.2f}, "
         f"mean flow {q_final.mean():.1f}", flush=True)


def strategy_swap():
    """s62: {f1,h1,i1,j1,n1,v1} - m1 replaced by v1, plain mean of six."""
    names = ["f1", "h1", "i1", "j1", "n1", "v1"]
    vs, qs = zip(*[arrays(n)[:2] for n in names])
    write("s62_t1_swap-v1", np.mean(vs, 0), np.mean(qs, 0), load_tree())


def strategy_soft_add():
    """s63: the untouched six at weight 1, v1 at weight 0.3 - a fractional
    voice instead of every earlier equal-weight add (which always cost ~0.0005)."""
    six = ["f1", "h1", "i1", "j1", "m1", "n1"]
    vs, qs = zip(*[arrays(n)[:2] for n in six])
    v6, q6 = np.sum(vs, 0), np.sum(qs, 0)
    v1, q1, _ = arrays("v1")
    w = 0.3
    v = (v6 + w * v1) / (6 + w)
    q = (q6 + w * q1) / (6 + w)
    write("s63_t1_soft-add-v1", v, q, load_tree())


def strategy_khead_swap(beta=0.3):
    """s64: {f1,h1,i1,j1,n1} plain + dn1 blended at its own best local beta,
    replacing m1 - the density mechanism tested directly on the board."""
    names = ["f1", "h1", "i1", "j1", "n1"]
    vs, qs = zip(*[arrays(n)[:2] for n in names])
    v_dn1, q_head, k_dn1 = arrays("dn1")
    q_dn1 = (1 - beta) * q_head + beta * (k_dn1 * v_dn1)
    v = (sum(vs) + v_dn1) / 6
    q = (sum(qs) + q_dn1) / 6
    write("s64_t1_dn1-swap", v, q, load_tree())


def strategy_trimmed():
    """s65: drop the highest and lowest prediction per cell across a nine-net
    pool, average the remaining seven - robust to any one member's bad cells,
    untried all session. Density-head nets contribute their plain flow head
    here (beta=0), since this strategy is about robustness, not the density
    mechanism."""
    pool = ["f1", "h1", "i1", "j1", "m1", "n1", "v1", "dn1", "dn2"]
    vs, qs = [], []
    for n in pool:
        v, q, _ = arrays(n)
        vs.append(v); qs.append(q)
    V = np.stack(vs); Q = np.stack(qs)
    def trimmed_mean(X):
        Xs = np.sort(X, axis=0)
        return Xs[1:-1].mean(0)          # drop one highest, one lowest
    write("s65_t1_trimmed-mean", trimmed_mean(V), trimmed_mean(Q), load_tree())


def strategy_combined(beta=0.3):
    """s66: the two individually-promising changes together, not a pair of
    density heads (the sweep showed pairing dn1+dn2 does not preserve the
    individual gain any better than one did - still flat, 0.44877-0.44884).
    Swap -m1 +v1 (the only clean win of twelve swaps tried, 2026-09-24) AND
    swap -h1 +dn1 at its own best beta (the density mechanism, tested on the
    board since local dilution cannot see it) at the same time - the biggest,
    most speculative bet of the five: two different real findings compounded."""
    names = ["f1", "i1", "j1", "n1", "v1"]
    vs, qs = zip(*[arrays(n)[:2] for n in names])
    v_dn1, q_head, k_dn1 = arrays("dn1")
    q_dn1 = (1 - beta) * q_head + beta * (k_dn1 * v_dn1)
    v = (sum(vs) + v_dn1) / 6
    q = (sum(qs) + q_dn1) / 6
    write("s66_t1_combined", v, q, load_tree())


STRATEGIES = {
    "swap": strategy_swap, "soft_add": strategy_soft_add,
    "khead_swap": strategy_khead_swap, "trimmed": strategy_trimmed,
    "combined": strategy_combined,
}


def main():
    names = sys.argv[1:] or ["all"]
    todo = list(STRATEGIES) if names == ["all"] else names
    for n in todo:
        STRATEGIES[n]()


if __name__ == "__main__":
    main()
