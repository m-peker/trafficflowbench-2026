"""Which transitions does the board's S_LWR score?  Board implies ~0.51.

For one field (the tree blend, fast; the conventions are what is compared):
    A  every finite transition (the local proxy all session; obs->obs pairs
       add zero residual but count in the denominator)
    C  transitions with at least one submitted (target) endpoint
    B  transitions with both endpoints submitted - what a scorer that only
       reads the state rows can compute

    python lwr_convention.py
"""
import nice  # noqa: F401
import warnings
import numpy as np
warnings.filterwarnings("ignore")
import ens_search as E
from t1_phys_compare import W


def main():
    rows = []
    for panel in E.PANELS[:4]:
        P = E.Panel(panel)
        tj, ti = P.tj, P.ti
        v = np.clip(P.mix(W, "speed"), 5.0, P.vf[tj] * 1.05)
        k = P.mix(W, "density")
        q = np.clip(((1 - E.FLOW_BLEND) * P.mix(W, "flow") + E.FLOW_BLEND * k * v)
                    * P.lanes[tj], 5.0, P.cap[tj] * 1.15)
        sp = P.obs_sp.astype(np.float64).copy(); fl = P.obs_fl.astype(np.float64).copy()
        sp[ti, tj] = v; fl[ti, tj] = q
        N = (fl / np.maximum(sp, 1.0)) * P.length[None, :]
        dN = N[1:] - N[:-1]
        tgt = np.zeros(sp.shape, bool); tgt[ti, tj] = True
        base = np.isfinite(dN) & P.dNt_ok
        conv = {"A": base, "C": base & (tgt[1:] | tgt[:-1]), "B": base & tgt[1:] & tgt[:-1]}
        out = []
        for name in ("A", "C", "B"):
            m = conv[name]
            e = np.abs(dN[m] - P.dNt[m]).sum() / (np.abs(P.dNt[m]).sum() + 1e-9)
            out.append((max(0.0, 1 - min(1.0, e)), m.sum() / base.sum()))
        rows.append(out)
        print(f"{panel:10s} " + "  ".join(f"{n}: S_lwr {s:.4f} ({f:.0%} of pairs)"
                                         for n, (s, f) in zip("ACB", out)), flush=True)
    r = np.array(rows)
    print("mean       " + "  ".join(f"{n}: S_lwr {r[:, i, 0].mean():.4f}" for i, n in enumerate("ACB")))


if __name__ == "__main__":
    main()
