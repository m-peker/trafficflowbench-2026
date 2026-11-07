"""Task 4 on the current release: NNLS anchored on the released weak prior.

This is the method that has scored best every time it has been measured, and the
five-point sweep that tried to beat it - moving the answer further from the prior
- lost at every distance. It is one line of substance:

    f = argmin ||A f - c||^2 + lam ||f - b||^2,  f >= 0,  lam = 0.05

with `b` the released weak prior and `c` the released link counts.

Two things changed in the 2026-09-12 rebuild and both point here. The link counts
for validation and private were regenerated, so the old answer is stale and has
to be recomputed - unlike Task 1 and Task 2, there is no reusing it. And the
organizers found they had been scoring Task 4 against the wrong month's weak
prior, which they have now fixed; they say scores will read higher for the same
method. It also means the distance sweep that closed this task was measured
against a target that no longer applies, so Task 4 is worth reopening once
there is a fresh baseline to compare against.

    python task4_run.py <out_dir>
"""
import os, sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import nice  # noqa: F401
from common import PANELS
from odme_fast import panel_inputs, fista, load_operator, s_link

LAM = 0.05
SPLITS = ["validation", "private"]


def main():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    records, diag = [], []
    for panel in PANELS:
        _, _, _, paths, _ = load_operator(panel)
        for split in SPLITS:
            P = panel_inputs(panel, split, anchor="weak_prior")
            A, c, b = P["A"], P["counts"], P["b"]
            f = fista(A, c, b, LAM)
            if os.environ.get("T4_REFINE") == "lbfgs":
                # FISTA stops ~0.8% L1 short of the optimum on the large corridors;
                # L-BFGS-B from its answer reaches the exact NNLS solution in seconds
                from scipy.optimize import minimize
                r = minimize(lambda x: (float(np.sum((A @ x - c) ** 2) + LAM * np.sum((x - b) ** 2)),
                                        2 * (A.T @ (A @ x - c)) + 2 * LAM * (x - b)),
                             f, jac=True, method="L-BFGS-B", bounds=[(0, None)] * len(b),
                             options=dict(maxiter=20000, ftol=1e-15, gtol=1e-10))
                f = np.maximum(r.x, 0.0)
            d = float(np.abs(f - b).sum() / b.sum())
            x = paths[["path_id", "origin_zone", "destination_zone"]].copy()
            x["panel"] = panel
            x["departure_time"] = P["dep"]
            x["path_flow"] = f
            records.append(x[["panel", "departure_time", "path_id",
                              "origin_zone", "destination_zone", "path_flow"]])
            diag.append(dict(panel=panel, split=split, D=d, S_link=s_link(A, f, c),
                             nonzero=float((f > 1e-6).mean())))
            print(f"[{panel}/{split}] D={d:.3f}  S_link={s_link(A, f, c):.5f}  "
                  f"nonzero={(f > 1e-6).mean():.1%}  ({time.time()-t0:.0f}s)",
                  flush=True)
    out = pd.concat(records, ignore_index=True)
    out.to_csv(out_dir / "odme_submission.csv", index=False)
    dd = pd.DataFrame(diag)
    dd.to_csv(out_dir / "odme_diag.csv", index=False)
    print(f"\nwrote {len(out):,} rows to {out_dir/'odme_submission.csv'}")
    print(f"mean D {dd.D.mean():.4f}   S_link mean {dd.S_link.mean():.5f} "
          f"min {dd.S_link.min():.5f}")


if __name__ == "__main__":
    main()
