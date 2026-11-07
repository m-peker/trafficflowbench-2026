import sys, numpy as np
from t1feat import load_grid
from t1build import ramp_matrices

DT = 5.0/60.0
panel = sys.argv[1] if len(sys.argv) > 1 else "D7_I10_E"
Gm = load_grid(panel, "train", masked=True)
Gt = load_grid(panel, "train", masked=False)
o = [Gt["link_ids"].index(l) for l in Gm["link_ids"]]
sp, fl = Gt["speed"][:, o], Gt["flow"][:, o]
R = ramp_matrices(panel, "train", Gm["link_ids"])
L = Gm["stat"]["length"][None, :]
N = (fl / np.maximum(sp, 1.0)) * L
dN = N[1:] - N[:-1]
ok0 = np.isfinite(dN) & (R["miss"][:-1] == 0)

print(f"{panel}: mean N = {np.nanmean(N):.2f} veh, mean |dN| = {np.nanmean(np.abs(dN)):.3f} veh")
print(f"0.75% of N would be {0.0075*np.nanmean(N):.3f} veh\n")

q_up = np.full_like(fl, np.nan); q_up[:, 1:] = fl[:, :-1]
q_dn = np.full_like(fl, np.nan); q_dn[:, :-1] = fl[:, 1:]
conventions = {
    "in=up,  out=self": (q_up, fl),
    "in=up,  out=down": (q_up, q_dn),
    "in=self,out=down": (fl, q_dn),
}
for name, (qi, qo) in conventions.items():
    rhs = (DT * (qi + R["on"] - qo - R["off"]))[:-1]
    m = ok0 & np.isfinite(rhs)
    m[:, 0] = False; m[:, -1] = False
    e = np.abs(dN[m] - rhs[m]).sum() / (np.abs(rhs[m]).sum() + 1e-9)
    c = np.corrcoef(dN[m], rhs[m])[0, 1]
    print(f"{name:20s} mean|rhs|={np.abs(rhs[m]).mean():7.3f}  e_lwr={e:7.4f}  "
          f"S_LWR={max(0,1-min(1,e)):.4f}  corr(dN,rhs)={c:+.4f}")
