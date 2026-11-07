import sys, numpy as np
from qwin import load_panel, enumerate_windows, window_iou, HIST, HOR

panel = sys.argv[1] if len(sys.argv) > 1 else "D7_I10_E"
D = load_panel(panel)
onset_all, _ = enumerate_windows(D, per_cond=0)
rng = np.random.default_rng(7)
perm = rng.permutation(len(onset_all))
fit_T = onset_all[perm[: len(perm) // 2]]
val_T = onset_all[perm[len(perm) // 2:]]
q, elig, nL = D["q"], D["elig"], D["nL"]

# empirical P(queued) per (link, horizon step) over FIT onset windows
P = np.zeros((HOR, nL))
for T in fit_T:
    sl = slice(T + 1, T + 1 + HOR)
    P += (q[sl] & elig[sl])
P /= len(fit_T)

rank = np.argsort(-P.max(0))
print("link rank by P:", [(D['link_ids'][i], round(float(P.max(0)[i]), 3)) for i in rank[:10]])
print("P by step for top link:", np.round(P[:, rank[0]], 3))

def ev(pred_fn, T_list):
    return float(np.mean([window_iou(D, T, pred_fn(T)) for T in T_list]))

print("\n--- top-K links, all steps ---")
for K in [2, 4, 5, 6, 7, 8, 10, 12]:
    sel = rank[:K]
    m = np.zeros((HOR, nL), bool); m[:, sel] = True
    print(f"  K={K:2d}: fit={ev(lambda T: m, fit_T):.4f} val={ev(lambda T: m, val_T):.4f}")

print("\n--- top-7 links, steps >= s0 ---")
for s0 in range(6):
    m = np.zeros((HOR, nL), bool); m[s0:, rank[:7]] = True
    print(f"  s0={s0}: val={ev(lambda T: m, val_T):.4f}")

print("\n--- threshold on P(link,step) ---")
for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    m = P >= th
    print(f"  th={th}: cells={m.sum():4d} fit={ev(lambda T: m, fit_T):.4f} val={ev(lambda T: m, val_T):.4f}")
