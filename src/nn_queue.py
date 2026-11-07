"""Task 2 queue forecaster as a spatio-temporal conv net, on the GPU.

Every LightGBM variant tried (thirteen of them) sits at ~0.70 out of sample, and
their averages go nowhere because the members correlate ~0.91: they all see the
same hand-built per-cell summary. This one sees the raw 12-step x link history
of the whole corridor at once and learns the upstream/downstream propagation
itself, so it can disagree with the trees in ways that averaging can use.

Input per window, one column per link (corridor order):
  history  speed/vf, flow/cap, occ, observed, pct/100, queued   6 x 12 steps
  static   vf, lanes, cap, length, rank                          5
  priors   P_on, P_ong, tod rate at each horizon step, link rate 19
  global   sin/cos time of day, weekend, condition               4
Output: logits (condition, step, link); loss on eligible cells only.

Training days are the same fit days every QVAL_DAYS=60 LightGBM model used,
minus the last 20 of them for checkpoint selection, so the out-of-sample
harness windows (last 60 train days) stay unseen.

    python nn_queue.py train NAME [--steps 30000]
    python nn_queue.py eval NAME          # writes out/nn_oos_{onset,ong}_NAME.npz
"""
import nice  # noqa: F401
import argparse, os, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings("ignore")

from common import OUT, QUEUE_PANELS
from qdata import load_panel, make_priors, origins, contract_ongoing
from qfeat import HIST, HOR

VAL = 60          # harness days, never trained on
SEL = 20          # checkpoint-selection days just before them
LMAX = 360
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(4)
import os
DROP = float(os.environ.get("NN_DROP", "0.1"))
WD = float(os.environ.get("NN_WD", "1e-2"))

# NN_LOSS_ALL=1 scores the loss on EVERY horizon cell, with ineligible ones
# labelled 0, instead of masking them out. The board counts predictions on
# cells whose truth it does not credit, so the masked loss trains an
# objective we are not scored on: 92.8% of our false positives sit on
# ineligible cells, and 70.5% of those are wrong even against the real
# (unmasked) queue. This teaches the model to be quiet where it cannot see.
LOSS_ALL = os.environ.get("NN_LOSS_ALL", "0") == "1"
# checkpoint-selection windows per panel; c1-c3 used 60/60
SEL_ON = int(os.environ.get("NN_SEL_ON", "60"))
SEL_ONG = int(os.environ.get("NN_SEL_ONG", "60"))


# ---------------------------------------------------------------- inputs
def make_input(sp, fl, oc, pc, stat, pri, tod0, dow, cond, occ_scale):
    """(C, nL) float32 from 12-step history arrays shaped (HIST, nL)."""
    vf, cap = stat["vf"][None, :], stat["cap"][None, :]
    obs = np.isfinite(sp)
    s = np.where(obs, sp / np.maximum(vf, 1.0), 0.0)
    f = np.where(np.isfinite(fl), fl / np.maximum(cap, 1.0), 0.0)
    o = np.where(np.isfinite(oc), oc / occ_scale, 0.0)
    p = np.nan_to_num(pc.astype(np.float32), nan=0.0) / 100.0
    qd = (obs & (s <= 0.6)).astype(np.float32)
    nL = sp.shape[1]
    hist = np.concatenate([s, f, o, obs.astype(np.float32), p, qd], 0)
    st = np.stack([stat["vf"] / 120.0, stat["lanes"] / 6.0, stat["cap"] / 12000.0,
                   stat["length"], stat["rank"] / max(nL, 1)])
    tods = (tod0 + np.arange(1, HOR + 1)) % 288
    pr = np.concatenate([pri["P_on"], pri["P_ong"], pri["rate_tod"][tods],
                         pri["rate_link"][None, :]], 0)
    if os.environ.get("NQ_NOPRIOR") == "1":
        # train-derived queue rates mislead on the separately generated scored
        # months (s104 vs s99: +0.0051 from a no-prior member) - blank them
        pr = np.zeros_like(pr)
    ang = 2 * np.pi * tod0 / 288
    g = np.array([np.sin(ang), np.cos(ang), float(dow >= 5), float(cond)])[:, None]
    g = np.repeat(g, nL, 1)
    return np.concatenate([hist, st, pr, g], 0).astype(np.float32)


def history_at(D, T, pid):
    """12-step history arrays; with QT_T=1 the window is T-55..T and the T row is
    thinned like the Task 1 mask thins the released row at the origin (see qdata)."""
    import os as _os
    if _os.environ.get("QT_T", "0") != "1":
        h = slice(T - HIST, T)
        return D["speed"][h], D["flow"][h], D["occ"][h], D["pct"][h]
    h = slice(T - HIST + 1, T + 1)
    hide = np.random.default_rng(int(T) * 7919 + int(pid)).random(D["nL"]) < float(_os.environ.get("QT_HIDE", "0.35"))
    sp, fl, oc, pc = (D[k][h].copy() for k in ("speed", "flow", "occ", "pct"))
    sp[-1, hide] = np.nan; fl[-1, hide] = np.nan; oc[-1, hide] = np.nan; pc[-1, hide] = 0
    return sp, fl, oc, pc


C_IN = 6 * HIST + 5 + 19 + 4


class Panel:
    def __init__(self, pid, panel):
        D = load_panel(panel)
        self.D, self.pid, self.panel = D, pid, panel
        nd = len(D["days"])
        fit = np.zeros(nd, bool); fit[:nd - VAL] = True
        self.pri = make_priors(D, fit)
        self.occ_scale = float(np.nanpercentile(D["occ"][::7], 99)) or 1.0
        q = D["q"]; nT = q.shape[0]
        anyq = q.any(1).astype(np.int32); cs = np.concatenate([[0], np.cumsum(anyq)])
        T = np.arange(HIST, nT - HOR - 1)
        day = T // 288
        on_std = T[((cs[T] - cs[T - HIST]) == 0) & (anyq[T] == 0)
                   & ((cs[T + HOR + 1] - cs[T + 1]) > 0)]
        on_s6 = T[((cs[T] - cs[T - HIST]) == 0) & (anyq[T] == 0)
                  & ((cs[T + HOR] - cs[T + 1]) == 0) & (anyq[T + HOR] == 1)]
        ong = contract_ongoing(D, T[anyq[T] == 1])
        tr = lambda a: a[(a // 288) < nd - VAL - SEL]
        se = lambda a: a[((a // 288) >= nd - VAL - SEL) & ((a // 288) < nd - VAL)]
        self.train = dict(on_std=tr(on_std), on_s6=tr(on_s6), ong=tr(ong))
        rng = np.random.default_rng(pid)
        pick = lambda a, n: np.sort(rng.choice(a, min(n, len(a)), replace=False))
        self.sel = ([(t, 0) for t in pick(se(on_s6), SEL_ON)]
                    + [(t, 1) for t in pick(se(ong), SEL_ONG)])
        self.nd = nd

    def _pri(self, T):
        """Train priors, or with QROLL=N the rolling same-split priors of T's day (qdata)."""
        if int(os.environ.get("QROLL", "0")):
            from qdata import rolling_priors
            return rolling_priors(self.panel, "train", T // 288)
        return self.pri

    def window(self, T, cond):
        D = self.D
        sp, fl, oc, pc = (NQ.history_at if "nn_queue.py" == "nn_queue2.py" else history_at)(D, T, self.pid)
        return make_input(sp, fl, oc, pc,
                          D["stat"], self._pri(T), T % 288, int(D["dow"][T]), cond,
                          self.occ_scale)

    def target(self, T):
        sl = slice(T + 1, T + 1 + HOR)
        return self.D["q"][sl].astype(np.float32), self.D["elig"][sl]


def batch(items):
    """Pad a list of (x, y, m, cond) to LMAX."""
    B = len(items)
    X = np.zeros((B, C_IN, LMAX), np.float32)
    Y = np.zeros((B, HOR, LMAX), np.float32)
    M = np.zeros((B, HOR, LMAX), np.float32)
    Cd = np.zeros(B, np.int64)
    for i, (x, y, m, c) in enumerate(items):
        L = x.shape[1]
        X[i, :, :L] = x; Cd[i] = c
        if y is not None:
            Y[i, :, :L] = y; M[i, :, :L] = m
    t = lambda a: torch.from_numpy(a).to(DEV, non_blocking=True)
    return t(X), t(Y), t(M), t(Cd)


def valid_mask(X, M):
    """(B, HOR, LMAX) 1.0 on real links. batch() pads every corridor to LMAX, so
    an all-cell loss would otherwise be mostly padding - 87% of it for a 44-link
    corridor. Padding columns are all-zero across channels, which is what this
    reads."""
    v = (X.abs().sum(1) > 0).float()            # (B, LMAX)
    return v[:, None, :].expand_as(M)


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, w, d, drop=0.0):
        super().__init__()
        self.n1 = nn.GroupNorm(8, w); self.c1 = nn.Conv1d(w, w, 3, padding=d, dilation=d)
        self.n2 = nn.GroupNorm(8, w); self.c2 = nn.Conv1d(w, w, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return x + self.c2(self.drop(F.gelu(self.n2(self.c1(F.gelu(self.n1(x)))))))


# Architectures other than the default, by run name. Task 1 showed that ensemble
# diversity comes from structure, not seeds (a fifth seed lost, a differently
# shaped net won), so the ongoing ensemble gets members that differ in width and
# in how far along the corridor they look.
ARCH = {
    "q1": dict(w=192, dil=(1, 2, 4, 8, 16, 32, 64, 1, 2, 4, 8, 1)),   # wide, far-seeing
    "q1npT": dict(w=192, dil=(1, 2, 4, 8, 16, 32, 64, 1, 2, 4, 8, 1)),  # q1np + the row at T (QT_T=1)
    "q1npT2": dict(w=192, dil=(1, 2, 4, 8, 16, 32, 64, 1, 2, 4, 8, 1)),  # second seed
    "q1npT3": dict(w=192, dil=(1, 2, 4, 8, 16, 32, 64, 1, 2, 4, 8, 1)),  # third seed
    "q1rollT": dict(w=192, dil=(1, 2, 4, 8, 16, 32, 64, 1, 2, 4, 8, 1)),  # rolling same-split priors + row T
    "q1np": dict(w=192, dil=(1, 2, 4, 8, 16, 32, 64, 1, 2, 4, 8, 1)),  # q1 without priors (NQ_NOPRIOR=1)
    "q2": dict(w=96, dil=(1, 2, 3, 1, 2, 3, 1, 2, 3, 1)),              # narrow, local
    # q1 (wide, far-seeing) paid ~3x its harness gain on the board (s53); more
    # members along that line, each shaped differently from the others:
    "q3": dict(w=256, dil=(1, 2, 4, 8, 16, 32, 64, 1, 2, 4, 8, 1)),   # wider still
    "q4": dict(w=192, dil=(1, 3, 9, 27, 81, 1, 3, 9, 27, 1)),          # base-3 reach
    "q5": dict(w=160, dil=(1, 2, 4, 8, 16, 32, 64, 1, 2, 4, 8, 16, 32, 64, 1, 1)),  # deep
    # q6 is q1's shape exactly - the only 1-D member that ever helped - so that
    # NN_LOSS_ALL is the single thing that differs from it.
    "q6": dict(w=192, dil=(1, 2, 4, 8, 16, 32, 64, 1, 2, 4, 8, 1)),
}


def make_net(name=None):
    cfg = ARCH.get(name, {})
    kw = {k: cfg[k] for k in ("w", "dil") if k in cfg}
    return Net(**kw)


class Net(nn.Module):
    def __init__(self, w=128, dil=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 1), drop=DROP):
        super().__init__()
        self.inp = nn.Conv1d(C_IN, w, 1)
        self.blocks = nn.Sequential(*[Block(w, d, drop) for d in dil])
        self.out = nn.Sequential(nn.GroupNorm(8, w), nn.GELU(), nn.Conv1d(w, 2 * HOR, 1))

    def forward(self, x, cond):
        z = self.out(self.blocks(self.inp(x)))              # (B, 12, L)
        z = z.view(z.shape[0], 2, HOR, -1)
        return z[torch.arange(z.shape[0], device=z.device), cond]   # (B, HOR, L)


def iou_at(p, y, m, thr=0.5):
    pr = p >= thr; t = (y > 0) & (m > 0)
    u = (pr | t).sum()
    return 1.0 if u == 0 else float((pr & t).sum() / u)


def selection_score(net, panels):
    net.eval(); on, ong = [], []
    with torch.no_grad():
        for P in panels:
            for k in range(0, len(P.sel), 64):
                chunk = P.sel[k:k + 64]
                items = []
                for T, c in chunk:
                    y, m = P.target(T); items.append((P.window(T, c), y, m, c))
                X, Y, M, Cd = batch(items)
                prob = torch.sigmoid(net(X, Cd)).cpu().numpy()
                Y, M = Y.cpu().numpy(), M.cpu().numpy()
                for i, (T, c) in enumerate(chunk):
                    if c == 0:
                        on.append(iou_at(prob[i, -1], Y[i, -1], M[i, -1]))
                    else:
                        ong.append(iou_at(prob[i], Y[i], M[i]))
    net.train()
    return float(np.mean(on)), float(np.mean(ong))


def train(name, steps, bs=48, lr=2e-3, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    panels = [Panel(i, p) for i, p in enumerate(QUEUE_PANELS)]
    for P in panels:
        print(f"  {P.panel}: on_std {len(P.train['on_std'])} on_s6 {len(P.train['on_s6'])} "
              f"ong {len(P.train['ong'])} sel {len(P.sel)}", flush=True)
    net = make_net(name).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    best_on, best_ong, t0, run = -1.0, -1.0, time.time(), []
    for step in range(1, steps + 1):
        items = []
        for _ in range(bs):
            P = panels[rng.integers(len(panels))]
            c = int(rng.random() < 0.5)
            pool = P.train["ong"] if c else (P.train["on_s6"] if rng.random() < 0.5
                                              else P.train["on_std"])
            T = int(pool[rng.integers(len(pool))])
            y, m = P.target(T)
            items.append((P.window(T, c), y, m, c))
        X, Y, M, Cd = batch(items)
        logit = net(X, Cd)
        W = valid_mask(X, M) if LOSS_ALL else M
        loss = (F.binary_cross_entropy_with_logits(logit, Y, reduction="none")
                * W).sum() / W.sum().clamp(min=1.0)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step(); run.append(loss.item())
        if step % 1000 == 0 or step == steps:
            on, ong = selection_score(net, panels)
            flag = ""
            if on > best_on:
                best_on = on; flag += " on*"
                torch.save(net.state_dict(), OUT / f"nn_queue_{name}_c0.pt")
            if ong > best_ong:
                best_ong = ong; flag += " ong*"
                torch.save(net.state_dict(), OUT / f"nn_queue_{name}_c1.pt")
            print(f"step {step:6d} loss {np.mean(run):.4f} sel onset {on:.3f} ongoing {ong:.3f}"
                  f" [{time.time()-t0:.0f}s]{flag}", flush=True)
            run = []


# ---------------------------------------------------------------- harness
def eval_oos(name):
    """Probabilities on exactly the oos_onset.py / oos_ongoing.py windows."""
    nets = []
    for c in (0, 1):
        n_ = make_net(name).to(DEV)
        n_.load_state_dict(torch.load(OUT / f"nn_queue_{name}_c{c}.pt", map_location=DEV))
        nets.append(n_.eval())
    P_on, P_ong = [], []
    for pid, panel in enumerate(QUEUE_PANELS):
        P = Panel(pid, panel); D = P.D; q = D["q"]; nT = q.shape[0]; nd = P.nd
        anyq = q.any(1).astype(np.int32); cs = np.concatenate([[0], np.cumsum(anyq)])
        T = np.arange(HIST, nT - HOR - 1)
        s6 = T[((cs[T] - cs[T - HIST]) == 0) & (anyq[T] == 0)
               & ((cs[T + HOR] - cs[T + 1]) == 0) & (anyq[T + HOR] == 1)]
        pool = s6[s6 // 288 >= nd - VAL]
        pick_on = np.random.default_rng(11).choice(pool, min(150, len(pool)), replace=False)
        To = T[(T // 288 >= nd - VAL) & q[T].any(1)]
        To = contract_ongoing(D, To)
        pick_ong = np.sort(np.random.default_rng(21).choice(To, min(120, len(To)), replace=False))
        with torch.no_grad():
            for picks, c, dest in ((pick_on, 0, P_on), (pick_ong, 1, P_ong)):
                for k in range(0, len(picks), 64):
                    ch = picks[k:k + 64]
                    X, _, _, Cd = batch([(P.window(int(t), c), None, None, c) for t in ch])
                    prob = torch.sigmoid(nets[c](X, Cd)).cpu().numpy()[:, :, :D["nL"]]
                    for i in range(len(ch)):
                        dest.append(prob[i, -1] if c == 0 else prob[i].reshape(-1))
        print(f"  {panel} done", flush=True)
    np.savez_compressed(OUT / f"nn_oos_onset_{name}.npz", p=np.array(P_on, dtype=object))
    np.savez_compressed(OUT / f"nn_oos_ong_{name}.npz", p=np.array(P_ong, dtype=object))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("name")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.name, a.steps, seed=a.seed)
    else:
        eval_oos(a.name)
