"""Task 2 queue forecaster as a true (time x link) conv net.

nn_queue.py calls itself spatio-temporal, but its first layer is
`Conv1d(C_IN, w, 1)`: the twelve history steps arrive as channels and are
collapsed at each link independently, after which every convolution runs over
links alone. The horizon exists only as `2 * HOR` output channels. So the model
has no weight sharing across time, and a queue front moving upstream - a
diagonal in (time, link) - has to be memorised separately at each offset rather
than learned once.

This is the same gap Task 1 had between hand-made per-cell tree features and the
2-D inpainting net, which was Task 1's single biggest gain. Here the window is
laid out as an 18 x L grid (HIST=12 observed rows, HOR=6 blank rows to fill) and
convolved in both axes, so one kernel sees propagation wherever it happens.

Channels, per (step, link):
  dynamic   speed/vf, flow/cap, occ, observed, pct/100, queued   6  (0 on horizon)
  horizon   1 on the rows to predict                             1
  static    vf, lanes, cap, length, rank                         5
  priors    P_on, P_ong, link rate; tod rate of that row         4
  global    sin/cos time of day, weekend, condition              4
  position  row index / (T - 1)                                  1

Trained and selected exactly as nn_queue.py does - same fit days, same
checkpoint-selection windows, same per-condition checkpoints - so its
probabilities drop into the same mixes.

    python nn_queue2.py train NAME [--steps 12000] [--seed 1]
    python nn_queue2.py eval NAME        # out/nn2_oos_{onset,ong}_NAME.npz
"""
import nice  # noqa: F401
import argparse, os, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings("ignore")

from common import OUT, QUEUE_PANELS
from qfeat import HIST, HOR
import nn_queue as NQ

DEV = NQ.DEV
LMAX = NQ.LMAX
TLEN = HIST + HOR
C_IN2 = 6 + 1 + 5 + 4 + 4 + 1
DROP = float(os.environ.get("NN2_DROP", "0.1"))
WD = float(os.environ.get("NN2_WD", "1e-2"))
PAD_BATCH = os.environ.get("NN2_PAD", "lmax") == "batch"
ONE_PANEL = os.environ.get("NN2_ONEPANEL", "0") == "1"   # every train batch from one corridor

# NN2_LOSS_ALL=1 scores the loss on EVERY horizon cell, with ineligible ones
# labelled 0, instead of masking them out. The board counts predictions on
# cells whose truth it does not credit, so the masked loss trains an
# objective we are not scored on: 92.8% of our false positives sit on
# ineligible cells, and 70.5% of those are wrong even against the real
# (unmasked) queue. This teaches the model to be quiet where it cannot see.
LOSS_ALL = os.environ.get("NN2_LOSS_ALL", "0") == "1"


# ---------------------------------------------------------------- inputs
def make_input2(sp, fl, oc, pc, stat, pri, tod0, dow, cond, occ_scale):
    """(C, HIST+HOR, nL): history rows carry values, horizon rows are blank."""
    nL = sp.shape[1]
    vf, cap = stat["vf"][None, :], stat["cap"][None, :]
    obs = np.isfinite(sp)
    s = np.where(obs, sp / np.maximum(vf, 1.0), 0.0)
    f = np.where(np.isfinite(fl), fl / np.maximum(cap, 1.0), 0.0)
    o = np.where(np.isfinite(oc), oc / occ_scale, 0.0)
    p = np.nan_to_num(pc.astype(np.float32), nan=0.0) / 100.0
    qd = (obs & (s <= 0.6)).astype(np.float32)

    x = np.zeros((C_IN2, TLEN, nL), np.float32)
    for i, a in enumerate((s, f, o, obs.astype(np.float32), p, qd)):
        x[i, :HIST] = a
    x[6, HIST:] = 1.0                                    # the rows to fill
    for i, a in enumerate((stat["vf"] / 120.0, stat["lanes"] / 6.0,
                           stat["cap"] / 12000.0, stat["length"],
                           stat["rank"] / max(nL, 1))):
        x[7 + i] = a[None, :]
    # P_on/P_ong are per horizon step, (HOR, nL) - they belong on the rows they
    # describe, not broadcast over the history
    x[12, HIST:] = pri["P_on"]
    x[13, HIST:] = pri["P_ong"]
    x[14] = pri["rate_link"][None, :]
    # each row gets the tod rate of its own step, history rows included
    tods = (tod0 + np.arange(-HIST + 1, HOR + 1)) % 288
    x[15] = pri["rate_tod"][tods]
    if os.environ.get("NQ_NOPRIOR") == "1":      # see nn_queue.make_input
        x[12:16] = 0.0
    ang = 2 * np.pi * tod0 / 288
    for i, v in enumerate((np.sin(ang), np.cos(ang), float(dow >= 5), float(cond))):
        x[16 + i] = v
    x[20] = (np.arange(TLEN, dtype=np.float32) / (TLEN - 1))[:, None]
    return x


class Panel(NQ.Panel):
    """NQ.Panel's day/window bookkeeping, with the 2-D window representation."""

    def window(self, T, cond):
        D = self.D
        sp, fl, oc, pc = (NQ.history_at if "nn_queue2.py" == "nn_queue2.py" else history_at)(D, T, self.pid)
        return make_input2(sp, fl, oc, pc,
                           D["stat"], self._pri(T), T % 288, int(D["dow"][T]), cond,
                           self.occ_scale)


def batch(items):
    """Pad a list of (x, y, m, cond) to LMAX links."""
    B = len(items)
    # NN2_PAD=batch pads only to the widest corridor IN the batch (single-panel
    # batches then carry no padding at all, so GroupNorm sees real links only);
    # the original behaviour pads every corridor to LMAX = 360 links.
    Lp = max(x.shape[2] for x, *_ in items) if PAD_BATCH else LMAX
    X = np.zeros((B, C_IN2, TLEN, Lp), np.float32)
    Y = np.zeros((B, HOR, Lp), np.float32)
    M = np.zeros((B, HOR, Lp), np.float32)
    Cd = np.zeros(B, np.int64)
    for i, (x, y, m, c) in enumerate(items):
        L = x.shape[2]
        X[i, :, :, :L] = x; Cd[i] = c
        if y is not None:
            Y[i, :, :L] = y; M[i, :, :L] = m
    t = lambda a: torch.from_numpy(a).to(DEV, non_blocking=True)
    return t(X), t(Y), t(M), t(Cd)


def valid_mask(X, M):
    """(B, HOR, LMAX) 1.0 on real links; see nn_queue.valid_mask. Here the input
    carries a time axis too, so the sum is over channels and steps."""
    v = (X.abs().sum((1, 2)) > 0).float()       # (B, LMAX)
    return v[:, None, :].expand_as(M)


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, w, dt, dl, drop=0.0):
        super().__init__()
        self.n = nn.GroupNorm(8, w)
        self.c1 = nn.Conv2d(w, w, 3, padding=(dt, dl), dilation=(dt, dl))
        self.c2 = nn.Conv2d(w, w, 1)
        self.drop = nn.Dropout2d(drop) if drop else nn.Identity()

    def forward(self, x):
        return x + self.c2(self.drop(F.gelu(self.c1(F.gelu(self.n(x))))))


# (time dilation, link dilation). Time only spans 18 rows, so it stays small;
# the link axis reaches far, because a corridor is hundreds of links long.
ARCH = {
    # r2rollT: the wider / deeper r2 shape (link reach 64) on the rolling-prior + T-row recipe
    "r2rollT2": dict(w=128, dil=((1, 1), (1, 2), (2, 4), (1, 8), (2, 16), (1, 32),
                                 (1, 64), (1, 1), (1, 2), (2, 4), (1, 8), (1, 1))),
    "r2rollT": dict(w=128, dil=((1, 1), (1, 2), (2, 4), (1, 8), (2, 16), (1, 32),
                                (1, 64), (1, 1), (1, 2), (2, 4), (1, 8), (1, 1))),
    "r1": dict(w=96, dil=((1, 1), (1, 2), (2, 4), (1, 8), (2, 16), (1, 32),
                          (1, 1), (1, 2), (2, 4), (1, 1))),
    "r2": dict(w=128, dil=((1, 1), (1, 2), (2, 4), (1, 8), (2, 16), (1, 32),
                           (1, 64), (1, 1), (1, 2), (2, 4), (1, 8), (1, 1))),
    "r4rollT2": dict(w=96, dil=((1, 1), (1, 2), (2, 4), (1, 8), (2, 16), (1, 32),
                               (1, 1), (1, 2), (2, 4), (1, 1))),
    "r4rollT": dict(w=96, dil=((1, 1), (1, 2), (2, 4), (1, 8), (2, 16), (1, 32),
                              (1, 1), (1, 2), (2, 4), (1, 1))),
    "r4npT3": dict(w=96, dil=((1, 1), (1, 2), (2, 4), (1, 8), (2, 16), (1, 32),
                              (1, 1), (1, 2), (2, 4), (1, 1))),
    "r4npT2": dict(w=96, dil=((1, 1), (1, 2), (2, 4), (1, 8), (2, 16), (1, 32),
                              (1, 1), (1, 2), (2, 4), (1, 1))),
    "r4npT": dict(w=96, dil=((1, 1), (1, 2), (2, 4), (1, 8), (2, 16), (1, 32),
                             (1, 1), (1, 2), (2, 4), (1, 1))),
    "r4np": dict(w=96, dil=((1, 1), (1, 2), (2, 4), (1, 8), (2, 16), (1, 32),
                            (1, 1), (1, 2), (2, 4), (1, 1))),
    "r4": dict(w=96, dil=((1, 1), (1, 2), (2, 4), (1, 8), (2, 16), (1, 32),
                          (1, 1), (1, 2), (2, 4), (1, 1))),
    "r3": dict(w=96, dil=((1, 1), (2, 1), (1, 3), (2, 9), (1, 27), (2, 1),
                          (1, 3), (1, 9), (1, 1))),
}


class Net(nn.Module):
    def __init__(self, w=96, dil=ARCH["r1"]["dil"], drop=DROP):
        super().__init__()
        self.inp = nn.Conv2d(C_IN2, w, 1)
        self.blocks = nn.Sequential(*[Block(w, dt, dl, drop) for dt, dl in dil])
        self.out = nn.Sequential(nn.GroupNorm(8, w), nn.GELU(), nn.Conv2d(w, 1, 1))

    def forward(self, x, cond=None):
        z = self.out(self.blocks(self.inp(x)))      # (B, 1, TLEN, L)
        return z[:, 0, HIST:]                        # (B, HOR, L)


def make_net(name=None):
    cfg = ARCH.get(name, ARCH["r1"])
    return Net(w=cfg["w"], dil=cfg["dil"])


def load_net(name, cond, device=None):
    f = OUT / f"nn2_queue_{name}_c{cond}.pt"
    if not f.exists():
        f = OUT / f"nn2_queue_{name}.pt"
    net = make_net(name).to(device or DEV)
    net.load_state_dict(torch.load(f, map_location=device or DEV))
    return net.eval()


# ---------------------------------------------------------------- train
def selection_score(net, panels):
    net.eval(); on, ong = [], []
    with torch.no_grad():
        for P in panels:
            for k in range(0, len(P.sel), 32):
                chunk = P.sel[k:k + 32]
                items = []
                for T, c in chunk:
                    y, m = P.target(T); items.append((P.window(T, c), y, m, c))
                X, Y, M, Cd = batch(items)
                prob = torch.sigmoid(net(X, Cd)).cpu().numpy()
                Y, M = Y.cpu().numpy(), M.cpu().numpy()
                for i, (T, c) in enumerate(chunk):
                    if c == 0:
                        on.append(NQ.iou_at(prob[i, -1], Y[i, -1], M[i, -1]))
                    else:
                        ong.append(NQ.iou_at(prob[i], Y[i], M[i]))
    net.train()
    return float(np.mean(on)), float(np.mean(ong))


def train(name, steps, bs=int(os.environ.get("NN2_BS", "16")), lr=2e-3, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    panels = [Panel(i, p) for i, p in enumerate(QUEUE_PANELS)]
    for P in panels:
        print(f"  {P.panel}: on_std {len(P.train['on_std'])} on_s6 {len(P.train['on_s6'])} "
              f"ong {len(P.train['ong'])} sel {len(P.sel)}", flush=True)
    net = make_net(name).to(DEV); net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=WD)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps,
                                              pct_start=0.1)
    best = [0.0, 0.0]
    t0 = time.time()
    for step in range(1, steps + 1):
        items = []
        P1 = panels[rng.integers(len(panels))]
        for _ in range(bs):
            P = P1 if ONE_PANEL else panels[rng.integers(len(panels))]
            c = int(rng.random() < 0.5)
            # the same pools and the same 50/50 onset split nn_queue.py uses,
            # so representation is the only thing that differs
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
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sch.step()
        if step % 1000 == 0 or step == steps:
            on, ong = selection_score(net, panels)
            tag = ""
            if on > best[0]:
                best[0] = on; torch.save(net.state_dict(), OUT / f"nn2_queue_{name}_c0.pt"); tag += " on*"
            if ong > best[1]:
                best[1] = ong; torch.save(net.state_dict(), OUT / f"nn2_queue_{name}_c1.pt"); tag += " ong*"
            print(f"step {step:6d} loss {loss.item():.4f} sel onset {on:.3f} "
                  f"ongoing {ong:.3f} [{time.time() - t0:.0f}s]{tag}", flush=True)
    print(f"best onset {best[0]:.4f} ongoing {best[1]:.4f}")


def evaluate(name):
    """Probabilities on the same OOS windows nn_queue.py eval writes."""
    import oos_onset as OON
    import oos_ongoing as OONG
    for cond, mod, tag in ((0, OON, "onset"), (1, OONG, "ong")):
        net = load_net(name, cond)
        P, Tr, Pa = [], [], []
        for pid, panel, D, pri, pick in mod.windows():
            occ = float(np.nanpercentile(D["occ"][::7], 99)) or 1.0
            for t0 in pick:
                h = slice(int(t0) - HIST, int(t0))
                x = make_input2(D["speed"][h], D["flow"][h], D["occ"][h], D["pct"][h],
                                D["stat"], pri, int(t0) % 288, int(D["dow"][int(t0)]),
                                cond, occ)
                X, _, _, Cd = batch([(x, None, None, cond)])
                with torch.no_grad():
                    pr = torch.sigmoid(net(X, Cd)).cpu().numpy()[0, :, :x.shape[2]]
                P.append(pr[-1].reshape(-1) if cond == 0 else pr.reshape(-1))
                sl = slice(int(t0) + 1, int(t0) + 1 + HOR)
                t = (D["q"][sl] & D["elig"][sl])
                Tr.append(t[-1].reshape(-1) if cond == 0 else t.reshape(-1))
                Pa.append(panel)
            print(f"  [{name}/{tag}] {panel} done", flush=True)
        np.savez_compressed(OUT / f"nn2_oos_{tag}_{name}.npz",
                            p=np.array(P, dtype=object),
                            truth=np.array(Tr, dtype=object), panel=np.array(Pa))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval"])
    ap.add_argument("name", nargs="?", default="r1")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.name, a.steps, seed=a.seed)
    else:
        evaluate(a.name)
