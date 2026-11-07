"""Task 1 as spatio-temporal inpainting, on the GPU.

The tree models read a handful of hand-made summaries at each target cell -
the flanking observations in time, the nearest observed link up and down the
corridor, a time-of-day profile, rolling means. The targets are scattered
single cells (blank runs average 1.2 steps, 15-38% of eligible cells by regime),
so almost every one of them is surrounded by observed values that those few
summaries compress away. A 2-D convolution over (time x link) sees the field.

What it is scored on, and therefore trained on:

    0.54 * ((v_hat - v)/25)^2  +  0.46 * ((q_hat - q)/600)^2     per lane

which is S_state's own weighting, plus an optional L1 term on the change in
accumulation N = (q/v)*L between consecutive steps - the quantity S_LWR reads.
On the board S_state is ~0.918 and S_LWR ~0.54 (s34), and 71% of the LWR error
sits on transitions with exactly one filled endpoint, where the error is just
the density error of a single cell. So both scores want the same thing: a
better cell.

    python t1_nn.py prep                 # float16 memmaps under work/nn/
    python t1_nn.py train NAME [--steps N] [--lwr 0.0]
    python t1_nn.py eval NAME            # S_state / e_lwr on held-out days
"""
import nice  # noqa: F401
import argparse, os, time, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings("ignore")

from common import WORK, OUT, PANELS
from t1feat import load_grid

NN = WORK / "nn"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(4)

VAL_DAYS = int(os.environ.get("T1_VAL_DAYS", "30"))   # last train days held out
# 30 for an honest comparison against the trees; 7 for the model that ships,
# since validation and private are separate months anyway and the trees use
# every train day too.
# patch: five-minute steps x links. Wider context costs memory, not much time,
# and a target cell's information is mostly local - but "local" spans a queue,
# which can be tens of links long and an hour deep.
TCROP = int(os.environ.get("T1_TCROP", "64"))
LCROP = int(os.environ.get("T1_LCROP", "48"))
WIDTH = int(os.environ.get("T1_WIDTH", "96"))
DILS = tuple(int(x) for x in os.environ.get("T1_DILS", "1,2,4,8,1,2,4,1").split(","))
REGIMES = (0.20, 0.30, 0.50)
# v, q, occ, observed, pct, vf, lanes, cap, tod  (+ on-ramp, off-ramp, ramp-miss
# when T1_RAMPS=1). Ramps are the physical cause of the flow changes the net has
# to infer, and Task 1 already reads them for its conservation terms.
RAMPS = os.environ.get("T1_RAMPS", "0") == "1"
# T1_KHEAD=1 adds a third output channel: a density head trained toward the
# true k = q/v ratio, with its OWN temporal-smoothness loss - decoupled from
# the v/q heads that carry S_state. p1's LWR term derived N_hat from q_hat/v_hat
# directly, so its gradient competed with the very same weights S_state depends
# on; a dedicated head lets the backbone learn density consistency without that
# tug-of-war. K_NORM is a per-link jam-density-like scale (lanes*100, matching
# the fd_parameters.csv k_jam constant measured on 3-lane links).
KHEAD = os.environ.get("T1_KHEAD", "0") == "1"
K_WEIGHT = float(os.environ.get("T1_KWEIGHT", "0.20"))
K_LWR_W = float(os.environ.get("T1_KLWR", "0.15"))
K_SCALE = float(os.environ.get("T1_KSCALE", "30.0"))
# T1_GATE=1 goes one step past T1_KHEAD: instead of testing a single global
# blend weight AFTER training (t1_khead_sweep.py's beta), the net outputs a
# FOURTH channel - a per-cell gate - and is trained end-to-end on the
# RECONCILED flow the gate produces, not on the flow head alone. dn1/dn2 (one
# scalar beta) showed a real individual-net S_lwr gain that a fixed weight
# could not turn into an ensemble win; a per-cell, learned gate can in
# principle use the density estimate only where it is genuinely more reliable
# (e.g. mid-outage) and the direct flow estimate elsewhere, rather than one
# global trade-off applied everywhere alike.
GATE = os.environ.get("T1_GATE", "0") == "1"
# T1_ARCH=unet swaps the whole architecture family - real downsampling and
# upsampling with skip connections, not another dilation schedule on the same
# fixed-resolution stack every other member uses. T1_UWIDTH/T1_UDEPTH size it.
ARCH_FAMILY = os.environ.get("T1_ARCH", "dilated")
UWIDTH = int(os.environ.get("T1_UWIDTH", "64"))
UDEPTH = int(os.environ.get("T1_UDEPTH", "3"))
# Generalisation knobs. The board scores months the net never saw, and s41 showed
# that fitting the train period harder loses there, so these push the other way:
#   T1_AUG=1   scale flow/occupancy (inputs and flow target) by U(0.85,1.15) per
#              patch - demand level shifts between months, shape should not - and
#              blank the time-of-day channel half the time, so the net cannot
#              lean on the clock instead of the neighbouring cells.
#   T1_DROP=p  channel dropout inside every residual block.
AUG = os.environ.get("T1_AUG", "0") == "1"
TOD_TRAIN = os.environ.get("T1_TOD_TRAIN", "ramp")   # "clock" = the real hour
DROP = float(os.environ.get("T1_DROP", "0"))
WD = float(os.environ.get("T1_WD", "1e-4"))
CH_IN = 12 if RAMPS else 9
GAPP = float(os.environ.get("T1_GAPP", "0"))
INELIG_TRAIN = os.environ.get("T1_INELIG_TRAIN", "0") == "1"
GAP_LEN = 18
# T1_GAP_EVENT=1 places the training gaps where the organizers place theirs:
# right after a queue-onset (step-6) or contract-ongoing origin, instead of at a
# random time. With T1_RAMPS=1 the net also reads the ramp counts, which the
# release keeps publishing inside the gaps.
GAP_EVENT = os.environ.get("T1_GAP_EVENT", "0") == "1"


# ---------------------------------------------------------------- data
def prep():
    """One float16 memmap per panel: normalised channels + the observed mask."""
    NN.mkdir(parents=True, exist_ok=True)
    for panel in PANELS:
        G = load_grid(panel, "train", masked=False)
        st = G["stat"]
        nT, nL = G["nT"], G["nL"]
        vf = st["vf"][None, :]; cap = st["cap"][None, :]
        lanes = np.maximum(st["lanes"], 1.0)[None, :]
        obs = np.isfinite(G["speed"]) & (G["pct"] >= 75)
        occ_scale = max(float(np.nanpercentile(G["occ"][::7], 99)), 1e-3)
        from t1build import ramp_matrices
        R = ramp_matrices(panel, "train", list(G["link_ids"]))
        a = np.zeros((nT, nL, 8), np.float16)
        a[..., 5] = np.nan_to_num(R["on"] / cap)
        a[..., 6] = np.nan_to_num(R["off"] / cap)
        a[..., 7] = (R["miss"] != 0) if "miss" in R else 0.0
        a[..., 0] = np.nan_to_num(G["speed"] / vf)
        a[..., 1] = np.nan_to_num(G["flow"] / cap)
        a[..., 2] = np.nan_to_num(G["occ"] / occ_scale)
        a[..., 3] = obs
        a[..., 4] = np.nan_to_num(G["pct"] / 100.0)
        np.save(NN / f"{panel}_grid.npy", a)
        np.savez(NN / f"{panel}_static.npz", vf=st["vf"], lanes=st["lanes"],
                 cap=st["cap"], length=st["length"], occ_scale=occ_scale,
                 nT=nT, nL=nL, days=np.array(G["days"]))
        print(f"  {panel}: {nT}x{nL}, observed {obs.mean():.3f}", flush=True)


def prep_masked(split):
    """Memmaps from a split's *masked* release: the cells the board scores are
    blank, every other observed cell is there. Self-supervised inpainting on
    these adapts a net to the months it will be scored on, without a label."""
    NN.mkdir(parents=True, exist_ok=True)
    for panel in PANELS:
        G = load_grid(panel, split, masked=True)
        st = G["stat"]
        vf = st["vf"][None, :]; cap = st["cap"][None, :]
        obs = np.isfinite(G["speed"]) & np.isfinite(G["flow"]) & (G["pct"] >= 75)
        z = np.load(NN / f"{panel}_static.npz", allow_pickle=True)
        occ_scale = float(z["occ_scale"])            # the train scale, as at inference
        a = np.zeros((G["nT"], G["nL"], 8), np.float16)
        a[..., 0] = np.nan_to_num(G["speed"] / vf) * obs
        a[..., 1] = np.nan_to_num(G["flow"] / cap) * obs
        a[..., 2] = np.nan_to_num(G["occ"] / occ_scale) * obs
        a[..., 3] = obs
        a[..., 4] = np.nan_to_num(G["pct"] / 100.0) * obs
        np.save(NN / f"{panel}_{split}M_grid.npy", a)
        print(f"  {panel}/{split}: {G['nT']}x{G['nL']}, observed {obs.mean():.3f}", flush=True)


class AdaptData:
    """Crops from masked grids of the given splits, optionally one day window.
    Only observed cells are hidden and scored, so no board label is ever read."""

    def __init__(self, splits, day_lo=None, day_hi=None):
        self.items = []
        for sp in splits:
            for p in PANELS:
                g = np.load(NN / f"{p}_{sp}M_grid.npy", mmap_mode="r")
                z = np.load(NN / f"{p}_static.npz", allow_pickle=True)
                self.items.append((p, g, {k: z[k] for k in z.files}))
        self.lo, self.hi = day_lo, day_hi

    def crop(self, rng):
        p, g, st = self.items[rng.integers(len(self.items))]
        nT, nL = g.shape[0], g.shape[1]
        lo = 0 if self.lo is None else self.lo * 288
        hi = nT if self.hi is None else min(nT, self.hi * 288)
        t0 = rng.integers(lo, hi - TCROP)
        j0 = rng.integers(0, max(nL - LCROP, 1))
        x = np.asarray(g[t0:t0 + TCROP, j0:j0 + LCROP], np.float32)
        sl = slice(j0, j0 + LCROP)
        stat = dict(vf=st["vf"][sl], lanes=np.maximum(st["lanes"][sl], 1.0),
                    cap=st["cap"][sl], length=st["length"][sl])
        if x.shape[1] < LCROP:
            pad = LCROP - x.shape[1]
            x = np.pad(x, ((0, 0), (0, pad), (0, 0)))
            stat = {k: np.pad(v, (0, pad), constant_values=1.0) for k, v in stat.items()}
        return x, stat


def adapt(init, name, splits, steps, day_lo=None, day_hi=None, lr=2e-4, bs=None, seed=0):
    """Self-supervised finetune of `init` on masked grids, saved as `name`."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    data = AdaptData(splits, day_lo, day_hi)
    net = load_net(init); net.train()
    bs = bs or int(os.environ.get("T1_BS", "10"))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=WD)
    t0, run = time.time(), []
    for step in range(1, steps + 1):
        xs, ys, hs, sts = [], [], [], []
        for _ in range(bs):
            g, stat = data.crop(rng)
            x, y, hide = make_example(g, stat, rng)
            xs.append(x); ys.append(y); hs.append(hide); sts.append(stat)
        t = lambda a: torch.from_numpy(np.stack(a)).to(DEV)
        loss, _, _, _ = losses(net(t(xs)), t(ys), torch.from_numpy(np.stack(hs)).to(DEV),
                              stack_stat(sts))
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); run.append(loss.item())
        if step % 250 == 0 or step == steps:
            print(f"adapt {step:5d} loss {np.mean(run):.5f} [{time.time()-t0:.0f}s]", flush=True)
            run = []
    torch.save(net.state_dict(), OUT / f"t1_nn_{name}.pt")
    ARCH[name] = ARCH.get(init, {})


class Data:
    def __init__(self):
        self.g, self.s = {}, {}
        for p in PANELS:
            self.g[p] = np.load(NN / f"{p}_grid.npy", mmap_mode="r")
            z = np.load(NN / f"{p}_static.npz", allow_pickle=True)
            self.s[p] = {k: z[k] for k in z.files}
        self.nd = {p: int(self.s[p]["nT"]) // 288 for p in PANELS}
        self.ev, self.last_gap = {}, None
        if GAP_EVENT:
            os.environ["QONSET_STEP6"] = "1"; os.environ["QONG_CONTRACT"] = "1"
            from qdata import load_panel, origins
            from common import QUEUE_PANELS
            for p in QUEUE_PANELS:
                on, ong = origins(load_panel(p))
                self.ev[p] = np.sort(np.r_[on, ong])
            print("gap events:", {p: len(v) for p, v in self.ev.items()}, flush=True)

    def crop(self, panel, rng, val=False):
        nT, nL = int(self.s[panel]["nT"]), int(self.s[panel]["nL"])
        hi = nT - 288 * VAL_DAYS
        self.last_gap = None
        t0 = None
        if GAP_EVENT and panel in self.ev and rng.random() < GAPP:
            ev = self.ev[panel]
            ev = ev[(ev >= hi + 40) & (ev < nT - TCROP)] if val else ev[(ev >= TCROP) & (ev < hi - TCROP)]
            if len(ev):
                a = int(rng.integers(14, TCROP - GAP_LEN - 8))
                t0 = int(ev[rng.integers(len(ev))]) + 1 - a
                self.last_gap = a
        if t0 is None:
            t0 = (rng.integers(hi, nT - TCROP) if val else rng.integers(0, hi - TCROP))
        j0 = rng.integers(0, max(nL - LCROP, 1))
        g = np.asarray(self.g[panel][t0:t0 + TCROP, j0:j0 + LCROP], np.float32)
        st = self.s[panel]
        sl = slice(j0, j0 + LCROP)
        stat = dict(vf=st["vf"][sl], lanes=np.maximum(st["lanes"][sl], 1.0),
                    cap=st["cap"][sl], length=st["length"][sl])
        if g.shape[1] < LCROP:
            # two corridors are narrower than the crop; pad with cells that are
            # never observed, so they carry no loss and no gradient
            pad = LCROP - g.shape[1]
            g = np.pad(g, ((0, 0), (0, pad), (0, 0)))
            stat = {k: np.pad(v, (0, pad), constant_values=1.0) for k, v in stat.items()}
        return g, stat, t0


def make_example(g, stat, rng, rate=None, t0=None, gap_at=None):
    """Hide a share of the observed cells; return (input, target, loss mask)."""
    T, L, _ = g.shape
    obs = g[..., 3] > 0.5
    rate = rate if rate is not None else REGIMES[rng.integers(len(REGIMES))]
    hide = obs & (rng.random((T, L)) < rate)
    # T1_GAPP=p: with probability p the crop also gets a Task 2 gap - every link
    # blank for GAP_LEN steps, the 12 steps before it fully published (the
    # release ships that history raw). Validation/private hold ~1.9% of the
    # scored Task 1 cells inside such blackouts and a net trained only on
    # scattered blanks fills them with free flow (t1_gap_probe.py).
    a = gap_at
    if a is None and GAPP and not GAP_EVENT and rng.random() < GAPP and T >= GAP_LEN + 24:
        a = int(rng.integers(12, T - GAP_LEN - 6))
    if a is not None:
        hide[a - 12:a] = False
        hide[a:a + GAP_LEN] = obs[a:a + GAP_LEN]
    seen = obs & ~hide
    if INELIG_TRAIN:
        # finite readings with pct_observed < 75 are released in every split and never
        # Task 1 targets: always visible (with their own pct), never hidden, never in
        # the loss - except inside a Task 2 gap, where the release blanks everything
        inel = ~obs & (g[..., 4] > 0) & (g[..., 0] > 0)
        if a is not None:
            inel[a:a + GAP_LEN] = False
        seen = seen | inel

    scale = rng.uniform(0.85, 1.15) if AUG else 1.0
    v = np.where(seen, g[..., 0], 0.0)
    q = np.where(seen, g[..., 1] * scale, 0.0)
    o = np.where(seen, g[..., 2] * scale, 0.0)
    # T1_TOD_TRAIN=clock feeds the real time of day, the same
    # `(t % 288) / 288` predict_grid uses, instead of the crop-position ramp.
    # The ramp carries almost no information - it says where you are in the
    # patch, not what hour it is - and it is not what inference shows, so the
    # nets were both starved of the diurnal pattern and tested off-distribution.
    if TOD_TRAIN == "clock" and t0 is not None:
        clock = (((t0 + np.arange(T)) % 288) / 288.0).astype(np.float32)
        tod = clock[:, None] * np.ones((1, L), np.float32)
    else:
        tod = np.linspace(0, 1, T, dtype=np.float32)[:, None] * np.ones((1, L), np.float32)
    if AUG and rng.random() < 0.5:
        tod = np.full_like(tod, 0.5)
    chans = [v, q, o, seen.astype(np.float32), np.where(seen, g[..., 4], 0.0),
             np.repeat(stat["vf"][None, :] / 120.0, T, 0),
             np.repeat(stat["lanes"][None, :] / 6.0, T, 0),
             np.repeat(stat["cap"][None, :] / 12000.0, T, 0),
             tod]
    if RAMPS:
        chans += [g[..., 5], g[..., 6], g[..., 7]]
    x = np.stack(chans, 0)
    if KHEAD or GATE:
        v_phys = np.maximum(g[..., 0] * stat["vf"][None, :], 5.0)
        q_phys = g[..., 1] * scale * stat["cap"][None, :]
        k_ratio = (q_phys / v_phys) / (stat["lanes"][None, :] * 100.0)
        y = np.stack([g[..., 0], g[..., 1] * scale, k_ratio], 0)
    else:
        y = np.stack([g[..., 0], g[..., 1] * scale], 0)  # true v/vf and q/cap
    return x.astype(np.float32), y.astype(np.float32), hide


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, w, d):
        super().__init__()
        self.n = nn.GroupNorm(8, w)
        self.c1 = nn.Conv2d(w, w, 3, padding=d, dilation=d)
        self.c2 = nn.Conv2d(w, w, 1)
        self.drop = nn.Dropout2d(DROP) if DROP else nn.Identity()

    def forward(self, x):
        return x + self.c2(self.drop(F.gelu(self.c1(F.gelu(self.n(x))))))


class Net(nn.Module):
    def __init__(self, w=WIDTH, dil=DILS, ch_in=None, ch_out=None):
        super().__init__()
        self.inp = nn.Conv2d(ch_in or CH_IN, w, 3, padding=1)
        self.body = nn.Sequential(*[Block(w, d) for d in dil])
        n_out = ch_out if ch_out is not None else (4 if GATE else (3 if KHEAD else 2))
        self.out = nn.Sequential(nn.GroupNorm(8, w), nn.GELU(), nn.Conv2d(w, n_out, 1))

    def forward(self, x):
        return self.out(self.body(self.inp(x)))


class DownBlock(nn.Module):
    """A residual conv at the current (cin-channel) resolution - the skip -
    then a stride-2 downsample that ALSO lifts the channel count to cout."""

    def __init__(self, cin, cout):
        super().__init__()
        self.block = Block(cin, 1)
        self.down = nn.Conv2d(cin, cout, 3, stride=2, padding=1)

    def forward(self, x):
        skip = self.block(x)
        return self.down(skip), skip


class UpBlock(nn.Module):
    """Upsample, concatenate the matching encoder skip, two residual convs."""

    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, stride=2)
        self.merge = nn.Conv2d(cout + cskip, cout, 1)
        self.b1 = Block(cout, 1)

    def forward(self, x, skip):
        x = self.up(x)
        # a crop shape not divisible by 8 (3 downsamples) leaves a 1-pixel
        # mismatch at some levels - trim the skip to match rather than pad,
        # since padding would hand the block a border of never-observed cells.
        if x.shape[-2:] != skip.shape[-2:]:
            skip = skip[..., :x.shape[-2], :x.shape[-1]]
        return self.b1(self.merge(torch.cat([x, skip], 1)))


class UNet(nn.Module):
    """Multi-scale alternative to Net's fixed-resolution dilated stack.

    Every member trained all session - f1 through v1, dn1/dn2, gt1 - is the
    same family: dilated residual convs at one fixed resolution, varying only
    width/depth/context/dilation schedule. A U-Net has a different inductive
    bias entirely: it can represent a short, local speed drop and a
    corridor-wide queue at DIFFERENT scales explicitly, via real
    downsampling/upsampling with skip connections, rather than asking one
    fixed-resolution kernel stack to do both.
    """

    def __init__(self, w=64, depth=3, ch_in=None, ch_out=None):
        super().__init__()
        n_out = ch_out if ch_out is not None else (4 if GATE else (3 if KHEAD else 2))
        cin = ch_in or CH_IN
        self.inp = nn.Conv2d(cin, w, 3, padding=1)
        # chans[i] is the skip's own channel count at encoder level i (full
        # res = chans[0], each level down doubles it); the bottleneck sits one
        # level below chans[-1].
        chans = [w * (2 ** i) for i in range(depth)]
        bott = w * (2 ** depth)
        down_io = list(zip(chans, chans[1:] + [bott]))
        self.downs = nn.ModuleList([DownBlock(ci, co) for ci, co in down_io])
        self.bott = nn.Sequential(Block(bott, 1), Block(bott, 2))
        up_io = list(zip([bott] + chans[:0:-1], chans[::-1]))
        self.ups = nn.ModuleList([UpBlock(ci, co, co) for ci, co in up_io])
        self.out = nn.Sequential(nn.GroupNorm(8, w), nn.GELU(), nn.Conv2d(w, n_out, 1))

    def forward(self, x):
        x = self.inp(x)
        skips = []
        for d in self.downs:
            x, skip = d(x)
            skips.append(skip)
        x = self.bott(x)
        for u, skip in zip(self.ups, reversed(skips)):
            x = u(x, skip)
        return self.out(x)


# Dilations live in the forward pass, not in the weights, so a checkpoint cannot
# say what shape it wants. Anything that is not the current default is recorded
# here; load_net() reads the width straight off the weights.
ARCH = {n: dict(dil=(1, 2, 4, 8, 16, 1, 2, 4, 8, 1))
        for n in ("f1", "g1", "f2", "G0", "G1", "G2", "h1", "i1", "j1", "k1", "l1", "m1", *(f"ft_{p}" for p in PANELS))}


ARCH["m1"] = dict(dil=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 1))
# gap-aware finetunes (T1_GAPP) keep their parent's shape
for _n in ("gf1", "gf1b", "gh1", "gi1", "gj1", "gn1", "ge1", "gr1", "gr2", "zz_smoke", "geh1", "gei1", "gen1"):
    ARCH[_n] = dict(dil=(1, 2, 4, 8, 16, 1, 2, 4, 8, 1))
ARCH["gm1"] = dict(dil=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 1))
ARCH["gem1"] = ARCH["gm1"]
# q*: the six members finetuned with finite-but-ineligible readings as inputs (T1_INELIG_TRAIN)
for _n in ("qf1", "qh1", "qi1", "qj1", "qn1"):
    ARCH[_n] = dict(dil=(1, 2, 4, 8, 16, 1, 2, 4, 8, 1))
ARCH["qm1"] = ARCH["gm1"]
# <gapnet>q: the 12 gap nets fine-tuned with ineligible readings as inputs (T1_INELIG_TRAIN)
for _n in ("gf1", "gh1", "gi1", "gm1", "gn1", "ge1", "gr1", "geh1", "gei1", "gem1", "gen1", "xs1"):
    if _n in ARCH:
        ARCH[_n + "q"] = ARCH[_n]
# xs1: from-scratch gap member - wide, deep, 16 h x 96 links, ramps, event gaps (s83)
ARCH["xs1"] = dict(dil=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32, 1))
ARCH["xs1q"] = ARCH["xs1"]
# Members that differ structurally, not by seed - s50 showed that is where
# ensemble diversity comes from: n1 sees twice the context, o1 is deeper.
ARCH["n1"] = dict(dil=(1, 2, 4, 8, 16, 1, 2, 4, 8, 1))
ARCH["o1"] = dict(dil=(1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1, 2))
# n1 (twice the context) paid in the ensemble, o1 (depth) did not: push
# context further - a 256x128 patch needs dilations to 32 to be seen whole.
ARCH["n2"] = dict(dil=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 1))
# p1/p2 are f1's shape trained on a different objective: the S_LWR term carries
# real weight, so they trade a little S_state for density accuracy. The earlier
# LWR-term run (c1, lwr_w 0.02) was judged as a replacement for f1 and read
# neutral; the per-member scan of 2026-09-22 shows ensembling lifts S_LWR far
# more than S_state (members 0.6501-0.6579 -> mean 0.6643 against 0.9529-0.9541
# -> 0.9547), so the objective is worth trying as a diversity axis instead.
ARCH["p1"] = dict(dil=(1, 2, 4, 8, 16, 1, 2, 4, 8, 1))
# w1 is f1's shape and recipe with T1_TOD_TRAIN=clock: the only difference is
# that its time-of-day channel is the real hour rather than the position inside
# the crop, so train and inference finally agree *and* the channel carries the
# diurnal pattern. Score it with T1_TOD=abs, which is that same convention.
ARCH["w1"] = dict(dil=(1, 2, 4, 8, 16, 1, 2, 4, 8, 1))
# x1/y1 are two more seeds of that recipe and z1 is m1's wider shape, all
# clock-trained, so a four-member clock ensemble mirrors the shape mix of the
# six ramp-trained members it would replace.
ARCH["x1"] = dict(dil=(1, 2, 4, 8, 16, 1, 2, 4, 8, 1))
# v1 sees a whole day at once: a 288 x 48 crop (24 hours by 48 links) against
# f1's 96 x 64 and n1's 192 x 96, with dilations to 64 so the reach covers it.
# Context is the one axis that still paid - n1 gave +0.00053 on the board - and
# it was never pushed to a full diurnal cycle.
ARCH["v1"] = dict(dil=(1, 2, 4, 8, 16, 32, 64, 1, 2, 4, 8, 1))
# v1 (full-day crop, 288x48) is the only structural change that has ever won a
# swap against the shipped six (2026-09-24, -m1 +v1, +0.00012 S_state on all
# ten panels). v2 is a second seed of the same recipe, to see if the win is
# reproducible or a lucky draw; v3 is the same crop and dilations at double the
# width, to test whether capacity was the limit at this context length
# specifically (every earlier capacity test was at the shorter 96x64 crop).
ARCH["v2"] = dict(dil=(1, 2, 4, 8, 16, 32, 64, 1, 2, 4, 8, 1))
ARCH["v3"] = dict(dil=(1, 2, 4, 8, 16, 32, 64, 1, 2, 4, 8, 1))
ARCH["y1"] = dict(dil=(1, 2, 4, 8, 16, 1, 2, 4, 8, 1))
ARCH["z1"] = dict(dil=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 1))
ARCH["p2"] = dict(dil=(1, 2, 4, 8, 16, 1, 2, 4, 8, 1))
# dn1: f1's exact recipe (128ch, 96x64, dilations to 16) plus T1_KHEAD=1 - the
# density-head mechanism, isolated from every other change, to test whether a
# decoupled smoothness objective succeeds where p1's coupled one did not.
ARCH["dn1"] = dict(dil=(1, 2, 4, 8, 16, 1, 2, 4, 8, 1))
# dn2 is h1's seed of the same recipe, dn3 is m1's wider shape - both with the
# density head, to test whether the +0.0042 S_lwr gain dn1 shows alone survives
# averaging when it is not diluted 1-in-6 among untouched members.
ARCH["dn2"] = dict(dil=(1, 2, 4, 8, 16, 1, 2, 4, 8, 1))
ARCH["dn3"] = dict(dil=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 1))
# gt1: f1's recipe with T1_GATE=1 - a learned per-cell gate reconciling the
# flow head with the density head, trained end-to-end on the reconciled
# output (not a fixed global beta swept after the fact, as dn1/dn2 used).
ARCH["gt1"] = dict(dil=(1, 2, 4, 8, 16, 1, 2, 4, 8, 1))


def load_net(name, device=None):
    """Build the right Net (or UNet) for a checkpoint and load it."""
    sd = torch.load(OUT / f"t1_nn_{name}.pt", map_location=device or DEV)
    w = sd["inp.weight"].shape[0]
    ch_in, ch_out = sd["inp.weight"].shape[1], sd["out.2.weight"].shape[0]
    if any(k.startswith("downs.") for k in sd):
        # a U-Net checkpoint - depth is however many "downs.N." indices exist,
        # width is read straight off the input conv same as the dilated family
        depth = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("downs."))
        net = UNet(w=w, depth=depth, ch_in=ch_in, ch_out=ch_out).to(device or DEV)
        net.load_state_dict(sd)
        return net.eval()
    # an adapted checkpoint "<base>__<tag>" keeps its base net's architecture
    dil = ARCH.get(name, ARCH.get(name.split("__")[0], {})).get("dil")
    if dil is None:
        n_blocks = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("body."))
        dil = DILS if len(DILS) == n_blocks else tuple(DILS[i % len(DILS)]
                                                       for i in range(n_blocks))
    net = Net(w=w, dil=dil, ch_in=ch_in, ch_out=ch_out).to(device or DEV)
    net.load_state_dict(sd)
    return net.eval()


def losses(pred, y, hide, stat, lwr_w=0.0):
    """S_state's own weighting, in the submission's units, plus the LWR term."""
    vf = stat["vf"][None, None, :]; cap = stat["cap"][None, None, :]
    lanes = stat["lanes"][None, None, :]; length = stat["length"][None, None, :]
    v_hat, v = pred[:, 0] * vf, y[:, 0] * vf
    q_head, q = pred[:, 1] * cap, y[:, 1] * cap
    q_hat = q_head
    kscale = lanes * 100.0
    k_hat = pred[:, 2] * kscale if pred.shape[1] >= 3 else None
    if GATE and pred.shape[1] >= 4:
        # The gate is part of the forward pass, not a post-hoc sweep: the
        # state loss is applied to the RECONCILED flow, so the backbone learns
        # directly what to submit rather than an independent flow head that a
        # fixed beta blends in afterwards (dn1/dn2's design).
        gate = torch.sigmoid(pred[:, 3])
        q_hat = (1 - gate) * q_head + gate * (k_hat * v_hat)
    m = hide.float()
    n = m.sum().clamp(min=1.0)
    l_v = (((v_hat - v) / 25.0) ** 2 * m).sum() / n
    l_q = (((q_hat - q) / (600.0 * lanes)) ** 2 * m).sum() / n
    loss = 0.54 * l_v + 0.46 * l_q
    if lwr_w:
        # Accumulation on the submitted field: observed cells keep their value.
        # Everything is clamped before the division - an early net emits
        # negative speeds and the raw ratio went to nan within 500 steps - and
        # the term is normalised by the truth's own |dN| so its scale matches
        # S_LWR (sum|resid| / sum|rhs|) rather than the units of a vehicle.
        fill = lambda a, b: torch.where(hide, a, b)
        v_s = fill(v_hat, v).clamp(min=5.0)
        q_s = fill(q_hat, q).clamp(min=5.0)
        N_hat = q_s / v_s * length
        N = q.clamp(min=5.0) / v.clamp(min=5.0) * length
        d_hat = N_hat[:, 1:] - N_hat[:, :-1]
        d = N[:, 1:] - N[:, :-1]
        loss = loss + lwr_w * ((d_hat - d).abs().sum()
                               / d.abs().sum().clamp(min=1e-3))
    l_k = torch.zeros((), device=pred.device)
    if pred.shape[1] >= 3:
        # A density head with its OWN smoothness loss, decoupled from the v/q
        # heads that S_state depends on - p1's LWR term derived N_hat from
        # q_hat/v_hat directly, so it competed with those same weights for
        # exactly the gradient S_state needs; this does not.
        k_true = y[:, 2] * kscale
        l_k = (((k_hat - k_true) / K_SCALE) ** 2 * m).sum() / n
        loss = loss + K_WEIGHT * l_k
        if K_LWR_W:
            fill = lambda a, b: torch.where(hide, a, b)
            k_s = fill(k_hat, k_true).clamp(min=1.0)
            N_hat_k = k_s * length
            N_true_k = k_true.clamp(min=1.0) * length
            dk_hat = N_hat_k[:, 1:] - N_hat_k[:, :-1]
            dk_true = N_true_k[:, 1:] - N_true_k[:, :-1]
            loss = loss + K_LWR_W * ((dk_hat - dk_true).abs().sum()
                                     / dk_true.abs().sum().clamp(min=1e-3))
    return loss, l_v.item(), l_q.item(), l_k.item()


def batch(data, rng, bs, val=False, panel=None):
    xs, ys, hs, sts = [], [], [], []
    for _ in range(bs):
        panel = panel or PANELS[rng.integers(len(PANELS))]
        g, stat, t0 = data.crop(panel, rng, val)
        x, y, hide = make_example(g, stat, rng, t0=t0, gap_at=data.last_gap)
        xs.append(x); ys.append(y); hs.append(hide); sts.append(stat)
    t = lambda a: torch.from_numpy(np.stack(a)).to(DEV)
    stat = {k: t([s[k] for s in sts]).float() for k in ("vf", "lanes", "cap", "length")}
    stat = {k: v[:, None, :] if v.dim() == 2 else v for k, v in stat.items()}
    return t(xs), t(ys), torch.from_numpy(np.stack(hs)).to(DEV), sts


def stack_stat(sts):
    out = {}
    for k in ("vf", "lanes", "cap", "length"):
        out[k] = torch.from_numpy(np.stack([s[k] for s in sts])).to(DEV).float()[:, None, :]
    return out


def evaluate(net, data, rng, n=64, panel=None):
    net.eval()
    sv, sq, cnt = 0.0, 0.0, 0
    with torch.no_grad():
        for _ in range(n):
            x, y, hide, sts = batch(data, rng, 8, val=True, panel=panel)
            pred = net(x)
            st = stack_stat(sts)
            vf, cap, lanes = st["vf"], st["cap"], st["lanes"]
            m = hide
            dv = ((pred[:, 0] - y[:, 0]) * vf)[m]
            dq = ((pred[:, 1] - y[:, 1]) * cap / lanes)[m]
            sv += float((dv ** 2).sum()); sq += float((dq ** 2).sum()); cnt += int(m.sum())
    net.train()
    rmse_v, rmse_q = (sv / cnt) ** 0.5, (sq / cnt) ** 0.5
    S = 0.54 * (1 - rmse_v / 25) + 0.46 * (1 - rmse_q / 600)
    return rmse_v, rmse_q, S


def train(name, steps, lwr_w, bs=int(os.environ.get("T1_BS", "16")), lr=2e-3, seed=0,
          init=None, only_panel=None):
    """init=NAME starts from that checkpoint; only_panel trains on one corridor.

    One net covers ten corridors that differ in length, lane count and capacity,
    so a short per-panel finetune is the cheapest way to let it specialise
    without giving up what the shared run learned.
    """
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    data = Data()
    if init:
        net = load_net(init)
        if CH_IN > net.inp.in_channels:
            # widen the input conv for the ramp channels, new weights at zero so
            # the finetune starts exactly where the parent was
            old = net.inp
            new = nn.Conv2d(CH_IN, old.out_channels, 3, padding=1).to(DEV)
            with torch.no_grad():
                new.weight.zero_(); new.weight[:, :old.in_channels] = old.weight
                new.bias.copy_(old.bias)
            net.inp = new
    elif ARCH_FAMILY == "unet":
        net = UNet(w=UWIDTH, depth=UDEPTH).to(DEV)
    else:
        net = Net().to(DEV)
    net.train()
    if init:
        lr = lr / 10.0            # a finetune should not undo the shared run
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.05)
    best, t0, run = -1, time.time(), []
    for step in range(1, steps + 1):
        x, y, hide, sts = batch(data, rng, bs, panel=only_panel)
        loss, lv, lq, lk = losses(net(x), y, hide, stack_stat(sts), lwr_w)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step(); run.append(loss.item())
        if step % 500 == 0 or step == steps:
            rv, rq, S = evaluate(net, data, np.random.default_rng(99), panel=only_panel)
            flag = ""
            if S > best:
                best = S; flag = " *"
                torch.save(net.state_dict(), OUT / f"t1_nn_{name}.pt")
            ktag = f"  l_k {lk:.4f}" if KHEAD else ""
            print(f"step {step:6d} loss {np.mean(run):.5f}  holdout speed RMSE {rv:.3f} "
                  f"flow/lane {rq:.2f}  S_state {S:.4f}{ktag} [{time.time()-t0:.0f}s]{flag}",
                  flush=True)
            run = []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "train", "eval", "prep_masked", "adapt"])
    ap.add_argument("--split", default="validation")
    ap.add_argument("--splits", default="validation,private")
    ap.add_argument("--day_lo", type=int)
    ap.add_argument("--day_hi", type=int)
    ap.add_argument("name", nargs="?", default="a")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lwr", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init")
    ap.add_argument("--panel")
    a = ap.parse_args()
    if a.cmd == "prep":
        prep()
    elif a.cmd == "prep_masked":
        prep_masked(a.split)
    elif a.cmd == "adapt":
        adapt(a.init, a.name, a.splits.split(","), a.steps, a.day_lo, a.day_hi, seed=a.seed)
    elif a.cmd == "train":
        train(a.name, a.steps, a.lwr, seed=a.seed, init=a.init, only_panel=a.panel)


# ---------------------------------------------------------------- inference

def tod_for(mode, tod, sl, n):
    """The time-of-day channel for one tile, under the chosen convention."""
    if mode == "const":
        return np.full(n, 0.5, np.float32)
    if mode == "tile":
        return np.linspace(0, 1, n, dtype=np.float32)
    return tod[sl].astype(np.float32)


def predict_grid(net, panel, split, tile_t=256, over=16, t_lo=0, t_hi=None):
    """Fill a whole panel-split grid, tiling over time with overlap.

    The masked release is the input: observed cells keep their values, blanked
    ones are what the net writes. Tiles overlap so no target cell sits at a
    tile edge with half its context missing.
    """
    G = load_grid(panel, split, masked=True)
    st = G["stat"]
    nT, nL = G["nT"], G["nL"]
    vf = st["vf"][None, :]; cap = st["cap"][None, :]
    # T1_PCTMIN: lowest pct_observed fed to the net as an observation (75 = only
    # score-eligible readings, as trained). Finite readings below 75 are ~7% of
    # cells and nearly as informative; T1_PCTCLIP shows them at that pct floor.
    pmin = float(os.environ.get("T1_PCTMIN", "75"))
    if pmin < 75:
        obs = np.isfinite(G["speed"]) & np.isfinite(G["flow"]) & (G["pct"] >= pmin)
    else:
        obs = np.isfinite(G["speed"]) & (G["pct"] >= 75)
    z = np.load(NN / f"{panel}_static.npz", allow_pickle=True)
    occ_scale = float(z["occ_scale"])
    v = np.nan_to_num(G["speed"] / vf) * obs
    q = np.nan_to_num(G["flow"] / cap) * obs
    o = np.nan_to_num(G["occ"] / occ_scale) * obs
    pcr = G["pct"] / 100.0
    if os.environ.get("T1_PCTCLIP"):
        pcr = np.maximum(pcr, float(os.environ["T1_PCTCLIP"]) / 100.0)
    pc = np.nan_to_num(pcr) * obs
    ramp = None
    if net.inp.weight.shape[1] == 12:
        from t1build import ramp_matrices
        R = ramp_matrices(panel, split, list(G["link_ids"]))
        ramp = (np.nan_to_num(R["on"] / cap), np.nan_to_num(R["off"] / cap),
                (R["miss"] != 0).astype(np.float32) if "miss" in R
                else np.zeros((nT, nL), np.float32))
    tile_t = int(os.environ.get("T1_TILE_T", tile_t))
    tile_l = int(os.environ.get("T1_TILE_L", "0")) or nL
    over_l = min(8, max(tile_l // 8, 1))
    n_out = net.out[-1].out_channels
    out_v = np.zeros((nT, nL), np.float32); out_q = np.zeros((nT, nL), np.float32)
    out_k = np.zeros((nT, nL), np.float32) if n_out >= 3 else None
    out_g = np.zeros((nT, nL), np.float32) if n_out >= 4 else None
    wsum = np.zeros((nT, nL), np.float32)
    # T1_TOD: which time-of-day signal to feed. make_example trains on a 0->1
    # ramp across the crop, not the clock, so "abs" hands the net a channel it
    # never saw. "tile" reproduces the training signal per tile and "const"
    # matches what the augmentation shows half the time.
    tod = (np.arange(nT) % 288) / 288.0
    TOD_MODE = os.environ.get("T1_TOD", "abs")
    net.eval()
    with torch.no_grad():
        hi = nT if t_hi is None else min(t_hi, nT)
        lo = max(0, t_lo)
        for t0 in range(lo, hi, tile_t - 2 * over):
            t1 = min(t0 + tile_t, hi)
            if t1 - t0 < 8:
                break
            sl = slice(t0, t1)
            for j0 in range(0, nL, max(tile_l - 2 * over_l, 1)):
                j1 = min(j0 + tile_l, nL)
                if j1 - j0 < 4 and j0 > 0:
                    break
                jl = slice(j0, j1)
                nj = j1 - j0
                chans = [v[sl, jl], q[sl, jl], o[sl, jl], obs[sl, jl].astype(np.float32),
                         pc[sl, jl],
                         np.repeat(st["vf"][None, jl] / 120.0, t1 - t0, 0),
                         np.repeat(np.maximum(st["lanes"], 1.0)[None, jl] / 6.0, t1 - t0, 0),
                         np.repeat(st["cap"][None, jl] / 12000.0, t1 - t0, 0),
                         np.repeat(tod_for(TOD_MODE, tod, sl, t1 - t0)[:, None], nj, 1)]
                if ramp is not None:
                    chans += [ramp[0][sl, jl], ramp[1][sl, jl], ramp[2][sl, jl]]
                x = np.stack(chans, 0)[None]
                p = net(torch.from_numpy(x.astype(np.float32)).to(DEV))[0].cpu().numpy()
                w = np.ones((t1 - t0, nj), np.float32)
                # ramp against the requested range, not the grid: a range-limited
                # call would otherwise fade its own first row to zero weight
                if t0 > lo:
                    w[:over] = np.minimum(w[:over], np.linspace(0, 1, over)[:, None])
                if t1 < hi:
                    w[-over:] = np.minimum(w[-over:], np.linspace(1, 0, over)[:, None])
                if j0 > 0:
                    w[:, :over_l] = np.minimum(w[:, :over_l], np.linspace(0, 1, over_l)[None, :])
                if j1 < nL:
                    w[:, -over_l:] = np.minimum(w[:, -over_l:], np.linspace(1, 0, over_l)[None, :])
                out_v[sl, jl] += p[0] * w; out_q[sl, jl] += p[1] * w; wsum[sl, jl] += w
                if out_k is not None:
                    out_k[sl, jl] += p[2] * w
                if out_g is not None:
                    out_g[sl, jl] += p[3] * w
                if j1 >= nL:
                    break
    out_v /= np.maximum(wsum, 1e-6); out_q /= np.maximum(wsum, 1e-6)
    speed = np.clip(out_v * vf, 5.0, vf * 1.05)
    if out_g is not None:
        # A learned per-cell gate reconciles the flow head with the density
        # head BEFORE anything is clipped - this net's flow is already the
        # final, submission-ready value, a drop-in replacement for a plain
        # 2-channel net anywhere in the pipeline (t1_field_cache.py, the
        # ensemble scripts, ...), no external beta sweep needed.
        out_k /= np.maximum(wsum, 1e-6)
        density = np.maximum(out_k, 0.0) * (np.maximum(st["lanes"], 1.0)[None, :] * 100.0)
        gate = 1.0 / (1.0 + np.exp(-out_g / np.maximum(wsum, 1e-6)))
        out_q_final = (1 - gate) * out_q * cap + gate * (density * speed)
        flow = np.clip(out_q_final, 5.0, cap * 1.15)
        return G, speed, flow, obs
    flow = np.clip(out_q * cap, 5.0, cap * 1.15)
    if out_k is not None:
        # density head's own units: ratio * (lanes*100), matching training's
        # k_ratio normalisation exactly (see make_example/losses)
        out_k /= np.maximum(wsum, 1e-6)
        density = np.maximum(out_k, 0.0) * (np.maximum(st["lanes"], 1.0)[None, :] * 100.0)
        return G, speed, flow, obs, density
    return G, speed, flow, obs


def write_submission(name, out_dir, blend=0.0, tree_csv=None):
    """state_submission.csv from the net(s), optionally averaged with a tree file.

    `name` is one checkpoint for every panel, or a callable panel -> list of
    checkpoints whose fills are averaged - which is how per-panel finetunes and
    two-net means are shipped.
    """
    import pandas as pd
    from common import ROOT
    pick = name if callable(name) else (lambda panel: [name])
    cache = {}
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for panel in PANELS:
        names = pick(panel)
        for n in names:
            if n not in cache:
                cache[n] = load_net(n)
        for split in ("validation", "private"):
            got = [predict_grid(cache[n], panel, split) for n in names]
            G = got[0][0]
            speed = np.mean([g[1] for g in got], 0)
            flow = np.mean([g[2] for g in got], 0)
            key = pd.read_csv(ROOT / "task1" / panel / split / "sample_submission_state.csv",
                              dtype={"link_id": str, "station_id": str})
            ts = pd.to_datetime(key.timestamp)
            di = {d: i for i, d in enumerate(G["days"])}
            li = {l: i for i, l in enumerate(G["link_ids"])}
            row = ts.dt.strftime("%Y-%m-%d").map(di).to_numpy() * 288 + \
                (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy()
            col = key.link_id.map(li).to_numpy()
            assert not np.isnan(row).any() and not pd.isna(col).any()
            k = key.copy()
            k["speed_kmh"] = speed[row.astype(int), col.astype(int)]
            k["flow_vph"] = flow[row.astype(int), col.astype(int)]
            parts.append(k)
            print(f"  [{panel}/{split}] {len(k):,} rows", flush=True)
    sub = pd.concat(parts, ignore_index=True)
    if tree_csv and blend > 0:
        tree = pd.read_csv(tree_csv, dtype={"link_id": str, "station_id": str})
        m = sub.merge(tree[["timestamp", "link_id", "panel", "speed_kmh", "flow_vph"]],
                      on=["timestamp", "link_id", "panel"], suffixes=("", "_t"))
        assert len(m) == len(sub), f"{len(m)} vs {len(sub)}"
        m["speed_kmh"] = (1 - blend) * m.speed_kmh + blend * m.speed_kmh_t
        m["flow_vph"] = (1 - blend) * m.flow_vph + blend * m.flow_vph_t
        sub = m.drop(columns=["speed_kmh_t", "flow_vph_t"])
    sub[["panel", "timestamp", "station_id", "link_id", "mask_regime",
         "speed_kmh", "flow_vph"]].to_csv(out_dir / "state_submission.csv", index=False)
    print("wrote", out_dir / "state_submission.csv", len(sub), "rows")
