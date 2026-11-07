"""Is the net sensitive to the time-of-day channel, and does the inference
convention hurt?

make_example trains it as `linspace(0, 1, TCROP)` - slope 1/96 for f1 - while
predict_grid feeds `(t % 288) / 288`, slope 1/288 from an arbitrary offset.
This scores the same held-out cells with the same net, changing only that one
channel, so any difference is the convention and nothing else.

    T1_TCROP=96 T1_LCROP=64 python t1_tod_probe.py f1 m1 n1
"""
import nice  # noqa: F401
import os, sys
import numpy as np
import torch
import t1_nn as NN

TOD_CH = 8
BS, NB = 8, 40


def variants(T, rng):
    base = np.linspace(0, 1, T, dtype=np.float32)          # as trained
    off = rng.random()
    clock = (off + np.arange(T, dtype=np.float32) / 288.0) % 1.0   # as shipped
    return {"train-ramp": base, "clock(t%288/288)": clock,
            "const 0.5": np.full(T, 0.5, np.float32)}


def main():
    names = sys.argv[1:] or ["f1"]
    data = NN.Data()
    print(f"TCROP={NN.TCROP} LCROP={NN.LCROP}  {NB} batches x {BS}")
    print(f"{'net':4s} {'tod channel':18s} {'speed RMSE':>10s} {'flow/lane':>10s} "
          f"{'S_state':>8s} {'S_lwr':>8s} {'0.35S+.1L':>9s}")
    for n in names:
        net = NN.load_net(n)
        acc = {}
        rng = np.random.default_rng(0)
        for _ in range(NB):
            x, y, hide, sts = NN.batch(data, rng, BS, val=True)
            st = NN.stack_stat(sts)
            for label, tod in variants(x.shape[2], rng).items():
                xv = x.clone()
                xv[:, TOD_CH] = torch.from_numpy(tod).to(x.device)[None, :, None]
                with torch.no_grad():
                    pred = net(xv)
                dv = ((pred[:, 0] - y[:, 0]) * st["vf"])[hide]
                dq = ((pred[:, 1] - y[:, 1]) * st["cap"] / st["lanes"])[hide]
                # S_LWR's own numerator and denominator, on the submitted field
                # (observed cells keep the truth), exactly as losses() forms it
                fill = lambda a_, b_: torch.where(hide, a_, b_)
                v_s = fill(pred[:, 0] * st["vf"], y[:, 0] * st["vf"]).clamp(min=5.0)
                q_s = fill(pred[:, 1] * st["cap"], y[:, 1] * st["cap"]).clamp(min=5.0)
                N_h = q_s / v_s * st["length"]
                N_t = (y[:, 1] * st["cap"]).clamp(min=5.0) / (y[:, 0] * st["vf"]).clamp(min=5.0) * st["length"]
                d_h = N_h[:, 1:] - N_h[:, :-1]
                d_t = N_t[:, 1:] - N_t[:, :-1]
                a = acc.setdefault(label, [0.0, 0.0, 0, 0.0, 0.0])
                a[0] += float((dv ** 2).sum()); a[1] += float((dq ** 2).sum())
                a[2] += int(hide.sum())
                a[3] += float((d_h - d_t).abs().sum()); a[4] += float(d_t.abs().sum())
        for label, (sv, sq, c, e_n, e_d) in acc.items():
            rv, rq = (sv / c) ** 0.5, (sq / c) ** 0.5
            S = 0.54 * (1 - rv / 25) + 0.46 * (1 - rq / 600)
            S_lwr = max(0.0, 1.0 - min(1.0, e_n / max(e_d, 1e-9)))
            part = 0.35 * S + 0.10 * S_lwr
            print(f"{n:4s} {label:18s} {rv:10.4f} {rq:10.3f} {S:8.5f} {S_lwr:8.4f} {part:9.5f}", flush=True)


if __name__ == "__main__":
    main()
