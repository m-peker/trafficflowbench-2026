"""Rebuild our final TrafficFlowBench submissions from the release and the archived models.

    python reproduce.py --data <release dir> --work <scratch dir> --final s111
    python reproduce.py --data <release dir> --work <scratch dir> --final s79 --reference submission_s79.zip

Every stage writes into --work and is skipped when its output already exists,
so an interrupted run resumes; --force STAGE reruns one stage. The chain:

    caches   dense (time x link) grids from the parquet release       CPU
    models   archived weights -> work/out                              -
    task4    ODME: NNLS anchored on the released weak prior            CPU
    task2    queue: 3 LightGBM + 3 conv nets, thresholds, rules        GPU/CPU
    trees    Task 1 tree ensemble (4 LightGBM members)                 CPU
    nets     Task 1: six conv nets, 0.85 / 0.15 blend with the trees   GPU
    gapnets  eleven gap-aware nets on the validation/private grids     GPU
    gapfill  Task 1 rows inside the Task 2 gaps (final-specific)       CPU
    package  merge onto sample_submission.csv, validate, zip           CPU
"""
import argparse, os, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"
GAPNETS = "gf1,gh1,gi1,gm1,gn1,ge1,gr1,geh1,gei1,gem1,gen1"
GAPNETS12 = GAPNETS + ",xs1"
QNETS = "qf1,qh1,qi1,qj1,qm1,qn1"
GAPQ = "gf1q,gh1q,gi1q,gm1q,gn1q,ge1q,gr1q,geh1q,gei1q,gem1q,gen1q,xs1q"
FINALS = {
    # s131 / s132: s124 / s125 rebuilt after the compliance fix (qdata.rolling_priors no longer
    # falls back to the window's own day on the first validation day) - the selected finals
    # s124 / s125: s122 / s123 with a second LightGBM seed (bothrollT2) in the LightGBM member
    "s124": dict(GAP_RULE="analog", GAP_MAXD="0.003", GAP_NET=GAPQ, GAP_NETW="1.0",
                 GAP_NETW_Q="1.0", GAP_SM="union", T1_SIX=QNETS, T1_PCTMIN="0", T4_REFINE="lbfgs",
                 GAPNET_ENV=dict(T1_PCTMIN="0"), T2="roll3Tbw", T2_R4ROLL="2", T2_BROLL2="1"),
    "s125": dict(GAP_RULE="mix", GAP_NET=GAPNETS, GAP_NETW="1.0", GAP_NETW_Q="1.0", T4_REFINE="lbfgs",
                 T2="roll3Tbw", T2_R4ROLL="2", T2_BROLL2="1"),
    # s122 / s123: s120 / s121 with a second rolling-prior 2-D net (r4rollT2) in the 2-D member
    "s122": dict(GAP_RULE="analog", GAP_MAXD="0.003", GAP_NET=GAPQ, GAP_NETW="1.0",
                 GAP_NETW_Q="1.0", GAP_SM="union", T1_SIX=QNETS, T1_PCTMIN="0", T4_REFINE="lbfgs",
                 GAPNET_ENV=dict(T1_PCTMIN="0"), T2="roll3Tbw", T2_R4ROLL="2"),
    "s123": dict(GAP_RULE="mix", GAP_NET=GAPNETS, GAP_NETW="1.0", GAP_NETW_Q="1.0", T4_REFINE="lbfgs",
                 T2="roll3Tbw", T2_R4ROLL="2"),
    # s120 / s121: s118 / s119 with r4rollT (2-D net on rolling priors) added to the 2-D member
    "s120": dict(GAP_RULE="analog", GAP_MAXD="0.003", GAP_NET=GAPQ, GAP_NETW="1.0",
                 GAP_NETW_Q="1.0", GAP_SM="union", T1_SIX=QNETS, T1_PCTMIN="0", T4_REFINE="lbfgs",
                 GAPNET_ENV=dict(T1_PCTMIN="0"), T2="roll3Tbw", T2_R4ROLL="1"),
    "s121": dict(GAP_RULE="mix", GAP_NET=GAPNETS, GAP_NETW="1.0", GAP_NETW_Q="1.0", T4_REFINE="lbfgs",
                 T2="roll3Tbw", T2_R4ROLL="1"),
    # s118 / s119: s116 / s117 with the LightGBM member using ROLLING priors - queue rates
    # from the 14 days before the window's day, on the window's own split's masked layer
    # (validation precedes private; all stamped before T) - instead of no priors
    "s118": dict(GAP_RULE="analog", GAP_MAXD="0.003", GAP_NET=GAPQ, GAP_NETW="1.0",
                 GAP_NETW_Q="1.0", GAP_SM="union", T1_SIX=QNETS, T1_PCTMIN="0", T4_REFINE="lbfgs",
                 GAPNET_ENV=dict(T1_PCTMIN="0"), T2="roll3Tbw"),
    "s119": dict(GAP_RULE="mix", GAP_NET=GAPNETS, GAP_NETW="1.0", GAP_NETW_Q="1.0", T4_REFINE="lbfgs",
                 T2="roll3Tbw"),
    # s116 / s117: s113 / s114 with the two no-prior nets seed-bagged (q1npT+q1npT2, r4npT+r4npT2)
    "s116": dict(GAP_RULE="analog", GAP_MAXD="0.003", GAP_NET=GAPQ, GAP_NETW="1.0",
                 GAP_NETW_Q="1.0", GAP_SM="union", T1_SIX=QNETS, T1_PCTMIN="0", T4_REFINE="lbfgs",
                 GAPNET_ENV=dict(T1_PCTMIN="0"), T2="np3Tbw"),
    "s117": dict(GAP_RULE="mix", GAP_NET=GAPNETS, GAP_NETW="1.0", GAP_NETW_Q="1.0", T4_REFINE="lbfgs",
                 T2="np3Tbw"),
    # s113 / s114: s111 / s112 with the three no-prior members retrained to read the
    # released row AT the origin T (QT_T=1: bothnpT, q1npT, r4npT)
    "s113": dict(GAP_RULE="analog", GAP_MAXD="0.003", GAP_NET=GAPQ, GAP_NETW="1.0",
                 GAP_NETW_Q="1.0", GAP_SM="union", T1_SIX=QNETS, T1_PCTMIN="0", T4_REFINE="lbfgs",
                 GAPNET_ENV=dict(T1_PCTMIN="0"), T2="np3Tw"),
    "s114": dict(GAP_RULE="mix", GAP_NET=GAPNETS, GAP_NETW="1.0", GAP_NETW_Q="1.0", T4_REFINE="lbfgs",
                 T2="np3Tw"),
    # s111 / s112: ongoing weights 0.25 ensemble / 0.25 anubisyy / 0.5 ours, ours = mean of
    # three no-prior members (bothnp LightGBM, q1np 1-D net, r4np 2-D net); s112 = hedge
    "s111": dict(GAP_RULE="analog", GAP_MAXD="0.003", GAP_NET=GAPQ, GAP_NETW="1.0",
                 GAP_NETW_Q="1.0", GAP_SM="union", T1_SIX=QNETS, T1_PCTMIN="0", T4_REFINE="lbfgs",
                 GAPNET_ENV=dict(T1_PCTMIN="0"), T2="np3w"),
    "s112": dict(GAP_RULE="mix", GAP_NET=GAPNETS, GAP_NETW="1.0", GAP_NETW_Q="1.0", T4_REFINE="lbfgs",
                 T2="np3w"),
    # s108: s105 with "ours" = mean of two no-prior members (bothnp LightGBM + q1np net)
    "s108": dict(GAP_RULE="analog", GAP_MAXD="0.003", GAP_NET=GAPQ, GAP_NETW="1.0",
                 GAP_NETW_Q="1.0", GAP_SM="union", T1_SIX=QNETS, T1_PCTMIN="0", T4_REFINE="lbfgs",
                 GAPNET_ENV=dict(T1_PCTMIN="0"), T2="public3np2"),
    # s105 / s107: s101 / s103 with our own no-prior LightGBM as a third ongoing member
    "s105": dict(GAP_RULE="analog", GAP_MAXD="0.003", GAP_NET=GAPQ, GAP_NETW="1.0",
                 GAP_NETW_Q="1.0", GAP_SM="union", T1_SIX=QNETS, T1_PCTMIN="0", T4_REFINE="lbfgs",
                 GAPNET_ENV=dict(T1_PCTMIN="0"), T2="public3"),
    "s107": dict(GAP_RULE="mix", GAP_NET=GAPNETS, GAP_NETW="1.0", GAP_NETW_Q="1.0", T4_REFINE="lbfgs",
                 T2="public3"),
    # s101 / s103: s95 / s79c Task 1 + Task 4, with Task 2 from two public Kaggle
    # pipelines retrained locally (src/t2_public.py, train/t2_public.sh)
    "s101": dict(GAP_RULE="analog", GAP_MAXD="0.003", GAP_NET=GAPQ, GAP_NETW="1.0",
                 GAP_NETW_Q="1.0", GAP_SM="union", T1_SIX=QNETS, T1_PCTMIN="0", T4_REFINE="lbfgs",
                 GAPNET_ENV=dict(T1_PCTMIN="0"), T2="public"),
    "s103": dict(GAP_RULE="mix", GAP_NET=GAPNETS, GAP_NETW="1.0", GAP_NETW_Q="1.0", T4_REFINE="lbfgs",
                 T2="public"),
    # s95: s90 with the analog gate at 0.003 and the 12 gap nets fine-tuned with the
    # ineligible readings as inputs (fed with their own pct) (0.87097)
    "s95": dict(GAP_RULE="analog", GAP_MAXD="0.003", GAP_NET=GAPQ, GAP_NETW="1.0",
                GAP_NETW_Q="1.0", GAP_SM="union", T1_SIX=QNETS, T1_PCTMIN="0", T4_REFINE="lbfgs",
                GAPNET_ENV=dict(T1_PCTMIN="0")),
    # s90 / s79c: s89 / s79 with Task 4 solved to the exact optimum (FISTA, then
    # L-BFGS-B; FISTA alone stopped up to ~3.5% L1 short on the large corridors)
    "s90": dict(GAP_RULE="analog", GAP_MAXD="0.002",
                GAP_NET=",".join(n + "_i" for n in (GAPNETS + ",xs1").split(",")), GAP_NETW="1.0",
                GAP_NETW_Q="1.0", GAP_SM="union", T1_SIX=QNETS, T1_PCTMIN="0", T4_REFINE="lbfgs",
                GAPNET_ENV=dict(GAPNET_TAG="_i", T1_PCTMIN="0", T1_PCTCLIP="75")),
    "s79c": dict(GAP_RULE="mix", GAP_NET=GAPNETS, GAP_NETW="1.0", GAP_NETW_Q="1.0", T4_REFINE="lbfgs"),
    # s89: s88 with the 12 gap nets also fed the ineligible readings (inference
    # only, pct shown as 75); their caches are gapnet_<net>_i_* (0.87061)
    "s89": dict(GAP_RULE="analog", GAP_MAXD="0.002",
                GAP_NET=",".join(n + "_i" for n in (GAPNETS + ",xs1").split(",")), GAP_NETW="1.0",
                GAP_NETW_Q="1.0", GAP_SM="union", T1_SIX=QNETS, T1_PCTMIN="0",
                GAPNET_ENV=dict(GAPNET_TAG="_i", T1_PCTMIN="0", T1_PCTCLIP="75")),
    # s88: s86 with the six Task 1 members fine-tuned to read finite-but-ineligible
    # readings (pct_observed < 75) as inputs, fed with their own pct (0.87057)
    "s88": dict(GAP_RULE="analog", GAP_MAXD="0.002", GAP_NET=GAPNETS + ",xs1", GAP_NETW="1.0",
                GAP_NETW_Q="1.0", GAP_SM="union", T1_SIX=QNETS, T1_PCTMIN="0"),
    # s86: s83 with a same-month analog library for the gap fill - per window, the
    # closer of the train library and the scored month's own other days (0.86995)
    "s86": dict(GAP_RULE="analog", GAP_MAXD="0.002", GAP_NET=GAPNETS + ",xs1", GAP_NETW="1.0",
                GAP_NETW_Q="1.0", GAP_SM="union"),
    # s83: s78 with the from-scratch net xs1 as a twelfth gap net (0.86983)
    "s83": dict(GAP_RULE="analog", GAP_MAXD="0.002", GAP_NET=GAPNETS12, GAP_NETW="1.0", GAP_NETW_Q="1.0"),
    # s78: analogs where the match is good (dist <= 0.002), else the mean of the
    # 11 gap-aware nets (trees retrained on the current release)
    "s78": dict(GAP_RULE="analog", GAP_MAXD="0.002", GAP_NET=GAPNETS, GAP_NETW="1.0", GAP_NETW_Q="1.0"),
    # s79: the 11 gap nets on every gap row, no analogs
    "s79": dict(GAP_RULE="mix", GAP_NET=GAPNETS, GAP_NETW="1.0", GAP_NETW_Q="1.0"),
}
FINALS["s131"] = dict(FINALS["s124"])
FINALS["s132"] = dict(FINALS["s125"])
TREE_WEIGHTS = ("speed:own=.51,pooled=.12,family=.21,alt=.16;"
                "flow:own=.47,pooled=.25,family=.17,alt=.12;"
                "density:own=.44,pooled=.16,family=.17,alt=.23")
PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S",
          "D12_I5_N", "D12_I5_S", "D12_I405_N", "D12_I405_S"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="unzipped competition release")
    ap.add_argument("--work", required=True, help="scratch dir (caches, models, outputs)")
    ap.add_argument("--final", choices=sorted(FINALS), required=True)
    ap.add_argument("--reference", help="the submitted zip, to verify against")
    ap.add_argument("--force", action="append", default=[])
    ap.add_argument("--models-dir", default=str(HERE / "models"), help="archived weights to use")
    a = ap.parse_args()
    data, work = Path(a.data).resolve(), Path(a.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    st = work / "stage"; st.mkdir(exist_ok=True)
    env = dict(os.environ, TFB_DATA=str(data), TFB_WORK=str(work),
               T1_TOD="const", T1_TILE_L="64", PYTHONIOENCODING="utf-8")
    log = open(work / f"reproduce_{a.final}.log", "a", encoding="utf-8")

    def run(name, cmd, done, extra=None):
        if done.exists() and name not in a.force:
            print(f"[skip] {name} ({done.name} exists)", flush=True)
            return
        t0 = time.time()
        print(f"[run ] {name}: {' '.join(map(str, cmd))}", flush=True)
        log.write(f"\n### {name}: {cmd}\n"); log.flush()
        subprocess.run([sys.executable, "-u", *map(str, cmd)], cwd=SRC, check=True,
                       env={**env, **(extra or {})}, stdout=log, stderr=subprocess.STDOUT)
        print(f"[done] {name} in {time.time() - t0:.0f}s", flush=True)

    cache = work / "cache"
    run("caches-train", ["build_cache.py"], cache / "D12_I405_S_train.npz")
    run("caches-masked", ["build_all_caches.py"], cache / "D12_I405_S_train_masked.npz")
    run("nn_static", ["repro_steps.py", "nn_static"], work / "nn" / "D12_I405_S_static.npz")

    out = work / "out"; out.mkdir(exist_ok=True)
    n_new = 0
    mdir = Path(a.models_dir)
    for f in mdir.iterdir():
        if f.is_dir():          # models/t2 is read in place by src/t2_public.py
            continue
        dst = out / f.name
        if not dst.exists() or dst.stat().st_size != f.stat().st_size:
            shutil.copy2(f, dst); n_new += 1
    print(f"[done] models ({n_new} copied, {len(list(mdir.iterdir()))} total)", flush=True)

    (st / "t4").mkdir(exist_ok=True); (st / "t2").mkdir(exist_ok=True); (st / "trees").mkdir(exist_ok=True)
    run("task4", ["task4_run.py", st / "t4"], st / "t4" / "odme_submission.csv")
    t4_dir = st / "t4"
    if FINALS[a.final].get("T4_REFINE"):
        t4_dir = st / "t4_refined"; t4_dir.mkdir(exist_ok=True)
        run("task4-refined", ["task4_run.py", t4_dir], t4_dir / "odme_submission.csv",
            extra={"T4_REFINE": FINALS[a.final]["T4_REFINE"]})
    run("task2-probs", ["qpredict2.py", st / "t2" / "raw_queue.csv",
                        "--onset", "both,both6,negall@0.60",
                        "--ongoing", "both,nn:c1,nn:c2,nn:q1@0.50",
                        "--dump", st / "t2" / "q2_probs.pkl"], st / "t2" / "q2_probs.pkl")
    run("task2", ["-c", "import sys; from qvariant import write; "
                  "write(sys.argv[1], sys.argv[2], 0.60, 0.50, onset_kmin=1)",
                  st / "t2" / "q2_probs.pkl", st / "t2" / "queue_submission.csv"],
        st / "t2" / "queue_submission.csv")
    run("trees", ["t1_ensemble_predict.py", st / "trees", TREE_WEIGHTS], st / "trees" / "state_submission.csv")
    run("nets", ["repro_steps.py", "nets", st / "trees" / "state_submission.csv",
                 st / "state_s55.csv"], st / "state_s55.csv")
    nets_csv = st / "state_s55.csv"
    nets_env = {k: v for k, v in FINALS[a.final].items() if k in ("T1_SIX", "T1_PCTMIN")}
    if nets_env:     # a final with its own Task 1 members / input rule
        nets_csv = st / f"state_{a.final}_nets.csv"
        run(f"nets-{a.final}", ["repro_steps.py", "nets", st / "trees" / "state_submission.csv",
                                nets_csv], nets_csv, extra=nets_env)
    genv = FINALS[a.final].get("GAPNET_ENV", {})
    tag = genv.get("GAPNET_TAG", "")
    for n in FINALS[a.final]["GAP_NET"].split(","):
        base = n[:-len(tag)] if tag and n.endswith(tag) else n
        run(f"gapnet-{n}", ["t1_gap_netcache.py", base], out / f"gapnet_{n}_D12_I5_S_private.npz",
            extra=genv or None)

    fin = work / f"final_{a.final}"; fin.mkdir(exist_ok=True)
    run("gapfill", ["t1_gapfill.py", nets_csv, fin / "state_submission.csv"],
        fin / "state_submission.csv",
        extra={k: v for k, v in FINALS[a.final].items() if isinstance(v, str)})
    t2_csv = st / "t2" / "queue_submission.csv"
    if FINALS[a.final].get("T2") == "public":
        (st / "t2pub").mkdir(exist_ok=True)
        t2_csv = st / "t2pub" / "queue_submission.csv"
        run("task2-public", ["t2_public.py", t2_csv], t2_csv)
    if FINALS[a.final].get("T2") == "roll3Tbw":
        (st / "t2npT").mkdir(exist_ok=True)
        pk = {}
        for nm, member, extra in (("bothrollT", "bothrollT", {"QT_T": "1", "QROLL": "14"}),
                                  ("q1npT", "nn:q1npT", {"QT_T": "1", "NQ_NOPRIOR": "1"}),
                                  ("q1npT2", "nn:q1npT2", {"QT_T": "1", "NQ_NOPRIOR": "1"}),
                                  ("r4npT", "nn2:r4npT", {"QT_T": "1", "NQ_NOPRIOR": "1", "NN2_PAD": "batch"}),
                                  ("r4npT2", "nn2:r4npT2", {"QT_T": "1", "NQ_NOPRIOR": "1", "NN2_PAD": "batch"})):
            pk[nm] = st / "t2npT" / f"q2_probs_{nm}.pkl"
            run(f"task2-{nm}", ["qpredict2.py", st / "t2npT" / f"raw_{nm}.csv", "--onset", "both,both6,negall@0.60",
                                "--ongoing", f"{member}@0.50", "--dump", pk[nm]], pk[nm], extra=extra)
        t2_csv = st / "t2npT" / "queue_submission_roll3Tbw.csv"
        ours = f"{pk['bothrollT']},{pk['q1npT']}+{pk['q1npT2']},{pk['r4npT']}+{pk['r4npT2']}"
        if FINALS[a.final].get("T2_R4ROLL"):
            pk["r4rollT"] = st / "t2npT" / "q2_probs_r4rollT.pkl"
            run("task2-r4rollT", ["qpredict2.py", st / "t2npT" / "raw_r4rollT.csv", "--onset", "both,both6,negall@0.60",
                                  "--ongoing", "nn2:r4rollT@0.50", "--dump", pk["r4rollT"]], pk["r4rollT"],
                extra={"QT_T": "1", "QROLL": "14", "NN2_PAD": "batch"})
            t2_csv = st / "t2npT" / "queue_submission_roll3Tbw_r4roll.csv"
            ours = f"{pk['bothrollT']},{pk['q1npT']}+{pk['q1npT2']},{pk['r4npT']}+{pk['r4npT2']}+{pk['r4rollT']}"
            if FINALS[a.final]["T2_R4ROLL"] == "2":
                pk["r4rollT2"] = st / "t2npT" / "q2_probs_r4rollT2.pkl"
                run("task2-r4rollT2", ["qpredict2.py", st / "t2npT" / "raw_r4rollT2.csv", "--onset", "both,both6,negall@0.60",
                                       "--ongoing", "nn2:r4rollT2@0.50", "--dump", pk["r4rollT2"]], pk["r4rollT2"],
                    extra={"QT_T": "1", "QROLL": "14", "NN2_PAD": "batch"})
                t2_csv = st / "t2npT" / "queue_submission_roll3Tbw_r4roll2.csv"
                ours += f"+{pk['r4rollT2']}"
        if FINALS[a.final].get("T2_BROLL2"):
            pk["bothrollT2"] = st / "t2npT" / "q2_probs_bothrollT2.pkl"
            run("task2-bothrollT2", ["qpredict2.py", st / "t2npT" / "raw_bothrollT2.csv", "--onset", "both,both6,negall@0.60",
                                     "--ongoing", "bothrollT2@0.50", "--dump", pk["bothrollT2"]], pk["bothrollT2"],
                extra={"QT_T": "1", "QROLL": "14"})
            t2_csv = t2_csv.with_name(t2_csv.stem + "_broll2.csv")
            ours = ours.replace(str(pk["bothrollT"]), f"{pk['bothrollT']}+{pk['bothrollT2']}", 1)
        run("task2-" + t2_csv.stem, ["t2_public.py", t2_csv], t2_csv,
            extra={"T2_W3": "0.25,0.25,0.5", "T2_ONG_THR": "0.45", "T2_OURS_PKL": ours})
    if FINALS[a.final].get("T2") == "np3Tbw":
        (st / "t2npT").mkdir(exist_ok=True)
        pk = {}
        for nm, member, extra in (("bothnpT", "bothnpT", {"QT_T": "1"}),
                                  ("q1npT", "nn:q1npT", {"QT_T": "1", "NQ_NOPRIOR": "1"}),
                                  ("q1npT2", "nn:q1npT2", {"QT_T": "1", "NQ_NOPRIOR": "1"}),
                                  ("r4npT", "nn2:r4npT", {"QT_T": "1", "NQ_NOPRIOR": "1", "NN2_PAD": "batch"}),
                                  ("r4npT2", "nn2:r4npT2", {"QT_T": "1", "NQ_NOPRIOR": "1", "NN2_PAD": "batch"})):
            pk[nm] = st / "t2npT" / f"q2_probs_{nm}.pkl"
            run(f"task2-{nm}", ["qpredict2.py", st / "t2npT" / f"raw_{nm}.csv", "--onset", "both,both6,negall@0.60",
                                "--ongoing", f"{member}@0.50", "--dump", pk[nm]], pk[nm], extra=extra)
        t2_csv = st / "t2npT" / "queue_submission_np3Tbw.csv"
        ours = f"{pk['bothnpT']},{pk['q1npT']}+{pk['q1npT2']},{pk['r4npT']}+{pk['r4npT2']}"
        run("task2-np3Tbw", ["t2_public.py", t2_csv], t2_csv,
            extra={"T2_W3": "0.25,0.25,0.5", "T2_ONG_THR": "0.45", "T2_OURS_PKL": ours})
    if FINALS[a.final].get("T2") == "np3Tw":
        (st / "t2npT").mkdir(exist_ok=True)
        dumps = []
        for nm, member, extra in (("bothnpT", "bothnpT", {"QT_T": "1"}),
                                  ("q1npT", "nn:q1npT", {"QT_T": "1", "NQ_NOPRIOR": "1"}),
                                  ("r4npT", "nn2:r4npT", {"QT_T": "1", "NQ_NOPRIOR": "1", "NN2_PAD": "batch"})):
            pk = st / "t2npT" / f"q2_probs_{nm}.pkl"
            run(f"task2-{nm}", ["qpredict2.py", st / "t2npT" / f"raw_{nm}.csv", "--onset", "both,both6,negall@0.60",
                                "--ongoing", f"{member}@0.50", "--dump", pk], pk, extra=extra)
            dumps.append(str(pk))
        t2_csv = st / "t2npT" / "queue_submission_np3Tw.csv"
        run("task2-np3Tw", ["t2_public.py", t2_csv], t2_csv,
            extra={"T2_W3": "0.25,0.25,0.5", "T2_ONG_THR": "0.45", "T2_OURS_PKL": ",".join(dumps)})
    if FINALS[a.final].get("T2") == "np3w":
        (st / "t2np").mkdir(exist_ok=True)
        dumps = []
        for nm, member, extra in (("bothnp", "bothnp", {}), ("q1np", "nn:q1np", {"NQ_NOPRIOR": "1"}),
                                  ("r4np", "nn2:r4np", {"NQ_NOPRIOR": "1", "NN2_PAD": "batch"})):
            pk = st / "t2np" / f"q2_probs_{nm}.pkl"
            run(f"task2-{nm}", ["qpredict2.py", st / "t2np" / f"raw_{nm}.csv", "--onset", "both,both6,negall@0.60",
                                "--ongoing", f"{member}@0.50", "--dump", pk], pk, extra=extra or None)
            dumps.append(str(pk))
        t2_csv = st / "t2np" / "queue_submission_np3w.csv"
        run("task2-np3w", ["t2_public.py", t2_csv], t2_csv,
            extra={"T2_W3": "0.25,0.25,0.5", "T2_ONG_THR": "0.45", "T2_OURS_PKL": ",".join(dumps)})
    if FINALS[a.final].get("T2") == "public3np2":
        (st / "t2np").mkdir(exist_ok=True)
        np_pkl = st / "t2np" / "q2_probs_bothnp.pkl"
        run("task2-bothnp", ["qpredict2.py", st / "t2np" / "raw_bothnp.csv", "--onset", "both,both6,negall@0.60",
                             "--ongoing", "bothnp@0.50", "--dump", np_pkl], np_pkl)
        q1_pkl = st / "t2np" / "q2_probs_q1np.pkl"
        run("task2-q1np", ["qpredict2.py", st / "t2np" / "raw_q1np.csv", "--onset", "both,both6,negall@0.60",
                           "--ongoing", "nn:q1np@0.50", "--dump", q1_pkl], q1_pkl, extra={"NQ_NOPRIOR": "1"})
        t2_csv = st / "t2np" / "queue_submission_np2.csv"
        run("task2-public3np2", ["t2_public.py", t2_csv], t2_csv,
            extra={"T2_W3": "0.3333,0.3333,0.3334", "T2_ONG_THR": "0.43", "T2_OURS_PKL": f"{np_pkl},{q1_pkl}"})
    if FINALS[a.final].get("T2") == "public3":
        (st / "t2np").mkdir(exist_ok=True)
        np_pkl = st / "t2np" / "q2_probs_bothnp.pkl"
        run("task2-bothnp", ["qpredict2.py", st / "t2np" / "raw_bothnp.csv", "--onset", "both,both6,negall@0.60",
                             "--ongoing", "bothnp@0.50", "--dump", np_pkl], np_pkl)
        t2_csv = st / "t2np" / "queue_submission.csv"
        run("task2-public3", ["t2_public.py", t2_csv], t2_csv,
            extra={"T2_W3": "0.3333,0.3333,0.3334", "T2_ONG_THR": "0.43", "T2_OURS_PKL": str(np_pkl)})
    shutil.copy(t2_csv, fin / "queue_submission.csv")
    shutil.copy(t4_dir / "odme_submission.csv", fin / "odme_submission.csv")
    zp = work / f"submission_{a.final}_reproduced.zip"
    run("package", ["repro_steps.py", "package", fin, zp], zp)
    if a.reference:
        run("verify", ["repro_steps.py", "verify", zp, Path(a.reference).resolve()],
            work / "__always__", )
        print((work / f"reproduce_{a.final}.log").read_text(encoding="utf-8").split("### verify")[-1])
    print(f"\nfinal file: {zp}")


if __name__ == "__main__":
    main()
