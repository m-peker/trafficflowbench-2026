# TrafficFlowBench 2026 - reproduction package

Reproduces our two selected final submissions (s146 + s148; s131, s132, s124, s125, s122, s123, s120, s121, s118, s119, s116, s117, s113, s114, s111, s112, s108, s107, s105, s101, s103, s95, s90, s89, s88, s86, s83, s79c, s79 and s78 kept for reference) from the released data (the
2026-09-11 rebuilt release) and the archived model weights:

| final | public score | submitted file md5 |
|---|---|---|
| s105 `submission_s105_t2-ong3-ens-pub-bothnp.zip` | 0.88157 | d3a4c8a86f9bc5c1841cc397836f2b7f |
| s146 `submission_s146_t2-original-onset.zip` | 0.88588 | e7411373287d09da68c148106d8e8ec3 |
| s148 `submission_s148_hedge-original-onset.zip` | 0.88481 | 36340d98a9e233788a6ad24977990549 |
| s131 `submission_s131_t2-fix-firstday.zip` | 0.88494 | c42ffb639631ab2a8b5f0d7d8a1cc5d5 |
| s132 `submission_s132_hedge-fix-firstday.zip` | 0.88387 | 3e62923df0a5ba6a5439ebdff27df21e |
| s124 `submission_s124_t2-roll-lgbseeds.zip` | 0.88640 | 16e8bf5f6cb21064836ce586ebbb9651 |
| s125 `submission_s125_hedge-noanalog-s124T2.zip` | 0.88532 | 4e697a16027c470a1da381431661bc6d |
| s122 `submission_s122_t2-roll-r4seeds.zip` | 0.88617 | e749a5555eecc64ca46bab529320ea56 |
| s123 `submission_s123_hedge-noanalog-s122T2.zip` | 0.88509 | 124d9c5c0273735653413f5e2190d475 |
| s120 `submission_s120_t2-rollprior-r4roll.zip` | 0.88587 | 423dfd8d78717e9347dba97ee3971636 |
| s121 `submission_s121_hedge-noanalog-s120T2.zip` | 0.88480 | 8b7dce15d6198101981f1264e8a2f055 |
| s118 `submission_s118_t2-rollprior.zip` | 0.88510 | 945ddac9291f5ac77a31c36cc7d35bb9 |
| s119 `submission_s119_hedge-noanalog-s118T2.zip` | 0.88402 | 33c6bc117ba79d94787f3ed80c518018 |
| s116 `submission_s116_t2-np3T-bag.zip` | 0.88411 | 7966cbdd68eb72fd7a5e29950e4187a8 |
| s117 `submission_s117_hedge-noanalog-s116T2.zip` | 0.88303 | 33d8a395a148a29cf70f28998ae8b992 |
| s113 `submission_s113_t2-ong-np3T-w50.zip` | 0.88370 | a3eb5b51ab88d2b086380ad2fccf9265 |
| s114 `submission_s114_hedge-noanalog-s113T2.zip` | 0.88262 | f0283c1da0bd213720e0c2beec378203 |
| s111 `submission_s111_t2-ong-np3-w50.zip` | 0.88233 | a06c58eeece84b894413abd2be8310f8 |
| s112 `submission_s112_hedge-noanalog-s111T2.zip` | 0.88125 | ebcfd49919e154871e5964adabd1be09 |
| s108 `submission_s108_t2-ong3-np2.zip` | 0.88175 | 966720c15c613bc03b45f9ea2ed3e2ef |
| s107 `submission_s107_hedge-noanalog-s105T2.zip` | 0.88050 | 9848d7db3f41646514db273d19be432f |
| s101 `submission_s101_t2-yeash-ongoing-avg.zip` | 0.88054 | b2b6520bdcd974a70cea9e22b562aefa |
| s103 `submission_s103_hedge-noanalog-newT2.zip` | 0.87946 | 7f745e584a05ff03ddd1cf160a03a821 |
| s95 `submission_s95_t1-gapnets-ft-gate003.zip` | 0.87097 | b1d238cd91825098f8792b6c4e53b38d |
| s90 `submission_s90_t4-converged.zip` | 0.87086 | 99bea8477c5d7fba8f155231407dc1e0 |
| s79c `submission_s79c_t4-converged-noanalog.zip` | 0.86989 | 5aa07caea905fe6d86fe38209abd3524 |
| s89 `submission_s89_t1-gapnets-inelig.zip` | 0.87061 | a6ec941d77fe0180971a8ea7b2827dc1 |
| s88 `submission_s88_t1-inelig-retrained.zip` | 0.87057 | 278c34ee147b432ddbea542b3095d0c3 |
| s86 `submission_s86_t1-gap-samemonth-union.zip` | 0.86995 | bd6798dab4dc77766525e7d8c913e753 |
| s83 `submission_s83_t1-gap-ens12-xs1.zip` | 0.86983 | b05e215cb2a22dda0fdee23fe013b93a |
| s78 `submission_s78_t1-newtrees-gap-ens11.zip` | 0.86980 | a32b38b1d55f36193fd8b38700425590 |
| s79 `submission_s79_t1-newtrees-gap-ens11-noanalog.zip` | 0.86965 | 4ac6a307fc212f33b475a4c6c2053b1e |

The Task 1 tree members were retrained on the current release (2026-09-27), so
no part of the chain depends on data from before the 2026-09-11 rebuild.

## Model weights

The archived weight files (71 + models/t2 + bothnp + q1np) (~2.4 GB, ~1.1 GB zipped) are distributed
separately as `TrafficFlowBench_models.zip`: https://github.com/m-peker/trafficflowbench-2026/releases/download/v1.0/TrafficFlowBench_models.zip. Unzip it so
the files sit directly in `models/` (for example `models/t1_nn_f1.pt`).
`models/MD5SUMS.txt` lists the md5 of every file.

## Run

```
pip install -r requirements.txt
python reproduce.py --data <unzipped release> --work <scratch dir> --final s131 [--reference submission_s131_t2-fix-firstday.zip]
python reproduce.py --data <unzipped release> --work <same scratch dir> --final s132 [--reference submission_s132_hedge-fix-firstday.zip]
```

`--data` is the folder holding `sample_submission.csv`, `submission_key.csv`,
`config/`, `corridors/`, `task1/`, `task2/`, `task4/`. The output is
`<work>/submission_<final>_reproduced.zip`, ready to upload. Stages that already
have their output are skipped, so a second final or an interrupted run resumes;
`--force <stage>` reruns one. A per-stage log is written to
`<work>/reproduce_<final>.log`.

**Verified (2026-09-27/28):** run from an empty work dir on the release, the chain
reproduces the submitted files value for value - all 6,980,503 rows identical in speed_kmh, flow_vph,
queue_pred and path_flow (max |diff| 0). Only the zip's md5 differs, because the
zip stores a file timestamp.

## Hardware and runtime

Measured on one laptop: Intel i7-12700H (20 threads), 32 GB RAM, NVIDIA RTX 3070
Ti Laptop (8 GB), Windows 11. Full chain for the first final ~3.2 h; the second
final reuses everything but the last two stages (~8 min).

| stage | what | time |
|---|---|---|
| caches | dense time x link grids from the parquet release (`build_cache.py`, `build_all_caches.py`) | 50 min |
| task4 | ODME: NNLS (lambda 0.05) anchored on the released weak prior (`task4_run.py`) | 1 min |
| task2 | queue: 3 LightGBM + 3 conv nets, onset 0.60 / ongoing 0.50, onset steps 1-5 empty, never-empty onset (`qpredict2.py`, `qvariant.py`) | 6 min |
| trees | Task 1 tree ensemble, 4 LightGBM members (`t1_ensemble_predict.py`) | 107 min |
| nets | Task 1: mean of six conv nets, blended 0.85 / 0.15 with the trees (`repro_steps.py nets`) | 9 min (GPU) |
| gapnets | eleven gap-aware conv nets on the validation/private grids (`t1_gap_netcache.py`) | 17 min (GPU) |
| gapfill | Task 1 rows inside the Task 2 gaps, final-specific (`t1_gapfill.py`) | 5 min |
| package | merge onto `sample_submission.csv`, validate, round, zip | 3 min |

A GPU is needed for reasonable speed (the nets fall back to CPU, much slower).
Disk: ~2.4 GB models, ~2.4 GB caches.

## What the two finals are

Both finals share Task 2 and Task 4. They differ in Task 1: s146 uses the six networks fine-tuned to read
ineligible values and fills the Task 2 blackout rows with gated analogs (match distance <= 0.003, library =
train or the scored month's own other days) and otherwise the mean of 12 gap-aware networks; the hedge s148 uses
the Task 1 of s79c (the original six networks, and the mean of 11 gap-aware networks on every blackout row, no
analogs). Older hedges (s103 ... s132) follow the same pattern.
- **s146** (final) / **s148** (final, hedge): s131 / s132 below with the public notebook's own onset labels on
  the 4 queue cells where our retrained replica of the public onset model differs (2 validation, 2 private).
  `reproduce.py --final s131` / `--final s132` reproduce them on every row except those 4 cells (the same 4 cells
  reported as "all but 4 queue cells" throughout this file); about 0.0009 of public score, far inside the
  1%-of-private tolerance.
- **s131** / **s132** (hedge): s124 / s125 produced by the current code. The
  organizers re-run Task 2 with every row stamped after each window's origin removed; the earlier
  code fell back to the window's own day for the rolling priors when the split had no earlier day
  (the first validation day, 16 windows - never a private window). It now uses the last 14 days of
  the train split's masked layer there. Board: s131 0.88494 vs s124 0.88640 (the same-day data had
  helped those validation windows); private predictions unchanged.
- **s124**: s122 below with a second LightGBM seed (bothrollT2, QSEED=2) -
  the LightGBM member is mean(bothrollT, bothrollT2). Reproduces the submitted file on
  all but 4 queue cells.
- **s125** (hedge): s79c's Task 1 + s124's Task 2.
- **s122**: s120 below with a second seed of the rolling-prior 2-D net
  (r4rollT2, seed 29) in the 2-D member (mean of r4npT, r4npT2, r4rollT, r4rollT2).
  Reproduces the submitted file on all but 4 queue cells.
- **s123** (hedge): s79c's Task 1 + s122's Task 2.
- **s120**: s118 below with r4rollT - the 2-D queue net trained on the same
  rolling priors (QROLL=14, QT_T=1) - added to the 2-D member (mean of r4npT, r4npT2,
  r4rollT). Reproduces the submitted file on all but 4 queue cells.
- **s121** (hedge): s79c's Task 1 + s120's Task 2.
- **s118**: s116 below with the LightGBM member using rolling priors -
  queue rates from the 14 days before the window's day, computed on the window's own
  split's Task 1 masked layer (validation precedes private; everything used is stamped
  before T, as the online rule allows) instead of no priors (bothrollT, QROLL=14).
  Reproduces the submitted file on all but 4 queue cells.
- **s119** (final, hedge): s79c's Task 1 + s118's Task 2.
- **s116**: s113 below with the two no-prior nets seed-bagged (q1npT +
  q1npT2, r4npT + r4npT2). Reproduces the submitted file on all but 4 queue cells.
- **s117** (final, hedge): s79c's Task 1 + s116's Task 2.
- **s113**: s111 below with the three no-prior members retrained to also
  read the released row AT the forecast origin T (allowed by the organizers' ruling:
  "timestamps at or before T, including the observations at T"); the row comes from
  the split's Task 1 masked layer and is thinned 35% in training (QT_T=1).
- **s114** (final, hedge): s79c's Task 1 + s113's Task 2.
- **s111**: s108 below with a third no-prior member, the r4np 2-D queue net
  (nn_queue2 r4 shape, NQ_NOPRIOR=1), and ongoing weights 0.25 ensemble / 0.25
  anubisyy / 0.5 mean(bothnp, q1np, r4np), threshold 0.45. Reproduces the
  submitted file on all but 4 queue cells.
- **s112** (hedge): s79c's Task 1 + s111's Task 2.
- **s108**: s105 below with "ours" = the mean of two no-prior members, the
  bothnp LightGBM and the q1np queue net (nn_queue q1 shape, NQ_NOPRIOR=1).
  Reproduces the submitted file on all but 4 queue cells.
- **s107** (final, hedge): s79c's Task 1 (no analogs) + s105's Task 2.
- **s105**: s101 below with a third ongoing member - our own LightGBM trained
  WITHOUT the train-derived queue-rate priors (models/queue_model_bothnp.pkl):
  ongoing p = mean(7-member ensemble, anubisyy LGB, ours) >= 0.43. Our prior-based
  models had been strong on train-month checks but weak on the separately generated
  scored months; the no-prior member fixed that (+0.0051 in a paired test).
  Reproduces the submitted file on all but 4 queue cells.
- **s107** (hedge): s103 below with s105's Task 2.
- **s101**: Task 1 and Task 4 = s95 below. Task 2 combines two public Kaggle
  pipelines, retrained locally from their published code (src_t2/, train/t2_public.sh,
  inference src/t2_public.py): onset = anubisyy's LightGBM + CatBoost; ongoing = the
  mean of yeashusemwal's 7-member LightGBM ensemble and anubisyy's ongoing LightGBM.
  The submitted s101 took its Task 2 labels from that notebook's own output; our
  retrained models reproduce them on all but 70 of 174,000 queue cells (Task 1 and
  Task 4 reproduce exactly) - far inside the 1%-of-private tolerance.
- **s103** (final, hedge): s79c's Task 1 (no analogs) + s101's Task 2 + converged Task 4.
- **s95**: s90 below with the analog gate at 0.003 instead of 0.002, and
  the 12 gap nets fine-tuned to read the ineligible readings too
  (train/t1_gapnets_inelig.sh, caches gapnet_<net>q_*).
- **s90**: s89 below, with Task 4 solved to the exact optimum of the
  documented objective (FISTA, then L-BFGS-B; T4_REFINE=lbfgs). FISTA alone had
  stopped up to ~3.5% L1 short of it on the large corridors.
- **s79c** (final, hedge): s79 with the same converged Task 4.
- **s89**: s88 below, with the 12 gap-fallback nets also fed the
  ineligible readings at inference (pct shown as 75; caches gapnet_<net>_i_*).
- **s88**: s86 below, with the six Task 1 members replaced by versions
  fine-tuned to also read the finite readings the release ships with
  pct_observed < 75 (~7% of cells; never targets, nearly as informative), fed at
  inference with their own pct (train/t1_inelig_ft.sh, T1_PCTMIN=0).
- **s86**: s83 below, but the analog library is chosen per gap window:
  the train truth or the scored month's OWN other days (its masked grid, starts
  more than one day from the window and clear of every gap span), whichever
  matches the gap's edges more closely (GAP_SM=union); the 0.002 gate and the
  12-net fallback are unchanged. The scored months come from a separate
  generation, so their own days match better (median distance 0.00224 vs 0.00326).
- **s83**: s78 below with a twelfth gap net, xs1, trained from scratch
  (train/t1_xs1.sh); only gap rows differ.
- **s78**: where the corridor state just before and after the gap matches a
  train moment closely (distance <= 0.002), the average of the 10 closest train
  trajectories; elsewhere the mean of eleven gap-aware nets.
- **s79**: the mean of the eleven gap-aware nets on every gap row.

Rules compliance: Task 2 forecasts read only `window_history.parquet` (at or
before the origin) and the train split. The gap fill is Task 1, which is offline,
and uses released observations on both sides of each gap - explicitly allowed.

## Contents

- `reproduce.py` - the end-to-end driver.
- `src/` - the code (`src/repro_steps.py` holds the driver's helper stages; the
  rest is our experiment code base as it stood when the finals were built; paths
  come from `TFB_DATA` / `TFB_WORK`).
- `models/` - 52 archived weight files; md5s match the build manifests.
- `TRAINING.md` - how every model was trained: commands, environment, seeds,
  held-out days, runtimes.
- `train/t1_gapnets*.sh` - the gap-aware finetunes (see TRAINING.md).
- `train/train_trees.sh` - trains the four Task 1 tree members on the current
  release (this is how the archived tree weights were produced, 2026-09-27).

## Declarations

- External data: none. Pretrained models: none. Paid or rate-limited APIs: none.
- Dependencies: numpy, pandas, pyarrow, scipy (BSD), LightGBM (MIT), PyTorch (BSD).
- Determinism: inference from the archived weights is exact (verified above).
  Retraining is seeded but not bit-exact: GPU training is not deterministic and
  LightGBM output depends on the thread count.
