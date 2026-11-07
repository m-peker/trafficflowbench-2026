# Training record: models behind the final submissions s73 and s74

TrafficFlowBench, IEEE Big Data Cup 2026. This file lists every trained model that the two
final submissions read. For each one it gives the exact command, the environment, the
seed, the held-out days, the runtime and the source of the record.

| final | public LB | submission md5 |
|---|---|---|
| `s73_t1-gap-ens7` | 0.86973 | `968ee8a4c196c35b0add81d4489c8196` |
| `s74_t1-gap-ens7-noanalog` | 0.86959 | see `runs/s74_t1-gap-ens7-noanalog/MANIFEST.json` |

s73 and s74 read the same models. They differ only in one inference setting of the Task 1
gap fill (`GAP_RULE=analog GAP_MAXD=0.002` for s73, `GAP_RULE=mix` for s74). No model was
trained for either submission specifically. All model md5s below are identical in both
MANIFESTs and match the files on disk in `exp/out/` (checked 2026-09-27).

Record-keeping conventions:
- **Confirmed** means the command appears verbatim in a shell script in `exp/` or in the
  operator's command log, and the log file it wrote agrees with it: step count, runtime
  and file time.
- **Reconstructed** means the command is inferred from code and logs. The reason is given
  each time.
- "Command log" is the session log of the development machine on which this work ran (not included).
  Timestamps are UTC. Local machine time, used in the `out/*.log` files, is UTC+3.

---

## 0. How the final files are built from the models

```
Task 4  task4_run.py (closed form, no model)                      -> s12 odme file (md5 5cae43cf...)
Task 2  qpredict2.py with LightGBM both/both6/negall and nets c1/c2/q1 -> out/q2_probs_s53.pkl
        -> runs/s53_q-ongoing-q1/make.py (qvariant.write, thresholds 0.60 / 0.50, onset_kmin=1)
Task 1  (a) trees: t1_ensemble_predict.py with own/pooled/family/alt  -> s12 state file (e40e71fc...)
            carried unchanged through s13 and s15 (runs/s15_onset-step6/out/state_submission.csv)
        (b) runs/make_t1_tod.py: 0.85 * mean(f1,h1,i1,j1,m1,n1) + 0.15 * trees -> runs/_t1_tod_state6.csv
            (= s55 state file)
        (c) t1_gap_netcache.py for gf1,gh1,gi1,gm1,gn1,ge1,gr1 -> out/gapnet_<net>_<panel>_<split>.npz
        (d) runs/s73*/make.py or runs/s74*/make.py: t1_gapfill.py rewrites the rows inside the
            Task 2 gaps -> final state file
```

The build scripts for these steps are `runs/s12_rebuild/make.py` with `exp/s12.sh`,
`runs/s53_q-ongoing-q1/make.py`, `runs/make_t1_tod.py`, `runs/s55_t1-tod-fix/make.py`,
`runs/s73_t1-gap-ens7/make.py` and `runs/s74_t1-gap-ens7-noanalog/make.py`. Each run folder
also has a `MANIFEST.json` with the md5 of every model and output.

---

## 1. Environment

**Hardware.** Windows 11 Pro (10.0.26200) laptop:
- CPU: Intel Core i7-12700H, 14 cores / 20 threads
- RAM: 32 GB
- GPU: NVIDIA GeForce RTX 3070 Ti Laptop, 8 GB, driver 551.61

All GPU training ran on this one card, and all CPU training on this CPU.

**Shell and working directory.** Commands were run from Git Bash (POSIX `sh`). Every
command below uses `D:\BigData Competition\exp\` as its working directory. `common.py`
hard-codes two paths:
- `ROOT = d:/BigData Competition/2026-ieee-big-data-traffic-flow-bench`: the data release.
- `WORK = exp/`: caches go in `exp/cache`, net memmaps in `exp/nn`, models and logs in
  `exp/out`.

**Python packages.** These were installed when this record was written, read with
`python -c`. The command log shows no `pip`/`conda` install during the competition
(2026-09-07 to 09-27), so they are believed to be the training-time versions. That belief
is inferred and was not recorded at the time.

| package | version |
|---|---|
| python | 3.11.5 (Anaconda, `C:\Users\musa.peker\AppData\Local\anaconda3`) |
| numpy | 1.26.4 |
| pandas | 2.0.3 |
| lightgbm | 4.6.0 |
| torch | 2.6.0+cu124 (CUDA 12.4, cuDNN 9.1.0 / 90100) |
| pyarrow | 11.0.0 |
| scipy | 1.11.1 |
| threadpoolctl | 2.2.0 |
| psutil | 5.9.0 |

`exp/requirements.txt` pins lightgbm, numpy, pandas, psutil, pyarrow and scipy to the same
versions.

**Thread settings.** They matter for LightGBM reproducibility.
- `exp/nice.py` was added on 2026-09-10 06:47 UTC. It is imported before numpy and
  LightGBM. It sets `OMP/OPENBLAS/MKL_NUM_THREADS` and the LightGBM `num_threads` from
  `TFB_THREADS` (default 8), and lowers process priority.
- All four Task 1 tree members, and the Task 2 models `both` and `negall`, were trained
  **before** `nice.py` existed. They used a hard-coded `num_threads=12`. To match them with
  today's code, set `TFB_THREADS=12`.
- `queue_model_both6.pkl` was trained after the change, with `TFB_THREADS` unset, so 8
  threads.
- LightGBM histogram building depends on the thread count, so a different count gives
  slightly different trees.

**Randomness.**
- LightGBM: no `seed` parameter is ever passed, so LightGBM's built-in default seeds apply.
  Row sampling and window sampling use fixed numpy seeds, listed per model below.
- PyTorch: `torch.manual_seed(seed)` and `np.random.default_rng(seed)` are set from
  `--seed`. cuDNN determinism is **not** enabled, so GPU runs are not bit-reproducible. The
  archived checkpoints (md5s below) are what reproduce the submitted numbers exactly.

**Train period.** 273 days, 2030-06-01 to 2031-02-28, in 5-minute steps (78,624 steps).
Validation is 2031-03 and private is 2031-04. Held-out windows used below:

| held out | from |
|---|---|
| last 45 days | 2031-01-15 |
| last 7 days | 2031-02-22 |
| last 2 days | 2031-02-27 |
| last 60 days | 2030-12-31 |
| last 80 days | 2030-12-11 |

---

## 2. Data caches (prerequisite for everything)

```
python build_cache.py train          # exp/cache/<panel>_train.npz  (unmasked train grids)
python build_all_caches.py           # exp/cache/<panel>_<split>_{masked,ramp}.npz, all splits
```

Sources: README text in the command log (2026-09-07); `exp/build_all_caches.py`.

The caches were first built on 2026-09-07. On 2026-09-12 the organizers rebuilt the
release. The old folder was renamed `2026-old-0912`. The ramp and masked caches were then
rebuilt with `nohup python build_all_caches.py > out/rebuild2.log` (command log,
2026-09-12 09:46 UTC). `exp/cache/<panel>_train.npz` dates from 2026-09-07 and is still
valid, because `train/mainline_states` is byte-identical in both releases.

**Important for Task 1 trees (Section 3).** `train/ramp_states` is **not** identical
between the two releases. Parquet md5s differ; this was checked on D12_I405_N 2030-06. The
trees were trained on 2026-09-08 to 09-10 from ramp caches built from the pre-rebuild
release, and they read ramp-derived features. Exact re-training therefore needs the train
split of the pre-2026-09-12 release. A local copy is at
`D:\BigData Competition\2026-old-0912`, and the organizers hold the original. With the
current release, retraining will give slightly different trees. Every other model (all
nets, all Task 2 models) either reads no ramp data or was trained after the rebuild.

---

## 3. Task 1: LightGBM tree members (own, pooled, family, alt)

> **Update 2026-09-28 - the archived tree weights in `models/` are a retrain.**
> The members first shipped (s12 .. s74) had been fitted on the train ramp files
> of the pre-rebuild release. For the finals s76 / s77 all four were retrained on
> the CURRENT (2026-09-11) release with the scripts, samples and seeds below,
> using `train/train_trees.sh` (`TFB_THREADS=8`, work dir with caches built by
> `reproduce.py` from the release). Holdout quality is unchanged (per-panel S
> within +-0.002) and the public scores match s73 / s74 to 1e-5. Logs:
> `train_own.log`, `train_pooled.log`, `train_family.log`, `train_alt.log` in that
> work dir. Nothing in the finals' chain reads pre-rebuild data.

**Used by.** `t1_ensemble_predict.py` loads the members through `ens_search.MEMBERS`
(`exp/ens_search.py:83-89`):

| member | files |
|---|---|
| own | `out/t1_model_<panel>.pkl` |
| pooled | `out/t1_pooled.pkl` |
| family | `out/t1_family_<family>.pkl` |
| alt | `out/t1_alt_<panel>.pkl` |

Each file holds three boosters: speed, flow and density.

**Inference.** The shipped tree file is s12's. It was built by `exp/s12.sh:25-30` with
per-target weights:

```
python t1_ensemble_predict.py "D:/BigData Competition/runs/s12_rebuild/out" \
  "speed:own=.51,pooled=.12,family=.21,alt=.16;flow:own=.47,pooled=.25,family=.17,alt=.12;density:own=.44,pooled=.16,family=.17,alt=.23"
```

That output is `runs/s12_rebuild/out/weights.txt` and state md5 `e40e71fc...`. It is
carried unchanged into `runs/s15_onset-step6/out/state_submission.csv`, which
`runs/make_t1_tod.py` reads as `TREES` at weight 0.15. The weights are the "global
per-target least squares" row of `ens_search.py search` (`out/ens_result.log`).

The following applies to all four members:
- Features come from `t1build.build(panel, "train", truth="train", sample=N, seed=S)`.
- Targets are speed, flow per lane and density.
- Early stopping (patience 60) runs on the **last 45 train days**, 2031-01-15 to
  2031-02-28. Rows from those days are used only to stop training, never for fitting.
- `num_threads=12` (hard-coded at the time; see Section 1).
- LightGBM default seeds.
- CPU only (i7-12700H).

The training code has not changed since training, apart from two things:
- `task1_run.py` now takes `NUM_THREADS` from `nice.THREADS` instead of the hard-coded 12.
- `t1_members.py` now writes its pickles atomically.

The command log records both edits (2026-09-10 05:42 and 06:48 UTC). Neither changes the
model.

### 3.1 own: per-panel model, 10 files `t1_model_<panel>.pkl`

- **Script.** `exp/task1_run.py`.
  - `PARAMS` at lines 36-38: l2 objective, lr 0.05, num_leaves 255, min_data_in_leaf 100,
    feature_fraction 0.75, bagging_fraction 0.8, bagging_freq 1.
  - `SAMPLE = 2_500_000` at line 27; `build(..., seed=1)` at line 70.
  - Split `day < nd-45` at line 74; at most 2500 rounds at line 82.
- **Commands (confirmed, command log):**
  ```
  python -W ignore task1_run.py D7_I10_E                     # 2026-09-08 08:51 UTC
  nohup python -W ignore task1_run.py > t1_full.log 2>&1 &   # 2026-09-08 09:03 UTC, the other 9 panels
  ```
  No extra environment variables. Run on the code version with `NUM_THREADS = 12`
  (to reproduce today: `TFB_THREADS=12 python -W ignore task1_run.py`).
- **Seed.** Row sample `seed=1`; LightGBM defaults.
- **Held out.** Last 45 train days (early stopping only).
- **Runtime.** About 12 min for D7_I10_E. 4,678 s for the other nine panels, including
  prediction. See `exp/t1_full.log`; per-panel holdout scores are in `exp/out/t1_diag.csv`.
  File times: 2026-09-08 12:03 to 13:21 local.
- **Side effect.** `task1_run.py` also writes `out/state_submission.csv` and `grids/`.
  Those outputs are not used by the finals.

### 3.2 pooled: one model over all ten panels, `t1_pooled.pkl`

- **Script.** `exp/pooled_t1.py`.
  - `PER_PANEL = 500_000` at line 29 (5.0M rows total); `build(..., seed=1)` at line 38.
  - `PARAMS` imported from `task1_run`.
  - Split `day < nd-45` at line 61; at most 3000 rounds at line 66.
- **Command (confirmed, command log, 2026-09-09):**
  ```
  nohup python -W ignore pooled_t1.py train > pooled_train.log 2>&1 &
  ```
- **Seed.** `seed=1`; LightGBM defaults.
- **Held out.** Last 45 train days (early stopping only).
- **Runtime.** 2,126 s after a 180 s matrix build (`exp/pooled_train.log`). Boosting
  rounds: speed 2999, flow 2231, density 2984. File time 2026-09-09 21:18 local.

### 3.3 family: one model per corridor family, 5 files `t1_family_<D7_I10|D7_I210|D7_I405|D12_I5|D12_I405>.pkl`

- **Script.** `exp/t1_members.py`.
  - `FAMILY_SAMPLE = 1_200_000` per panel, at line 55.
  - `collect(panels, FAMILY_SAMPLE, seed=1)` at line 146.
  - `fit(..., PARAMS, 3000, ...)` at line 148; `day < nd-45` at line 111.
- **Command (confirmed).** From `exp/overnight.sh`, run in the night of 2026-09-09/10:
  ```
  python t1_members.py family > out/t1_family.log 2>&1
  ```
- **Seed.** `seed=1`; LightGBM defaults.
- **Held out.** Last 45 train days (early stopping only).
- **Runtime.** 01:59 to 04:05 local, 2026-09-10, about 2 h 06 min for all five
  (`out/overnight.log`).
- **Note.** The original training log was overwritten when `resume.sh` re-ran the command
  at 12:44 local. The current `out/t1_family.log` only shows "already trained, skipping".
  Per-model iteration counts are therefore not preserved.

### 3.4 alt: shallow, randomised per-panel model, 10 files `t1_alt_<panel>.pkl`

- **Script.** `exp/t1_members.py`.
  - `ALT_PARAMS` at line 59: l2, lr 0.035, num_leaves 63, min_data_in_leaf 40,
    feature_fraction 0.55, bagging_fraction 0.7, bagging_freq 1, `extra_trees=True`,
    lambda_l2 1.0.
  - `ALT_SAMPLE = 1_500_000` at line 56; `collect([panel], ALT_SAMPLE, seed=11)` at line 165.
  - At most 4000 rounds, line 167.
- **Command (confirmed).** From `exp/overnight.sh`:
  ```
  python t1_members.py alt > out/t1_alt.log 2>&1
  ```
- **Seed.** Row sample `seed=11`; LightGBM defaults.
- **Held out.** Last 45 train days (early stopping only).
- **Runtime.** 04:05 to 07:38 local, 2026-09-10, about 3 h 33 min for ten panels. The log
  was overwritten in the same way as family's.

(`t1_phys_*.pkl` models exist but are **not** used by s73/s74, which carry zero weight on
them.)

---

## 4. Task 1: conv-net inpainting members f1, h1, i1, j1, m1, n1

**Used by.** `runs/make_t1_tod.py` (`SIX = ["f1","h1","i1","j1","m1","n1"]`), run as
`T1_TOD=const T1_TILE_L=64 python make_t1_tod.py` from `runs/` (log `runs/_t1_tod.log`).
Each net's weights are `exp/out/t1_nn_<name>.pt`. Architecture is read from the weights
(width) plus the `ARCH` table in `t1_nn.py:417-429` (dilations).

### 4.1 Prerequisite: memmaps

```
python t1_nn.py prep          # exp/nn/<panel>_grid.npy (float16) and <panel>_static.npz
```

The command log shows prep run twice:
- 2026-09-17 07:39 UTC. This was a 5-channel memmap, used by f1.
- 2026-09-18 06:43 UTC, `python -u t1_nn.py prep > out/t1_prep_ramps.log 2>&1`. This added
  three ramp channels (8 channels); every later net used it.

Channels 0-4 are computed by identical code in both versions (checked against the original
`t1_nn.py`), and nets without `T1_RAMPS=1` read only channels 0-4. So today's `prep`
reproduces f1's inputs too. It reads `exp/cache/<panel>_train.npz` (unmasked grids) and
the current train ramp cache.

### 4.2 Recipe shared by all six

All six nets share this configuration:
- Config is read from environment variables at import (`t1_nn.py:41-102`).
- `train()` at line 599: AdamW lr 2e-3, OneCycle with 5% warm-up, weight decay
  `T1_WD` = 1e-4, grad-clip 1.0.
- Every 500 steps the net is scored on the held-out last `T1_VAL_DAYS` days: 64 batches
  of 8, fixed `default_rng(99)`, line 638. The checkpoint is saved when the holdout
  S_state improves, so the saved file is the best-scoring step, not the last one.
- Unset variables take their defaults:
  - `T1_RAMPS=0`, `T1_TOD_TRAIN=ramp`, `T1_AUG=0`, `T1_DROP=0`, `T1_GAPP=0`,
    `T1_KHEAD=0`, `T1_GATE=0`, `T1_ARCH=dilated`.
  - `--lwr 0`.
- GPU: RTX 3070 Ti Laptop.

| net | command (run in `exp/`) | seed | held out | steps / best | runtime | source | md5 |
|---|---|---|---|---|---|---|---|
| **f1** | `T1_VAL_DAYS=7 T1_WIDTH=128 T1_TCROP=96 T1_LCROP=64 T1_BS=10 T1_DILS=1,2,4,8,16,1,2,4,8,1 python -u t1_nn.py train f1 --steps 26000 --seed 5 > out/t1_nn_f1.log 2>&1` | 5 | last 7 days | 26000 / 25500 | 13,859 s (3.9 h), 2026-09-18 02:11 to 06:02 local | `exp/t1_queue3.sh:16-18`; `out/t1_queue3.log` | a43cf2c10bb5bcf29c91c09378f265dc |
| **h1** | `T1_VAL_DAYS=2 T1_WIDTH=128 T1_TCROP=96 T1_LCROP=64 T1_BS=10 T1_DILS=1,2,4,8,16,1,2,4,8,1 python -u t1_nn.py train h1 --steps 26000 --seed 31 > out/t1_nn_h1.log 2>&1` | 31 | **last 2 days** | 26000 / 25000 | 7,730 s, finished 2026-09-19 02:44 local | command log 2026-09-18 21:35 UTC; FINDINGS.md:896,905-908 ("h1 = f1 architecture with only 2 days held out") | 6e9929ae93821a542048ec16c7ebae91 |
| **i1** | `T1_VAL_DAYS=7 T1_WIDTH=128 T1_TCROP=96 T1_LCROP=64 T1_BS=10 T1_DILS=1,2,4,8,16,1,2,4,8,1 python -u t1_nn.py train i1 --steps 26000 --seed 41 > out/t1_nn_i1.log 2>&1` | 41 | last 7 days | 26000 / 26000 | 5,685 s, 2026-09-19 09:46 to 11:21 local | `exp/t1_seeds.sh:5-10` (spec `i1:41`); `out/t1_seeds.log` | e372d523b6bbf29d3f3a396708071fc8 |
| **j1** | same as i1 with `train j1 ... --seed 51` | 51 | last 7 days | 26000 / 26000 | 7,413 s, 2026-09-19 11:21 to 13:24 local | `exp/t1_seeds.sh:5-10` (spec `j1:51`); `out/t1_seeds.log` | 9713e7f2c3031f832ab37086a4515855 |
| **m1** | `T1_VAL_DAYS=7 T1_WIDTH=192 T1_TCROP=96 T1_LCROP=64 T1_BS=6 T1_DILS=1,2,4,8,16,32,1,2,4,8,16,1 python -u t1_nn.py train m1 --steps 30000 --seed 81 > out/t1_nn_m1.log 2>&1` | 81 | last 7 days | 30000 / 28500 | 10,027 s, finished 2026-09-20 18:46 local | command log 2026-09-20 12:57 UTC; FINDINGS.md:940-945 (192 ch, dilations to 32, 30k steps); `ARCH["m1"]` t1_nn.py:421 | ceb29bdd6d1861d814ce23d1cd57c4de |
| **n1** | `T1_VAL_DAYS=7 T1_WIDTH=128 T1_TCROP=192 T1_LCROP=96 T1_BS=4 T1_DILS=1,2,4,8,16,1,2,4,8,1 python -u t1_nn.py train n1 --steps 22000 --seed 91 > out/t1_nn_n1.log 2>&1` | 91 | last 7 days | 22000 / 22000 | 6,874 s, 2026-09-20 20:57 to 22:52 local | `exp/t1_arch.sh:7-8`; `out/t1_arch.log` | 4d413b012efe1537e49bfdfc1d5b4063 |

Notes:
- The `.sh` scripts also append `&& touch out/t1_nn_<name>.done`. That is bookkeeping only.
- h1 and m1 were launched by hand, not from a script. That is why no `.sh` file has them.
  In both cases the step count, the runtime and the final holdout line in the log match the
  command.
  - h1 was launched twice within 23 s (21:35:01 with `nohup ... &`, and 21:35:24). The log
    shows one clean 26,000-step run. Which of the two launches produced it is not recorded;
    the command is identical either way.
- The m1 batch size is 6. The gap finetune gm1 later used `T1_BS=8` for the same shape, so
  do not copy m1's batch size from `t1_gapnets.sh`.
- Code differences since each net was trained (gap, clock and gate options) sit behind
  environment variables whose defaults are no-ops. Their random draws short-circuit when
  the option is off, so with the settings above today's `t1_nn.py` follows the same code
  path. This was checked by reading the code, not by re-running.

---

## 5. Task 1: gap-aware finetunes gf1, gh1, gi1, gm1, gn1, ge1, gr1

**Used by.** `t1_gapfill.py` through `GAP_NET=gf1,gh1,gi1,gm1,gn1,ge1,gr1`, with
`GAP_NETW=1.0 GAP_NETW_Q=1.0`, in both s73 and s74.

How the finetunes differ from Section 4.2:
- `--init <parent>` loads the parent checkpoint; width and dilations come from the parent.
  Any `T1_WIDTH`/`T1_DILS` in the environment is ignored for these runs.
- The learning rate is divided by 10 (`t1_nn.py:626`).
- `T1_GAPP=p` adds a full-corridor 18-step blackout to a crop with probability p.
- `T1_GAP_EVENT=1` places the blackouts at queue-onset (step-6) or contract-ongoing
  origins from `qdata.origins`. The code sets `QONSET_STEP6=1 QONG_CONTRACT=1` itself
  (`t1_nn.py:224-231`).
- `T1_RAMPS=1` widens the input conv from 9 to 12 channels, with the new weights
  initialised to zero.
- Selection and checkpointing work as in Section 4.2.

| net | command (run in `exp/`) | parent | seed | held out | steps / best | runtime | source | md5 |
|---|---|---|---|---|---|---|---|---|
| **gf1** | `T1_VAL_DAYS=7 T1_WIDTH=128 T1_TCROP=96 T1_LCROP=64 T1_BS=10 T1_GAPP=0.5 T1_DILS=1,2,4,8,16,1,2,4,8,1 python -u t1_nn.py train gf1 --init f1 --steps 6000 --seed 7 > out/t1_nn_gf1.log 2>&1` | f1 | 7 | last 7 days | 6000 / 6000 | 1,191 s, 2026-09-26 ~20:29 to 20:49 local | command log 2026-09-26 (verbatim) | 80c740c0ff146cc49cde25aa101bd3f4 |
| **gh1** | `T1_VAL_DAYS=7 T1_GAPP=0.5 T1_TCROP=96 T1_LCROP=64 T1_BS=10 python -u t1_nn.py train gh1 --init h1 --steps 6000 --seed 11 > out/t1_nn_gh1.log 2>&1` | h1 | 11 | last 7 days | 6000 / 5500 | 1,064 s, 21:56 to 22:14 local | `exp/t1_gapnets.sh:7-9`; `out/t1_gapnets.log` | c628e446073331b7df03c3cfb33bae2f |
| **gi1** | `T1_VAL_DAYS=7 T1_GAPP=0.5 T1_TCROP=96 T1_LCROP=64 T1_BS=10 python -u t1_nn.py train gi1 --init i1 --steps 6000 --seed 12 > out/t1_nn_gi1.log 2>&1` | i1 | 12 | last 7 days | 6000 / 5500 | 1,281 s | `exp/t1_gapnets.sh:10` | b2b51f057e92b04abf2871a32064bdc6 |
| **gm1** | `T1_VAL_DAYS=7 T1_GAPP=0.5 T1_TCROP=96 T1_LCROP=64 T1_BS=8 python -u t1_nn.py train gm1 --init m1 --steps 6000 --seed 13 > out/t1_nn_gm1.log 2>&1` | m1 | 13 | last 7 days | 6000 / 5000 | 2,403 s | `exp/t1_gapnets.sh:11` | 3b85859ad443b810b076e5d999a54e05 |
| **gn1** | `T1_VAL_DAYS=7 T1_GAPP=0.5 T1_TCROP=192 T1_LCROP=96 T1_BS=4 python -u t1_nn.py train gn1 --init n1 --steps 6000 --seed 14 > out/t1_nn_gn1.log 2>&1` | n1 | 14 | last 7 days | 6000 / 6000 | 1,591 s, finished 23:42 local | `exp/t1_gapnets.sh:12` | d81c08a03e2d0f04290a81ef72439481 |
| **ge1** | `T1_VAL_DAYS=7 T1_TCROP=96 T1_LCROP=64 T1_BS=10 T1_GAPP=0.6 T1_GAP_EVENT=1 python -u t1_nn.py train ge1 --init f1 --steps 8000 --seed 21 > out/t1_nn_ge1.log 2>&1` | f1 | 21 | last 7 days | 8000 / 8000 | 1,642 s, 2026-09-27 00:14 to 00:42 local | `exp/t1_gapnets2.sh:4-6`; `out/t1_gapnets2.log` | 5001c695b3493ed337ad03410f9c0b49 |
| **gr1** | `T1_VAL_DAYS=7 T1_TCROP=96 T1_LCROP=64 T1_BS=10 T1_GAPP=0.6 T1_GAP_EVENT=1 T1_RAMPS=1 python -u t1_nn.py train gr1 --init f1 --steps 8000 --seed 22 > out/t1_nn_gr1.log 2>&1` | f1 | 22 | last 7 days | 8000 / 6500 | 1,521 s, 00:42 to 01:08 local | `exp/t1_gapnets2.sh:4,9` | ff4feef81c18d6788eecd70f27d08406 |

Dates: gh1 to gn1 ran on 2026-09-26, and ge1 and gr1 on 2026-09-27. The gh1 to gn1 rows
expand the scripts' `run()` helper (`env $C $2 python -u t1_nn.py train $1 --init $3 --steps
6000 --seed $4`) and the ge1/gr1 rows expand `env $B ...`.

**Inference cache (not training, but required).** Before `t1_gapfill.py` runs, each gap
net's predictions on the validation and private grids are cached as
`out/gapnet_<net>_<panel>_<split>.npz`, one net at a time:

```
TFB_THREADS=6 python -u t1_gap_netcache.py <net>
```

That script sets `T1_TOD=const T1_TILE_L=64` itself. `predict_grid` feeds ramp channels
automatically to a 12-channel checkpoint (gr1). Source: command log, 2026-09-26/27.

**Other Task 1 inference steps, for completeness.**
- The s73 analog fill (`t1_gap_analog.analog_fill`, `k=10`) is a deterministic
  nearest-neighbour lookup in the train grids. It has no fitted state and no random numbers.
- The train time-of-day profiles in `t1_gapfill.train_profile` are plain means.

---

## 6. Task 2: queue models

**Used by.** This command, run in `exp/` (command log 2026-09-22; log `out/q2_s53.log`):

```
python -u qpredict2.py out/q2_s53.csv --onset both,both6,negall@0.60 \
    --ongoing both,nn:c1,nn:c2,nn:q1@0.50 --dump out/q2_probs_s53.pkl > out/q2_s53.log 2>&1
```

After it, `runs/s53_q-ongoing-q1/make.py` calls `qvariant.write(q2_probs_s53.pkl, ...,
0.60, 0.50, onset_kmin=1)`. The s73/s74 queue file is s53's, md5 `e0fd118a...`.

How `qpredict2.py` loads the models:
- `load_tree(n)` reads `out/queue_model_<n>.pkl`.
- `load_nn(n)` builds `nn_queue.make_net(n)` and loads `out/nn_queue_<n>_c0.pt` for
  onset and `_c1.pt` for ongoing. It falls back to `nn_queue_<n>.pt` if those are missing.

### 6.1 LightGBM `queue_model_both.pkl`, `queue_model_negall.pkl`, `queue_model_both6.pkl`

The following applies to all three:
- **Script.** `exp/qtrain.py`, binary objective (lines 157-162): lr 0.05, num_leaves 127,
  min_data_in_leaf 200, feature_fraction 0.8, bagging 0.8/1, lambda_l2 5.0, **500 fixed
  rounds**, no early stopping. One booster per condition (onset / ongoing).
- **Seeds.** Fixed numpy seeds: window sampling `default_rng(100 + panel_index)` (line 68),
  negative subsampling `default_rng(7 + panel_index)` (line 86), and `RNG = default_rng(0)`.
  LightGBM uses its default seeds.
- **Held out.** `QVAL_DAYS=60` (default): the last 60 train days (2030-12-31 to
  2031-02-28) are excluded from fitting and priors and used only to tune a threshold. That
  tuned threshold is not what shipped; the shipped values are 0.60 / 0.50.
- **Data.** Only the eight Task 2 panels (`QUEUE_PANELS`). No ramp data, so the
  2026-09-12 release rebuild does not affect these models.
- **CPU.**

| model | command (run in `exp/`) | threads | runtime | source | md5 |
|---|---|---|---|---|---|
| **negall** | `env QNEG=1.0 QMODEL=queue_model_negall.pkl python qtrain.py > out/qtrain_negall.log 2>&1` | 12 (hard-coded then) | ~23 min (19:07 to 20:30 local, 2026-09-10) | `exp/qbig.sh:17` (via `run()`, lines 8-14); `out/qbig.log` | c0fbf043bcfe6cda054ed07bf6cfa3df |
| **both** | `env QWIN_ON=2800 QWIN_ONG=2800 QNEG=1.0 QMODEL=queue_model_both.pkl python qtrain.py > out/qtrain_both.log 2>&1` | 12 (hard-coded then) | 3,313 s (20:30 to 21:25 local, 2026-09-10) | `exp/qbig.sh:18`; `out/qtrain_both.log` | a0c7110ba6f1b484cf3182496db8e602 |
| **both6** | `env QWIN_ON=2800 QWIN_ONG=2800 QNEG=1.0 QONSET_STEP6=1 QMODEL=queue_model_both6.pkl python qtrain.py > out/qtrain_both6.log 2>&1` | `nice.THREADS` = 8 (`TFB_THREADS` unset) | 2,046 s (10:04 to 10:42 local, 2026-09-13) | `exp/qstep6.sh:6-7`; `out/qstep6.log` | 0aecb774eae137531bdba0b320586aad |

- Unset variables were at their defaults: `QSHOCK=0`, `QNOPRIOR=0`, `QTARGET=flag`,
  `QRAMP=0`, `QPARAMS` unset, `QONSET_RELAX=0`, `QONG_CONTRACT=0`, `QHOLD_FIRST=0`.
- `qtrain.py`, `qdata.py` and `qfeat.py` were edited after `both`/`negall` were trained, on
  2026-09-12 and 09-13 (`QNOPRIOR`, `QONSET_RELAX`, `QTARGET`, `QRAMP`, `QPARAMS`,
  `nice.THREADS`). The command log shows every such addition is gated by an environment
  variable whose default keeps the earlier behaviour. The one real change is the thread
  count: use `TFB_THREADS=12` for `both` and `negall`.

### 6.2 Conv nets c1, c2, q1 (`exp/nn_queue.py`)

The following applies to all three:
- 1-D dilated conv over links; input is the 12-step history plus static, prior and global
  channels.
- `train()` at line 220: batch 48, AdamW lr 2e-3, OneCycle 5%, grad-clip 1.0.
- **Held out.** `VAL = 60` days (last 60, never trained on). The 20 days before them
  (`SEL = 20`, 2030-12-11 to 2030-12-30) are used for checkpoint selection. Training
  windows therefore come only from days up to 2030-12-10.
- Selection windows are drawn with `default_rng(panel_index)` (line 102). The net is
  scored every 1000 steps.
- GPU.

| net | command (run in `exp/`) | seed | code version / selection | steps, selected checkpoints | runtime | source | md5 (c0 / c1) |
|---|---|---|---|---|---|---|---|
| **c1** | `python nn_queue.py train c1 --steps 30000 > out/nn_c1.log 2>&1 && python nn_queue.py eval c1 >> out/nn_c1.log 2>&1`, then `cp out/nn_queue_c1.pt out/nn_queue_c1_c0.pt; cp out/nn_queue_c1.pt out/nn_queue_c1_c1.pt` | 0 (default) | **first version of nn_queue.py**: no dropout, weight decay 1e-4 hard-coded, one checkpoint picked by mean(onset, ongoing) selection IoU with predictions masked to eligible cells, 60/60 selection windows | 30000; one checkpoint at step 11000 used for both conditions | 2,912 s, 2026-09-14 ~08:14 to 09:05 UTC | command log 2026-09-14 08:14 and 09:06 UTC; `out/nn_c1.log` | 7b87024c... / 7b87024c... (both are the same file) |
| **c2** | `python nn_queue.py train c2 --steps 14000 > out/nn_c2.log 2>&1 && python nn_queue.py eval c2 >> out/nn_c2.log 2>&1` | 0 (default) | current code, defaults `NN_DROP=0.1 NN_WD=1e-2`, per-condition checkpoints, 60/60 selection windows (`NN_SEL_ON`/`NN_SEL_ONG` did not exist yet; today's defaults are equal) | 14000; c0 = step 4000, c1 = step 9000 | 1,373 s, from 2026-09-14 09:06 UTC | command log 2026-09-14 09:06 UTC; `out/nn_c2.log` | 93835edb81b4bb18e6eee66df40ee256 / d995d3701033e6badeb74ef70347eb77 |
| **q1** | `NN_SEL_ONG=150 python -u nn_queue.py train q1 --steps 14000 --seed 201 > out/nn_q1.log 2>&1`, then `python -u nn_queue.py eval q1 >> out/nn_q1.log 2>&1` | 201 | `ARCH["q1"]`: w=192, dilations (1,2,4,8,16,32,64,1,2,4,8,1); `NN_DROP=0.1 NN_WD=1e-2 NN_LOSS_ALL=0 NN_SEL_ON=60` | 14000; c0 = step 4000, c1 = step 8000 | 1,690 s, 2026-09-22 16:12 to 16:42 local | `exp/q_arch.sh:5-10` (spec `q1:201`); `out/q_arch.log` | 131a6402e2709ab0ab3482f3ae42a045 / 88d3ce04d8691fd5b7af852f17063f68 |

- c1 and c2 use the default `Net(w=128, dil=(1,2,4,8,16,32,1,2,4,8,16,1))`. The `eval`
  step writes out-of-sample harness files only; it does not affect the checkpoints.
- **c1 cannot be re-run exactly with today's code.** Its training loop, weight decay and
  checkpoint rule were replaced on 2026-09-14 09:06 UTC, straight after it finished.
  - The closest equivalent today is
    `NN_DROP=0 NN_WD=1e-4 python nn_queue.py train c1 --steps 30000`. Today's code keeps the
    best onset and the best ongoing checkpoint separately, while c1 used a single
    checkpoint chosen by the mean of the two scores (step 11000).
  - Dropout has no parameters, so the archived c1 weights load into today's `Net`
    unchanged.
  - This is recorded in the s22 repro bundle, `submissions/repro_s22_ens3-nn_0.83209.zip`,
    in `REPRODUCE.md`.

---

## 7. Task 4: ODME, no trained model

**Confirmed.** `exp/task4_run.py` is closed form and deterministic:

```
python task4_run.py "D:/BigData Competition/runs/s12_rebuild/out"   # exp/s12.sh:21-22, log out/s12_odme.log
```

For each panel and split it solves

    f = argmin ||A f - c||^2 + lam * ||f - b||^2,   f >= 0

with the following inputs and settings:
- `A` is the path-link incidence matrix from `odme_fast.load_operator`.
- `c` is the released `synthetic_link_counts`.
- `b` is the released `synthetic_weak_prior` (`anchor="weak_prior"`).
- `lam = LAM = 0.05` (`task4_run.py:33`).
- The solver is projected FISTA (`odme_fast.fista`, `iters=4000, tol=1e-9`,
  `odme_fast.py:43`).

There is no random number, seed, training data or fitted parameter; `lam` was chosen by
leaderboard probes in the first week. Output md5 `5cae43cf9fd661fc64439d009a10efad`
(s12, unchanged through s73/s74). Runtime is about 1 minute on CPU.

---

## 8. What could not be established with certainty

1. **Tree members vs. the current release.** All four were trained on the pre-2026-09-12
   train ramp data (Section 2), and they read ramp features. Retraining on the current
   release will not reproduce them exactly. Use the train split of the 2026-09-10 release
   (local copy `D:\BigData Competition\2026-old-0912`) for exact re-training.
2. **family and alt per-model logs.** They were overwritten, so only wall-clock times
   survive (`out/overnight.log`). The commands and all settings are certain; they come from
   the code and `overnight.sh`.
3. **LightGBM bit-reproducibility.** Results depend on the thread count. It was 12 for the
   Task 1 trees and for `both`/`negall`, and 8 for `both6`. LightGBM seeds were never set
   explicitly (library defaults).
4. **GPU nets.** They are not bit-deterministic (cuDNN non-deterministic kernels). Seeds
   and commands are exact, but only the archived checkpoints reproduce the submitted files
   exactly.
5. **c1 (Task 2).** It was trained by a superseded version of `nn_queue.py`. The exact
   command is known (`python nn_queue.py train c1 --steps 30000`, seed 0), but today's code
   needs `NN_DROP=0 NN_WD=1e-4` and still differs in its checkpoint rule (Section 6.2).
6. **h1 and m1.** These have no `.sh` script. Their commands come from the command log, and
   the log files agree with them.
7. **Package versions.** These are the currently installed ones. No install happened during
   the competition as far as the command log shows, but the training-time versions were not
   written down at the time.

## 9. Model checksums (md5, identical in both s73 and s74 MANIFEST.json and on disk)

```
t1_model_D7_I10_E.pkl    37f8383a2e758da09bfd667e360cbfd2   t1_alt_D7_I10_E.pkl    c87daac7061031162567703909b87881
t1_model_D7_I10_W.pkl    82e5fa7d3d62d98ad5d9a73a19669012   t1_alt_D7_I10_W.pkl    6c13e49c07884cc6d4b4b7f91c7a8856
t1_model_D7_I210_E.pkl   24c4cd08a6839877f80c4c2cccce444d   t1_alt_D7_I210_E.pkl   c30857136cdfe5415b0f4376e7d5ae51
t1_model_D7_I210_W.pkl   da292c41b3d635dcd74f7a64af44f20f   t1_alt_D7_I210_W.pkl   c7983a331f8c8cbda4fbf310fd8c79c9
t1_model_D7_I405_N.pkl   8c693f3a5b9a8d5699bc8caec9238c7e   t1_alt_D7_I405_N.pkl   8b2a6e62009c27d20f95a463a12326ab
t1_model_D7_I405_S.pkl   947dd280f668c68365b51b2995b8453c   t1_alt_D7_I405_S.pkl   3ac52c3d41f33d5e5235fe2c93d8910b
t1_model_D12_I5_N.pkl    6dbc47c4f7910193c133bdc932835ecd   t1_alt_D12_I5_N.pkl    aace7495ebabdb9af221c92dbab5c847
t1_model_D12_I5_S.pkl    dba92b9aa4a1893c6ea5f523861ca7a7   t1_alt_D12_I5_S.pkl    01554455637bb75e375e18d40fb997f1
t1_model_D12_I405_N.pkl  b803281af96fc7f2ff2518cc9208ffaf   t1_alt_D12_I405_N.pkl  913fb6796047b7992dc79be3100cf4c5
t1_model_D12_I405_S.pkl  aaae8b7fdb2898b48906d1ff23bd93cb   t1_alt_D12_I405_S.pkl  26db897db72b950046efd005be60f8c2
t1_pooled.pkl            44b26556c86e12f899a863afb405ec98
t1_family_D7_I10.pkl     9d586ecf86ad8f3ba148087812d3e037   t1_family_D12_I5.pkl   739b6543c5f6b8e37c13a1838ae2ec32
t1_family_D7_I210.pkl    ad579ca186800eab7a632bc5be90d58d   t1_family_D12_I405.pkl a849ca198d6ff5a4195a88f05568a50d
t1_family_D7_I405.pkl    57768932263e356a1558d83bca561caa
t1_nn_f1.pt  a43cf2c10bb5bcf29c91c09378f265dc   t1_nn_gf1.pt 80c740c0ff146cc49cde25aa101bd3f4
t1_nn_h1.pt  6e9929ae93821a542048ec16c7ebae91   t1_nn_gh1.pt c628e446073331b7df03c3cfb33bae2f
t1_nn_i1.pt  e372d523b6bbf29d3f3a396708071fc8   t1_nn_gi1.pt b2b51f057e92b04abf2871a32064bdc6
t1_nn_j1.pt  9713e7f2c3031f832ab37086a4515855   t1_nn_gm1.pt 3b85859ad443b810b076e5d999a54e05
t1_nn_m1.pt  ceb29bdd6d1861d814ce23d1cd57c4de   t1_nn_gn1.pt d81c08a03e2d0f04290a81ef72439481
t1_nn_n1.pt  4d413b012efe1537e49bfdfc1d5b4063   t1_nn_ge1.pt 5001c695b3493ed337ad03410f9c0b49
                                                t1_nn_gr1.pt ff4feef81c18d6788eecd70f27d08406
queue_model_both.pkl     a0c7110ba6f1b484cf3182496db8e602
queue_model_both6.pkl    0aecb774eae137531bdba0b320586aad
queue_model_negall.pkl   c0fbf043bcfe6cda054ed07bf6cfa3df
nn_queue_c1_c0.pt        7b87024ca2c778c3eb4c434d88f1d8f0   (identical to nn_queue_c1_c1.pt and nn_queue_c1.pt)
nn_queue_c2_c0.pt        93835edb81b4bb18e6eee66df40ee256   nn_queue_c2_c1.pt  d995d3701033e6badeb74ef70347eb77
nn_queue_q1_c0.pt        131a6402e2709ab0ab3482f3ae42a045   nn_queue_q1_c1.pt  88d3ce04d8691fd5b7af852f17063f68
q2_probs_s53.pkl         dc584591b2d195d9b1f923d9130b08c6   (Task 2 probability dump the queue file is cut from)
```

Content-addressed copies of these files are kept under `submissions/models/<md5>`. Full
bundles are `submissions/repro_s73_t1-gap-ens7_0.86973.zip` and
`submissions/repro_s74_t1-gap-ens7-noanalog_0.86959.zip`.


## Addendum 2026-09-28: four more event-centred gap finetunes (finals s78 / s79)

Same recipe as ge1 (event-centred gaps: `T1_GAP_EVENT=1`, `T1_GAPP=0.6`,
8000 steps, last 7 train days held out), from the other shipped parents.
Script: `train/t1_gapnets3.sh` (run from `src/`, needs `python t1_nn.py prep`):

```
B="T1_VAL_DAYS=7 T1_GAPP=0.6 T1_GAP_EVENT=1"
env $B T1_TCROP=96  T1_LCROP=64 T1_BS=10 python -u t1_nn.py train geh1 --init h1 --steps 8000 --seed 31
env $B T1_TCROP=96  T1_LCROP=64 T1_BS=10 python -u t1_nn.py train gei1 --init i1 --steps 8000 --seed 32
env $B T1_TCROP=96  T1_LCROP=64 T1_BS=8  python -u t1_nn.py train gem1 --init m1 --steps 8000 --seed 33
env $B T1_TCROP=192 T1_LCROP=96 T1_BS=4  python -u t1_nn.py train gen1 --init n1 --steps 8000 --seed 34
```

Runtimes on the RTX 3070 Ti Laptop: 24, 26, 54, 39 min. Final holdout (event
gaps in 60% of crops): S_state 0.8829 / 0.8808 / 0.8784 / 0.9101. The finals use
all eleven gap nets: gf1, gh1, gi1, gm1, gn1, ge1, gr1, geh1, gei1, gem1, gen1.

## Addendum 2026-10-01: from-scratch twelfth gap net xs1 (final s83)

`train/t1_xs1.sh` trains xs1 from random initialisation (not a finetune):
width 160, 13 dilated blocks (1,2,4,8,16,32 twice, then 1), ramp inputs
(T1_RAMPS=1, 12 channels), 192-step x 96-link crops, batch 4, 30% of crops carry
an organiser-style event blackout (T1_GAPP=0.3, T1_GAP_EVENT=1), 24000 steps,
seed 101, ~4.9 h. Best holdout checkpoint (gap-mixed holdout S 0.9271) is kept.

Alone it is weaker than the shipped members (normal-cell S 0.9488 vs f1 0.9517;
gated gap S 0.8360 vs 0.8400 for the 11 nets), but as a twelfth member of the
gap fallback mean it reads 0.8402 locally and scored **0.86983** on the board
(s83, +0.00003 over s78). `reproduce.py --final s83` = s78 with GAP_NET = the 11
nets + xs1; only Task 1 gap rows differ from s78 (72,898 rows).

## Addendum 2026-10-02: same-month analog library (final s86)

No training. `src/t1_gapfill.py` with GAP_SM=union adds a second analog library
for each Task 2 gap window: the split's own masked grid (gap spans blanked,
library starts clear of every gap span and more than one day from the window).
`src/t1_samemonth_dist.scores` computes both libraries' match distance (mean of
the 10 best edge-signature distances); the closer library fills the window when
its distance is <= 0.002, otherwise the 12-net fallback does. 59 of 160 windows
use the month's own days. Board: s86 0.86995 (+0.00012 over s83); the
conservative variant (same month only where train fails, GAP_SM=rescue) 0.86993.

## Addendum 2026-10-05: ineligible readings as inputs (final s88)

The release ships finite readings with pct_observed < 75 (~7% of cells, pct
20-67) in every split; they are never Task 1 targets, and are nearly as
informative about a neighbouring hidden cell as eligible readings (temporal
neighbour |dv| 1.57 vs 1.52 km/h). The original nets only took pct >= 75 as
observed.

- `src/t1_nn.py` `T1_PCTMIN=0` feeds them to the net at inference with their
  own pct (`T1_PCTCLIP=75` would show them at 75 instead).
- `src/t1_nn.py` `T1_INELIG_TRAIN=1` makes them visible during training (never
  hidden, never in the loss, blanked inside simulated Task 2 gaps).
- `train/t1_inelig_ft.sh` fine-tunes the six members into qf1, qh1, qi1, qj1,
  qm1, qn1 (6000 steps each from the parent checkpoint, the parent's crop
  recipe, seeds 101-106). The archived qm1 is the best checkpoint of a run that
  was stopped at step 5000.
- `reproduce.py --final s88` runs the nets step with `T1_SIX=<q-nets>
  T1_PCTMIN=0` into `stage/state_s88_nets.csv`, then s86's gap fill.

Local (last 7 days, 3 panels, 0.35 S_state + 0.15 S_physics): shipped 0.44381,
old nets + ineligible inputs 0.44402, q-nets 0.44421. Board: s87 (old nets,
pct shown as 75) 0.87044; s88 0.87057.

## Addendum 2026-10-05: gap nets with ineligible inputs (final s89)

No training. `reproduce.py --final s89` = s88, but the 12 gap nets' caches are
built with `GAPNET_TAG=_i T1_PCTMIN=0 T1_PCTCLIP=75` (src/t1_gap_netcache.py),
i.e. the finite pct < 75 readings are inputs, shown at pct 75. Fair gap harness:
gated in-gap S 0.8402 -> 0.8410. Board: 0.87061 (+0.00004 over s88).

## Addendum 2026-10-11: Task 4 solved to convergence (finals s90 / s79c)

No training. `src/task4_run.py` with `T4_REFINE=lbfgs` refines the FISTA answer
with L-BFGS-B (ftol 1e-15, gtol 1e-10) to the exact minimiser of
||Af - c||^2 + 0.05 ||f - b||^2, f >= 0 (matches scipy nnls to 1e-7 L1 on the
corridors checked). Path flows move 1.18% L1 overall. Board: s90 0.87086
(+0.00025 over s89), s79c 0.86989 (+0.00024 over s79).

## Addendum 2026-10-12: looser analog gate + fine-tuned gap nets (final s95)

- `train/t1_gapnets_inelig.sh`: the 12 gap nets fine-tuned 3000 steps each from
  their own checkpoints with their own gap recipes and T1_INELIG_TRAIN=1 ->
  gf1q ... xs1q (seeds 201-212). Their caches are built with T1_PCTMIN=0.
- Analog gate GAP_MAXD 0.002 -> 0.003 (both libraries).
Fair gap harness: gated in-gap S 0.8410 (0.002) -> 0.8599 (0.003); the
fine-tuned nets alone 0.8015 -> 0.8034. Board: s93 (gate only) 0.87095, s95
(gate + nets) 0.87097, s96 (nets only) 0.87088 vs s90 0.87086.

## Addendum 2026-10-14: Task 2 from two public pipelines (finals s101 / s103)

Our own Task 2 forecaster turned out far weaker than public ones: swapping in the
public anubisyy Task 2 gained +0.0073 composite (s97); adding yeashusemwal's ongoing
ensemble average +0.0023 more (s101). Both are retrained here from their published
code (src_t2/pub, src_t2/yeash; train/t2_public.sh) and run from archived weights
(models/t2) by src/t2_public.py:
- anubisyy ongoing LightGBM (36 features, 700 rounds, seed 20260930): reproduces that
  notebook's published ongoing labels on 100% of 87,000 cells.
- anubisyy onset LightGBM (two-stage) + CatBoost (GPU), 0.5/0.5, p >= 0.55 at T+30,
  else argmax: 99.97% of 14,500 onset cells.
- yeashusemwal ongoing ensemble: task2_v11.py seeds 11-14, task2_v15.py seeds 15-16,
  task2_v18.py seed 20 (default arguments; their own seeds/args are not public),
  blended 0.8 x [0.5 mean(v11..v14) + 0.5 mean(v15, v16)] + 0.2 x v20 over both heads;
  final ongoing p = 0.5 ensemble + 0.5 anubisyy, >= 0.40.
Only the 60-minute window history (and the masked T column for the ensemble) is
read - data at or before the forecast origin.

## Addendum 2026-10-14: our no-prior LightGBM as a third ongoing member (s105 / s107)

`models/queue_model_bothnp.pkl` = qtrain.py with QWIN_ON=2800 QWIN_ONG=2800 QNEG=1.0
QNOPRIOR=1 (exp/qdiverse.sh). reproduce.py dumps its ongoing probabilities with
qpredict2.py and src/t2_public.py averages it in equal thirds with the ensemble and
anubisyy's LGB (T2_W3, threshold 0.43). Board: s104 (public + bothnp) 0.88109 vs s99
(public + our prior-based mix) 0.87601; s105 0.88157.

## Addendum 2026-10-15: a second no-prior member (final s108)

`models/nn_queue_q1np_c0.pt / _c1.pt`: `NQ_NOPRIOR=1 python nn_queue.py train q1np
--steps 10000 --seed 7` (q1's shape; the 19 prior channels are zeroed). Its ongoing
probabilities (qpredict2.py with NQ_NOPRIOR=1) are averaged 50/50 with bothnp's to
form the third ongoing member. OOS harness, ongoing alone at 0.50: bothnp 0.6991,
q1np 0.6970, mean 0.7053. Board: s108 0.88175 vs s105 0.88157.

## Addendum 2026-10-15: r4np and weights (s111 / s112)

`models/nn2_queue_r4np_c0.pt / _c1.pt`: `NQ_NOPRIOR=1 NN2_PAD=batch NN2_ONEPANEL=1
NN2_WD=1e-3 NN2_DROP=0.1 python nn_queue2.py train r4np --steps 8000 --seed 17`.
OOS ongoing alone 0.7024; mean(bothnp, q1np, r4np) 0.7093. Board: s110 (thirds)
0.88222, s111 (0.25/0.25/0.5, thr 0.45) 0.88233.

## Addendum 2026-10-16: the row at T (finals s113 / s114)

Organizer ruling (Kaggle discussion): Task 2 may use "timestamps at or before T,
including the observations at T"; window_history covers T-60..T-5 and the row at T
is released (in the Task 1 masked layer). QT_T=1 (src/qdata.py window_xy,
src/nn_queue.py history_at, src/qpredict2.py): history = T-55..T, row T from the
masked layer at inference, thinned 35% at random in training (51.7% of row-T cells
are eligible-visible at the scored origins vs 76.3% in window_history).
Retrained: bothnpT (qtrain QT_T=1 + the bothnp recipe), q1npT (nn_queue, 10000 steps,
seed 7), r4npT (nn_queue2, 8000 steps, seed 17). OOS ongoing triple 0.7093 -> 0.7167.
Board: s113 0.88370 vs s111 0.88233.

## Addendum 2026-10-17: seed bagging (finals s116 / s117)

q1npT2 (nn_queue, seed 23) and r4npT2 (nn_queue2, seed 29), same recipes as q1npT /
r4npT. "Ours" = mean(bothnpT, mean(q1npT, q1npT2), mean(r4npT, r4npT2)). OOS triple
0.7167 -> 0.7188; board s116 0.88411 vs s113 0.88370. Further seeds (bothnpT2,
r4npT3) add ~0.0005 IoU locally - below the board's resolution, not used.

## Addendum 2026-10-17: rolling same-split priors (finals s118 / s119)

Train-derived queue-rate priors misled on the separately generated scored months; the
same priors computed from the scored month itself carry its real pattern. QROLL=14
(src/qdata.py rolling_priors, src/qpredict2.py): rate_link, rate_tod, P_on, P_ong from
the 14 days before the window's day, on the window's split's masked layer (validation
precedes private). Training builds them identically from train's masked layer.
`models/queue_model_bothrollT.pkl` = qtrain.py with QROLL=14 QT_T=1 QWIN_ON=2800
QWIN_ONG=2800 QNEG=1.0 (priors kept). Board: s118 0.88510 vs s116 0.88411.

## Addendum 2026-10-18: r4rollT (finals s120 / s121)

`models/nn2_queue_r4rollT_c0.pt / _c1.pt`: `QT_T=1 QROLL=14 NN2_PAD=batch NN2_ONEPANEL=1
NN2_WD=1e-3 NN2_DROP=0.1 python nn_queue2.py train r4rollT --steps 8000 --seed 17` (priors
kept, rolling). Added to the 2-D member: mean(r4npT, r4npT2, r4rollT). Board: s120 0.88587
vs s118 0.88510.

## Addendum 2026-10-18: r4rollT2 (finals s122 / s123)

`models/nn2_queue_r4rollT2_c0.pt / _c1.pt`: `QT_T=1 QROLL=14 NN2_PAD=batch NN2_ONEPANEL=1
NN2_WD=1e-3 NN2_DROP=0.1 python nn_queue2.py train r4rollT2 --steps 8000 --seed 29` (same
recipe as r4rollT, second seed). 2-D member: mean(r4npT, r4npT2, r4rollT, r4rollT2)
(`T2_R4ROLL=2`). Board: s122 0.88617 vs s120 0.88587; hedge s123 0.88509.

## Addendum 2026-10-18: bothrollT2 (finals s124 / s125)

`models/queue_model_bothrollT2.pkl`: `QROLL=14 QT_T=1 QWIN_ON=2800 QWIN_ONG=2800 QNEG=1.0
QSEED=2 QMODEL=queue_model_bothrollT2.pkl python qtrain.py` (bothrollT recipe, LightGBM seed 2;
held-out S_queue 0.8387 vs 0.8389). LightGBM member: mean(bothrollT, bothrollT2) (`T2_BROLL2=1`).
Board: s124 0.88640 vs s122 0.88617; hedge s125 0.88532 vs s123 0.88509.
