"""Keep the machine usable while these jobs run.

Two separate problems, and the second one was silent.

Priority. A batch job at normal priority competes with whatever the person at
the keyboard is doing, so a run that saturates the cores makes the desktop
stutter even though nothing is actually short of memory. Dropping to
below-normal costs almost nothing in wall-clock when the machine is otherwise
idle - the scheduler still hands over every spare slice - and hands the cores
straight back the moment a foreground app wants them.

Threads. `task1_run.PARAMS` sets `num_threads=12`, which looked like the cap and
was not: it applies to *training only*. A booster saved through
`model_to_string` carries no thread setting, so every `predict()` call falls
back to the OpenMP default, which is every logical core. The long jobs here are
mostly prediction, so in practice nothing was capped at all - the s07 run was
observed with 48 threads on a 20-thread box. NumPy's BLAS adds its own pool on
top.

So the cap has to be applied in three places, and this module is the one that
knows the number:

    THREADS            - the budget, from TFB_THREADS, default 6
    limit_blas()       - caps NumPy/BLAS pools at runtime via threadpoolctl
    predict_kwargs()   - the kwargs every Booster.predict call needs

Import it first thing in a script. The environment variables it sets only bite
if it is imported before NumPy and LightGBM, which is why `common` imports it on
line one; the runtime calls work regardless of order and are what actually
carries the guarantee.
"""
import os

# 8 of 20 logical threads. Prediction is memory-bandwidth bound, so the top end
# of the thread count buys little: measured on a 300k-row predict, 20 threads
# beat 8 by well under the factor the core count suggests. Raise it with
# TFB_THREADS=20 when nobody is at the keyboard, lower it to 4 when they are.
THREADS = int(os.environ.get("TFB_THREADS", "8"))

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, str(THREADS))


def below_normal():
    """Drop this process to below-normal priority. No-op off Windows."""
    try:
        import ctypes
        from ctypes import wintypes
        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # restype matters: the default c_int truncates a 64-bit handle
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.GetCurrentProcess.argtypes = []
        k32.SetPriorityClass.restype = wintypes.BOOL
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        return bool(k32.SetPriorityClass(k32.GetCurrentProcess(),
                                         BELOW_NORMAL_PRIORITY_CLASS))
    except Exception:
        try:
            os.nice(10)
            return True
        except Exception:
            return False


_LIMITER = None


def limit_blas(n=None):
    """Cap NumPy/BLAS thread pools, whatever the import order was.

    The returned limiter must be kept alive: threadpoolctl restores the original
    limits when it is garbage collected, so calling this and dropping the result
    silently undoes it.
    """
    global _LIMITER
    try:
        import threadpoolctl
        _LIMITER = threadpoolctl.threadpool_limits(n or THREADS)
        return True
    except Exception:
        return False


def predict_kwargs():
    """What every Booster.predict call needs so it does not grab every core."""
    return {"num_threads": THREADS}


def apply():
    """Everything at once. Safe to call more than once."""
    ok_p, ok_b = below_normal(), limit_blas()
    return {"threads": THREADS, "below_normal": ok_p, "blas_limited": ok_b}


_STATE = apply()
