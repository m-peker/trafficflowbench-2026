import sys
from pathlib import Path
from common import PANELS, WORK
from build_cache import build_masked
from rampcache import build as build_ramp

CACHE = WORK / "cache"
for split in ["validation", "private", "train"]:
    for p in PANELS:
        if not (CACHE / f"{p}_{split}_ramp.npz").exists():
            build_ramp(p, split)
        if not (CACHE / f"{p}_{split}_masked.npz").exists():
            build_masked(p, split)
print("ALL CACHES DONE")
