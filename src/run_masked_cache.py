import sys
from build_cache import build_masked
from common import PANELS
for split in ["validation", "private", "train"]:
    for p in PANELS:
        build_masked(p, split)
