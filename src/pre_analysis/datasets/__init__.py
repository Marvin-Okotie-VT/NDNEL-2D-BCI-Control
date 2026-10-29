"""Dataset registry"""

from . import (bci3_iiia, bnci2014_001, kaya2018_halt, physionet_mi,
               wang2025, wang2026)

REGISTRY = {s.name: s for s in (
    bnci2014_001.SPEC,
    bci3_iiia.SPEC,
    kaya2018_halt.SPEC,
    wang2025.SPEC,
    wang2026.SPEC,
    physionet_mi.SPEC,
    
)}