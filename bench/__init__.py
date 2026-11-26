"""T6 throughput and smoke tooling for SLBR-Next.

The modules in this package intentionally avoid importing ``src.models`` at
module import time. The GPU scripts import the training stack lazily inside
their worker entry points, and the matrix helpers are pure standard-library +
``torch`` so the CPU test suite can validate them without creating the model.
"""
