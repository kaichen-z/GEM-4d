"""Pure retention predicate for GEM4D checkpoint pruning.

Split out of utils.py so it can be imported and unit-tested without pulling
torch / accelerate / diffusers at import time.

Policy (plan.md 17-04):
    A checkpoint at step N is retained iff it is among the largest K
    values in the current set (top-K newest) OR N is a positive multiple
    of M (milestone cadence).

K=None or K<=0 disables the top-K rule.
M=0 or None disables the milestone rule.
"""

from typing import Iterable, Optional, Set


def retained_checkpoints(
    step_nums: Iterable[int],
    K: Optional[int],
    M: Optional[int],
) -> Set[int]:
    """Return the subset of step_nums to keep under top-K ∪ multiples-of-M."""
    steps = set(step_nums)
    if not steps:
        return set()

    top_k = (
        set(sorted(steps)[-K:]) if (K is not None and K > 0) else set()
    )
    multiples = (
        {n for n in steps if n % M == 0} if (M and M > 0) else set()
    )
    return top_k | multiples
