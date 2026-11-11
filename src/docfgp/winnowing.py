"""Select deterministic rightmost-minimum passage anchors."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence


def anchors(hashes: Sequence[str], window: int = 32) -> tuple[str, ...]:
    """Winnow ordered shingle hashes with a rightmost minimum tie rule.

    Args:
        hashes: Word-shingle hashes in document order.
        window: Number of consecutive hashes per selection window.

    Returns:
        Distinct anchor hashes in first occurrence order.
    """
    if not hashes:
        return ()

    window = max(1, window)
    first_full_window_end = min(window, len(hashes)) - 1
    # Indexes of hashes that can still become a window minimum, in increasing hash order.
    candidates: deque[int] = deque()
    selected: dict[str, None] = {}

    for index, shingle_hash in enumerate(hashes):
        while candidates and hashes[candidates[-1]] >= shingle_hash:
            candidates.pop()
        candidates.append(index)
        while candidates[0] <= index - window:
            candidates.popleft()
        if index >= first_full_window_end:
            selected.setdefault(hashes[candidates[0]], None)
    return tuple(selected)
