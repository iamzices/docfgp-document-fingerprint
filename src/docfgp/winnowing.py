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
    queue: deque[int] = deque()
    selected: dict[str, None] = {}
    for index, value in enumerate(hashes):
        while queue and hashes[queue[-1]] >= value:
            queue.pop()
        queue.append(index)
        while queue and queue[0] <= index - window:
            queue.popleft()
        if index >= min(window, len(hashes)) - 1:
            selected.setdefault(hashes[queue[0]], None)
    return tuple(selected)
