"""Interval algebra over (start, end) second pairs.

Every list handled here is kept normalised: sorted, non-overlapping, and with
touching intervals fused. All operations return new lists.
"""

from __future__ import annotations

EPS = 1e-6

Interval = tuple[float, float]


def normalize(intervals, gap: float = 0.0) -> list[Interval]:
    """Sort, drop empties, and fuse intervals separated by <= gap."""
    items = sorted(
        (float(s), float(e)) for s, e in intervals if float(e) - float(s) > EPS
    )
    out: list[Interval] = []
    for start, end in items:
        if out and start - out[-1][1] <= gap + EPS:
            if end > out[-1][1]:
                out[-1] = (out[-1][0], end)
        else:
            out.append((start, end))
    return out


def union(*groups) -> list[Interval]:
    merged: list[Interval] = []
    for group in groups:
        merged.extend(group)
    return normalize(merged)


def intersect(a, b) -> list[Interval]:
    a, b = normalize(a), normalize(b)
    out: list[Interval] = []
    i = j = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end - start > EPS:
            out.append((start, end))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def subtract(a, b) -> list[Interval]:
    """Everything in a that is not in b."""
    a, b = normalize(a), normalize(b)
    out: list[Interval] = []
    for start, end in a:
        cursor = start
        for bs, be in b:
            if be <= cursor + EPS:
                continue
            if bs >= end - EPS:
                break
            if bs > cursor + EPS:
                out.append((cursor, min(bs, end)))
            cursor = max(cursor, be)
            if cursor >= end - EPS:
                break
        if end - cursor > EPS:
            out.append((cursor, end))
    return normalize(out)


def complement(intervals, lo: float, hi: float) -> list[Interval]:
    return subtract([(lo, hi)], intervals)


def pad(intervals, amount: float, lo: float, hi: float) -> list[Interval]:
    """Grow every interval by amount on both sides, clamped to [lo, hi]."""
    return normalize(
        (max(lo, s - amount), min(hi, e + amount)) for s, e in intervals
    )


def shrink(intervals, amount: float) -> list[Interval]:
    """Contract every interval by amount on both sides; collapsed ones vanish."""
    return normalize((s + amount, e - amount) for s, e in intervals)


def total(intervals) -> float:
    return sum(e - s for s, e in intervals)


def overlaps(interval: Interval, intervals, tolerance: float = 0.0) -> bool:
    start, end = interval
    for s, e in intervals:
        if s >= end - tolerance:
            break
        if e > start + tolerance:
            return True
    return False


def overlap_amount(interval: Interval, intervals) -> float:
    return total(intersect([interval], intervals))


def contains(intervals, t: float) -> bool:
    return any(s - EPS <= t <= e + EPS for s, e in intervals)


class Timeline:
    """Maps original timestamps onto the rendered timeline.

    Built from the list of kept intervals. Times inside a removed region
    collapse onto the splice point, and `dropped()` reports whether a given
    instant survives at all.
    """

    def __init__(self, keep):
        self.keep = normalize(keep)
        self._offsets: list[float] = []
        elapsed = 0.0
        for start, end in self.keep:
            self._offsets.append(elapsed)
            elapsed += end - start
        self.duration = elapsed

    def dropped(self, t: float) -> bool:
        return not contains(self.keep, t)

    def map(self, t: float) -> float:
        """Original time -> output time (monotonic, never negative)."""
        if not self.keep:
            return 0.0
        if t <= self.keep[0][0]:
            return 0.0
        for (start, end), offset in zip(self.keep, self._offsets):
            if t < start:
                return offset
            if t <= end:
                return offset + (t - start)
        return self.duration

    def map_span(self, start: float, end: float) -> tuple[float, float]:
        return self.map(start), self.map(max(start, end))
