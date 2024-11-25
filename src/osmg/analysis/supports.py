"""Node supports."""

from __future__ import annotations


class FixedSupport(tuple[bool, ...]):
    """Fixed support."""

    __slots__: list[str] = []


class ElasticSupport(tuple[float, ...]):
    """Flexible support."""

    __slots__: list[str] = []
