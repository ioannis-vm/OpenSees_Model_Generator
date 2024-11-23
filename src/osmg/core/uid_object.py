"""
UID Object.

Parent class for all objects that have a unique identifier (UID).
"""

from dataclasses import dataclass

from osmg.creators.uid import UIDGenerator


@dataclass
class UIDObject:
    """Base class for objects with a unique identifier (UID)."""

    uid_generator: UIDGenerator

    def __post_init__(self) -> None:
        """Post-initialization."""
        self.uid = self.uid_generator.new(self)
