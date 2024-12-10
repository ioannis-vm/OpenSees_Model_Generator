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
        self._uid = self.uid_generator.new(self)

    @property
    def uid(self) -> int:
        """Get the UID."""
        return self._uid

    def __hash__(self) -> int:
        """Return the hash of the object based on its UID."""
        return hash(self.uid)

    def __eq__(self, other: object) -> bool:
        """
        Check equality based on the UID.

        Returns:
          True if it is equal, False otherwise.
        """
        if not isinstance(other, UIDObject):
            return False
        return self.uid == other.uid
