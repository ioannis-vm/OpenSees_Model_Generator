"""Unit tests for UIDObject."""

from dataclasses import dataclass

from osmg.core.uid_object import UIDObject
from osmg.creators.uid import UIDGenerator


@dataclass
class _TestChild(UIDObject):
    """Test child class inheriting from UIDObject."""

    name: str


class TestUIDObject:
    """Unit tests for the UIDObject class."""

    def setup_method(self) -> None:
        """Set up test environment."""
        self.uid_generator = UIDGenerator()

    def test_uid_generation(self) -> None:
        """Test that a UID is generated and assigned correctly."""
        child1 = _TestChild(self.uid_generator, name='child1')
        child2 = _TestChild(self.uid_generator, name='child2')

        assert child1.uid == 0, 'UID for the first object should be 0.'
        assert child2.uid == 1, 'UID for the second object should be 1.'

    def test_unique_uid(self) -> None:
        """Test that each object gets a unique UID."""
        child1 = _TestChild(self.uid_generator, name='child1')
        child2 = _TestChild(self.uid_generator, name='child2')
        child3 = _TestChild(self.uid_generator, name='child3')

        uids = {child1.uid, child2.uid, child3.uid}
        assert len(uids) == 3, 'Each object should have a unique UID.'

    def test_post_init_uid_assignment(self) -> None:
        """Test that UID is assigned during post-init."""
        child = _TestChild(self.uid_generator, name='test_child')
        assert hasattr(child, 'uid'), "UIDObject should have a 'uid' attribute."
        assert child.uid == 0, 'UID should be correctly assigned during post-init.'
