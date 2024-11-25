"""Unit tests for supports."""

from osmg.analysis.supports import ElasticSupport, FixedSupport


def test_fixed_support_initialization() -> None:
    """Test initialization of FixedSupport.

    Ensures that FixedSupport is correctly initialized with the
    specified degrees of freedom restraints.
    """
    support = FixedSupport((True, False, True))
    assert isinstance(support, FixedSupport)
    assert support == FixedSupport((True, False, True))


def test_elastic_support_initialization() -> None:
    """Test initialization of ElasticSupport.

    Ensures that ElasticSupport is correctly initialized with the
    specified degrees of freedom restraints.
    """
    support = ElasticSupport((10.0, 5.0, 0.0))
    assert isinstance(support, ElasticSupport)
    assert support == ElasticSupport((10.0, 5.0, 0.0))
