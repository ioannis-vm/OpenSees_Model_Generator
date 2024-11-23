"""Unit tests for Vertex, Edge, and Halfedge classes."""

import numpy as np
import pytest

from osmg.geometry.mesh import Edge, Halfedge, Vertex


class TestVertex:
    """Tests for the Vertex class."""

    def test_vertex_initialization(self) -> None:
        """Test that a Vertex initializes correctly."""
        v = Vertex((0.0, 0.0))
        assert v.coordinates == (0.0, 0.0)
        assert v.edges == []
        assert v.halfedges == []
        assert isinstance(v.uid, int)

    def test_vertex_equality(self) -> None:
        """Test equality comparisons for Vertex objects."""
        v1 = Vertex((0, 0))
        v2 = Vertex((1, 1))
        v3 = Vertex((0, 0))
        assert v1 == v1  # noqa: PLR0124 Same object
        assert v1 != v2  # Different vertices
        assert v1 != v3  # Different UIDs despite same coordinates

    def test_vertex_hash(self) -> None:
        """Test that Vertex objects have unique hash values."""
        v1 = Vertex((0, 0))
        v2 = Vertex((1, 1))
        assert hash(v1) != hash(v2)


class TestEdge:
    """Tests for the Edge class."""

    def setup_method(self) -> None:
        """Set up vertices for Edge tests."""
        self.v1 = Vertex((0.0, 0.0))
        self.v2 = Vertex((1.0, 1.0))
        self.v3 = Vertex((2.0, 2.0))

    def test_edge_initialization(self) -> None:
        """Test that an Edge initializes correctly."""
        e = Edge(self.v1, self.v2)
        assert e.v_i == self.v1
        assert e.v_j == self.v2
        assert e in self.v1.edges
        assert e in self.v2.edges

    def test_edge_repr(self) -> None:
        """Test the string representation of an Edge."""
        e = Edge(self.v1, self.v2)
        assert repr(e) == f'(E{e.uid} @ V{self.v1.uid}, V{self.v2.uid}) '

    def test_define_halfedge(self) -> None:
        """Test defining halfedges for an Edge."""
        e = Edge(self.v1, self.v2)
        h1 = e.define_halfedge(self.v1)
        h2 = e.define_halfedge(self.v2)
        assert h1.vertex == self.v1
        assert h2.vertex == self.v2
        assert e.h_i == h1
        assert e.h_j == h2

        # Test ValueError if halfedge is already defined
        with pytest.raises(ValueError, match='Halfedge h_i already defined'):
            e.define_halfedge(self.v1)

    def test_other_vertex(self) -> None:
        """Test getting the other vertex of an Edge."""
        e = Edge(self.v1, self.v2)
        assert e.other_vertex(self.v1) == self.v2
        assert e.other_vertex(self.v2) == self.v1

        # Test ValueError if vertex is not part of the Edge
        with pytest.raises(
            ValueError, match='The edge is not connected to the given vertex'
        ):
            e.other_vertex(self.v3)


class TestHalfedge:
    """Tests for the Halfedge class."""

    def setup_method(self) -> None:
        """Set up vertices and edges for Halfedge tests."""
        self.v1 = Vertex((0.0, 0.0))
        self.v2 = Vertex((1.0, 1.0))
        self.edge = Edge(self.v1, self.v2)

    def test_halfedge_initialization(self) -> None:
        """Test that a Halfedge initializes correctly."""
        h = Halfedge(self.v1, self.edge)
        assert h.vertex == self.v1
        assert h.edge == self.edge
        assert h.nxt is None

    def test_halfedge_repr(self) -> None:
        """Test the string representation of a Halfedge."""
        h1 = Halfedge(self.v1, self.edge)
        h2 = Halfedge(self.v2, self.edge)
        h1.nxt = h2
        assert (
            repr(h1)
            == f'(H{h1.uid} from E{h1.edge.uid} to E{h2.edge.uid} next H{h2.uid})'
        )

    def test_halfedge_direction(self) -> None:
        """Test the direction calculation for a Halfedge."""
        h = Halfedge(self.v1, self.edge)
        direction = h.direction()
        expected_direction = np.arctan2(1.0, 1.0)  # From (0, 0) to (1, 1)
        assert direction == pytest.approx(expected_direction)
