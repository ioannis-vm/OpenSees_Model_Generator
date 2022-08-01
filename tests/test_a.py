"""
Basic Tests
"""

# import pytest
from osmg.model import Model
from osmg.gen.node_gen import NodeGenerator
from osmg.graphics.preprocessing_3D import show


def test_add_levels():
    """
    test adding levels
    """
    mdl = Model('test_mdl')
    mdl.add_level(0, 0.00)
    mdl.add_level(1, 3.00)


def test_add_node():
    """
    test adding nodes
    """
    mdl = Model('test_mdl')
    mdl.add_level(0, 0.00)
    mdl.add_level(1, 3.00)
    mdl.add_level(2, 6.00)
    mdl.levels.set_active([0, 1, 2])
    ndg = NodeGenerator(mdl)
    ndg.add_node_lvl(0.0, 0.0, 1)
    ndg.add_node_lvl(0.0, 0.0, 2)
    show(mdl)


if __name__ == '__main__':
    test_add_node()
