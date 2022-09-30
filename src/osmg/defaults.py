"""
Model Generator for OpenSees ~ defaults
"""

#                          __
#   ____  ____ ___  ____ _/ /
#  / __ \/ __ `__ \/ __ `/ /
# / /_/ / / / / / / /_/ /_/
# \____/_/ /_/ /_/\__, (_)
#                /____/
#
# https://github.com/ioannis-vm/OpenSees_Model_Generator

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import numpy.typing as npt
from .physical_material import PhysicalMaterial
from . import common
from .ops.uniaxial_material import Elastic
from .ops.uniaxial_material import Steel02
from .ops.section import ElasticSection
from .gen.mesh_shapes import rect_mesh
if TYPE_CHECKING:
    from .model import Model

nparr = npt.NDArray[np.float64]


def load_util_rigid_elastic(model: Model):
    """
    Adds a default rigid elastic beamcolumn element
    to the model
    """
    new_uid = model.uid_generator.new('section')
    sec = ElasticSection(
        name='rigid_link_section',
        uid=new_uid,
        outside_shape=None,
        snap_points=None,
        e_mod=common.STIFF,
        area=1.00,
        i_y=1.00,
        i_x=1.00,
        g_mod=common.STIFF,
        j_mod=1.00,
        sec_w=0.00)
    model.elastic_sections.add(sec)


def load_default_elastic(model: Model, sec_name: str):
    """
    Adds default non-rigid elastic beamcolumn element
    sections to the model
    """
    new_uid = model.uid_generator.new('section')
    sec = ElasticSection(
        name=sec_name,
        uid=new_uid,
        outside_shape=None,
        snap_points=None,
        e_mod=1.00,
        area=1.00,
        i_y=1.00,
        i_x=1.00,
        g_mod=1.00,
        j_mod=1.00,
        sec_w=0.00)
    if model.settings.imperial_units:
        y_max = +10.00
        y_min = -10.00
        z_max = +6.00
        z_min = -6.00
        sec.outside_shape = rect_mesh(12.0, 20.0)
    else:
        y_max = +0.25
        y_min = -0.25
        z_max = +0.15
        z_min = -0.15
        sec.outside_shape = rect_mesh(0.30, 0.50)
    snap_points: dict[str, nparr] = {
        'centroid': np.array([0., 0.]),
        'top_center': np.array([0., -y_max]),
        'top_left': np.array([-z_min, -y_max]),
        'top_right': np.array([-z_max, -y_max]),
        'center_left': np.array([-z_min, 0.]),
        'center_right': np.array([-z_max, 0.]),
        'bottom_center': np.array([0., -y_min]),
        'bottom_left': np.array([-z_min, -y_min]),
        'bottom_right': np.array([-z_max, -y_min])
    }
    sec.snap_points = snap_points
    model.elastic_sections.add(sec)


def load_default_steel(model: Model):
    """
    Adds a default steel material to the model
    """
    # TODO: add SI units some day
    uniaxial_mat = Steel02(
        model.uid_generator.new('uniaxial material'), 'default steel',
        55000.00, 29000000.00, 11153846.15, 0.01, 15.0, 0.925, 0.15)
    physical_mat = PhysicalMaterial(
        model.uid_generator.new('physical material'),
        'default steel',
        'A992-Fy50',
        0.2835648148148148/common.G_CONST_IMPERIAL,
        29000000.00,
        11153846.15,
        55000.00)
    model.uniaxial_materials.add(uniaxial_mat)
    model.physical_materials.add(physical_mat)


def load_default_fix_release(model: Model):
    """
    Loads default fix and release elastic uniaxial materials
    used to simulate moment releases using zerolength elements.
    """
    uniaxial_mat = Elastic(
        uid=model.uid_generator.new('uniaxial material'),
        name='fix',
        e_mod=common.STIFF_ROT)
    model.uniaxial_materials.add(uniaxial_mat)
    uniaxial_mat = Elastic(
        uid=model.uid_generator.new('uniaxial material'),
        name='release',
        e_mod=common.TINY)
    model.uniaxial_materials.add(uniaxial_mat)
