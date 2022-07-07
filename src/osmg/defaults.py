"""
Some defaults
"""
from __future__ import annotations
from typing import TYPE_CHECKING
from .physical_material import PhysicalMaterial
from . import common
from .ops.uniaxialMaterial import Steel02
from .ops.section import ElasticSection
from .gen.mesh_shapes import rect_mesh
import numpy as np
if TYPE_CHECKING:
    from .model import Model


def load_default_elastic(model: Model, sec_name: str):
    new_uid = model.uid_generator.new('section')
    sec = ElasticSection(
        name=sec_name,
        uid=new_uid,
        outside_shape=None,
        snap_points=None,
        E=1.00,
        A=1.00,
        Iy=1.00,
        Ix=1.00,
        G=1.00,
        J=1.00,
        W=0.00)
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
    snap_points = {
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
    # todo: add SI units some day
    uniaxial_mat = Steel02(
        uid=model.uid_generator.new('uniaxial material'),
        name='default steel',
        Fy=55000.00,
        E0=29000000.00,
        b=0.01,
        params=(19.0, 0.925, 0.15),
        a1=0.12,
        a2=0.90,
        a3=0.18,
        a4=0.90,
        sigInit=0.0,
        G=11153846.15)
    physical_mat = PhysicalMaterial(
        model.uid_generator.new('physical material'),
        'default steel',
        'A992-Fy50',
        0.2835648148148148/common.G_CONST_IMPERIAL,
        29000000.00,
        11153846.15)
    model.uniaxial_materials.add(uniaxial_mat)
    model.physical_materials.add(physical_mat)
