"""
Introduces some default objects to a model.

"""

#
#   _|_|      _|_|_|  _|      _|    _|_|_|
# _|    _|  _|        _|_|  _|_|  _|
# _|    _|    _|_|    _|  _|  _|  _|  _|_|
# _|    _|        _|  _|      _|  _|    _|
#   _|_|    _|_|_|    _|      _|    _|_|_|
#
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
from .gen.section_gen import SectionGenerator


if TYPE_CHECKING:
    from .model import Model

nparr = npt.NDArray[np.float64]


def load_util_rigid_elastic(model: Model) -> None:
    """
    Adds a default rigid elastic beamcolumn element
    to the model.

    """

    new_uid = model.uid_generator.new("section")
    sec = ElasticSection(
        name="rigid_link_section",
        uid=new_uid,
        outside_shape=None,
        snap_points=None,
        e_mod=common.STIFF,
        area=1.00,
        i_y=1.00,
        i_x=1.00,
        g_mod=common.STIFF,
        j_mod=1.00,
        sec_w=0.00,
    )
    model.elastic_sections.add(sec)


def load_default_elastic(model: Model, sec_name: str) -> None:
    """
    Adds default non-rigid elastic beamcolumn element
    sections to the model.

    """

    # intantiate a section generator object for the model
    sgen = SectionGenerator(model)
    # generate a default elastic section and add it to the model
    sgen.generate_generic_elastic(
        name=sec_name,
        e_times_a=1.00,
        e_times_i=1.00,
        g_times_j=1.00
    )


def load_default_steel(model: Model) -> None:
    """
    Adds a default steel material to the model.

    Note: If different properties are required, the values provided
    here can be altered, or the materials can be defined at the
    user-side and added to the model instead of calling this method.

    """

    if model.settings.imperial_units:
        # force: lb, length: in
        uniaxial_mat = Steel02(
            model.uid_generator.new("uniaxial material"),
            "default steel",
            55000.00,
            29000000.00,
            11153846.15,
            0.01,
            15.0,
            0.925,
            0.15,
        )
        physical_mat = PhysicalMaterial(
            model.uid_generator.new("physical material"),
            "default steel",
            "A992-Fy50",
            0.2835648148148148 / common.G_CONST_IMPERIAL,
            29000000.00,
            11153846.15,
            55000.00,
        )
    else:
        # force: kN, length: m, but it's just the values above,
        # converted to SI, instead of the properties of 'typical'
        # european steel.
        uniaxial_mat = Steel02(
            model.uid_generator.new("uniaxial material"),
            "default steel",
            379211.65,
            199947961.12,
            76903062.09,
            0.01,
            15.0,
            0.925,
            0.15,
        )
        physical_mat = PhysicalMaterial(
            model.uid_generator.new("physical material"),
            "default steel",
            "A992-Fy50",
            773.37062406 / common.G_CONST_IMPERIAL,
            199947961.12,
            76903062.09,
            379211.65,
        )
    model.uniaxial_materials.add(uniaxial_mat)
    model.physical_materials.add(physical_mat)


def load_default_fix_release(model: Model) -> None:
    """
    Loads default fix and release elastic uniaxial materials
    used to simulate moment releases using zerolength elements.

    """

    uniaxial_mat = Elastic(
        uid=model.uid_generator.new("uniaxial material"),
        name="fix",
        e_mod=common.STIFF_ROT,
    )
    model.uniaxial_materials.add(uniaxial_mat)
    uniaxial_mat = Elastic(
        uid=model.uid_generator.new("uniaxial material"),
        name="release",
        e_mod=common.TINY,
    )
    model.uniaxial_materials.add(uniaxial_mat)
