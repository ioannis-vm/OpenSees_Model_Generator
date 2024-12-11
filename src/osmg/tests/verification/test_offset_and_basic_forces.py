"""
Offset: basic force verification.

This simple verification check ensures that basic forces are correctly
assigned using a single beamcolumn element in a variety of
arrangements.

Written: Thu Nov 28 06:45:08 PM PST 2024

"""

from __future__ import annotations

import numpy as np

from osmg.analysis.common import PointLoad
from osmg.analysis.load_case import LoadCaseRegistry
from osmg.analysis.supports import FixedSupport
from osmg.core.model import Model2D, Model3D
from osmg.creators.component import BeamColumnCreator
from osmg.graphics.plotly import (
    BasicForceConfiguration,
    DeformationConfiguration,
    Figure3D,
    Figure3DConfiguration,
)
from osmg.model_objects.node import Node
from osmg.model_objects.section import ElasticSection


def no_offset_2d() -> None:
    """Axially loaded cantilever column, 2D, no offset."""
    column_model = Model2D(name='Test model', dimensionality='2D Frame')
    n1 = Node(uid_generator=column_model.uid_generator, coordinates=((0.00, 0.00)))
    n2 = Node(uid_generator=column_model.uid_generator, coordinates=((0.00, 5.00)))
    column_model.nodes.add(n1)
    column_model.nodes.add(n2)
    simple_section = ElasticSection(
        column_model.uid_generator,
        'Test Section',
        e_mod=1e3,
        area=1e3,
        i_y=1.00,
        i_x=1.00,
        g_mod=1.00,
        j_mod=1.00,
        sec_w=0.00,
    )
    bcg = BeamColumnCreator(column_model, 'elastic')
    bcg.generate_plain_component_assembly(
        tags={'column'},
        node_i=n2,
        node_j=n1,
        n_sub=1,
        eo_i=np.array((0.00, 0.00)),
        eo_j=np.array((0.00, 0.00)),
        section=simple_section,
        transf_type='Linear',
    )
    load_case_registry = LoadCaseRegistry(column_model)
    fixed_support = FixedSupport((True, True, True))
    load_case_registry.dead['test'].fixed_supports[n1.uid] = fixed_support
    load_case_registry.dead['test'].load_registry.nodal_loads[n2.uid] = PointLoad(
        (0.00, -10.00, 0.00)  # lb
    )
    load_case_registry.run()

    axial_df, shear_y_df, shear_z_df, torsion_df, moment_y_df, moment_z_df = (
        load_case_registry.dead['test'].calculate_basic_forces(
            'default_basic_force',
            column_model.components.get_line_element_lengths(),
            ndm=2,
            num_stations=12,
        )
    )
    assert np.allclose(axial_df.to_numpy(), -10.00)
    assert np.allclose(shear_y_df.to_numpy(), 0.00)
    assert np.allclose(shear_z_df.to_numpy(), 0.00)
    assert np.allclose(torsion_df.to_numpy(), 0.00)
    assert np.allclose(moment_y_df.to_numpy(), 0.00)
    assert np.allclose(moment_z_df.to_numpy(), 0.00)

    displacements = (
        load_case_registry.dead['test'].analysis.recorders['default_node'].get_data()
    )
    assert np.allclose(
        displacements.to_numpy(), np.array((0.0, 0.0, 0.0, 0.0, -5.0e-05, 0.0))
    )
    # plot(column_model, load_case_registry)  # appears correct.


def offset_2d() -> None:
    """Axially loaded cantilever column, 2D, offset."""
    column_model = Model2D(name='Test model', dimensionality='2D Frame')
    n1 = Node(uid_generator=column_model.uid_generator, coordinates=((0.00, 0.00)))
    n2 = Node(uid_generator=column_model.uid_generator, coordinates=((0.00, 5.00)))
    column_model.nodes.add(n1)
    column_model.nodes.add(n2)
    simple_section = ElasticSection(
        column_model.uid_generator,
        'Test Section',
        e_mod=1e3,
        area=1e3,
        i_y=1.00,
        i_x=1.00,
        g_mod=1.00,
        j_mod=1.00,
        sec_w=0.00,
    )
    bcg = BeamColumnCreator(column_model, 'elastic')
    bcg.generate_plain_component_assembly(
        tags={'column'},
        node_i=n2,
        node_j=n1,
        n_sub=1,
        eo_i=np.array((1.00, 0.00)),
        eo_j=np.array((1.00, 0.00)),
        section=simple_section,
        transf_type='Linear',
    )
    load_case_registry = LoadCaseRegistry(column_model)
    fixed_support = FixedSupport((True, True, True))
    load_case_registry.dead['test'].fixed_supports[n1.uid] = fixed_support
    load_case_registry.dead['test'].load_registry.nodal_loads[n2.uid] = PointLoad(
        (0.00, -10.00, 0.00)  # lb
    )
    load_case_registry.run()

    axial_df, shear_y_df, shear_z_df, torsion_df, moment_y_df, moment_z_df = (
        load_case_registry.dead['test'].calculate_basic_forces(
            'default_basic_force',
            column_model.components.get_line_element_lengths(),
            ndm=2,
            num_stations=12,
        )
    )
    assert np.allclose(axial_df.to_numpy(), -10.00)
    assert np.allclose(shear_y_df.to_numpy(), 0.00)
    assert np.allclose(shear_z_df.to_numpy(), 0.00)
    assert np.allclose(torsion_df.to_numpy(), 0.00)
    assert np.allclose(moment_y_df.to_numpy(), 0.00)
    assert np.allclose(moment_z_df.to_numpy(), -10.00)

    displacements = (
        load_case_registry.dead['test'].analysis.recorders['default_node'].get_data()
    )
    assert np.allclose(
        displacements.to_numpy(), np.array((0.0, 0.0, 0.0, -0.125, -0.05005, 0.05))
    )
    # plot(column_model, load_case_registry)  # appears correct.


def no_offset_3d() -> None:
    """Axially loaded cantilever column, 3D, no offset."""
    column_model = Model3D(name='Test model', dimensionality='3D Frame')
    n1 = Node(
        uid_generator=column_model.uid_generator, coordinates=((0.00, 0.00, 0.00))
    )
    n2 = Node(
        uid_generator=column_model.uid_generator, coordinates=((0.00, 0.00, 5.00))
    )
    column_model.nodes.add(n1)
    column_model.nodes.add(n2)
    simple_section = ElasticSection(
        column_model.uid_generator,
        'Test Section',
        e_mod=1e3,
        area=1e3,
        i_y=1.00,
        i_x=1.00,
        g_mod=1.00,
        j_mod=1.00,
        sec_w=0.00,
    )
    bcg = BeamColumnCreator(column_model, 'elastic')
    bcg.generate_plain_component_assembly(
        tags={'column'},
        node_i=n2,
        node_j=n1,
        n_sub=1,
        eo_i=np.array((0.00, 0.00, 0.0)),
        eo_j=np.array((0.00, 0.00, 0.0)),
        section=simple_section,
        transf_type='Linear',
        angle=90.00 / 360.00 * 2.00 * np.pi,
    )
    load_case_registry = LoadCaseRegistry(column_model)
    fixed_support = FixedSupport((True, True, True, True, True, True))
    load_case_registry.dead['test'].fixed_supports[n1.uid] = fixed_support
    load_case_registry.dead['test'].load_registry.nodal_loads[n2.uid] = PointLoad(
        (0.00, 0.00, -10.00, 0.00, 0.00, 0.00)  # lb
    )
    load_case_registry.run()
    axial_df, shear_y_df, shear_z_df, torsion_df, moment_y_df, moment_z_df = (
        load_case_registry.dead['test'].calculate_basic_forces(
            'default_basic_force',
            column_model.components.get_line_element_lengths(),
            ndm=3,
            num_stations=12,
        )
    )
    assert np.allclose(axial_df.to_numpy(), -10.00)
    assert np.allclose(shear_y_df.to_numpy(), 0.00)
    assert np.allclose(shear_z_df.to_numpy(), 0.00)
    assert np.allclose(torsion_df.to_numpy(), 0.00)
    assert np.allclose(moment_y_df.to_numpy(), 0.00)
    assert np.allclose(moment_z_df.to_numpy(), 0.00)

    displacements = (
        load_case_registry.dead['test'].analysis.recorders['default_node'].get_data()
    )
    assert np.allclose(
        displacements.to_numpy(),
        np.array((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -5.0e-05, 0.0, 0.0, 0.0)),
    )
    plot(column_model, load_case_registry)  # appears correct.


def offset_3d() -> None:
    """Axially loaded cantilever column, 3D, offset."""
    column_model = Model3D(name='Test model', dimensionality='3D Frame')
    n1 = Node(
        uid_generator=column_model.uid_generator, coordinates=((0.00, 0.00, 0.00))
    )
    n2 = Node(
        uid_generator=column_model.uid_generator, coordinates=((0.00, 0.00, 5.00))
    )
    column_model.nodes.add(n1)
    column_model.nodes.add(n2)
    simple_section = ElasticSection(
        column_model.uid_generator,
        'Test Section',
        e_mod=1e3,
        area=1e3,
        i_y=1.00,
        i_x=1.00,
        g_mod=1.00,
        j_mod=1.00,
        sec_w=0.00,
    )
    bcg = BeamColumnCreator(column_model, 'elastic')
    bcg.generate_plain_component_assembly(
        tags={'column'},
        node_i=n2,
        node_j=n1,
        n_sub=1,
        eo_i=np.array((1.00, 0.00, 0.0)),
        eo_j=np.array((1.00, 0.00, 0.0)),
        section=simple_section,
        transf_type='Linear',
        angle=90.00 / 360.00 * 2.00 * np.pi,
    )
    load_case_registry = LoadCaseRegistry(column_model)
    fixed_support = FixedSupport((True, True, True, True, True, True))
    load_case_registry.dead['test'].fixed_supports[n1.uid] = fixed_support
    load_case_registry.dead['test'].load_registry.nodal_loads[n2.uid] = PointLoad(
        (0.00, 0.00, -10.00, 0.00, 0.00, 0.00)  # lb
    )
    load_case_registry.run()
    axial_df, shear_y_df, shear_z_df, torsion_df, moment_y_df, moment_z_df = (
        load_case_registry.dead['test'].calculate_basic_forces(
            'default_basic_force',
            column_model.components.get_line_element_lengths(),
            ndm=3,
            num_stations=12,
        )
    )
    assert np.allclose(axial_df.to_numpy(), -10.00)
    assert np.allclose(shear_y_df.to_numpy(), 0.00)
    assert np.allclose(shear_z_df.to_numpy(), 0.00)
    assert np.allclose(torsion_df.to_numpy(), 0.00)
    assert np.allclose(moment_y_df.to_numpy(), 0.00)
    assert np.allclose(moment_z_df.to_numpy(), 10.00)

    displacements = (
        load_case_registry.dead['test'].analysis.recorders['default_node'].get_data()
    )
    assert np.allclose(
        displacements.to_numpy(),
        np.array(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.125, 0.0, -0.05005, 0.0, -0.05, 0)
        ),
    )
    # plot(column_model, load_case_registry)  # appears correct.


def plot(model: Model2D | Model3D, load_case_registry: LoadCaseRegistry) -> None:
    """Plot the deformed shape and basic forces."""
    if isinstance(model, Model2D):
        ndm = 2
        ndf = 3
    else:
        ndm = 3
        ndf = 6
    deformation_configuration = DeformationConfiguration(
        reference_length=model.reference_length(),
        ndf=ndf,
        ndm=ndm,
        data=load_case_registry.dead['test']
        .analysis.recorders['default_node']
        .get_data(),
        step=0,
        amplification_factor=None,  # Figure it out.
    )
    basic_force_configuration = BasicForceConfiguration(
        reference_length=model.reference_length(),
        ndf=ndf,
        ndm=ndm,
        data=load_case_registry.dead['test'].calculate_basic_forces(
            'default_basic_force',
            model.components.get_line_element_lengths(),
            ndm=ndm,
            num_stations=12,
        ),
        step=-1,
        force_to_length_factor=0.1,
        moment_to_length_factor=0.1,
    )

    fig = Figure3D(Figure3DConfiguration(ndm=ndm))  # type: ignore
    fig.add_nodes(list(model.nodes.values()), 'primary', overlay=True)
    fig.add_components(list(model.components.values()), overlay=True)
    fig.add_nodes(list(model.nodes.values()), 'primary', deformation_configuration)
    fig.add_components(list(model.components.values()), deformation_configuration)
    fig.add_supports(
        model.nodes,
        load_case_registry.dead['test'].fixed_supports,
        symbol_size=12.00 / 120.0,
    )
    fig.add_udl(
        load_case_registry.dead['test'].load_registry.component_udl,
        model.components,
        force_to_length_factor=0.10,
        offset=0.00,
    )
    fig.add_loads(
        load_case_registry.dead['test'].load_registry.nodal_loads,
        model.nodes,
        force_to_length_factor=0.10,
        offset=0.0,
        head_length=24.0 / 120.0,
        head_width=24.0 / 120.0,
        base_width=5.0 / 120.0,
    )
    fig.add_basic_forces(
        components=list(model.components.values()),
        basic_force_configuration=basic_force_configuration,
    )
    fig.show()
