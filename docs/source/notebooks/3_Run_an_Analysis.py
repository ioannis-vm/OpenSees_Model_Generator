# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Running an analysis

The `solver` module is used to interact with OpenSees.

The most convenient approach is using a child of the `Analysis` class from those already defined in the `solver` module for the specific analysis that is needed. This will run the analysis and store the requested results of each analysis step.

Alternatively, a generic `Analysis` object can be defined (from the parent class), and its generic methods for model definition can be used together with OpenSees commands issued directly in the analysis script. If such an analysis is repeated a lot, a new `Analysis` child class can be defined in the `solver` module.
"""

# %% [markdown]
"""
## Analysis examples

First, we need to define a model.

"""


# %%
# This cell defines the model from notebook 2_Define_a_Model
import numpy as np
from osmg import model
import osmg.defaults as defaults
from osmg.gen.section_gen import SectionGenerator
from osmg.ops.section import ElasticSection
from osmg.gen.component_gen import BeamColumnGenerator
from osmg.ops.element import ElasticBeamColumn
from osmg.gen.zerolength_gen import gravity_shear_tab
from osmg.load_case import LoadCase
from osmg.preprocessing.self_weight_mass import self_weight
from osmg.preprocessing.self_weight_mass import self_mass
from osmg.graphics.preprocessing_3d import show
mdl = model.Model('example_model')
for i in range(3):
    mdl.add_level(i, 144.00*(i))
defaults.load_default_steel(mdl)
defaults.load_default_fix_release(mdl)
defaults.load_util_rigid_elastic(mdl)
steel_phys_mat = mdl.physical_materials.retrieve_by_attr('name', 'default steel')
secg = SectionGenerator(mdl)
secg.load_aisc_from_database(
    'W',
    ["W24X94"],
    'default steel',
    'default steel',
    ElasticSection)
mdl.levels.set_active([1, 2])
p1 = np.array((0.00, 0.00))
p2 = np.array((360., 0.00))
p3 = np.array((360., 360.))
p4 = np.array((0.00, 360.00))
mcg = BeamColumnGenerator(mdl)
sec = mdl.elastic_sections.retrieve_by_attr('name', 'W24X94')
for pt in [p1, p2, p3, p4]:
    mcg.add_pz_active(
        pt[0], pt[1],
        sec,
        steel_phys_mat,
        0.00,
        24.00,
        24.00,
        0.00,
        0.02)
    mcg.add_vertical_active(
        x_coord=pt[0], y_coord=pt[1],
        offset_i=np.zeros(3), offset_j=np.zeros(3),
        transf_type='Corotational',
        n_sub=1,
        section=sec,
        element_type=ElasticBeamColumn,
        placement='centroid',
        angle=0.00)
snap_i_list = ['centroid', 'middle_front', 'centroid', 'middle_back']
snap_j_list = ['centroid', 'middle_back', 'centroid', 'middle_front']
for i, pair in enumerate([(p1, p2), (p2, p3), (p3, p4), (p4, p1)]):
    mcg.add_horizontal_active(
        xi_coord=pair[0][0],
        yi_coord=pair[0][1],
        xj_coord=pair[1][0],
        yj_coord=pair[1][1],
        offset_i=np.zeros(3),
        offset_j=np.zeros(3),
        snap_i=snap_i_list[i],
        snap_j=snap_j_list[i],
        transf_type='Linear',
        n_sub=4,
        section=sec,
        element_type=ElasticBeamColumn,
        placement='top_center',
        angle=0.00,
        method='generate_hinged_component_assembly',
        additional_args={
            'zerolength_gen_i': gravity_shear_tab,
            'zerolength_gen_args_i': {
                'consider_composite': True,
                'section': sec,
                'physical_material': steel_phys_mat,
                'distance': 10.00,
                'n_sub': 2
            },
            'zerolength_gen_j': gravity_shear_tab,
            'zerolength_gen_args_j': {
                'consider_composite': True,
                'section': sec,
                'physical_material': steel_phys_mat,
                'distance': 10.00,
                'n_sub': 2
            }
        }
    )
for node in mdl.levels[0].nodes.values():
    node.restraint = [True]*6
testcase = LoadCase('test', mdl)
self_weight(mdl, testcase)
self_mass(mdl, testcase)
testcase.rigid_diaphragms([1, 2])


# %%
show(mdl, testcase, extrude=True)

# %% [markdown]
"""
### Static Analysis
"""

# %%
from osmg import solver


# %%
# instantiate analysis object
static_anl = solver.StaticAnalysis(mdl, {testcase.name: testcase})


# %%
static_anl.run()


# %% [markdown]
"""
#### Retrieving results

Analysis results are stored in dictionaries. The keys are the unique identifiers of the elements that the results correspond to, and the values contain the results.

"""

# %% [markdown]
"""

Show all node displacement results

"""

# %%
# DANGER: Don't do this for a large model.
static_anl.results[testcase.name].node_displacements.items()


# %%
# Displacement of the parent node of the top story,
# in the Y direction.
analysis_step = 0
direction = 1
node_id = testcase.parent_nodes[2].uid
static_anl.results[testcase.name].node_displacements[
    node_id][analysis_step][direction]


# %% [markdown]
"""

Note: Multiple load cases and analysis objects can be defined using the same model. The results will be stored independently in the various analysis objects.

"""

# %% [markdown]
"""
#### Visualizing results

The following visualization methods work for all analysis methods. However, some require specifying the analysis step to visualize. Static analyses only have a single step, so we specify 0.

"""

# %%
from osmg.graphics.postprocessing_3d import show_deformed_shape
from osmg.graphics.postprocessing_3d import show_basic_forces


# %%
help(show_deformed_shape)


# %%
show_deformed_shape(static_anl, testcase.name, 0, 0.00, True)
# I'm not sure why it's not showing here in the docs, it should work
# on your machine!

# %%
help(show_basic_forces)


# %%
show_basic_forces(static_anl, testcase.name, 0, 1.00, 1.00, 1.00, 1.00, 1.00, 10, 1.00, 1.00, False)


# %% [markdown]
"""
### Modal Analysis

"""

# %%
modalcase = LoadCase('modal', mdl)
self_mass(mdl, modalcase)
modalcase.rigid_diaphragms([1, 2])


# %%
modal_analysis = solver.ModalAnalysis(mdl, {modalcase.name: modalcase}, num_modes=4)
modal_analysis.run()


# %%
print(modal_analysis.results[modalcase.name].periods)


# %% [markdown]
"""
for modal analyses, step corresponds to mode

"""


# %%
show_deformed_shape(modal_analysis, modalcase.name, 3, 0.00, extrude=False, animation=False)


# %% [markdown]
"""
More analysis methods are available, including static pushover and transient time-history. See `solver.py` and the tests.

"""

# %% [markdown]
"""
### Time-history analysis

"""

# %%
nlth_anl = solver.NLTHAnalysis(mdl, {testcase.name: testcase})


# %%
help(nlth_anl.run)


# %%
nlth_anl.run(
    0.01,
    'groundmotions/1xa.txt',
    'groundmotions/1ya.txt',
    None,
    0.005,
    damping={'type': 'rayleigh', 'ratio': 0.05, 'periods': [1.00, 0.30]},
    print_progress=True
)


# %%
parent_node_lvl2 = testcase.parent_nodes[2]


# %%
nlth_anl.plot_node_displacement_history(testcase.name, parent_node_lvl2, 0, plotly=True)


# %% [markdown]
"""
### Other types of structural analysis

- The analysis objects themselves are going to be renamed soon, because the terms `linear` and `nonlinear` are used unjustifiably. One can use a linear model to run a NLTHAnalyisis and vice versa. The new names will be closer to what OpenSees calls them.

- Currently it is unlcear what the intent of each model is (i.e. a linear model used for design or an advanced model used for performance evaluation purposes). There is already support for design-related analyses, including linear statis and modal response spectrum analysis, and definition of load combinations, which can support a design workflow. Examples will be added in the future, and there are plans to further enhance the design procedures.  

- Planned future support includes modal response history analysis.

"""
