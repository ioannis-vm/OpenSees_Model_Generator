"""
Run nonlinear time-history analysis to get the building's response
"""

import os
import importlib
import argparse
import numpy as np
import pandas as pd
from osmg import solver

# ~~~~~~~~~~~~~~~~~~~~~~ #
# set up argument parser #
# ~~~~~~~~~~~~~~~~~~~~~~ #

parser = argparse.ArgumentParser()
parser.add_argument("--archetype")

args = parser.parse_args()
archetype = args.archetype

analysis_dt = 0.01
progress_bar = True

# load archetype building
archetypes_module = importlib.import_module("archetypes")
try:
    archetype_builder = getattr(archetypes_module, archetype)
except AttributeError as exc:
    raise ValueError(f"Invalid archetype code: {archetype}") from exc

mdl, loadcase = archetype_builder()

# from osmg.graphics.preprocessing_3d import show
# show(mdl, loadcase, extrude=True)

num_levels = len(mdl.levels) - 1
level_heights = []
for level in mdl.levels.values():
    level_heights.append(level.elevation)
level_heights = np.diff(level_heights)

lvl_nodes = []
base_node = list(mdl.levels[0].nodes.values())[0].uid
lvl_nodes.append(base_node)
for i in range(num_levels):
    lvl_nodes.append(loadcase.parent_nodes[i + 1].uid)

specific_nodes = lvl_nodes + [n.uid for n in mdl.levels[0].nodes.values()]



ag_x = np.genfromtxt('groundmotions/1xa.txt')
# ensure that the time-histories have the same ground motion dt
gm_dt = 0.005

damping = "modal"

output_folder = '/tmp/out'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# from osmg.graphics.preprocessing_3d import show
# show(mdl, loadcase, extrude=True)
# show(mdl, loadcase, extrude=False)

#
# modal analysis
#

modal_analysis = solver.ModalAnalysis(
    mdl, {loadcase.name: loadcase}, num_modes=num_levels*3)
modal_analysis.settings.store_forces = False
modal_analysis.settings.store_fiber = False
modal_analysis.settings.restrict_dof = [False]*6
modal_analysis.run()

t_bar = modal_analysis.results[loadcase.name].periods[0]

# for per in modal_analysis.results[loadcase.name].periods:
#     print(per)
# print(modal_analysis.results[loadcase.name].periods)

# from osmg.graphics.postprocessing_3d import show_deformed_shape
# show_deformed_shape(
#     modal_analysis, loadcase.name, 8, 0.00,
#     extrude=False, animation=True)

# mnstar = modal_analysis.modal_participation_factors(loadcase.name, 'x')[1]
# np.cumsum(mnstar)


#
# time-history analysis
#

if damping == "rayleigh":
    damping_input = {
        "type": "rayleigh",
        "ratio": 0.02,
        "periods": [t_bar, t_bar / 10.00],
    }
elif damping == "modal":
    damping_input = {
        "type": "modal+stiffness",
        "num_modes": (num_levels) * 3.00,
        "ratio_modal": 0.02,
        "period": t_bar / 10.00,
        "ratio_stiffness": 0.001,
    }
else:
    raise ValueError(f"Invalid damping type: {damping}")


# define analysis object
nlth = solver.THAnalysis(mdl, {loadcase.name: loadcase})
nlth.settings.log_file = f"{output_folder}/log"
if damping == "rayleigh":
    nlth.settings.solver = "Umfpack"
nlth.settings.store_fiber = False
nlth.settings.store_forces = False
nlth.settings.store_reactions = True
nlth.settings.store_release_force_defo = False
nlth.settings.specific_nodes = specific_nodes

# run the nlth analysis
nlth.run(
    analysis_dt,
    ag_x,
    None,
    None,
    gm_dt,
    damping=damping_input,
    print_progress=progress_bar,
    drift_check=0.10,  # 10% drift
    time_limit=71.95,  # hours
)

# store response quantities

df = pd.DataFrame()
df["time--"] = np.array(nlth.time_vector)
for lvl in range(num_levels + 1):
    df[[f"FA-{lvl}-{j}" for j in range(1, 3)]] = nlth.retrieve_node_abs_acceleration(
        lvl_nodes[lvl], loadcase.name
    ).loc[:, "abs ax":"abs ay"]
    df[[f"FV-{lvl}-{j}" for j in range(1, 3)]] = nlth.retrieve_node_abs_velocity(
        lvl_nodes[lvl], loadcase.name
    ).loc[:, "abs vx":"abs vy"]
    if lvl > 0:
        us = nlth.retrieve_node_displacement(lvl_nodes[lvl], loadcase.name).loc[
            :, "ux":"uy"
        ]
        if lvl == 1:
            dr = us / level_heights[lvl - 1]
        else:
            us_prev = nlth.retrieve_node_displacement(
                lvl_nodes[lvl - 1], loadcase.name
            ).loc[:, "ux":"uy"]
            dr = (us - us_prev) / level_heights[lvl - 1]
        df[[f"ID-{lvl}-{j}" for j in range(1, 3)]] = dr

df["Vb-0-1"] = nlth.retrieve_base_shear(loadcase.name)[:, 0]
df["Vb-0-2"] = nlth.retrieve_base_shear(loadcase.name)[:, 1]

df.columns = pd.MultiIndex.from_tuples([x.split("-") for x in df.columns.to_list()])
df.sort_index(axis=1, inplace=True)

df.to_csv(f"{output_folder}/results.csv")
