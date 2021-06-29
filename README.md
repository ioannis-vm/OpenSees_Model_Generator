# OpenSeesPy Building Modeler

![Screenshot](/img/teaser_image.png)

The purpose of this module is to assist the definition, analysis, and post-processing of OpenSees models of 3D buildings.
The module is in constant development, and new functionality is added as needed for my research. No backwards compatibility is maintained. Anyone is free and welcome to use, fork, extend and redistribute the code.

Currently, the following functionality is supported:

#### Modeling

- Building levels ~ definition of elements on multiple levels with a single line of code
- Groups
- Tributary area analysis for floor-to-beam load distribution
- Element self-weight, self-mass
- Accelerated element definitions using gridlines
- Element seleciton and modification
- AISC steel sections
- Fiber generation of arbitrary sections


#### Analysis

- Linear static
- Modal
- Nonlinear pushover
- Nonlinear response-history

#### Post processing

- Visualizing the defined components in 3D with metadata in hover boxes, with or without frame extrusions.
- Visualizing the deformed shape of a given analysis step, with automatic determination of an appropriate scaling factor, with or without frame extrusions.
- Visualizing the basic forces of the elements for a given analysis step.

