# OpenSeesPy Building Modeler

![Screenshot](/img/teaser_image.png)

The purpose of this module is to assist the definition, analysis, and post-processing of OpenSees models of 3D buildings.
The module is in constant development, and new functionality is added as needed for my research. No backwards compatibility is maintained. Anyone is free and welcome to use, fork, extend and redistribute the code.

Currently, the following functionality is supported:

### Features

#### Modeling

- Organizing building components in building `levels`
- Organizing building components in `groups`
- Automated generation of floor tributary areas (based on the closed regions defined by the level's beams), which are used to distribute the floor's UDL on the components
- Automated element self-weight and mass
- Using `gridline` objects to define beams and columns
- Ability to define multiple elements at once on all `active` levels
- Support for importing AISC steel sections
- Fiber sections of any shape
- Specifying element offsets and placement point relative to the section


#### Analysis

- Linear static
- Modal
- Nonlinear pushover
- Nonlinear time-history

#### Post processing

- Visualizing the defined components in 3D with metadata in hover boxes, with or without frame extrusions.
- Visualizing the deformed shape of a given analysis step, with automatic determination of an appropriate scaling factor, with or without extruding the frame elements.
- Visualizing the basic forces of the elements for a given analysis step.

