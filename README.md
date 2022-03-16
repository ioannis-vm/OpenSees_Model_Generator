# OpenSees Model Builder

*Analyze structures like a scientist!*

![Screenshot](/img/teaser_image.png)

Fancy user interfaces, while convenient, have limited functionality. This restriction often traps the engineer in a tedious process of seemingly endless pointing and clicking to accomplish a relatively simple task. Using python to conduct structural analysis opens the gate to unlimited customization, extensibility, and automation. Beautiful plain text and the sky is the limit!

This module aims to assist the definition, analysis, and post-processing of OpenSees models of 3D buildings.
The module is in constant development, and new functionality is added as needed for my research. No backward compatibility is maintained. Anyone is free and welcome to use, fork, extend and redistribute the code.

## Current Functionality

#### Modeling

- Building levels ~ definition of elements on multiple levels with a single line of code
- Tributary area analysis for floor-to-beam load distribution
- Element self-weight, self-mass
- AISC steel sections
- Section polygon subdivision for fiber elements

#### Analysis

- Linear static
- Modal
- Nonlinear pushover
- Nonlinear response-history

#### Post processing

- Visualizing the defined components in 3D with metadata in hover boxes, with or without frame extrusions.
- Visualizing the deformed shape of a given analysis step, with automatic determination of an appropriate scaling factor, with or without frame extrusions.
- Visualizing the basic forces of the elements for a given analysis step.

## FAQ

1. Why isn't this on Pypi?

The code is not meant to be used as a module. At least not yet. There are still parts of the code that require tinkering, depending on the intended functionality. For this reason, it's better to use the code by relative import rather than by installing it as a module.

2. Your examples are confusing

I'm sorry! I plan to improve the examples at some point, replacing them with a tutorial using a series of annotated jupyter notebooks. My main job is focusing on my research, so you have to figure out things on your own until I do that. I would advise using pdb and reading the docstrings.

3. This is what I was looking for! Can I use this for my research/project/practice?

This code is *free* code. See `LICENSE`. It relies on OpenSees, which has its own [license](https://opensees.berkeley.edu/OpenSees/copyright.php). Keep in mind, though, that future changes to this code might shamelessly break backward compatibility. I.e., your code might not run if you pull the latest changes in the future. I need that freedom to be able to experiment and add/drop functionality as needed.

4. I found a bug! / You really should have used XYZ / I would like to share an improvement

Thank you for bringing this to my attention! Please send me an email. I am `iοαnnis_νm@berκeley.edυ`, but replace the greek characters, otherwise they will start dancing Zorbas and drinking ouzo. Opa!
