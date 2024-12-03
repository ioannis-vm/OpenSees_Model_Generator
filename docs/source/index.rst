.. osmg documentation master file, created by
   sphinx-quickstart on Tue Jan 10 15:35:33 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the documentation for `osmg`
=======================================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Home <self>
   Tips <tips>
   Introductory Examples <notebooks>
   API reference <_autosummary/osmg>


Introduction
------------
   
:literal:`OpenSees_Model_Generator` (`osmg`) is a Python package
designed to facilitate the definition and analysis of `OpenSees`_
models. Its main goal is to simplify the process of creating
large-scale models of complex geometry, by providing a set of useful
functions and classes. It is worth noting that osmg currently supports
only a subset of the capabilities of OpenSees, but it can be easily
extended to add more functionality on an as-needed basis. This makes
it a great tool for users who are already familiar with OpenSees and
are looking for a more streamlined model creation and management
workflow. OpenSees is incredibly powerful, and the goal of `osmg` is
to encourage greater adoption by technical users, by providing
commonly used functionality as boilerplate code, thus reducing the
time and effort involved in model creation and analysis.

.. note::
   This module is intended for advanced users who can understand the
   source code and are able to extend it to meet their specific
   needs. Familiarity with Python, including object-oriented
   programming using `dataclasses`_, and previous experience using
   `OpenSees`_ is required.

.. warning::
   Please note that ``osmg`` is currently in beta and may contain
   bugs. While every effort has been made to ensure its stability, it
   is advised to use caution when using it in production
   environments. Additionally, it should be kept in mind that future
   versions of this module may introduce breaking changes, which could
   affect code that relies on earlier versions. To minimize the risk
   of issues, it is recommended to keep up to date with the latest
   version and adjust the syntax as instructed in the change logs. If
   you encounter any issues with the documentation or the code itself,
   please consider reporting them by opening an `issue
   <https://github.com/ioannis-vm/OpenSees_Model_Generator/issues>`_
   on Github.

The `osmg` Workflow
-------------------

The general workflow enabled by the module is the following:

#. Instantiate model objects and define their components.
#. Define load cases.
#. Perform preprocessing operations, including calculating
   self-weight, load distributions, and other operations meant to be
   performed after the definition of components.
#. Run the analyses.
#. Post-process and visualize the analysis results.
#. Revise and repeat if needed.

Actual interaction with OpenSees only happens at step #4.

Supported model types
---------------------

#. 2D Truss (``ndm=2``, ``ndf=2``)
#. 3D Truss (``ndm=3``, ``ndf=3``)
#. 2D Frame (``ndm=2``, ``ndf=3``)
#. 3D Frame (``ndm=3``, ``ndf=6``)

Supported analysis types
------------------------

#. Static analysis: Calculate basic forces and displacements for a
   variety of load cases and under load combinations. These results
   can then be used to perform design verification checks.
#. Modal analysis: Calculate free vibration mode shapes and
   periods. In addition to the deformed shapes, ``osmg`` also imposes
   the resulting displacements on the structure with separate
   analyses, capturing the resulting basic forces, which can be used
   for modal response spectrum analysis.
#. Cyclic pushover analysis, including the option to unload in between
   the predefined control node displacement targets or in the end.
#. Time-history analysis, including the option to dampen out the
   residual free vibration.
   

Installing `osmg`
-----------------

It is recommended to use a virtual environment to manage the
dependencies of this module. Conda, Mamba, or simply ``venv`` are some
valid options.

``osmg`` is available on `PyPI`_, but it is recommended to install it in
development mode instead, and potentially make changes to its source
code as needed.

Installing from PyPI
********************

.. code-block:: bash

   # create a conda environment
   conda create --name your_env_name_here python=3.13
   # activate the environment
   conda activate your_env_name_here
   # install osmg from PyPI in that environment
   python -m pip install osmg

Installing in development mode
******************************

.. code-block:: bash

   mkdir parent_directory_where_you_would_like_to_have_osmg
   cd parent_directory_where_you_would_like_to_have_osmg
   git clone https://github.com/ioannis-vm/OpenSees_Model_Generator
   cd OpenSees_Model_Generator
   conda create --name your_env_name_here python=3.13
   conda activate your_env_name_here
   python -m pip install -e .[dev]


.. _OpenSees: https://opensees.berkeley.edu/
.. _dataclasses: https://docs.python.org/3/library/dataclasses.html
.. _scikit-geometry: https://github.com/scikit-geometry/scikit-geometry
.. _PyPI: https://pypi.org/
.. _pdb: https://docs.python.org/3/library/pdb.html
.. _xref: https://www.gnu.org/software/emacs/manual/html_node/emacs/Xref.html
.. _Gnu Emacs: https://www.gnu.org/software/emacs/
.. |osmg| replace:: :literal:`osmg`
