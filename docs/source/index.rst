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
designed to facilitate the definition and analysis of 3D models using
`OpenSees`_. Its main goal is to simplify the process of creating
large models of complex geometry, by providing a set of useful
functions and classes.  It is worth noting that osmg currently
supports only a subset of the capabilities of OpenSees, but it can be
easily extended to add more functionality as needed. This makes it a
great tool for users who are already familiar with OpenSees and are
looking for a more streamlined model creation process. OpenSees is
incredibly powerful, and the goal of `osmg` is to encourage greater
adoption by technical users, by providing commonly used functionality
as boilerplate code, thus reducing the time and effort involved in
model creation and analysis.

.. note::
   This module is intended for advanced users who are capable of
   understanding the source code and have the ability to modify and
   extend it to meet their specific needs. A strong understanding of
   `dataclasses`_ and previous experience using `OpenSees`_ is
   required for proper use of this module. Those who are not familiar
   with these concepts and tools will benefit from learning more about
   them before using this module.

.. warning::
   Please note that `osmg` is currently in beta and may contain
   bugs. While every effort has been made to ensure its stability, it
   is advised to use caution when using it in production
   environments. Additionally, it should be kept in mind that future
   versions of this module may introduce breaking changes, which could
   affect code that relies on earlier versions. To minimize the risk
   of issues, it is recommended to keep up to date with the latest
   version and adjust the syntax as instructed in the changelogs.  If you encounter
   any issues with the documentation or the code itself, please
   consider reporting them by opening an `issue
   <https://github.com/ioannis-vm/OpenSees_Model_Generator/issues>`_
   on Github. Note that the documentation is a work in progress.

The `osmg` Workflow
-------------------

The general workflow enabled by the module is the following:

#. Instantiate model objects and define their components.
#. Perform preprocessing operations.
#. Define load case and analysis objects.
#. Run the analyses.
#. Post-process the analysis results.

Actual interaction with OpenSees only happens at step #4.

Installing `osmg`
-----------------

It is recommended to use a virtual environment to manage the
dependencies of this module. Anaconda or Miniconda are some valid
options. Installation instructions can be found
`here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.

`osmg` is available on `PyPI`_, but it is recommended to install it in
development mode instead.

Installing from PyPI
********************

.. code-block:: bash

   # create a conda environment
   conda create --name your_env_name_here python=3.9
   # activate the environment
   conda activate your_env_name_here
   # install scikit-geometry
   conda install -c conda-forge scikit-geometry
   # install osmg from PyPI in that environment
   python -m pip install osmg

It is recommended to install `scikit-geometry`_ before installing
`osmg`. This dependency is used to perform tributary area analyses for
load distribution calculations, but it is optional.

Installing in development mode
******************************

.. code-block:: bash

   mkdir parent_directory_where_you_would_like_to_have_osmg
   cd parent_directory_where_you_would_like_to_have_osmg
   git clone https://github.com/ioannis-vm/OpenSees_Model_Generator
   cd OpenSees_Model_Generator
   conda create --name your_env_name_here python=3.9
   conda activate your_env_name_here
   python -m pip install -r requirements_dev.txt
   conda install scikit-geometry -c conda-forge
   python -m pip install -e .


.. _OpenSees: https://opensees.berkeley.edu/
.. _dataclasses: https://docs.python.org/3/library/dataclasses.html
.. _scikit-geometry: https://github.com/scikit-geometry/scikit-geometry
.. _PyPI: https://pypi.org/
.. _pdb: https://docs.python.org/3/library/pdb.html
.. _xref: https://www.gnu.org/software/emacs/manual/html_node/emacs/Xref.html
.. _Gnu Emacs: https://www.gnu.org/software/emacs/
.. |osmg| replace:: :literal:`osmg`
