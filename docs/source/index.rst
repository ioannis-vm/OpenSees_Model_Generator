.. osmg documentation master file, created by
   sphinx-quickstart on Tue Jan 10 15:35:33 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the documentation for `osmg`
=======================================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Home page <self>
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
   version and test any changes before deployment.

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

Becoming familiar with `osmg`
-----------------------------

The examples can help
*********************

`osmg` is composed of several interconnected modules, which may make
it difficult to understand at first. A helpful resource for gaining a
better understanding is the set of quick examples provided in the next
section. They are by no means comprehensive, but by running these
examples and following the execution of the code, the main logic
behind the package will become clearer.

API reference and pydoc
***********************

The package's API reference provides in-depth information about the
package's functions and classes. It is automatically generated from
the docstrings in the code. This information can also be accessed
offline by using the pydoc command in the terminal, which allows
viewing the documentation for any specific module or package that is
installed. pydoc can be used from the command line as follows:

.. code-block:: bash

   pydoc <module_name>

   # some examples:
   pydoc osmg
   pydoc osmg.model
   pydoc osmg.model.Model
   pydoc osmg.model.Model.add_level  # and so on
   pydoc numpy.array
   pydoc matplotlib.pyplot.plot

pydoc can also be used to start an HTTP server that allows viewing the
documentation in a web browser, using the following command:

.. code-block:: bash

   pydoc -p <some_port_number>

   # e.g.
   pydoc -p 9090

Note that this does not require network access.
   
Use `pdb`, it's fantastic
*************************

Another helpful tool for understanding the package's code is the
`pdb`_ library. This library allows running the code in debugging
mode, which enables stepping through the code line by line and viewing
the current state of the execution. This can be extremely useful for
understanding how the functions and classes of this (or any other)
package interact with one another. Many well-known integrated
development environments (IDEs) utilize the `pdb` library natively. A
breakpoint can also be forced anywhere by simply adding the following
lines at the intended location:

.. code-block:: python

   import pdb
   pdb.set_trace()

.. note::

   Pro tip: These two lines can be added under an if block,
   effectively creating a conditional breakpoint.

This will start the debugger at the point where the script is
executed. Once the debugger is running, the following commands are
available:

- n (next): Execute the current line and move to the next one
- s (step): Step into a function call
- c (continue): Continue execution until a breakpoint is reached
- l (list): List the source code of the current file
- b (break): Set a breakpoint at a specific line


A good IDE can make a big difference
************************************

An incredibly useful feature of many IDEs is called "jump to
definition". This feature allows you to quickly navigate to the
location in the code where a specific function or variable is
defined. If you are not currently utilizing this feature, it is highly
recommended that you look into it, as it can greatly increase your
productivity. In `Gnu Emacs`_, this functionality is provided by
`xref`_.
  
.. _OpenSees: https://opensees.berkeley.edu/
.. _dataclasses: https://docs.python.org/3/library/dataclasses.html
.. _scikit-geometry: https://github.com/scikit-geometry/scikit-geometry
.. _PyPI: https://pypi.org/
.. _pdb: https://docs.python.org/3/library/pdb.html
.. _xref: https://www.gnu.org/software/emacs/manual/html_node/emacs/Xref.html
.. _Gnu Emacs: https://www.gnu.org/software/emacs/
.. |osmg| replace:: :literal:`osmg`
