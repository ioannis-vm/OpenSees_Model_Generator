Becoming familiar with `osmg`
-----------------------------

The examples can help
*********************

`osmg` is composed of several interconnected modules, which may make
it difficult to understand at first, but this modularity offers
enhanced development flexibility that will allow the project to scale
more easily over time. A helpful resource for gaining a better
understanding is the set of quick examples provided in the next
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

   # or, alternatively:
   breakpoint()

.. note::

   Pro tip: These two lines can be added under an if block,
   effectively creating a conditional breakpoint.

This will start the debugger at the point where the script is
executed. Once the debugger is running, the following commands are
available:

* n (next): Execute the current line and move to the next one.
* s (step): Step into a function call.
* c (continue): Continue execution until a breakpoint is reached.
* l (list): List the source code of the current file.
* b (break): Set a breakpoint at a specific line.


A good IDE can make a big difference
************************************

An incredibly useful feature of many IDEs is called "jump to
definition". This feature allows for quick navigation to the location
in the code where a specific function or variable is defined (even if
it is in another file). Typically there is also the option to jump to
the previous location. If you are not currently utilizing these
features, it is highly recommended that you look into it, as it can
greatly increase your productivity. In `Gnu Emacs`_, this
functionality is provided by `xref`_.


.. _OpenSees: https://opensees.berkeley.edu/
.. _dataclasses: https://docs.python.org/3/library/dataclasses.html
.. _scikit-geometry: https://github.com/scikit-geometry/scikit-geometry
.. _PyPI: https://pypi.org/
.. _pdb: https://docs.python.org/3/library/pdb.html
.. _xref: https://www.gnu.org/software/emacs/manual/html_node/emacs/Xref.html
.. _Gnu Emacs: https://www.gnu.org/software/emacs/
.. |osmg| replace:: :literal:`osmg`
