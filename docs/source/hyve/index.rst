``hyve``
========

A compositional visualisation engine for neuroimaging data
**********************************************************

``hyve`` (:doc:`hypercoil </hypercoil/index>` visualisation engine) is a Python package for interactive and static 3D visualisation of functional brain mapping data. It was originally designed to be used in conjunction with the ``hypercoil`` project for differentiable programming in the context of functional brain mapping, but can be used independently.

This system is currently under development, and the API is accordingly subject to sweeping changes without notice. Documentation is also extremely sparse, but will be added in the near future. To get a sense of how the package might look and feel when it is more mature, you can take a look at the test cases in the tests directory.

``hyve`` allows for the visualisation of 3D data in a variety of formats, including volumetric data, surface meshes, and 3-dimensional network renderings. It is built using a rudimentary quasi-functional programming paradigm, allowing users to compose new plotting utilities for their data by chaining together functional primitives. The system is designed to be modular and extensible, and can be easily extended to support new data types and visualisation techniques. It is built on top of the pyvista library and therefore uses VTK as its rendering backend. The system is also capable of combining visualisations as panels of a SVG figure.


Installation
------------

``hyve`` can be installed from PyPI using pip::

    pip install hyve

Alternatively, you can install the latest development version from GitHub::

    pip install git+https://github.com/hypercoil/hyve.git

Some examples also require installation of the hyve-examples package, which can be installed from PyPI using pip::

    pip install hyve-examples

.. raw:: html

  <iframe src="../_static/scalars-parcellation+pain_hemisphere-left_mode-face_scene.html" height="345px" width="100%"></iframe>

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   plotdef
   transforms
   examples
