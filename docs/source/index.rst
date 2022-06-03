.. hypercoil documentation master file, created by
   sphinx-quickstart on Tue Jun 29 13:00:08 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Differentiable programming for brain mapping
============================================

The ``hypercoil`` software library includes tools for writing fully differentiable programs for use in brain mapping. At the current stage of development, most basic components of the functional connectivity software stack are available.

.. image:: https://github.com/rciric/hypercoil/actions/workflows/ci.yml/badge.svg

.. image:: https://github.com/rciric/hypercoil/actions/workflows/doc.yml/badge.svg

.. image:: https://codecov.io/gh/rciric/hypercoil/branch/main/graph/badge.svg?token=FVJVK6AFQG
  :target: https://codecov.io/gh/rciric/hypercoil


Currently, because this project is still in the incipient stage, this documentation hub is fairly minimal (principally focused on documenting the fairly substantial :doc:`API <modules>` for :doc:`functionals <functional>`,  :doc:`neural network modules <nn>`, :doc:`initialisation schemes <init>`, and :doc:`loss functions <loss>`). Usage guidelines and simple examples will be ported into the documentation from test cases soon, and API details will eventually be added for currently unstable submodules, including data engineering workflows, evaluation and benchmarks, and visualisation.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   diffprog
   usage
   support
   modules
   examples



Indices and tables
******************

* :ref:`search`
