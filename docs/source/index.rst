.. hypercoil documentation master file, created by
   sphinx-quickstart on Tue Jun 29 13:00:08 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

A differentiable program for mapping brain function
===================================================

The ``hypercoil`` software library includes tools for writing fully differentiable programs for use in brain mapping. It is build on the `JAX <https://jax.readthedocs.io/en/latest/>`_ framework, which combines a NumPy-like API with automatic differentiation and support for GPU acceleration. At the current stage of development, most basic components of the functional connectivity software stack are available.

.. image:: https://github.com/hypercoil/hypercoil/actions/workflows/ci.yml/badge.svg
  :target: https://github.com/hypercoil/hypercoil/actions/workflows/ci.yml
  :alt: Continuous Integration

.. image:: https://github.com/hypercoil/hypercoil/actions/workflows/doc.yml/badge.svg
  :target: https://github.com/hypercoil/hypercoil/actions/workflows/doc.yml
  :alt: Documentation

.. image:: https://codecov.io/gh/hypercoil/hypercoil/branch/main/graph/badge.svg?token=FVJVK6AFQG
  :target: https://codecov.io/gh/hypercoil/hypercoil
  :alt: Code Coverage

.. image:: https://img.shields.io/badge/GitHub-hypercoil-662299?logo=github
  :target: https://github.com/hypercoil/hypercoil/
  :alt: GitHub

.. image:: https://img.shields.io/badge/License-Apache_2.0-informational?logo=openaccess
  :target: https://opensource.org/licenses/Apache-2.0
  :alt: License

.. image:: https://img.shields.io/badge/cite-preprint-red?logo=arxiv
  :target: https://hypercoil.github.io/portal.html
  :alt: Preprint

.. image:: https://raw.githubusercontent.com/hypercoil/hypercoil/xrecore/docs/source/_static/logo.png
  :width: 600px
  :align: center

About
-----

In functional neuroimaging and adjacent fields, the advent of large, public data repositories has brought with it a proliferation of instruments and methods for analysing these data. This has introduced new challenges for the field: How can we ensure that our analyses are reproducible? How can we ensure that our analyses are valid? Conditioned on our dataset and our scientific question, how do we choose from among the available methods to design an analytic workflow in a principled way? How can we know that the workflow we've designed is suited to answering the questions we are asking?

We typically approach the problem of designing a scientific workflow *combinatorially*: we begin from a set of analytic options, and we choose from among these to configure our workflow. The ``hypercoil`` library provides a framework for going beyond this combinatorial approach, and for designing principled workflows that are *differentiable*. Instead of selecting from among a set of available methods, we can *learn* the (locally) best workflow for our dataset and our scientific question. This approach is particularly well-suited to the functional neuroimaging domain, where the data are high-dimensional and the scientific questions are often complex.

Built upon the same principles that power deep neural networks, this library provides software instruments for designing, deploying, and evaluating both differentiable programs and GPU-accelerated workflows. Our current focus is on applications related to fMRI-derived brain connectivity, but the library is designed to be eventually generalisable to other domains.

**Public warning:** At this time, this software should be used as if it were in a pre-alpha state. Many operations are fragile or incompletely documented. Edge cases might not be covered, and there are certainly bugs lurking in the code base. *Expect breaking changes* as the code base is further expanded and refined. Contributions or ideas for improvement are always welcome.

Remarks
-------

Currently, because this project is still in the incipient stage, this documentation hub is fairly minimal (principally focused on documenting the fairly substantial :doc:`API <modules>` for :doc:`functionals <functional>`,  :doc:`neural network modules <nn>`, :doc:`initialisation schemes <init>`, and :doc:`loss functions <loss>`). Usage guidelines and simple examples will be ported into the documentation from test cases soon, and API details will eventually be added for currently unstable submodules, including data engineering workflows, evaluation and benchmarks, and visualisation.

.. warning::

   At this time, this software should be used as if it were in a pre-alpha state. Many operations are fragile or incompletely documented. Edge cases might not be covered, and there are certainly bugs lurking in the code base. *Expect breaking changes* as the code base is further expanded and refined. Contributions or ideas for improvement are always welcome.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   overview
   diffprog
   usage
   support
   modules
   examples



Indices and tables
******************

* :ref:`search`
