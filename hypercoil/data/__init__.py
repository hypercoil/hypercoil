# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Data interfaces and data engineering workflows.

The ``data`` submodule contains functionality for efficiently interacting with
and ingesting data into the differentiable pipeline. The recommended data
ingestion workflow proceeds as follows:

.. image:: _images/ingestion.svg

- Creating a ``pybids``-inspired representation of a dataset's layout on
  disk. This process uses a data grabber to locate files whose names match an
  expected pattern set and then stores their paths in a layout object.
- Mapping each data observation to a ``DataReference`` object. Each reference
  comprises a set of variables (corresponding to feature groups of the data
  observation, e.g. BOLD or confound time series).
- Transforming data references to tensors. Each of the variables that
  comprise a ``DataReference`` is equipped with a set of ``torchvision``-
  inspired transformations that
  form an instruction set for mapping the filesystem path retrieved in the
  dataset layout to a tensor block.
- Saving the transformed references in a ``.tar`` archive for compatibility
  with the ``webdataset`` package. This process includes creating dataset
  splits (which can be later used for defining training, validation, and test
  sets) and ``.tar``-formatted shards.

In practice, the API is designed to combine the first three steps into a
single function call when using a BIDS- or HCP-formatted dataset. In these
cases, using the
:doc:`fmriprep_references <api/hypercoil.data.bids.fmriprep_references>` and
:doc:`hcp_references <api/hypercoil.data.hcp.hcp_references>` convenience
functions, respectively, is sufficient to implement the first three workflow
stages. The final stage is handled by passing the references thereby retrieved
to the
:doc:`make_wds <api/hypercoil.data.wds.make_wds>` function.

.. warning::
    The data engineering workflow remains fairly brittle and unstable at this
    time. Many atoms are currently untested, and test coverage is overall
    poor.

.. note::
    The data grabbing and ``webdataset`` sharding procedures can be quite slow
    when datasets are large and disk read/write speed is limited. The workflow
    is likely suboptimal; we would welcome ideas for improvement or
    contributions. Parallelisation is not implemented, but the rate-limiting
    factor is likely the disk read/write head, so it's not clear how helpful
    it would actually be. In our experience it can take close to an hour to
    initialise the layout for a large dataset, and upwards of a day to write
    a large webdataset to a slow hard disk.
"""
from .bids import (
    fmriprep_references,
    LightBIDSObject,
    LightBIDSLayout,
    fMRIPrepDataset
)
from .dataset import (
    ReferencedDataset,
    ReferencedDataLoader
)
from .grabber import (
    LightGrabber
)
from .hcp import (
    hcp_references,
    HCPObject,
    HCPLayout,
    HCPDataset
)
from .neuro import (
    fMRIDataReference
)
from .variables import (
    VariableFactory,
    CategoricalVariable,
    ContinuousVariable,
    NeuroImageBlockVariable,
    TableBlockVariable,
    DataObjectVariable,
    DataPathVariable
)
