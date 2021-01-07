# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Model specifications
~~~~~~~~~~~~~~~~~~~~
Specifications for selecting particular columns from a data frame and adding
expansion terms if necessary. Largely adapted and updated from niworkflows.
"""
from .expression import Expression


class ModelSpec(object):
    """
    Model specification.

    The specification comprises a model formula together with shorthand rules
    and variable transformations that can be used to compose the model from a
    provided DataFrame and optionally associated metadata.

    Parameters/Attributes
    ---------------------
    spec : str
        Formula specifying the model to be built. Variables and transformation
        instructions are written additively; the model is specified as their
        sum. Each term in the sum is either a variable or a transformation of
        another variable or sum of transformations/variables.
    name : str or None (default None)
        Model specification name. Included for functions that make use of
        multiple model specifications, so that they can use this as a key for
        hashing. If this is not provided, it will be set to the formula spec
        by default.
    shorthand : Shorthand object or None (default None)
        Object specifying keywords and metadata filters in the specification
        to be expanded or used to select additional variables. Consult the
        Shorthand documentation for further information and the source code of
        the FCShorthand class for an example implementation.
    transforms : list(ColumnTransform objects) or None (default None)
        List containing the column-wise transforms to be parsed in the
        formula. Consult the ColumnTransform documentation for further
        information and the source code of the DerivativeTransform and
        PowerTransform classes for example implementations.

    Methods
    -------
    __call__(df, metadata=None)
        Recursively parse the model formula, and then select and transform the
        columns of a data frame required to build the model specified by the
        formula.

        Parameters
        ----------
        df : DataFrame
            DataFrame containing all necessary variables for building the
            specified model. Relevant variables will be selected, transformed,
            and composed from the columns of the provided DataFrame.
        metadata : dict or None (default None)
            Dictionary containing additional metadata for particular columns,
            usable by the Shorthand object to filter variables that satisfy
            particular criteria. Not used if no Shorthand is provided.

        Returns
        -------
        data: pandas DataFrame
            All variables and values in the specified model, built by
            selecting, transforming, and composing specified columns of the
            input DataFrame.
    """
    def __init__(self, spec, name=None, shorthand=None, transforms=None):
        self.spec = spec
        self.name = name or spec
        self.shorthand = shorthand
        self.transforms = transforms

    def __call__(self, df, metadata=None):
        if self.shorthand:
            formula = self.shorthand(self.spec, df.columns, metadata)
        else:
            formula = self.spec
        expr = Expression(formula, self.transforms)
        return expr(df)

    def __repr__(self):
        return f'ModelSpec(name={self.name}, formula={self.spec})'
