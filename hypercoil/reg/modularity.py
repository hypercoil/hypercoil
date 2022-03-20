# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Modularity penalty
~~~~~~~~~~~~~~~~~~
Penalise a weight according to the quality of community structure it induces.
"""
import torch
from torch.nn import Module
from .base import Loss
from ..functional.graph import relaxed_modularity, girvan_newman_null


class ModularityLoss(Loss):
    def __init__(self, nu, affiliation_xfm=None, exclude_diag=True, gamma=1,
                 null=girvan_newman_null, normalise_modularity=True,
                 normalise_coaffiliation=True, directed=False, sign='+',
                 name=None, **params):
        super(ModularityLoss, self).__init__()
        if affiliation_xfm is None:
            affiliation_xfm = lambda x: x
        self.affiliation_xfm = affiliation_xfm
        self.nu = nu
        self.exclude_diag = exclude_diag
        self.gamma = gamma
        self.null = null
        self.normalise_modularity = normalise_modularity
        self.normalise_coaffiliation = normalise_coaffiliation
        self.directed = directed
        self.sign = sign
        self.name = name
        self.params = params

    def forward(self, A, C, C_o=None, L=None):
        if C_o is None:
            C_o = C
        return -self.nu * relaxed_modularity(
            A=A,
            C=self.affiliation_xfm(C),
            C_o=self.affiliation_xfm(C_o),
            L=L,
            exclude_diag=self.exclude_diag,
            gamma=self.gamma,
            null=self.null,
            normalise_modularity=self.normalise_modularity,
            normalise_coaffiliation=self.normalise_coaffiliation,
            directed=self.directed,
            sign=self.sign,
            **self.params
        ).squeeze()
