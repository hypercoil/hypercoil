# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import torch
from ..functional import (
    corr,
    pairedcorr,
    sym2vec
)


class AtlasBenchmark:
    def __init__(self, mask,
                 evaluate_homogeneity=True,
                 evaluate_variance=True,
                 evaluate_varexp=True):
        self.voxel_assignments = {}
        self.mask = mask
        self.evaluate_homogeneity = evaluate_homogeneity
        self.evaluate_variance = evaluate_variance
        self.evaluate_varexp = evaluate_varexp
        self.cur_id = 0
        self.homogeneity = {}
        self.variance = {}
        self.varexp = {}

    def add_voxel_assignment(self, name, asgt):
        self.voxel_assignments[name] = asgt

    def add_voxel_assignment_from_maps(self, name, maps, compartments):
        asgt = torch.zeros((self.mask.sum()))
        offset = 0
        for k, v in maps.items():
            mask = compartments[k]
            asgt_compartment = v.argmax(0)
            asgt[mask] = asgt_compartment + offset
            offset += asgt_compartment.max()
        self.voxel_assignments[name] = asgt

    def telescope(self, asgt):
        n_labels = len(asgt.unique())
        n_voxels = len(asgt)
        maps = torch.empty((n_labels, n_voxels))
        for i, l in enumerate(labels):
            maps[i] = (asgt == l)
        return maps

    def internal_homogeneity(self, matrix, id):
        self.homogeneity[id] = {}
        for name, asgt in self.voxel_assignments.items():
            labels = asgt.unique()
            self.homogeneity[id][name] = [None for _ in labels]
            for i, label in enumerate(labels):
                label_mask = (asgt == label)
                parcel = matrix[label_mask, :]
                parcel_size = len(parcel)
                cor = sym2vec(corr(parcel)).mean()
                self.homogeneity[id][name][i] = cor

    def internal_variance(self, matrix, id):
        self.variance[id] = {}
        for name, asgt in self.voxel_assignments.items():
            labels = asgt.unique()
            self.variance[id][name] = [None for _ in labels]
            for i, label in enumerate(labels):
                label_mask = (asgt == label)
                parcel = matrix[label_mask, :]
                var = parcel.var(0).mean()
                self.variance[id][name][i] = var

    def variance_explained(self, matrix, id):
        self.varexp[id] = {}
        for name, asgt in self.voxel_assignments.items():
            labels = asgt.unique()
            label_ts = self.telescope(asgt) @ ts
            pmatrix = pairedcorr(label_ts, ts)
            self.varexp[id][name] = [None for _ in labels]
            for i, label in enumerate(labels):
                label_mask = (asgt == label)
                parcel = matrix[label_mask, :]
                theta, _, _, _ = torch.linalg.lstsq(pmatrix.T, parcel.T)
                parcel_hat = (pmatrix.T @ theta).T
                vex = torch.diagonal(pairedcorr(parcel, parcel_hat)).mean()
                self.varexp[id][name][i] = vex

    def evaluate(self, ts, id=None):
        if self.mask is not None:
            ts = ts[self.mask]
        if id is None:
            id = self.cur_id
            self.cur_id += 1
        matrix = corr(ts)
        if self.evaluate_homogeneity:
            self.internal_homogeneity(matrix, id)
        if self.evaluate_variance:
            self.internal_variance(matrix, id)
        if self.evaluate_varexp:
            self.variance_explained(matrix, id)
