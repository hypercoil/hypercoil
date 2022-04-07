# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import torch
from collections import OrderedDict
from hypercoil.functional import (
    corr,
    pairedcorr,
    sym2vec
)


class AtlasBenchmark:
    def pretransform(self, asgt, matrix, ts):
        pass

    def transform(self, parcel, pretransformation=None):
        pass


class InternalHomogeneity(AtlasBenchmark):
    name = 'homogeneity'

    def transform(self, parcel, pretransformation=None):
        return sym2vec(corr(parcel)).mean()


class InternalVariance(AtlasBenchmark):
    name = 'variance'

    def transform(self, parcel, pretransformation=None):
        return parcel.var(0).mean()


class VarianceExplained(AtlasBenchmark):
    name = 'varexp'

    def pretransform(self, asgt, matrix, ts):
        label_ts = self.telescope(asgt) @ ts
        return pairedcorr(label_ts, ts)

    def transform(self, parcel, pretransformation=None):
        theta, _, _, _ = torch.linalg.lstsq(pretransformation.T, parcel.T)
        parcel_hat = (pretransformation.T @ theta).T
        return (torch.diagonal(pairedcorr(parcel, parcel_hat)) ** 2).mean()

    def telescope(self, asgt):
        labels = asgt.unique()
        n_labels = len(labels)
        n_voxels = len(asgt)
        maps = torch.empty((n_labels, n_voxels))
        for i, l in enumerate(labels):
            maps[i] = (asgt == l)
        return maps


class AtlasEval:
    def __init__(self, mask,
                 evaluate_homogeneity=True,
                 evaluate_variance=True,
                 evaluate_varexp=True):
        self.voxel_assignments = OrderedDict()
        self.mask = mask
        self.cur_id = 0
        self.results = {}
        self.benchmarks = []
        if evaluate_homogeneity:
            benchmark = InternalHomogeneity()
            self.benchmarks += [benchmark]
            self.results[benchmark.name] = {}
        if evaluate_variance:
            benchmark = InternalVariance()
            self.benchmarks += [benchmark]
            self.results[benchmark.name] = {}
        if evaluate_varexp:
            benchmark = VarianceExplained()
            self.benchmarks += [benchmark]
            self.results[benchmark.name] = {}

    def add_voxel_assignment(self, name, asgt):
        self.voxel_assignments[name] = asgt.squeeze()

    def add_voxel_assignment_from_maps(self, name, maps, compartments):
        asgt = torch.zeros((self.mask.sum()))
        offset = 0
        for k, v in maps.items():
            mask = compartments[k]
            asgt_compartment = v.argmax(0)
            asgt[mask] = asgt_compartment + offset
            offset += asgt_compartment.max()
        self.voxel_assignments[name] = asgt

    def evaluate(self, ts, id=None, verbose=False):
        if self.mask is not None:
            ts = ts[self.mask]
        if id is None:
            id = self.cur_id
            self.cur_id += 1
        if verbose:
            print(f'[ Computing full spatial correlation ]')
        matrix = corr(ts)

        for b in self.benchmarks:
            self.results[b.name][id] = {}

        for name, asgt in self.voxel_assignments.items():
            if verbose:
                print(f'[ Assignment {name} ]')
            labels = asgt.unique()
            pretransformation = {}
            for b in self.benchmarks:
                pretransformation[b.name] = b.pretransform(
                    asgt=asgt,
                    matrix=matrix,
                    ts=ts
                )
                self.results[b.name][id][name] = [
                    None for _ in labels]
            for i, label in enumerate(labels):
                if verbose:
                    print(f'[ Label {label} ]')
                label_mask = (asgt == label)
                parcel = matrix[label_mask, :]
                for b in self.benchmarks:
                    self.results[b.name][id][name][i] = (
                        b.transform(
                            parcel=parcel,
                            pretransformation=pretransformation[b.name]
                        ))