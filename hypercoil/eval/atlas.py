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
    def pretransform(self, asgt, ts):
        pass

    def transform(self, parcel, pretransformation=None):
        pass


class InternalHomogeneity(AtlasBenchmark):
    name = 'homogeneity'

    def transform(self, parcel, pretransformation=None):
        return sym2vec(corr(parcel)).mean().item()


class InternalVariance(AtlasBenchmark):
    name = 'variance'

    def transform(self, parcel, pretransformation=None):
        return parcel.var(0).mean().item()


class VarianceExplained(AtlasBenchmark):
    name = 'varexp'
    lstsq_driver = 'gels'

    def pretransform(self, asgt, ts):
        label_ts = self.telescope(asgt, dtype=ts.dtype, device=ts.device) @ ts
        return pairedcorr(label_ts, ts)

    def transform(self, parcel, pretransformation=None):
        theta, _, _, _ = torch.linalg.lstsq(
            pretransformation.T, parcel.T,
            driver=self.lstsq_driver
        )
        parcel_hat = (pretransformation.T @ theta).T
        return (
            torch.diagonal(pairedcorr(parcel, parcel_hat)) ** 2
        ).mean().item()

    def telescope(self, asgt, dtype=None, device=None):
        labels = asgt.unique()
        n_labels = len(labels)
        n_voxels = len(asgt)
        maps = torch.empty((n_labels, n_voxels), dtype=dtype, device=device)
        for i, l in enumerate(labels):
            maps[i] = (asgt == l)
        return maps


##TODO: flexible null label instead of fixed to 0.
class AtlasEval:
    def __init__(self, mask,
                 evaluate_homogeneity=True,
                 evaluate_variance=True,
                 evaluate_varexp=True,
                 lstsq_driver='gels'):
        self.voxel_assignments = OrderedDict()
        self.mask = mask
        self.cur_id = 0
        self.results = {}
        self.results['sizes'] = {}
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
            benchmark.lstsq_driver = lstsq_driver
            self.benchmarks += [benchmark]
            self.results[benchmark.name] = {}

    def add_voxel_assignment(self, name, asgt):
        self.voxel_assignments[name] = asgt.squeeze()
        self.get_parcel_sizes(name)

    def add_voxel_assignment_from_maps(self, name, maps, compartments):
        asgt = torch.zeros(self.mask.sum())
        offset = 0
        for k, v in maps.items():
            mask = compartments[k]
            asgt_compartment = v.argmax(0)
            asgt[mask] = asgt_compartment + offset
            offset += asgt_compartment.max()
        self.add_voxel_assignment(name, asgt)

    def get_parcel_sizes(self, name):
        asgt = self.voxel_assignments[name]
        labels = asgt.unique()
        sizes = [None for _ in labels]
        for i, label in enumerate(labels):
            if label == 0:
                continue
            sizes[i] = (asgt == label).sum().item()
        self.results['sizes'][name] = sizes

    def evaluate(self, ts, id=None, verbose=False, compute_full_matrix=False):
        if self.mask is not None:
            ts = ts[self.mask]
        if id is None:
            id = self.cur_id
            self.cur_id += 1
        if compute_full_matrix:
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
                    ts=ts
                )
                self.results[b.name][id][name] = [
                    None for _ in labels]
            for i, label in enumerate(labels):
                if label == 0:
                    continue
                if verbose:
                    print(f'[ Label {label} ]')
                label_mask = (asgt == label)
                if compute_full_matrix:
                    parcel = matrix[label_mask, :]
                else:
                    parcel_ts = ts[label_mask, :]
                    parcel = pairedcorr(parcel_ts, ts)
                for b in self.benchmarks:
                    self.results[b.name][id][name][i] = (
                        b.transform(
                            parcel=parcel,
                            pretransformation=pretransformation[b.name]
                        ))
