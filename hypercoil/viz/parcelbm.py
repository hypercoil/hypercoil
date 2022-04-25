# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Atlas benchmark plot
~~~~~~~~~~~~~~~~~~~~
Plot atlas benchmark results. This is going to be very non-generalisable for
the moment.
"""
import json
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches
from ..engine import Sentry


class AtlasBenchmarkPlotter(Sentry):
    def __init__(self, key='homogeneity', min_size=20):
        self.atlases = {}
        self.key = key
        self.min_size = min_size

    def load_benchmark(self, path, xfm=np.log):
        with open(path) as f:
            data = json.load(f)
        name = list(data['sizes'].keys())[0]
        sizes = data['sizes'][name]
        subjects = list(data[self.key].keys())
        if sizes[0] is None:
            sizes = sizes[1:]
        sz = np.array(sizes)
        bm = np.zeros_like(sizes, dtype=float)
        for s in subjects:
            benchmark = data[self.key][s][name]
            if benchmark[0] is None:
                benchmark = benchmark[1:]
            bm += np.array(benchmark)
        bm /= len(subjects)
        df = pd.DataFrame({
            'name': [f'parcel-{i + 1}_{name}' for i in range(len(sz))],
            'sizes': sz,
            self.key: bm,
            'szxfm': xfm(sz),
            'i': np.ones_like(bm)
        })
        df = df[df['sizes'] > self.min_size]
        return df

    def fit_curve(self, df, minimum=0.1, xfm=np.log):
        if xfm is None:
            X = df[['i', 'sizes']].values
            xfms = [np.ones_like, lambda x: x]
        else:
            X = df[['i', 'sizes', 'szxfm']].values
            xfms = [np.ones_like, lambda x: x, xfm]
        y = df[self.key].values
        theta = np.linalg.lstsq(X, y, rcond=None)[0]
        x = np.linspace(minimum, df['sizes'].max(), 1000)
        y = theta @ np.stack([f(x) for f in xfms])
        return x, y

    def add_atlas_set(self, root, name, path, xfm=np.log):
        df = None
        paths = pathlib.Path(root).glob(path)
        for path in paths:
            path = str(path)
            df_new = self.load_benchmark(
                f'{path}',
                xfm=xfm
            )
            if df is None:
                df = df_new
            else:
                df = df.merge(df_new, 'outer')
        self.atlases[name] = df

    def __call__(self, root, pathdef, xfm=np.log, save=None):
        fig, ax = plt.subplots(figsize=(10, 10))
        l = {}
        max_x = -float('inf')
        max_y = -float('inf')
        colours = ['#66DDFF', '#FF9966', '#66FF66', 'purple', 'red']
        for name, path in pathdef.items():
            try:
                colour = colours.pop()
            except IndexError:
                colour = None
            self.add_atlas_set(
                root=root,
                name=name,
                path=path,
                xfm=xfm
            )
            benchmark_data = self.atlases[name]
            if name == 'null':
                sns.kdeplot(
                    ax=ax, x='sizes', y=self.key,
                    levels=20, thresh=0, fill=True,
                    data=benchmark_data)
                name = 'Null model'
                fit_min = 0.1
            else:
                if len(benchmark_data > 1000):
                    scatter_size = 0.5
                    scatter_marker = '.'
                else:
                    scatter_size = 2.5
                    scatter_marker = 'o'
                ax.scatter(
                    x=benchmark_data['sizes'],
                    y=benchmark_data[self.key],
                    c=colour,
                    s=scatter_size,
                    marker=scatter_marker,
                    label='_')
                fit_min = benchmark_data['sizes'].min()
            max_x = max(max_x, benchmark_data['sizes'].max())
            max_y = max(max_y, benchmark_data[self.key].max())
            ax.plot(*self.fit_curve(benchmark_data, minimum=fit_min, xfm=xfm),
                    c=colour)
            l[name] = patches.Patch(color=colour, label=name)

        ax.set_xscale('log')
        ax.set_xlabel('Parcel size')
        ax.set_ylabel(f'Parcel {self.key}')
        ax.set_xlim(10, max_x)
        ax.set_ylim(0, max_y)
        plt.legend(handles=[h for h in l.values()])
        if save is not None:
            fig.savefig(save, bbox_inches='tight')
