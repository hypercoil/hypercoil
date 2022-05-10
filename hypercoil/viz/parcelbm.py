#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Atlas benchmark plot
~~~~~~~~~~~~~~~~~~~~
Plot atlas benchmark results. This is going to be very non-generalisable for
the moment.

CLI usage example:
viz/parcelbm.py \
    -r /tmp/connhomogeneity/ \
    -p 'null:desc-spatialnull*.json' \
    -p 'MMP:desc-glasser*.json' \
    -p 'boundary map:desc-gordon*.json' \
    -p 'gwMRF:desc-schaefer*.json' \
    -o /tmp/test.png
"""
import click
import json
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches
from hypercoil.engine import Sentry


KDE_CMAP = sns.cubehelix_palette(
    start=0.45, rot=0.3, dark=0, light=.3, reverse=True, as_cmap=True
)


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

    def fit_curve(self, df, minimum=0.1, xfm=np.log, return_params=False):
        if xfm is None:
            X = df[['i', 'sizes']].values
            xfms = [np.ones_like, lambda x: x]
        else:
            X = df[['i', 'sizes', 'szxfm']].values
            xfms = [np.ones_like, lambda x: x, xfm]
        y = df[self.key].values
        theta = np.linalg.lstsq(X, y, rcond=None)[0]
        if return_params:
            return theta
        x = np.linspace(minimum, df['sizes'].max(), 1000)
        y = theta @ np.stack([f(x) for f in xfms])
        return x, y

    def adjust_points(self, theta, xfm=np.log, df=None, x=None, y=None):
        if xfm is None:
            xfms = [np.ones_like, lambda x: x]
        else:
            xfms = [np.ones_like, lambda x: x, xfm]
        if df is not None:
            if xfm is None:
                x = df[['i', 'sizes']].values
            else:
                x = df[['i', 'sizes', 'szxfm']].values
            y = df[self.key].values
            y_hat = theta @ x.T
        else:
            y_hat = theta @ np.stack([f(x) for f in xfms])
        return y - y_hat

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

    def plot_adjusted(self, name, benchmark_data, ax, max_x, max_y,
                      min_y, l, colour, style=None, theta=None, xfm=np.log):
        # null *must* be first for this to work.
        if name == 'null':
            theta = self.fit_curve(
                benchmark_data,
                xfm=xfm,
                return_params=True)
        benchmark_data[f'{self.key}_adj'] = self.adjust_points(
            theta=theta,
            xfm=xfm,
            df=benchmark_data
        )
        if name == 'null':
            sns.kdeplot(
                ax=ax, x='sizes', y=f'{self.key}_adj',
                levels=20, thresh=0, fill=True,
                data=benchmark_data, cmap=KDE_CMAP)
            name = 'Null model'
            fit_min = 0.1
        else:
            if len(benchmark_data) > 1000:
                scatter_size = 0.5
                scatter_marker = '.'
            else:
                scatter_size = 2.5
                scatter_marker = 'o'
            ax.scatter(
                x=benchmark_data['sizes'],
                y=benchmark_data[f'{self.key}_adj'],
                c=colour,
                s=scatter_size,
                marker=scatter_marker,
                label='_')
            fit_min = benchmark_data['sizes'].min()
        max_x = max(max_x, benchmark_data['sizes'].max())
        max_y = max(max_y, benchmark_data[f'{self.key}_adj'].max())
        min_y = min(min_y, benchmark_data[f'{self.key}_adj'].min())
        fit_x, fit_y = self.fit_curve(
            benchmark_data,
            minimum=fit_min,
            xfm=xfm)
        fit_y_adj = self.adjust_points(theta=theta, xfm=xfm, x=fit_x, y=fit_y)
        ax.plot(fit_x, fit_y_adj, c=colour, ls=style, linewidth=3)
        l[name] = patches.Patch(color=colour, label=name)
        return max_x, max_y, min_y, l, theta

    def __call__(self, root, pathdef, xfm=np.log, save=None, adjusted=False):
        fig, ax = plt.subplots(figsize=(10, 10))
        l = {}
        min_x = self.min_size / 2
        min_y = 0
        max_x = -float('inf')
        max_y = -float('inf')
        theta = None
        ##TODO: not gonna work with more than 5 now
        colours = ['#FF99CC', '#DD77FF', '#66DDFF', '#66FF66', 'red', ]
        styles = [(0, (3, 1, 1, 1, 1, 1)),
                  'dashdot', 'dashed',
                  'dotted', 'solid']
        for name, path in pathdef.items():
            try:
                colour = colours.pop()
                style = styles.pop()
            except IndexError:
                colour = None
                style=None
            self.add_atlas_set(
                root=root,
                name=name,
                path=path,
                xfm=xfm
            )
            benchmark_data = self.atlases[name]
            if adjusted:
                max_x, max_y, min_y, l, theta = self.plot_adjusted(
                    name, benchmark_data, ax, max_x, max_y, min_y, l,
                    xfm=xfm, theta=theta, colour=colour, style=style)
                continue
            if name == 'null':
                sns.kdeplot(
                    ax=ax, x='sizes', y=self.key,
                    levels=20, thresh=0, fill=True,
                    data=benchmark_data, cmap=KDE_CMAP)
                name = 'Null model'
                fit_min = 0.1
            else:
                if len(benchmark_data) > 1000:
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
                    c=colour, ls=style, linewidth=3)
            l[name] = patches.Patch(color=colour, label=name)

        ax.set_xscale('log')
        ax.set_xlabel('Parcel size')
        ax.set_ylabel(f'Parcel {self.key}')
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        plt.legend(handles=[h for h in l.values()])
        if save is not None:
            fig.savefig(save, bbox_inches='tight')


@click.command()
@click.option('-r', '--root', required=True, type=str)
@click.option('-p', '--pathdef', multiple=True, required=True, type=str)
@click.option('-o', '--out', required=True, type=str)
@click.option('-k', '--key', default='homogeneity', type=str)
@click.option('-s', '--min-size', default=20, type=int)
@click.option('-a', '--adjusted', default=False, type=bool, is_flag=True)
def main(root, pathdef, out, key, min_size, adjusted):
    pathdef = {k : v for k, v in [p.split(':') for p in pathdef]}
    a = AtlasBenchmarkPlotter(key=key, min_size=min_size)
    a(root=root, pathdef=pathdef, save=out, adjusted=adjusted)

if __name__ == '__main__':
    main()
