# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Neuroimaging data search
~~~~~~~~~~~~~~~~~~~~~~~~
A lightweight pybids alternative. Not robust; a temporary solution.
"""
import re, glob, pathlib
import pandas as pd
from itertools import product


class LightGrabberBase(object):
    """
    Extremely lightweight grabbit-like system for representation of data
    organised in a structured way across multiple files.

    A lightweight pybids alternative for fMRIPrep-like processed data that
    isn't as particular about leading zeros. It's not very robust and is
    likely to be a temporary solution until pybids is stabilised.

    Parameters
    ----------
    root : str
        Path to the root directory of the dataset on the file system.
    template : VariableFactory
        Factory for producing variables to assign each path. Consider using a
        factory for a subclass of `DataPathVariable` if the data instances
        have associated metadata.
    queries : list(DataQuery)
        Query objects defining the variables to extract from the dataset via
        query. Each query should specify a string pattern to constrain the
        scope of the layout. If queries are provided, then the layout will not
        include any files that do not match at least one pattern.

    Attributes
    ----------
    refs : List(DatasetVariable)
        List of the variables produced by the factory provided as `template`.
    """
    def __init__(self, root, template, patterns):
        self.refs = []
        self.root = root
        self.template = template
        self.patterns = patterns
        try:
            self.attr_regex = template.regex
        except AttributeError:
            self.attr_regex = {}
        self._iterate_and_assign(patterns)

    def find_files(self, pattern):
        """
        Find files in the dataset directory (and any nested directories) that
        match a specified pattern.
        """
        return [
            pathlib.Path(p) for p in
            glob.glob(f'{self.root}/**/{pattern}', recursive=True)
        ]

    def getall(self, entity):
        """
        Find all values that a particular data entity (e.g., an identifier)
        takes in the dataset files included in the layout.

        Parameters
        ----------
        entity : str
            Name of the entity (e.g., 'subject' or 'run') to enumerate.

        Returns
        -------
        list
            List of all unique values that the specified entity assumes in the
            dataset.
        """
        try:
            out = list(set([r.get(entity) for r in self.refs]))
            try:
                out.sort()
            except TypeError:
                pass
            if len(out) == 1 and out[0] is None:
                return []
            return out
        except AttributeError:
            return []

    def get(self, **filters):
        """
        Find dataset objects that match the specified filters.
        """
        obj = self.refs
        for k, v in filters.items():
            if (self.attr_regex) and (k not in self.attr_regex.keys()):
                continue
            obj = [r for r in obj if r.get(k) == v]
        return obj

    def __len__(self):
        return len(self.refs)

    def __repr__(self):
        s = f'{type(self).__name__}(root={self.root}, '
        s += f'results={len(self)}, '
        s += f'template={type(self.template).__name__})'
        return s


class _StringPatternMixin:
    def _iterate_and_assign(self, patterns):
        for pattern in patterns:
            files = self.find_files(pattern)
            for f in files:
                f_var = self.template(name=f.name)
                f_var.assign(f)
                self.refs += [f_var]


class _QueryMixin:
    def _iterate_and_assign(self, queries):
        for query in queries:
            pattern = query.pattern
            template = query.template or self.template
            files = self.find_files(pattern)
            for f in files:
                f_var = template(name=f.name)
                f_var.assign(f)
                self.refs += [f_var]


class _IndexMixin:
    def generate_index(self, observations, levels, ignore=None):
        ident = list(observations) + list(levels)
        entities = self._assemble_entities(ident, ignore)
        index, observations, levels, entities = self._collate_product(
            entities, observations, levels)
        return index, (observations, levels, entities)

    def _assemble_entities(self, ident, ignore=None):
        """
        Assemble all entities corresponding to the specified identifiers from a
        layout.

        Parameters
        ----------
        layout : object
            Object representing a dataset layout with a `getall` method that
            returns a list of all values of its argument entity present in the
            dataset.
        ident : list(str)
            List of identifier variables present in the dataset and its layout.
        ignore : dict(str: list) or None (default None)
            Dictionary indicating identifiers to be ignored. Currently this
            doesn't support any logical composition and takes logical OR over all
            ignore specifications. In other words, data will be ignored if they
            satisfy any of the ignore criteria.

        Returns
        -------
        entities : dict
            Dictionary enumerating the existing levels of each identifier variable
            in the dataset.
        """
        entities = {}
        ignore = ignore or {}
        for i in ident:
            if ignore.get(i) is None:
                ignore[i] = []

        for i in ident:
            entities[i] = self.getall(i)
            entities[i] = [n for n in entities[i] if n not in ignore[i]]

        return entities

    def _collate_product(self, values, observations, levels):
        """
        Collate a product of all possible combinations of within-observation
        variable assignments.

        This returns all possible combinations, including potentially invalid ones
        that are absent in the actual dataset. Additional functions like
        `delete_null_levels` and `delete_null_obs` can help remove the invalid
        combinations returned by `collate_product`.

        Parameters
        ----------
        values : dict
            Dictionary enumerating the existing levels of each identifier variable
            in the dataset.
        observations : list(str)
            List of identifier variables that distinguish observations from one
            another.
        levels : list(str)
            List of identifier variables that denote sub-levels of a single
            observation.

        Returns
        -------
        index : MultiIndex
            Pandas MultiIndex object enumerating all possible combinations of
            variable assignments in the input `values`.
        observations : list
            The input `observations`, filtered to exclude levels absent from the
            data.
        levels : list
            The input `levels`, filtered to exclude levels absent from the data.
        entities : list
            List of all identifier levels present in the data.
        """
        entities = []
        prod_gen = []
        for name, l in values.items():
            if l:
                entities += [name]
                prod_gen += [l]
        observations = [o for o in observations if o in entities]
        levels = [l for l in levels if l in entities]
        index = pd.MultiIndex.from_product(prod_gen, names=entities)
        return index, observations, levels, entities


class _IndexQueryMixin(_IndexMixin, _QueryMixin):
    def query_all(self, index, queries=None, **filters):
        """
        Query a layout for all values assigned to entities in an index.

        Parameters
        ----------
        index : MultiIndex
            Pandas MultiIndex object enumerating all possible combinations of
            variable assignments among dataset identifiers.
        queries : list(DataQuery)
            Data queries used to locate files in the dataset. Each entry locates a
            different file for each unique identifier combination.
        filters : dict
            Additional filters to apply for each query.

        Returns
        -------
        results : dict
            Results for each query in `queries` submitted to the `layout` after
            applying any `filters` additionally specified.
        """
        queries = queries or self.patterns
        results = {q.name: [] for q in queries}
        entities = index.names
        for ident in index:
            ident_filters = dict(zip(entities, ident))
            for q in queries:
                result = q(self, **ident_filters, **filters)
                if not result: result = [None]
                results[q.name] += result
        return results

    def dataset(self, observations, levels,
                ignore=None, queries=None, additional_tables=None, **filters):
        ident = list(observations) + list(levels)
        index, (observations, levels, entities) = self.generate_index(
            observations, levels, ignore)
        data = self.query_all(index, queries, **filters)
        df = pd.DataFrame(
            data=data,
            index=index)
        df_aux = __class__._read_additional_tables(additional_tables, entities)
        #TODO
        # Determine if we should concat first or delete nulls first. There's not
        # a great sensible way to ensure null deletion occurs on the basis of data
        # missing from the query frame vs. data missing from additional frames, if
        # this is even desirable. This order would additionally impact the concat
        # join type.
        df = __class__._concat_frames(df, df_aux)
        nd = NullDrop(observations, levels)
        return nd(df)

    @staticmethod
    def _read_additional_tables(paths, entities):
        """
        Read a set of TSV-formatted data files into DataFrames with a specified
        set of index columns.

        Parameters
        ----------
        paths : list(str)
            List of paths to additional TSV-formatted data files.
        entities : list
            List of identifier entities corresponding to index columns in the
            data files. These must be the first columns in the data files.

        Returns
        -------
        list(DataFrame)
            List of DataFrames read from the files in `paths`.
        """
        if paths is None:
            return []
        idx = list(range(len(entities)))
        return [pd.read_csv(p, sep='\t', index_col=idx) for p in paths]

    @staticmethod
    def _concat_frames(df, df_aux):
        """
        Concatenate a data frame with a list of additional data frames.
        """
        dfs = [df] + df_aux
        return pd.concat(dfs, axis=1)

class NullDrop:
    """
    Object for pruning unnecessary rows from data frames. See `drop_null` for
    a brief overview.
    """
    def __init__(self, observations, levels):
        self.observations = observations
        self.levels = levels

    def __call__(self, df):
        return self.drop_null(df)

    def drop_null(self, df):
        """
        Remove from a data frame any rows corresponding to either
        - observations missing from the data at all levels
        - identifier combinations (levels) that do not exist for any
          observation

        For instance, if subject 137 has no scans, it will be dropped. If no
        subject has a third run of rest data, then all rows corresponding to
        that run are dropped. (...assuming subjects correspond to observations,
        and runs x tasks correspond to levels.)
        """
        df = self._delete_null_levels(df)
        df = self._delete_null_obs(df)
        return df

    def _delete_null_levels(self, df):
        """Remove from the data frame any identifier combinations that are
        absent from all observations."""
        return self._delete_nulls(df, self._all_levels)

    def _delete_null_obs(self, df):
        """Remove from the data frame any missing observations."""
        return self._delete_nulls(df, self._all_observations)

    def _delete_nulls(self, df, all_combinations):
        """
        Remove from the data frame any identifier combinations with missing data
        across all other levels. Abstract function called by `delete_null_levels`
        and `delete_null_obs`.
        """
        for c in all_combinations(df.index):
            if __class__._level_null(df, c):
                df = df.drop(df.loc(axis=0)[c].index)
        return df

    def _all_levels(self, index):
        """
        Obtain all extant within-observation identifier combinations in a
        DataFrame MultiIndex.
        """
        return __class__._all_combinations(
            index, combine=self.levels, control=self.observations)

    def _all_observations(self, index):
        """
        Obtain all extant observation identifier combinations in a DataFrame
        MultiIndex.
        """
        return __class__._all_combinations(
            index, combine=self.observations, control=self.levels)

    @staticmethod
    def _all_combinations(index, combine, control):
        """
        Obtain slices into all subsets of a DataFrame that share common MultiIndex
        subset assignments. Abstract function called by `all_levels` and
        `all_observations`. This function only obtains slicing tuples; it does not
        index the DataFrame to extract them.

        Parameters
        ----------
        index : MultiIndex
            Pandas MultiIndex from which all salient combinations are sliced.
        combine : list(str)
            Indexing variables to combine when creating each slice.
        control : list(str)
            Indexing variables to ignore when creating each slice.

        Returns
        -------
        list(tuple)
            List of indices into all unique MultiIndex assignments of `combine`
            indexing variables.
        """
        idx = []
        gen = []
        for name, c in zip(index.names, index.levels):
            if name in combine:
                idx += [None]
                gen += [list(c)]
            elif name in control:
                idx += [slice(None)]
        combinations = product(*gen)
        return [tuple(__class__._fill_idx_pattern(c, idx)) for c in combinations]

    @staticmethod
    def _fill_idx_pattern(level, idx):
        """
        Complete the indexing pattern in `idx` using the items in `level`. Any
        instance of None in `idx` is replaced by the first unused element of
        `level`. Helper function for `_all_combinations` routines.
        """
        level = list(level)
        return [i if i is not None else level.pop(0) for i in idx]

    @staticmethod
    def _level_null(df, level):
        """
        Return True if the entry corresponding to a specified index level is null
        for at least one variable across all rows of a DataFrame.

        Parameters
        ----------
        df : DataFrame
            DataFrame to check for a null level.
        level : tuple
            Tuple slicing into the DataFrame's index. The tuple should specify
            index values for the input DataFrame.

        Returns
        -------
        bool
            Indicates whether the input level is null, i.e., whether the entry is
            null for at least one variable in every row of the DataFrame
            corresponding to that level.
        """
        any_null = False
        for v in df.columns:
            v_null = df.loc(axis=0)[level][v].isnull()
            any_null = (any_null | v_null)
        return any_null.all()

class LightGrabber(LightGrabberBase, _IndexQueryMixin):
    pass
