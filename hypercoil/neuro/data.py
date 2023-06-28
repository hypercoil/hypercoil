# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Data transformations.
"""
import dataclasses, os, glob, string, time, re, tarfile
import json, pickle
import datalad.api as datalad
import numpy as np
import pandas as pd
import tensorflow as tf
from io import BytesIO
from typing import (
    Any, Callable, Literal, Mapping, Optional, Sequence, Tuple, Type, Union,
)


class AbsentFromInstance:
    """Sentinel object for absent values"""


@dataclasses.dataclass
class Categoricals:
    definition: Mapping[str, Sequence[Any]]

    def decode(self, var: str, code: Union[int, np.ndarray]) -> Any:
        if isinstance(code, int):
            code = [code]
        elif isinstance(code, np.ndarray):
            code = code.nonzero()[-1].tolist()
        return [self.definition[var][c] for c in code]

    def encode(
        self,
        var: str,
        value: Union[Any, Sequence[Any]]
    ) -> np.ndarray:
        if isinstance(value, str):
            value = [value]
        try:
            iter(value)
        except TypeError:
            value = [value]
        value = np.array([self.definition[var].index(v) for v in value])
        return np.eye(self.level_count(var))[value]

    def level_count(self, var: str) -> int:
        return len(self.definition[var])

    def __getitem__(self, var: str) -> Sequence[Any]:
        return self.definition[var]

    def __setitem__(self, var: str, values: Sequence[Any]):
        self.definition[var] = values


@dataclasses.dataclass
class Header:
    categoricals: Categoricals = dataclasses.field(default=Categoricals({}))

    @property
    def constructor(self):
        _constructor = {}
        for field in dataclasses.fields(self):
            if field.name == 'categoricals':
                _constructor[field.name] = self.categoricals.definition
            else:
                _constructor[field.name] = getattr(self, field.name)
        return _constructor

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.constructor, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            constructor = pickle.load(f)
        return cls(**constructor)


@dataclasses.dataclass
class FunctionalMRIDataHeader(Header):
    surface_mask: Optional[np.ndarray] = dataclasses.field(default=None)
    volume_mask: Optional[np.ndarray] = dataclasses.field(default=None)


@dataclasses.dataclass
class AgnosticSchema:
    schema: Mapping[str, Any]

    def __getitem__(self, key):
        return self.schema[key]

    def __setitem__(self, key, value):
        self.schema[key] = value

    def __contains__(self, key):
        return key in self.schema

    def __iter__(self):
        return iter(self.schema)

    def __len__(self):
        return len(self.schema)

    def __repr__(self):
        return repr(self.schema)

    def __str__(self):
        return str(self.schema)

    @property
    def constructor(self):
        return {
            k: (type(v).__name__, v.dtype)
            for k, v in self.schema.items()
        }

    def to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.constructor, f, indent=4)

    @classmethod
    def from_json(cls, path):
        with open(path, 'r') as f:
            schema = json.load(f)
        for k, (t, d) in schema.items():
            if t == 'DataArray':
                schema[k] = DataArray(dtype=d)
            elif t == 'InexactArray':
                schema[k] = InexactArray(dtype=d)
            elif t == 'ExactNumeric':
                schema[k] = ExactNumeric(dtype=d)
            elif t == 'InexactNumeric':
                schema[k] = InexactNumeric(dtype=d)
            elif t == 'String':
                schema[k] = String()
        return cls(schema)


@dataclasses.dataclass
class DataArray:
    dtype: str = 'bool'

    @property
    def dtype_tf(self):
        return tf.string

    def encode_tf(self, data, serialise=True):
        if serialise:
            data = tf.io.serialize_tensor(data).numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))

    def decode_tf(self, feature):
        return tf.io.parse_tensor(feature, out_type=getattr(tf, self.dtype))

    def encode_tar(self, data):
        stream = BytesIO()
        np.lib.format.write_array(stream, np.asarray(data), allow_pickle=False)
        return stream.getvalue()


class InexactArray(DataArray):
    dtype: str = 'float32'
    def decode_tf(self, feature):
        # Why double? No idea
        return tf.io.parse_tensor(feature, out_type=tf.double)

@dataclasses.dataclass
class Numeric:
    @property
    def dtype_tf(self):
        return getattr(tf, self.dtype)

    def decode_tf(self, feature):
        return tf.cast(feature, dtype=self.dtype_tf)

    def encode_tar(self, data):
        return str(data).encode('utf-8')


@dataclasses.dataclass
class InexactNumeric(Numeric):
    dtype: str = 'float32'

    def encode_tf(self, data, serialise=True):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[data]))


@dataclasses.dataclass
class ExactNumeric(Numeric):
    dtype: str = 'int64'

    def encode_tf(self, data, serialise=True):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[data]))


@dataclasses.dataclass
class String(DataArray):
    dtype: str = 'string'

    def encode_tf(self, data, serialise=True):
        data = data.encode('utf-8')
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))

    def decode_tf(self, feature):
        return tf.cast(feature, dtype=tf.string)

    def encode_tar(self, data):
        return data.encode('utf-8')


#lru cache?
def filesystem_dataset(
    *,
    root: str,
    references: Sequence[str],
    regex: Mapping[str, str],
    dtypes: Mapping[str, str],
    entities: set[str],
    pivot: Optional[str] = None,
    tables: Optional[Sequence[pd.DataFrame]] = None,
    filters: Optional[Sequence[callable]] = None,
    **params,
) -> Mapping:
    dtypes = {k: v for k, v in dtypes.items() if k in references}

    # start = time.time()
    # fnames = (
    #     glob.glob(os.path.join(root, '**', pattern), recursive=True)
    #     for pattern in dtypes.values()
    # )
    # fnames = list(chain(*fnames))
    # t = time.time() - start
    # print(f'globbing took {t} seconds and found {len(fnames)} files')

    start = time.time()
    # Some notes:
    # (1) I've tried wrapping this in a list vs a generator, and the generator
    #     seems to be slightly faster.
    # (2) I've also tried using pathlib.PurePath(f).match(pattern) instead of
    #     glob.fnmatch.fnmatch(f, pattern) and it's substantially slower.
    # (3) Finally, I've tried using pathlib.Path(f, dir) instead of
    #     os.path.join(f, dir) and it's also marginally slower.
    # (4) I've tried using os.walk instead of glob.glob and it's substantially
    #     faster (nearly 3x) when we set up walk as we have done here.
    fnames = (
        os.path.join(dir, f)
        for dir, _, files in os.walk(root) for f in files
        if any(
            glob.fnmatch.fnmatch(f, pattern[0])
            for pattern in dtypes.values()
        )
    )
    fnames = list(fnames)
    t = time.time() - start
    print(f'os.walk took {t} seconds and found {len(fnames)} files')

    start = time.time()
    fnames_parsed = {fname: {
        k: v for i in (match_or_null(expr, fname)
        for expr in regex.values())
        for k, v in i.items()
    } for fname in fnames}
    t = time.time() - start
    print(f'entity extraction took {t} seconds and '
          f'parsed {len(fnames_parsed)} filenames')

    start = time.time()
    cols = (set(e.keys()) for e in fnames_parsed.values())
    cols = set.union(*cols)
    filedict = {col: [None for _ in range(len(fnames))] for col in cols}
    filedict['fname'] = fnames
    for i, (fname, instance) in enumerate(fnames_parsed.items()):
        for col in cols:
            filedict[col][i] = instance.get(col, None)
        filedict['fname'][i] = fname
    files = pd.DataFrame(filedict)
    t = time.time() - start
    print(f'building a dataframe took {t} seconds and registered '
          f'{len(files)} files')

    start = time.time()
    entities = tuple(e for e in entities if e in files.columns)
    if pivot is None:
        files = pd.DataFrame(
            files.pivot(index=entities, columns=()).fname)
    else:
        pivot_key = {v[1]: k for k, v in dtypes.items()}
        files['dtype'] = [pivot_key[k] for k in files[pivot]]
        files = files.pivot(index=entities, columns='dtype', values='fname')
        files.columns.name = None
    t = time.time() - start
    print(f'pivoting took {t} seconds and found {len(files)} instances')

    start = time.time()
    if tables is not None:
        for table in tables:
            index = [e for e in entities if e in table.columns]
            table = table.set_index(index)
            files = files.join(table, how='left', validate='m:1')
    t = time.time() - start
    print(f'merging tables took {t} seconds')

    start = time.time()
    if filters is not None:
        files = files.reset_index()
        for f in filters:
            files = f(files)
        files = files.set_index([*entities])
    t = time.time() - start
    print(f'filtering took {t} seconds and left {len(files)} instances')

    return {
        'fnames': fnames,
        'parsed': fnames_parsed,
        'dataset': files,
        'filetypes': list(dtypes.keys()),
        **params
    }


def match_or_null(pattern: str, string: str) -> Optional[Mapping[str, str]]:
    match = re.match(pattern, string)
    if match:
        return match.groupdict()
    else:
        return {}
