# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Data transformations.
"""
import dataclasses, os, glob, string, time, re, tarfile
import json, pickle
import datalad.api as datalad
import nibabel as nb
import numpy as np
import pandas as pd
import tensorflow as tf
import templateflow.api as tflow
from contextlib import ExitStack
from io import BytesIO
from typing import (
    Any, Callable, Literal, Mapping, Optional, Sequence, Tuple, Type, Union,
)
from hypercoil.formula.dfops import ConfoundFormulaGrammar
from hypercoil.neuro.const import template_dict, neuromaps_fetch_fn
from hypercoil.viz.flows import replicate, direct_transform


NIMG_ENTITIES = {'subject', 'session', 'run', 'task'}
BIDS_REGEX = {
    'datatype': '.*/(?P<datatype>[^/]*)/[^/]*',
    'subject': '.*/[^/]*sub-(?P<subject>[^_]*)[^/]*',
    'session': '.*/[^/]*ses-(?P<subject>[^_]*)[^/]*',
    'run': '.*/[^/]*run-(?P<run>[^_]*)[^/]*',
    'task': '.*/[^/]*task-(?P<task>[^_]*)[^/]*',
    'space': '.*/[^/]*space-(?P<space>[^_]*)[^/]*',
    'desc': '.*/[^/]*desc-(?P<desc>[^_]*)[^/]*',
    'suffix': '.*/[^/]*_(?P<suffix>[^/_\.]*)\..*',
    'extension': '.*/[^/\.]*(?P<extension>\..*)$'
}
BIDS_DTYPES = {
    'surface_L': ('*_desc-preproc*_bold.L.func.gii', '.L.func.gii'),
    'surface_R': ('*_desc-preproc*_bold.R.func.gii', '.R.func.gii'),
    'volume': ('*_desc-preproc*_bold.nii.gz', '.nii.gz'),
    'confounds': ('*_desc-confounds*_timeseries.tsv', '.tsv'),
    'confounds_metadata': ('*_desc-confounds*_timeseries.json', '.json'),
}
BIDS_DTYPES_LEGACY = {
    'surface_L': ('*_bold*{space}.L.func.gii', '.L.func.gii'),
    'surface_R': ('*_bold*{space}.R.func.gii', '.R.func.gii'),
    'volume': ('*_bold*{space}*_preproc.nii.gz', '.nii.gz'),
    'confounds': ('*_bold*_confounds.tsv', '.tsv'),
}
HCP_REGEX = {}
HCP_DTYPES = {}


def _null_op_one_arg(arg: Any) -> Any:
    return arg


def _null_op(**params: Mapping) -> Mapping:
    return params


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


def configure_transforms(
    *,
    dataset: pd.DataFrame,
    filetypes: Sequence[str],
    header: Optional[Type[Header]] = None,
    header_writers: Optional[Mapping[str, callable]] = None,
    instance_transforms: Optional[Sequence[Tuple[callable, int]]] = None,
    path_transform: callable = _null_op_one_arg,
    categorical: Optional[Sequence[str]] = None,
    infer_categorical: bool = True,
    **params,
) -> Mapping:
    index = dataset.index.names
    dataset = dataset.reset_index()
    filetypes = set(list(filetypes) + ['fname']).intersection(dataset.columns)
    scalars = set(dataset.columns) - filetypes

    scalar_dtypes = {
        k: v for k, v in dataset.dtypes.to_dict().items()
        if k in scalars
    }
    inexact_dtypes = {'float16', 'float32', 'float64', 'float128'}
    schema = {}
    for k, v in scalar_dtypes.items():
        # We assume any object dtype is a string.
        if isinstance(v, str) or str(v) == 'object':
            schema[k] = String()
        elif str(v) in inexact_dtypes:
            schema[k] = InexactNumeric(dtype=str(v))
        else:
            schema[k] = ExactNumeric(dtype=str(v))

    if infer_categorical and categorical is None:
        categorical = [
            k for k, v in scalar_dtypes.items()
            if not any(v == t for t in inexact_dtypes)
        ]
    if categorical is not None:
        categoricals = {
            k: dataset[k].unique().tolist()
            for k in categorical
        }
        categoricals = Categoricals(definition=categoricals)
    else:
        categorical = []
        categoricals = None

    header = header or Header
    header_writers = header_writers or {}
    header = header(
        **{k: v() for k, v in header_writers.items()},
        categoricals=categoricals,
    )

    instance_transform = scalar_transform(scalars=scalars)(header)
    if instance_transforms is not None:
        # Two reversals are necessary to ensure that the transforms are
        # applied in the correct order. The reverse sort is necessary in order
        # to apply transforms in order of their priority. Reversing the list
        # of transforms is necessary because the first transform in the list
        # is the one that is applied last.
        instance_transforms = sorted(
            reversed(instance_transforms),
            key=lambda e: e[1],
            reverse=True
        )
        for transform, _ in instance_transforms:
            instance_transform = transform(
                header=header,
                path_transform=path_transform,
                f=instance_transform,
            )

    dataset = dataset.set_index(index)
    return {
        'dataset': dataset,
        'header': header,
        #'filetypes': filetypes,
        'instance_transform': instance_transform,
        'schema': schema,
        **params,
    }


def write_records(
    *,
    write_path: str,
    dataset: pd.DataFrame,
    header: Header,
    instance_transform: callable,
    instance_writers: Optional[Sequence[callable]] = None,
    shard_writers: Optional[Sequence[callable]] = None,
    schema: Mapping[str, Any],
    write_header: bool = True,
    write_schema: bool = True,
    fname_entities: Optional[Mapping[str, str]] = None,
    **params,
) -> Mapping[str, Any]:
    HEADER_FNAME = 'dataset.hdr'
    SCHEMA_FNAME = 'schema.json'
    header_path = os.path.join(write_path, HEADER_FNAME)
    schema_path = os.path.join(write_path, SCHEMA_FNAME)

    os.makedirs(write_path, exist_ok=True)
    schema = AgnosticSchema(schema)

    if write_header:
        header.save(header_path)
    if write_schema:
        schema.to_json(schema_path)

    instance_writers = instance_writers or []
    shard_writers = shard_writers or []
    instance_writers = [
        (writer(schema.schema), *pattern(dir=write_path, **fname_entities))
        for writer, pattern in instance_writers
    ]
    with ExitStack() as stack:
        instance_writers = tuple(
            (writer, stack.enter_context(context), filename)
            for writer, context, filename in instance_writers
        )
        for i, instance in dataset.reset_index().iterrows():
            instance = instance_transform(dataset=instance)
            instance_id = [(e, instance[e]) for e in dataset.index.names]
            instance_id = '_'.join(['-'.join(e) for e in instance_id])
            instance_id = instance_id or f"instance_{i}"
            for writer, context, filename in instance_writers:
                writer(
                    instance=instance,
                    context=context,
                    instance_id=instance_id
                )
        for writer, filename in shard_writers:
            writer(shard=dataset, filename=filename)
    return {
        'write_path': write_path,
        'dataset': dataset,
        'header': header,
        'instance_transform': instance_transform,
        'schema': schema,
        **params,
    }


def match_or_null(pattern: str, string: str) -> Optional[Mapping[str, str]]:
    match = re.match(pattern, string)
    if match:
        return match.groupdict()
    else:
        return {}


def scalar_transform(scalars: Optional[Sequence[str]] = None) -> callable:
    def close_header(header: Header) -> callable:
        categoricals = header.categoricals
        def transform_dataset(
            dataset: pd.Series,
            **features,
        ) -> Mapping[str, Any]:
            transformed = dataset[list(scalars)].to_dict()
            #for var in categoricals.definition:
            #    transformed[var] = categoricals.encode(var, transformed[var])
            return {**features, **transformed}
        return transform_dataset
    return close_header


def fmri_dataset_transform(
    f_configure: callable,
    xfm: callable = direct_transform,
) -> callable:
    def transformer_f_configure(
        *,
        header: Optional[Type[Header]] = None,
    ) -> Mapping:
        return {
            'header': header or FunctionalMRIDataHeader,
        }

    def f_configure_transformed(
        *,
        header: Optional[Type[Header]] = None,
        **params,
    ) -> Mapping:
        return xfm(f_configure, transformer_f_configure)(**params)(
            header=header,
        )

    return f_configure_transformed


def fmriprep_dataset():
    def transform(
        f_index: callable = filesystem_dataset,
        f_configure: callable = configure_transforms,
        f_record: callable = write_records,
        xfm: callable = direct_transform,
    ) -> callable:
        def transformer_f_index(
            regex: Optional[Mapping[str, str]] = None,
            dtypes: Optional[Mapping[str, str]] = None,
            entities: Optional[set[str]] = None,
            pivot: Optional[str] = None,
        ) -> Mapping:
            return {
                'dtypes': dtypes or {**BIDS_DTYPES},
                'regex': regex or {**BIDS_REGEX},
                'entities': entities or NIMG_ENTITIES,
                'pivot': pivot or 'extension',
            }

        def f_index_transformed(
            *,
            regex: Optional[Mapping[str, str]] = None,
            dtypes: Optional[Mapping[str, str]] = None,
            entities: Optional[set[str]] = None,
            pivot: Optional[str] = None,
            **params,
        ):
            return xfm(f_index, transformer_f_index)(**params)(
                regex=regex,
                dtypes=dtypes,
                entities=entities,
                pivot=pivot,
            )

        f_configure_transformed = fmri_dataset_transform(f_configure, xfm)

        return (
            f_index_transformed,
            f_configure_transformed,
            f_record,
        )
    return transform


def hcp_dataset():
    def transform(
        f_index: callable = filesystem_dataset,
        f_configure: callable = configure_transforms,
        f_record: callable = write_records,
        xfm: callable = direct_transform,
    ) -> callable:
        def transformer_f_index(
            regex: Optional[Mapping[str, str]] = None,
            dtypes: Optional[Mapping[str, str]] = None,
            entities: Optional[Sequence[str]] = None,
        ) -> Mapping:
            return {
                'dtypes': dtypes or {**HCP_DTYPES},
                'regex': regex or {**HCP_REGEX},
                'entities': entities or NIMG_ENTITIES,
            }

        def f_index_transformed(
            *,
            regex: Optional[Mapping[str, str]] = None,
            dtypes: Optional[Mapping[str, str]] = None,
            entities: Optional[Sequence[str]] = None,
            **params,
        ):
            return xfm(f_index, transformer_f_index)(**params)(
                regex=regex,
                dtypes=dtypes,
                entities=entities,
            )

        f_configure_transformed = fmri_dataset_transform(f_configure, xfm)

        return (
            f_index_transformed,
            f_configure_transformed,
            f_record,
        )
    return transform


def transformer_bids_specs(
    reference_name: Union[str, Sequence[str]],
    spec: Mapping[str, str]
) -> callable:
    if isinstance(reference_name, str):
        refnames = (reference_name,)
    else:
        refnames = reference_name

    def format_ref(ref: str) -> str:
        base = {
            field: '*' for _, field, _, _
            in string.Formatter().parse(ref)
        }
        args = {**base, **spec}
        return ref.format(**args)

    def transformer(
        dtypes: Mapping[str, str],
        references: Optional[Sequence[str]] = None,
    ) -> Mapping:
        refs = [dtypes[refname] for refname in refnames]
        refs = {
            refname: (format_ref(ref[0]), ref[1])
            for ref, refname in zip(refs, refnames)
        }
        dtypes = {**dtypes, **refs}
        if references is None:
            references = refnames
        else:
            references = tuple(list(references) + list(refnames))
        return {
            'dtypes': dtypes,
            'references': references,
        }
    return transformer


def record_bold_surface(
    mask: bool = True,
    priority: int = 0,
    **spec: Mapping[str, str],
) -> callable:
    """
    Add surface datasets in the specified space to a data record.
    """
    def header_writer():
        def tflow_cfg():
            return {
                'query': template.TFLOW_MASK_QUERY,
                'fetch': tflow.get,
            }
        def nmaps_cfg():
            return {
                'query': template.NMAPS_MASK_QUERY,
                'fetch': neuromaps_fetch_fn,
            }

        space = spec.get('space', 'fsaverage5')
        density = spec.get('density', None)
        template = template_dict()[space]
        for cfg in (tflow_cfg, nmaps_cfg):
            try:
                fetch_fn = cfg()['fetch']
                mask_query = cfg()['query']
                if density is not None:
                    mask_query['density'] = density
                lh_mask, rh_mask = (
                    fetch_fn(**mask_query, hemi="L"),
                    fetch_fn(**mask_query, hemi="R")
                )
                return {
                    'L': nb.load(lh_mask).darrays[0].data.astype(bool),
                    'R': nb.load(rh_mask).darrays[0].data.astype(bool),
                }
            except Exception as e:
                continue
        raise e

    def instance_transform(
        header: Header,
        path_transform: callable,
        f: callable,
        xfm: callable = direct_transform
    ) -> callable:
        def transformer_f(dataset: pd.Series) -> Mapping:
            path_L = path_transform(dataset['surface_L'])
            path_R = path_transform(dataset['surface_R'])
            L = nb.load(path_L)
            sample_time = float(L.darrays[0].meta['TimeStep'])
            if sample_time > 10: # assume milliseconds
                sample_time /= 1000
            L = np.stack([e.data for e in L.darrays], axis=-1)
            R = nb.load(path_R)
            R = np.stack([e.data for e in R.darrays], axis=-1)
            if mask:
                surface_mask = header.surface_mask
                L = L[surface_mask['L']]
                R = R[surface_mask['R']]
            surface = np.concatenate([L, R], axis=0)
            return {
                'dataset': dataset,
                'surface': surface,
                'sample_time': sample_time,
            }

        def f_transformed(
            *,
            dataset: pd.Series,
            **params,
        ):
            return xfm(f, transformer_f)(**params)(dataset=dataset)

        return f_transformed

    def transform(
        f_index: callable = filesystem_dataset,
        f_configure: callable = configure_transforms,
        f_record: callable = write_records,
        xfm: callable = direct_transform,
    ) -> callable:
        transformer_f_index = transformer_bids_specs(
            ('surface_L', 'surface_R'), spec)

        def f_index_transformed(
            *,
            dtypes: Mapping[str, str],
            references: Optional[Sequence[str]] = None,
            **params,
        ):
            return xfm(f_index, transformer_f_index)(**params)(
                dtypes=dtypes,
                references=references,
            )

        def transformer_f_configure(
            *,
            header_writers: Optional[Mapping[str, callable]] = None,
            instance_transforms: Optional[Sequence[Tuple[callable, int]]] = None,
        ) -> Mapping:
            if mask:
                if header_writers is None:
                    header_writers = {}
                header_writers['surface_mask'] = header_writer
            if instance_transforms is None:
                instance_transforms = []
            instance_transforms.append((instance_transform, priority))
            return {
                'header_writers': header_writers,
                'instance_transforms': instance_transforms,
            }

        def f_configure_transformed(
            *,
            header_writers: Optional[Mapping[str, callable]] = None,
            instance_transforms: Optional[Sequence[Tuple[callable, int]]] = None,
            **params,
        ):
            return xfm(f_configure, transformer_f_configure)(**params)(
                header_writers=header_writers,
                instance_transforms=instance_transforms,
            )

        def transformer_f_record(
            schema: Mapping = None,
        ) -> Mapping:
            if schema is None:
                schema = {}
            schema["surface"] = InexactArray()
            schema["sample_time"] = InexactNumeric('float32')
            return {
                "schema": schema,
            }

        def f_record_transformed(
            *,
            schema: Mapping = None,
            **params,
        ):
            return xfm(f_record, transformer_f_record)(**params)(
                schema=schema,
            )

        return (
            f_index_transformed,
            f_configure_transformed,
            f_record_transformed,
        )
    return transform


def record_bold_volume(
    mask: Union[bool, str, Mapping] = True,
    priority: int = 0,
    **spec: Mapping[str, str],
) -> callable:
    """
    Add volumetric datasets in the specified space to a data record.
    """
    #TODO: Allow specification of explicit mask, for instance grey matter
    #      voxels only
    def header_writer():
        if isinstance(mask, str):
            brain_mask = mask
        elif isinstance(mask, dict):
            space = (
                mask.get('space', None) or
                spec.get('space', 'MNI152NLin2009cAsym')
            )
            brain_mask = tflow.get({**mask, **{'space': space}})
        else:
            space = spec.get('space', 'MNI152NLin2009cAsym')
            resolution = spec.get('resolution', 2)
            brain_mask = tflow.get(
                template=space,
                desc='brain',
                suffix='mask',
                resolution=resolution,
            )
        return nb.load(brain_mask).get_fdata().astype(bool)

    def instance_transform(
        header: Header,
        path_transform: callable,
        f: callable,
        xfm: callable = direct_transform
    ) -> callable:
        def transformer_f(dataset: pd.Series) -> Mapping:
            path_volume = path_transform(dataset['volume'])
            volume = nb.load(path_volume)
            sample_time = volume.header.get_zooms()[-1]
            if sample_time > 10: # assume milliseconds
                sample_time /= 1000
            volume = volume.get_fdata()
            if mask:
                volume = volume[header.volume_mask]
            return {
                'dataset': dataset,
                'volume': volume,
                'sample_time': sample_time,
            }

        def f_transformed(
            *,
            dataset: pd.Series,
            **params,
        ):
            return xfm(f, transformer_f)(**params)(dataset=dataset)

        return f_transformed

    def transform(
        f_index: callable = filesystem_dataset,
        f_configure: callable = configure_transforms,
        f_record: callable = write_records,
        xfm: callable = direct_transform,
    ) -> callable:
        transformer_f_index = transformer_bids_specs('volume', spec)

        def f_index_transformed(
            *,
            dtypes: Mapping[str, str],
            references: Optional[Sequence[str]] = None,
            **params,
        ):
            return xfm(f_index, transformer_f_index)(**params)(
                dtypes=dtypes,
                references=references,
            )

        def transformer_f_configure(
            *,
            header_writers: Optional[Mapping[str, callable]] = None,
            instance_transforms: Optional[Sequence[Tuple[callable, int]]] = None,
        ) -> Mapping:
            if mask:
                if header_writers is None:
                    header_writers = {}
                header_writers['volume_mask'] = header_writer
            if instance_transforms is None:
                instance_transforms = []
            instance_transforms.append((instance_transform, priority))
            return {
                'header_writers': header_writers,
                'instance_transforms': instance_transforms,
            }

        def f_configure_transformed(
            *,
            header_writers: Optional[Mapping[str, callable]] = None,
            instance_transforms: Optional[Sequence[Tuple[callable, int]]] = None,
            **params,
        ):
            return xfm(f_configure, transformer_f_configure)(**params)(
                header_writers=header_writers,
                instance_transforms=instance_transforms,
            )

        def transformer_f_record(
            schema: Mapping = None,
        ) -> Mapping:
            if schema is None:
                schema = {}
            schema["volume"] = InexactArray()
            schema["sample_time"] = InexactNumeric('float32')
            return {
                "schema": schema,
            }

        def f_record_transformed(
            *,
            schema: Mapping = None,
            **params,
        ):
            return xfm(f_record, transformer_f_record)(**params)(
                schema=schema,
            )

        return (
            f_index_transformed,
            f_configure_transformed,
            f_record_transformed,
        )
    return transform


def record_confounds(
    model: Optional[str] = None,
    priority: int = 0,
) -> callable:
    """
    Add confound datasets to a data record.
    """
    def instance_transform(
        header: Header,
        path_transform: callable,
        f: callable,
        xfm: callable = direct_transform
    ) -> callable:
        def transformer_f(dataset: pd.Series) -> Mapping:
            path_confounds = path_transform(dataset['confounds'])
            confounds = pd.read_csv(path_confounds, sep='\t')
            metadata = dataset.get('confounds_metadata', None)
            if metadata is not None:
                with open(metadata, 'r') as f:
                    metadata = json.load(f)
            if model is not None:
                model_f = ConfoundFormulaGrammar().compile(model)
                confounds = model_f(confounds, metadata)
            return {
                'dataset': dataset,
                'confounds': confounds.values.T,
                'confounds_names': list(confounds.columns),
            }

        def f_transformed(
            *,
            dataset: pd.Series,
            **params,
        ):
            return xfm(f, transformer_f)(**params)(dataset=dataset)

        return f_transformed

    def transform(
        f_index: callable = filesystem_dataset,
        f_configure: callable = configure_transforms,
        f_record: callable = write_records,
        xfm: callable = direct_transform,
    ) -> callable:
        def transformer_f_index(
            references: Optional[Sequence[str]] = None,
        ) -> Mapping:
            if references is None:
                references = ('confounds', 'confounds_metadata')
            else:
                references = tuple(
                    list(references) +
                    ['confounds', 'confounds_metadata']
                )
            return {
                'references': references,
            }

        def f_index_transformed(
            *,
            references: Optional[Sequence[str]] = None,
            **params,
        ):
            return xfm(f_index, transformer_f_index)(**params)(
                references=references,
            )

        def transformer_f_configure(
            *,
            instance_transforms: Optional[Sequence[Tuple[callable, int]]] = None,
        ) -> Mapping:
            if instance_transforms is None:
                instance_transforms = []
            instance_transforms.append((instance_transform, priority))
            return {'instance_transforms': instance_transforms}

        def f_configure_transformed(
            *,
            instance_transforms: Optional[Sequence[Tuple[callable, int]]] = None,
            **params,
        ):
            return xfm(f_configure, transformer_f_configure)(**params)(
                instance_transforms=instance_transforms,
            )

        def transformer_f_record(
            schema: Mapping = None,
        ) -> Mapping:
            if schema is None:
                schema = {}
            schema["confounds"] = InexactArray()
            schema["confounds_names"] = DataArray('string')
            return {
                "schema": schema,
            }

        def f_record_transformed(
            *,
            schema: Mapping = None,
            **params,
        ):
            return xfm(f_record, transformer_f_record)(**params)(
                schema=schema,
            )

        return (
            f_index_transformed,
            f_configure_transformed,
            f_record_transformed,
        )
    return transform


def record_tsv(*paths: Sequence[str]) -> callable:
    """
    Add a TSV dataset to a data record.
    """
    def transform(
        f_index: callable = filesystem_dataset,
        f_configure: callable = configure_transforms,
        f_record: callable = write_records,
        xfm: callable = direct_transform,
    ) -> callable:
        def transformer_f_index(
            tables: Optional[Sequence[pd.DataFrame]] = None,
            entities: Optional[Sequence[str]] = None,
        ) -> Mapping:
            tsv_tables = (
                pd.read_csv(
                    path,
                    sep='\t',
                    converters={e: str for e in entities},
                )
                for path in paths
            )
            if tables is None:
                tables = tuple(tsv_tables)
            else:
                tables = tuple(list(tables) + list(tsv_tables))
            return {
                'tables': tables,
                'entities': entities,
            }

        def f_index_transformed(
            *,
            tables: Optional[Sequence[pd.DataFrame]] = None,
            entities: Optional[Sequence[str]] = None,
            **params,
        ):
            return xfm(f_index, transformer_f_index)(**params)(
                tables=tables,
                entities=entities,
            )

        return (
            f_index_transformed,
            f_configure,
            f_record,
        )
    return transform


def polynomial_detrend(
    *,
    order: Union[int, Literal['auto']] = 1,
    features: Optional[Sequence[str]] = ('volume', 'surface', 'confounds'),
    axes: Optional[Sequence[int]] = None,
    priority: int = 1,
) -> callable:
    def instance_transform(
        header: Header,
        path_transform: callable,
        f: callable,
        xfm: callable = direct_transform,
    ) -> callable:
        def transformer_f(**transformed_features) -> Mapping:
            print('detrending time series')
            _axes = axes or [-1] * len(features)
            for ax, feature in zip(_axes, features):
                data = transformed_features.get(feature, None)
                if data is None:
                    continue
                size = data.shape[ax]
                _order = order
                if order == 'auto':
                    sample_time = header['sample_time']
                    _order = int(1 + sample_time * size / 150)
                basis = [
                    np.polynomial.legendre.Legendre.basis(
                        d, (1, size)
                    ).linspace(size) for d in range(_order + 1)
                ]
                basis = np.stack([b for _, b in basis]).T
                if ax != 0:
                    data = data.swapaxes(0, ax)
                coefs = np.linalg.lstsq(basis, data, rcond=None)[0]
                data = data - basis @ coefs
                if ax != 0:
                    data = data.swapaxes(0, ax)
                print(feature, data.shape)
                transformed_features[feature] = data
            return transformed_features

        def f_transformed(dataset: pd.Series, **params):
            return xfm(f, transformer_f)(dataset=dataset)(**params)

        return f_transformed

    def transform(
        f_index: callable = filesystem_dataset,
        f_configure: callable = configure_transforms,
        f_record: callable = write_records,
        xfm: callable = direct_transform,
    ) -> Tuple[callable, callable, callable]:
        def transformer_f_configure(
            *,
            instance_transforms: Optional[Sequence[Tuple[callable, int]]] = None,
        ) -> Mapping:
            if instance_transforms is None:
                instance_transforms = []
            instance_transforms.append((instance_transform, priority))
            return {'instance_transforms': instance_transforms}

        def f_configure_transformed(
            *,
            instance_transforms: Optional[Sequence[Tuple[callable, int]]] = None,
            **params,
        ):
            return xfm(f_configure, transformer_f_configure)(**params)(
                instance_transforms=instance_transforms,
            )

        return (
            f_index,
            f_configure_transformed,
            f_record,
        )
    return transform


def standardise(
    *,
    features: Optional[Sequence[str]] = ('volume', 'surface', 'confounds'),
    axes: Optional[Sequence[int]] = None,
    priority: int = 1,
) -> callable:
    def instance_transform(
        header: Header,
        path_transform: callable,
        f: callable,
        xfm: callable = direct_transform,
    ) -> callable:
        def transformer_f(**transformed_features) -> Mapping:
            print('z-scoring time series')
            _axes = axes or [-1] * len(features)
            for ax, feature in zip(_axes, features):
                data = transformed_features.get(feature, None)
                if data is None:
                    continue
                if ax != 0:
                    data = data.swapaxes(0, ax)
                data = (
                    (data - data.mean(axis=0, keepdims=True)) /
                    data.std(axis=0, keepdims=True)
                )
                if ax != 0:
                    data = data.swapaxes(0, ax)
                transformed_features[feature] = data
            return transformed_features

        def f_transformed(dataset: pd.Series, **params):
            return xfm(f, transformer_f)(dataset=dataset)(**params)

        return f_transformed

    def transform(
        f_index: callable = filesystem_dataset,
        f_configure: callable = configure_transforms,
        f_record: callable = write_records,
        xfm: callable = direct_transform,
    ) -> Tuple[callable, callable, callable]:
        def transformer_f_configure(
            *,
            instance_transforms: Optional[Sequence[Tuple[callable, int]]] = None,
        ) -> Mapping:
            if instance_transforms is None:
                instance_transforms = []
            instance_transforms.append((instance_transform, priority))
            return {'instance_transforms': instance_transforms}

        def f_configure_transformed(
            *,
            instance_transforms: Optional[Sequence[Tuple[callable, int]]] = None,
            **params,
        ):
            return xfm(f_configure, transformer_f_configure)(**params)(
                instance_transforms=instance_transforms,
            )

        return (
            f_index,
            f_configure_transformed,
            f_record,
        )
    return transform


def nanfill(
    *,
    features: Optional[Sequence[str]] = ('volume', 'surface', 'confounds'),
    fill_value_nan: Union[float, Sequence[float]] = 0.0,
    fill_value_inf: Optional[Union[float, Sequence[float]]] = None,
    record_mask: bool = False,
    priority: int = 1,
) -> callable:
    def instance_transform(
        header: Header,
        path_transform: callable,
        f: callable,
        xfm: callable = direct_transform,
    ) -> callable:
        def transformer_f(**transformed_features) -> Mapping:
            print('populating nan and inf values')
            _fillnan_all = fill_value_nan
            _fillinf_all = fill_value_inf
            if not isinstance(fill_value_nan, Sequence):
                _fillnan_all = [fill_value_nan] * len(features)
            if not isinstance(fill_value_inf, Sequence):
                _fillinf_all = [fill_value_inf] * len(features)
            _fillinf_pos = _fillinf_neg = fill_value_inf
            for feature, _fillinf, _fillnan in zip(
                features, _fillinf_all, _fillnan_all,
            ):
                data = transformed_features.get(feature, None)
                if data is None:
                    continue
                if _fillinf is not None:
                    if _fillinf > 0:
                        _fillinf_neg *= -1
                    else:
                        _fillinf_pos *= -1
                if record_mask:
                    nanmask = np.isnan(data)
                    infmask = np.isinf(data)
                    feature_nanmask = f'{feature}_nanmask'
                    feature_infmask = f'{feature}_infmask'
                    transformed_features[feature_nanmask] = nanmask
                    transformed_features[feature_infmask] = infmask
                data = np.nan_to_num(
                    data,
                    nan=_fillnan,
                    posinf=_fillinf_pos,
                    neginf=_fillinf_neg,
                )
                transformed_features[feature] = data
            return transformed_features

        def f_transformed(dataset: pd.Series, **params):
            return xfm(f, transformer_f)(dataset=dataset)(**params)

        return f_transformed

    def transform(
        f_index: callable = filesystem_dataset,
        f_configure: callable = configure_transforms,
        f_record: callable = write_records,
        xfm: callable = direct_transform,
    ) -> Tuple[callable, callable, callable]:
        def transformer_f_configure(
            *,
            instance_transforms: Optional[Sequence[Tuple[callable, int]]] = None,
        ) -> Mapping:
            if instance_transforms is None:
                instance_transforms = []
            instance_transforms.append((instance_transform, priority))
            return {'instance_transforms': instance_transforms}

        def f_configure_transformed(
            *,
            instance_transforms: Optional[Sequence[Tuple[callable, int]]] = None,
            **params,
        ):
            return xfm(f_configure, transformer_f_configure)(**params)(
                instance_transforms=instance_transforms,
            )

        def transformer_f_record(
            schema: Mapping = None,
        ) -> Mapping:
            if record_mask:
                if schema is None:
                    schema = {}
                for feature in features:
                    schema[f'{feature}_nanmask'] = DataArray('bool')
                    schema[f'{feature}_infmask'] = DataArray('bool')
            return {
                "schema": schema,
            }

        def f_record_transformed(
            *,
            schema: Mapping = None,
            **params,
        ):
            return xfm(f_record, transformer_f_record)(**params)(
                schema=schema,
            )

        return (
            f_index,
            f_configure_transformed,
            f_record_transformed,
        )
    return transform


def filtering_transform(filtering_f: callable) -> callable:
    def transform(
        f_index: callable = filesystem_dataset,
        f_configure: callable = configure_transforms,
        f_record: callable = write_records,
        xfm: callable = direct_transform,
    ) -> callable:
        def transformer_f_index(
            filters: Optional[Sequence[callable]] = None,
        ) -> Mapping:
            if filters is None:
                filters = (filtering_f,)
            else:
                filters = tuple(list(filters) + [filtering_f])
            return {
                'filters': filters,
            }

        def f_index_transformed(
            *,
            filters: Optional[Sequence[str]] = None,
            **params,
        ):
            return xfm(f_index, transformer_f_index)(**params)(
                filters=filters,
            )

        return (
            f_index_transformed,
            f_configure,
            f_record,
        )
    return transform


def filter_missing(
    reference: Optional[str] = None,
    anchor: Optional[str] = None,
    pivot: Optional[Union[str, Sequence[str]]] = None,
) -> callable:
    def filtering_f(table: pd.DataFrame) -> pd.DataFrame:
        df = table
        if anchor is not None and pivot is not None:
            df = df.pivot(index=anchor, columns=pivot)
        if reference is None:
            df = df[~df.isnull().any(axis=1)]
        else:
            df = df[~df[reference].isnull()]
        if anchor is not None and pivot is not None:
            df = df.stack(pivot).reset_index()
        return df

    return filtering_transform(filtering_f)


def filter_entity(
    entity: str,
    include: Optional[Sequence] = None,
    exclude: Optional[Sequence] = None,
) -> callable:
    def filtering_f(table: pd.DataFrame) -> pd.DataFrame:
        drop = None
        _include = include
        _exclude = exclude
        unique = table[entity].unique()
        if isinstance(_include, str):
            _include = (_include,)
        if isinstance(_exclude, str):
            _exclude = (_exclude,)
        if _include is not None:
            drop = tuple(set(unique) - set(_include))
        if _exclude is not None:
            drop = tuple(set(_exclude))
        if drop is not None:
            table = table[~table[entity].isin(drop)]
        return table

    return filtering_transform(filtering_f)


def filter_subjects(
    include: Optional[Sequence] = None,
    exclude: Optional[Sequence] = None,
) -> callable:
    return filter_entity('subject', include, exclude)


def filter_sessions(
    include: Optional[Sequence] = None,
    exclude: Optional[Sequence] = None,
) -> callable:
    return filter_entity('session', include, exclude)


def filter_runs(
    include: Optional[Sequence] = None,
    exclude: Optional[Sequence] = None,
) -> callable:
    return filter_entity('run', include, exclude)


def filter_tasks(
    include: Optional[Sequence] = None,
    exclude: Optional[Sequence] = None,
) -> callable:
    return filter_entity('task', include, exclude)


def instance_as_tf_example(schema: Mapping) -> callable:
    def f_package_tf_example(
        *,
        instance: Mapping,
        **params,
    ) -> Mapping:
        features = {}
        for k, v in schema.items():
            feature = instance.get(k, AbsentFromInstance())
            if isinstance(feature, AbsentFromInstance):
                continue
            features[k] = v.encode_tf(instance[k])
        return tf.train.Example(
            features=tf.train.Features(feature=features),
        ).SerializeToString()
    return f_package_tf_example


def tf_example_as_instance(schema: Mapping) -> callable:
    def f_unpackage_tf_example(
        *,
        example: tf.train.Example,
        **params,
    ) -> Mapping:
        tf_example = tf.io.parse_single_example(example, schema_to_tfrecords(schema))
        instance = {}
        for k, v in schema.items():
            instance[k] = v.from_feature(tf_example[k])
        return instance
    return f_unpackage_tf_example


def schema_to_tfrecords(schema: Mapping) -> Mapping:
    return {
        k: tf.io.FixedLenFeature([], dtype=v.dtype_tf)
        for k, v in schema.items()
    }


def write_instance_as_tf_example(schema: Mapping) -> callable:
    make_example = instance_as_tf_example(schema)

    def f_write_instance_as_tf_example(
        *,
        instance: Mapping,
        context: str,
        instance_id: str,
        **params,
    ) -> None:
        print(f'Writing to {context}')
        example = make_example(instance=instance)
        context.write(example)
    return f_write_instance_as_tf_example


def write_instance_as_tarfile(schema: Mapping) -> callable:
    def f_write_instance_as_tarfile(
        *,
        instance: Mapping,
        context: str,
        instance_id: str,
        **params,
    ) -> None:
        print(f'Writing to {context}')
        for k, v in schema.items():
            feature = instance.get(k, AbsentFromInstance())
            if isinstance(feature, AbsentFromInstance):
                continue
            encoded = v.encode_tar(feature)
            tarinfo = tarfile.TarInfo(f'{instance_id}.{k}')
            tarinfo.size = len(encoded)
            context.addfile(tarinfo, BytesIO(encoded))
    return f_write_instance_as_tarfile


def write_tfrecord(subdir: str = 'tfrecord') -> callable:
    def fname_pattern(dir: str, **params):
        param_fields = '_'.join(f'{k}-{v}' for k, v in params.items())
        fname = f'{dir}/{subdir}/{param_fields}.tfrecord'
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        return open(fname, 'wb'), fname

    def transform(
        f_index: callable = filesystem_dataset,
        f_configure: callable = configure_transforms,
        f_record: callable = write_records,
        xfm: callable = direct_transform,
    ) -> callable:
        def transformer_f_record(
            instance_writers: Optional[Sequence[callable]],
        ) -> Mapping:
            if instance_writers is None:
                instance_writers = []
            instance_writers += [(
                write_instance_as_tf_example,
                fname_pattern,
            )]
            return {
                'instance_writers': instance_writers,
            }

        def f_record_transformed(
            *,
            instance_writers: Optional[Sequence[callable]] = None,
            **params,
        ):
            return xfm(f_record, transformer_f_record)(**params)(
                instance_writers=instance_writers,
            )

        return (
            f_index,
            f_configure,
            f_record_transformed,
        )
    return transform


def write_tarball(subdir: str = 'tar') -> callable:
    def fname_pattern(dir: str, **params):
        param_fields = '_'.join(f'{k}-{v}' for k, v in params.items())
        fname = f'{dir}/{subdir}/{param_fields}.tar'
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        return tarfile.open(fname, 'w'), fname

    def transform(
        f_index: callable = filesystem_dataset,
        f_configure: callable = configure_transforms,
        f_record: callable = write_records,
        xfm: callable = direct_transform,
    ) -> callable:
        def transformer_f_record(
            instance_writers: Optional[Sequence[callable]],
        ) -> Mapping:
            if instance_writers is None:
                instance_writers = []
            instance_writers += [(
                write_instance_as_tarfile,
                fname_pattern,
            )]
            return {
                'instance_writers': instance_writers,
            }

        def f_record_transformed(
            *,
            instance_writers: Optional[Sequence[callable]] = None,
            **params,
        ):
            return xfm(f_record, transformer_f_record)(**params)(
                instance_writers=instance_writers,
            )

        return (
            f_index,
            f_configure,
            f_record_transformed,
        )
    return transform


def datalad_source(root: str) -> callable:
    def path_transform(path: str) -> str:
        datalad.get(path, dataset=root)
        return path

    def transform(
        f_index: callable = filesystem_dataset,
        f_configure: callable = configure_transforms,
        f_record: callable = write_records,
        xfm: callable = direct_transform,
    ) -> callable:
        def transformer_f_configure() -> Mapping:
            return {
                'path_transform': path_transform,
            }

        def f_configure_transformed(**params):
            return xfm(f_configure, transformer_f_configure)(**params)()

        return (
            f_index,
            f_configure_transformed,
            f_record,
        )
    return transform


def datapipe(*pparams):
    def configure_pipeline(
        *,
        f_index: callable = filesystem_dataset,
        f_configure: callable = configure_transforms,
        f_record: callable = write_records,
    ) -> callable:
        for transform in reversed(pparams):
            f_index, f_configure, f_record = transform(
                f_index, f_configure, f_record,
            )

        def execute_pipeline(**params) -> Mapping:
            return f_record(**f_configure(**f_index(**params)))

        return execute_pipeline
    return configure_pipeline
