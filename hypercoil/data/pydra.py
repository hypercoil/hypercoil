# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pydra-wrapped transforms
~~~~~~~~~~~~~~~~~~~~~~~~
Dataset transformations with pydra marks.
"""
import torch
import pydra
import typing as ty
from hypercoil.data import functional as F


@pydra.mark.task
def get_data(data):
    return data.data


@pydra.mark.task
def get_metadata(data):
    return data.metadata


@pydra.mark.task
def read_neuro_image(data):
    return F.read_neuro_image(data)


@pydra.mark.task
def read_data_frame(data, sep='\t'):
    return F.read_data_frame(data, sep=sep)


@pydra.mark.task
def metadata_key(data, key):
    return data[key]


@pydra.mark.task
def make_var_dict(ref, vars):
    return {k.name: k.assignment for k in ref.variables if k.name in vars}


#TODO
# Please tell me there's a better way of making this work with pydra...
@pydra.mark.task
def compile_dict(
    key0, value0,
    key1=None, value1=None,
    key2=None, value2=None,
    key3=None, value3=None,
    key4=None, value4=None,
    key5=None, value5=None,
    key6=None, value6=None,
    key7=None, value7=None,
    key8=None, value8=None,
    key9=None, value9=None,):
    pass


@pydra.mark.task
def keys(dict):
    return list(dict.keys())


@pydra.mark.task
def apply_model_specs(models, data, metadata):
    from hypercoil.formula import ModelSpec, FCConfoundModelSpec
    models = [FCConfoundModelSpec(m, name=m)
              if isinstance(m, str) else m
              for m in models]
    return F.apply_model_specs(models, data, metadata)


@pydra.mark.task
def to_tensor(data, dtype=torch.FloatTensor):
    return F.to_tensor(data, dtype=dtype, dim='auto')


@pydra.mark.task
def conform_time_axis(data):
    return {k: v.values.T for k, v in data.items()}


@pydra.mark.task
@pydra.mark.annotate({
    'return': {'length': int, 'start': int}
})
def random_window(data, window_length, axis=-1):
    return F.random_window(data, window_length, axis)


@pydra.mark.task
def window(data, window_length, window_start=0):
    if isinstance(data, dict):
        return {k : F.window(v, window_length, window_start)
                for k, v in data.items()}
    return F.window(data, window_length, window_start)


@pydra.mark.task
def standardise(data, axis=-1):
    if isinstance(data, dict):
        return {k : F.standardise(v, axis)
                for k, v in data.items()}
    return F.standardise(data, axis)


@pydra.mark.task
def unzip_blocked_dict(block):
    blk, _ = F.ravel(lst=block, stack=[])
    if isinstance(blk[0], dict):
        return F.unzip_blocked_dict(block)
    else:
        return block


@pydra.mark.task
@pydra.mark.annotate({
    'return': {'out': ty.Any, 'mask': ty.Any}
})
def nanfill(data, fill='mean'):
    if isinstance(data, dict):
        filled = {k : F.nanfill(v, fill)
                  for k, v in data.items()}
        return ({k : v[0] for k, v in filled.items()},
                {k : v[1] for k, v in filled.items()})
    return F.nanfill(data, fill)


@pydra.mark.task
def fillnan(data, mask):
    if isinstance(data, dict):
        return {k : F.fillnan(v, mask[k])
                for k, v in data.items()}
    return F.fillnan(data, mask)


@pydra.mark.task
def polynomial_detrend(data, order=0):
    if isinstance(data, dict):
        return {k : F.polynomial_detrend(v, order)
                for k, v in data.items()}
    return F.polynomial_detrend(data, order)


@pydra.mark.task
@pydra.mark.annotate({
    'return': {'list': list, 'stack': list}
})
def ravel(lst, stack=[], max_depth=None):
    return F.ravel(lst=lst, stack=stack, max_depth=max_depth)


@pydra.mark.task
@pydra.mark.annotate({
    'return': {'list': list, 'stack': list}
})
def fold(lst, stack):
    return F.fold(lst=lst, stack=stack)


"""@pydra.mark.task
def add(x, y):
    return x + y

t0 = torch.tensor([1])
t1 = torch.tensor([-1])
task = add(x=t0, y=t1)
task()"""


models = ['(dd1(rps + wm + csf + gsr))^^2']
tmask = []


def deploy_wf(embed, variable_names):
    wf = pydra.Workflow(
        name='deploy',
        input_spec=['variables']
    )
    wf.add(embed)
    for v in variable_names:
        wf.add(metadata_key(name=f'vars_{v}', data=wf.lzin.variables, key=v))
        wf.add(ravel(
            name=f'ravel_{v}',
            lst=wf.name2obj[f'vars_{v}'].lzout.out
        ))
        #TODO: This is probably NOT the way to do this.
        embed.inputs.__dict__[v] = wf.name2obj[f'ravel_{v}'].lzout.list
        embed_out = embed.lzout
        embed_out.field = v
        wf.add(fold(
            name=f'fold_{v}',
            lst=embed_out,
            stack=wf.name2obj[f'ravel_{v}'].lzout.stack
        ))
        wf.add(unzip_blocked_dict(
            name=f'unzip_{v}',
            block=wf.name2obj[f'fold_{v}'].lzout.list
        ))

    wf.set_output([
        (f'{v}', wf.name2obj[f'unzip_{v}'].lzout.out) for v in variable_names
    ])
    return wf


def imaging_wf(models, tmask, window_length=None, detrend_order=0):
    wf = pydra.Workflow(
        name='transform',
        input_spec=['images', 'confounds']
    )
    wf.add(get_data(name='idata', data=wf.lzin.images))
    wf.add(get_metadata(name='imeta', data=wf.lzin.images))
    wf.add(metadata_key(
        name='trep',
        data=wf.imeta.lzout.out,
        key='RepetitionTime'
    ))
    wf.add(read_neuro_image(name='rdimg', data=wf.idata.lzout.out))

    wf.add(get_data(name='cdata', data=wf.lzin.confounds))
    wf.add(get_metadata(name='cmeta', data=wf.lzin.confounds))
    wf.add(read_data_frame(name='rddf', data=wf.cdata.lzout.out))
    wf.add(apply_model_specs(name='model',
                             models=models,
                             data=wf.rddf.lzout.out,
                             metadata=wf.cmeta.lzout.out))
    wf.add(conform_time_axis(name='ctpos', data=wf.model.lzout.out))

    wf.add(apply_model_specs(name='tmask',
                             models=tmask,
                             data=wf.rddf.lzout.out,
                             metadata=wf.cmeta.lzout.out))
    wf.add(conform_time_axis(name='ttpos', data=wf.tmask.lzout.out))

    wf.add(random_window(
        name='wleng',
        data=wf.rdimg.lzout.out,
        window_length=window_length
    ))
    wf.add(window(
        name='iwndw',
        data=wf.rdimg.lzout.out,
        window_length=wf.wleng.lzout.length,
        window_start=wf.wleng.lzout.start
    ))
    wf.add(window(
        name='cwndw',
        data=wf.ctpos.lzout.out,
        window_length=wf.wleng.lzout.length,
        window_start=wf.wleng.lzout.start
    ))
    wf.add(window(
        name='twndw',
        data=wf.ttpos.lzout.out,
        window_length=wf.wleng.lzout.length,
        window_start=wf.wleng.lzout.start
    ))

    wf.add(nanfill(name='inanf', data=wf.iwndw.lzout.out))
    wf.add(nanfill(name='cnanf', data=wf.cwndw.lzout.out))
    wf.add(polynomial_detrend(
        name='idmdt',
        data=wf.inanf.lzout.out,
        order=detrend_order
    ))
    wf.add(polynomial_detrend(
        name='cdmdt',
        data=wf.cnanf.lzout.out,
        order=detrend_order
    ))
    wf.add(fillnan(
        name='ifnan',
        data=wf.idmdt.lzout.out,
        mask=wf.inanf.lzout.mask
    ))
    wf.add(fillnan(
        name='cfnan',
        data=wf.cdmdt.lzout.out,
        mask=wf.cnanf.lzout.mask
    ))

    wf.add(standardise(name='izscr', data=wf.ifnan.lzout.out))
    wf.add(standardise(name='czscr', data=wf.cfnan.lzout.out))

    wf.set_output([
        ('images', wf.izscr.lzout.out),
        ('t_r', wf.trep.lzout.out),
        ('confounds', wf.czscr.lzout.out),
        ('tmask', wf.twndw.lzout.out),
    ])
    return wf


ref = 'hcp'
if ref == 'bids':
    from hypercoil.data.bids import fmriprep_references
    data_dir = '/mnt/pulsar/Research/Development/hypercoil/hypercoil/examples/ds-synth/'
    layout, images, confounds, refs = fmriprep_references(
        data_dir,
        model=['(dd1(rps + wm + csf + gsr))^^2']
    )
    df = layout.dataset(
        observations=('subject',),
        levels=('task', 'session', 'run')
    )


if ref == 'hcp':
    from hypercoil.data.hcp import hcp_references
    data_dir = '/mnt/andromeda/Data/HCP_subsubsample/'
    layout, images, confounds, refs = hcp_references(
        data_dir,
        model=['(dd1(rps + wm + csf + gsr))^^2']
    )
    df = layout.dataset(
        observations=('subject',),
        levels=('task', 'session', 'run')
    )


class WorkflowFactory:
    def __init__(self, wf, **params):
        self.wf = wf
        self.params = params

    def __call__(self):
        return self.wf(**self.params)


ref = refs[0]
vars = ('images', 'confounds')
var_dict = {k.name: k.assignment for k in ref.variables}


embed = imaging_wf(models, tmask)
embed.split(('images', 'confounds'))
embed.combine(['images', 'confounds'])
#embed(images=var_dict['images'], confounds=var_dict['confounds'])

#deploy = deploy_wf(embed, ['images', 'confounds', 't_r', 'tmask'])
#deploy(variables=var_dict)
deploy = WorkflowFactory(
    deploy_wf,
    embed=embed,
    variable_names=['images', 'confounds', 't_r', 'tmask']
)

from hypercoil.data.dataset import ReferencedDataset
ds = ReferencedDataset(refs, deploy)
