# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Model argument
~~~~~~~~~~~~~~
Convenience mappings representing arguments to a model or loss function.
Also aliased to loss arguments.
"""
import torch
from collections.abc import Mapping


class ModelArgument(Mapping):
    """
    Effectively this is currently little more than a prettified dict.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

    def __setitem__(self, k, v):
        self.__setattr__(k, v)

    def __getitem__(self, k):
        return self.__dict__[k]

    def __delitem__(self, k):
        del self.__dict__[k]

    def __len__(self):
        return len(self.__dict__)

    def __iter__(self):
        return iter(self.__dict__)

    def _fmt_tsr_repr(self, tsr):
        return f'<tensor of dimension {tuple(tsr.shape)}>'

    def __repr__(self):
        s = f'{type(self).__name__}('
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                v = self._fmt_tsr_repr(v)
            elif isinstance(v, list):
                v = f'<list with {len(v)} elements>'
            elif isinstance(v, tuple):
                fmt = [self._fmt_tsr_repr(i)
                       if isinstance(i, torch.Tensor)
                       else i for i in v]
                v = f'({fmt})'
            s += f'\n    {k} : {v}'
        s += ')'
        return s

    def update(self, *args, **kwargs):
        for k, v in args:
            self.__setitem__(k, v)
        self.__dict__.update(kwargs)

    @classmethod
    def all_except(cls, arg, remove):
        arg = {k: v for k, v in arg.items() if k not in remove}
        return cls(**arg)

    @classmethod
    def replaced(cls, arg, replace):
        arg = {k: (replace[k] if replace.get(k) is not None else v)
               for k, v in arg.items()}
        return cls(**arg)

    @classmethod
    def swap(cls, arg, swap, val):
        old, new = swap
        arg = {k: v for k, v in arg.items() if k != old}
        arg.update({new : val})
        return cls(**arg)


class UnpackingModelArgument(ModelArgument):
    """
    ``ModelArgument`` variant that is automatically unpacked when it is the
    output of an ``apply`` call.
    """
    pass
