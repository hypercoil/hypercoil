# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Accumulator
~~~~~~~~~~~
Gradient accumulator for throughput control.
"""
import torch
from torch.autograd import Function
from collections import OrderedDict
from hypercoil.loss import LossArgument as ModelArgument


class AccumulatingFunction(Function):
    @staticmethod
    def forward(
        ctx,
        model,
        gradient,
        backward,
        retain_dims,
        argmap,
        throughput,
        batch_size,
        data_source,
        *params
    ):
        ctx.backward = backward
        acc = Accumulator(
            model=model,
            gradient=gradient,
            retain_dims=retain_dims,
            model_params=[],
            reduce_dims='mean'
        )
        sampled = 0
        out = []
        output = {}
        while sampled < batch_size:
            sample = data_source.sample(throughput)
            arg = argmap(sample)
            out += [acc(arg)]
            sampled += throughput
        acc_vals = [v for v in acc.acc.values()]
        ctx.save_for_backward(*acc_vals)
        acc.reset()
        keys = out[0].keys()
        for k in keys:
            output[k] = torch.cat([o[k] for o in out])
        ret = [o for o in output.values()]
        return tuple(ret)

    @staticmethod
    def backward(ctx, *grad_output):
        ret = (
            None, #model
            None, #gradient
            None, #backward
            None, #retain_dims
            None, #argmap
            None, #throughput
            None, #batch_size
            None, #data_source
            *ctx.backward(*grad_output, *ctx.saved_tensors) # params
        )
        return ret


class Accumulator(torch.nn.Module):
    def __init__(self, model, gradient, retain_dims,
                 reduce_dims=None, autouse_domain_gradient=False,
                 model_outputs=None, model_params=None):
        super().__init__()
        if model_outputs is None:
            model_outputs = ['out']
        if model_params is None:
            model_params = ['weight']
        if reduce_dims is None or reduce_dims == 'mean':
            reduce_dims = 'mean'
            reduction = torch.mean
        elif reduce_dims == 'sum':
            reduction = torch.sum
        if autouse_domain_gradient:
            raise NotImplementedError(
                'Automatic domain gradients not yet implemented. '
                'Provide domain gradients explicitly.'
            )
        self.model = model
        self.gradient = gradient
        self.retain_dims = retain_dims
        self.reduce_dims = reduce_dims
        self.reduction = reduction
        self.model_outputs = model_outputs
        self.model_params = model_params
        self.reset()

    def reset(self):
        self.acc = {}
        self.acc_weight = {}

    def _get_reduced_dims(self, tensor):
        reduced_dims = [True for _ in range(tensor.dim())]
        for ax in self.retain_dims:
            try:
                reduced_dims[ax] = False
            except IndexError:
                pass
        reduced_weight = torch.tensor(tensor.shape)[reduced_dims].prod().item()
        return [i for i, d in enumerate(reduced_dims) if d], reduced_weight

    def _conform_dims(self, grad):
        reduced_weight = {}
        for k, v in grad.items():
            reduced_dims, r_weight = self._get_reduced_dims(v)
            if len(reduced_dims) > 0:
                grad[k] = self.reduction(v, keepdim=True, axis=reduced_dims)
            else:
                grad[k] = v
            reduced_weight[k] = r_weight
        return grad, reduced_weight

    def _accumulate(self, grad, reduced_weight):
        for k, v in grad.items():
            if self.acc.get(k) is None:
                self.acc[k] = v
                self.acc_weight[k] = reduced_weight[k]
            elif self.reduce_dims == 'sum':
                self.acc[k] = self.acc[k] + v
            elif self.reduce_dims == 'mean':
                acc_weight = self.acc_weight.get(k) or 0
                new_weight = reduced_weight.get(k) or 0
                total_weight = acc_weight + new_weight
                self.acc[k] = (
                    (acc_weight / total_weight) * self.acc[k] +
                    (new_weight / total_weight) * v
                )
                self.acc_weight[k] = total_weight
            else:
                raise NotImplementedError(
                    f'Proposed reduction {self.reduce_dims} not implemented'
                )

    def forward(self, arg):
        with torch.no_grad():
            out = self.model(**arg)
            if not isinstance(out, torch.Tensor):
                out = OrderedDict(zip(self.model_outputs, out))
            else:
                out = {self.model_outputs[0]: out}
            grad_arg = ModelArgument(**arg, **out)
            grad, reduced_weight = self._conform_dims(self.gradient(**grad_arg))
            self._accumulate(grad, reduced_weight)
        return out
