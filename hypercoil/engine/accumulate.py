# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Accumulator
~~~~~~~~~~~
Gradient accumulator for throughput control.

Data conveyances through the atlas layer are, in general, throughput-limited.
The atlas layer operates on voxel-wise or vertex-wise datasets, which
necessitates a large memory footprint. On a consumer-grade GPU, fewer than 10
samples -- and in many cases only 1 or 2 -- can reasonably be expected to fit
in memory with the model.

This poses a significant problem for downstream modules. In particular, we
would like to impose a penalty on relationships between extracted features and
nuisance measures (for example, between connectivity estimates and subject
movement). Accurately and stably estimating these relationships (as with
Pearson correlations across the batch dimension) requires a sufficient batch
size that cannot be attained under the throughput constraints of the atlas
layer.

Currently, we circumvent this problem in a rather inefficient but operational
way. The inefficiency arises because it usually necessitates an additional
forward pass through the atlas layer.

.. note::
    There is no principled reason for this inefficiency; it appears to be a
    limitation of the way we interact with autograd. It would be great to work
    out a more streamlined approach, perhaps involving forward-mode
    autodifferentiation. The AI community likely already has many better
    solutions, but as a general rule is challenging to get information from.

Our solution works by first fragmenting each batch according to the throughput
limit of the high-memory-demand function. It then passes the data fragments
(which need not all be loaded into memory at once) one by one through a
throughput-limited
``AccumulatingFunction`` that subclasses
``torch.autograd.Function``. Each ``AccumulatingFunction`` caches only the
accumulated batch-average local gradient with respect to each parameter for
the backward pass, enabling memory savings of a factor equal to the batch size
or batch size to throughput ratio.

.. warning::
    Note that this solution only works if the local gradient over the batch is
    equal to the batch-average local gradient. Additionally, because it marks
    the input as non differentiable and thereby blocks further backpropagation,
    it will only work as an input layer.

Because of the compartmentalised environment of ``autograd.Function``s, it is
not possible to interact with the per-fragment inputs and outputs outside of
that environment, and inside the environment there is no history tracking;
automatic differentiation is therefore inactive. This introduces a second
challenge: we often want to define loss functions at the per-fragment stage.
(For example,
:doc:`second-moment losses <hypercoil.loss.secondmoment>`
have little chance of fitting into GPU memory at the full batch level.) Those
loss functions, however, require access to tensor history -- which, again,
cannot be tracked at the fragment level in the ``AccumulatingFunction``'s
forward pass.

The working solution is to implement two parallel forward passes in a
``Module`` that wraps ``AccumulatingFunction``, the ``Accumuline``. For each
batch fragment, the ``Accumuline`` runs one forward pass inside the
``AccumulatingFunction`` and a second outside it. The forward pass run outside
of the ``AccumulatingFunction`` can then interact with local loss functions.
"""
import torch
from torch.autograd import Function
from collections import OrderedDict
from hypercoil.loss import LossArgument as ModelArgument


class Accumuline(torch.nn.Module):
    def __init__(
        self,
        model,
        gradient,
        backward,
        retain_dims,
        argmap,
        throughput,
        batch_size
    ):
        super().__init__()
        self.model = model
        self.gradient = gradient
        self.backward = backward
        self.retain_dims = retain_dims
        self.argmap = argmap
        self.throughput = throughput
        self.batch_size = batch_size
        self.acc = Accumulator(
            model=self.model,
            gradient=self.gradient,
            retain_dims=self.retain_dims,
            model_params=[],
            reduce_dims='mean'
        )

    def forward(self, data_source, *params):
        sampled = 0
        out = []
        record = []
        accfwd = AccumulatingFunction.apply
        terminate = False
        while not terminate:
            if sampled > self.batch_size:
                terminate = True
                sample = None
            else:
                sample = data_source.sample(self.throughput)
                sampled += self.throughput
            out = accfwd(
                self.acc,
                self.backward,
                self.argmap,
                sample,
                out,
                record,
                terminate,
                *params
            )
        return out


class AccumulatingFunction(Function):
    @staticmethod
    def forward(
        ctx,
        acc,
        backward,
        argmap,
        sample,
        out,
        record,
        terminate,
        *params
    ):
        ctx.terminate = terminate
        if terminate:
            ctx.backward = backward
            acc_vals = [v for v in acc.acc.values()]
            ctx.save_for_backward(*acc_vals)
            acc.reset()
            keys = out[0].keys()
            ret = [torch.cat([o[k] for o in out]) for k in keys]
            return tuple(ret)
        arg = argmap(sample)
        out = list(out) + [acc(arg)]
        return out

    def backward(ctx, *grad_output):
        if ctx.terminate:
            ret = (
                None, #acc
                None, #backward
                None, #argmap
                None, #sample
                None, #out
                None, #record
                None, #terminate
                *ctx.backward(*grad_output, *ctx.saved_tensors) # params
            )
        else:
            ret = (None, None, None, None, None, None, None, None)
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
