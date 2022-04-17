# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Local gradient accumulator for throughput control.

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
    autodifferentiation. There are likely already many better solutions in
    existence.

Our solution works by first fragmenting each batch according to the throughput
limit of the high-memory-demand function. It then passes the data fragments
(which need not all be loaded into memory at once) one by one through a
throughput-limited
:doc:`AccumulatingFunction <hypercoil.engine.accumulate.AccumulatingFunction>`
that subclasses ``torch.autograd.Function``. Each ``AccumulatingFunction``
caches only the accumulated batch-average local gradient with respect to each
parameter for the backward pass, enabling memory savings of a factor equal to
the batch size or batch size-to-throughput ratio.

.. warning::
    Note that this solution only works if the local gradient over the batch is
    equal to the batch-average local gradient. Additionally, because it marks
    the input as non differentiable and thereby blocks further backpropagation,
    it will only work as an input layer.

Because of the compartmentalised environment of ``autograd.Function``, it is
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
``Module`` that wraps ``AccumulatingFunction``,
:doc:`Accumuline <hypercoil.engine.accumulate.Accumuline>`. For each
batch fragment, the ``Accumuline`` runs one forward pass inside the
``AccumulatingFunction`` and a second outside it. The forward pass run outside
of the ``AccumulatingFunction`` can then interact with local loss functions.
"""
import torch
from torch.autograd import Function
from collections import OrderedDict
from hypercoil.engine.argument import ModelArgument


class Accumuline(torch.nn.Module):
    def __init__(
        self,
        model,
        gradient,
        backward,
        retain_dims,
        argmap,
        throughput,
        batch_size,
        reduction='mean',
        loss=None,
        loss_argmap=None
    ):
        super().__init__()
        self.model = model
        self.gradient = gradient
        self.backward = backward
        self.retain_dims = retain_dims
        self.argmap = argmap
        self.throughput = throughput
        self.batch_size = batch_size
        self.loss = loss
        self.loss_argmap = loss_argmap
        self.acc = Accumulator(
            model=self.model,
            gradient=self.gradient,
            retain_dims=self.retain_dims,
            model_params=[],
            reduce_dims=reduction
        )

    def forward(self, data_source, *params):
        sampled = 0
        out = []
        accfwd = AccumulatingFunction.apply
        terminate = False
        while not terminate:
            if sampled >= self.batch_size:
                terminate = True
                sample = None
            else:
                #TODO: sometimes, we'll need to sample fewer than the max
                # throughput to get the exact batch size we want
                sample = data_source.sample(self.throughput)
                sampled += self.throughput
                if self.loss is not None:
                    sample_out = self.model(sample)
                    loss_arg = self.loss_argmap(
                        sample, sample_out, self.model
                    )
                    l = self.loss(loss_arg, verbose=True)
                    l.backward()
            out = accfwd(
                self.acc,
                self.backward,
                self.argmap,
                sample,
                out,
                terminate,
                *params
            )
        return out


class AccumulatingFunction(Function):
    """
    ``autograd.Function`` that supports local gradient accumulation.

    For many use cases, it will make most sense to use an
    :doc:`Accumuline <hypercoil.engine.accumulate.Accumuline>`
    instead of directly interfacing with this class.

    To use an ``AccumulatingFunction``, we need to begin by defining an
    :doc:`Accumulator <hypercoil.engine.accumulate.Accumulator>`
    to perform the accumulation::

        acc = Accumulator(
            model=model,
            gradient=model_grad,
            retain_dims=(-1, -2)
        )

    We will also need a function to operationalise the backward pass through
    the ``AccumulatingFunction``. The backward function should take as inputs
    (i) all of the tensors that encode the back-propagated gradient of the
    loss node with respect to each output of the ``AccumulatingFunction`` and
    (ii) all of the local derivatives accumulated by the ``Accumulator`` we've
    created. For the case of matrix-matrix multiplication, we can reasonably
    use::

        def accbwd(grad_output, grad_local):
            return grad_output @ grad_local,

    We'll also need a quick function to map from samples input to the
    ``AccumulatingFunction`` to arguments to the ``Accumulator``'s specified
    ``model``::

        argmap = lambda T: {'X' : T}

    We can now apply our ``AccumulatingFunction`` to input samples. Here we're
    accumulating the local derivative of the matrix multiplication operation
    with respect to one of the matrices being multiplied, ``W``::

        out = []
        T0 = torch.random.rand(4, 4, 10, 100)
        accfwd = AccumulatingFunction.apply
        out = accfwd(acc, None, argmap, T0, out, False, W)
        T1 = torch.random.rand(6, W.shape[-1], 100)
        out = accfwd(acc, None, argmap, T1, out, False, W)

    Each call to ``accfwd`` above will accumulate the local derivative in the
    provided ``acc`` object and will also append any outputs to the ``out``
    iterable that we provided in our call. When we've collected all of the
    outputs we need in ``out``, we make a terminal call to ``accfwd``::

        out = accfwd(acc, bwd, None, None, out, True, W)

    The terminal call is made by setting the ``terminate`` argument to True.
    We should not pass new data to the accumulating function during the
    terminal call, as it will not be used. The terminal call caches the
    accumulated derivative for the backward pass and then clears all
    accumulated data from the ``Accumulator``.

    .. note::
        Only the terminal call of an ``AccumulatingFunction`` is capable of
        back-propagating gradients. All other calls will not track tensor
        history and will block any gradients that they receive from being
        further propagated.

    We can use a ``partial`` to simplify usage::

        accfwd = partial(AccumulatingFunction.apply, acc, argmap)

    Parameters
    ----------
    acc : ``Accumulator``
        Accumulator object for accumulating the local derivative over calls
        to the function.
    backward : callable
        Backward pass through the function. This should take as inputs (i) all
        of the tensors that encode the back-propagated gradient of the
        loss node with respect to each output of the ``AccumulatingFunction``
        and (ii) all of the local derivatives accumulated by the
        ``Accumulator`` and should return the back-propagated gradient with
        respect to each input parameter provided in ``params``.
    argmap : callable
        Map from samples input to the function to mappings representing
        arguments to the ``Accumulator``.
    sample : ``tensor`` or iterable(``tensor``)
        Input sample to be processed. The gradient with respect to the input
        sample is not returned.
    out : iterable
        Iterable containing accumulated outputs thus far. The output created
        at each call is appended to this iterable.
    terminate : bool
        Indicates that accumulation should be terminated, and a new node
        containing the accumulated local derivative should be added to the
        computational graph. Only the terminal call will retain tensor
        history, and only the terminal node accordingly backpropagates
        received gradient. However, the terminal call does not do any data
        processing or new accumulation, so it should not receive previously
        unseen input data.
    params : ``tensor``
        Parameters with respect to which the accumulated gradient should be
        computed and backpropagated. The ``gradient`` parameter provided to
        the input ``Accumulator`` should define local gradients with respect
        to each of these parameters.

    Returns
    -------
    out : iterable
        Iterable containing all previously accumulated outputs provided as
        parameters to ``out``, together with a new output from the current
        call to the internal ``model`` of the input ``Accumulator``. For the
        terminal call, this will instead be a tuple of collated tensors.
    """
    @staticmethod
    def forward(
        ctx,
        acc,
        backward,
        argmap,
        sample,
        out,
        terminate,
        *params
    ):
        ctx.terminate = terminate
        if terminate:
            ctx.acc = acc
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

    @staticmethod
    def backward(ctx, *grad_output):
        if ctx.terminate:
            grad = [g for g in grad_output]
            for i, g in enumerate(grad_output):
                reduced_dims, _ = ctx.acc._get_reduced_dims(g)
                if len(reduced_dims) > 0:
                    grad[i] = torch.sum(
                        g,
                        keepdim=True,
                        axis=reduced_dims
                    )
            ret = (
                None, #acc
                None, #backward
                None, #argmap
                None, #sample
                None, #out
                None, #terminate
                *ctx.backward(*grad, *ctx.saved_tensors) # params
            )
        else:
            ret = (None, None, None, None, None, None, None)
        return ret


class Accumulator(torch.nn.Module):
    """
    Additive local gradient accumulator module.

    We can implement batch-averaged local additive gradient accumulation for
    matrix-matrix multiplication as follows. Let's assume that one matrix
    ``W`` we are multiplying is also the parameter for which we want to
    accumulate gradient.

    First, we define a ``model`` for the forward call. This is the matrix
    multiply operation::

        model = lambda X: W @ X

    Next, we specify the local gradient function -- the derivative of the
    model output with respect to the parameter ``W`` that we are interested
    in. The gradient should return a mapping over parameters of interest::

        def model_grad(X, *args, **kwargs):
            return {'W': X.transpose(-1, -2)}

    By default, an ``Accumulator`` will reduce any dimensions of the local
    gradient tensor not explicitly protected with a declaration to the
    argument ``retain_dims``. For matrix multiplication, we must protect
    the matrix slice, corresponding to the last two dimensions of the
    gradient tensors. We can now define our ``Accumulator``::

        acc = Accumulator(
            model=model,
            gradient=model_grad,
            retain_dims=(-1, -2)
        )

    To use this ``Accumulator``, we pass it an argument mapping containing
    inputs to the ``model`` we specified::

        T0 = torch.random.rand(4, 4, W.shape[-1], 100)
        out0 = acc(arg={'X' : T0})
        T1 = torch.random.rand(6, W.shape[-1], 100)
        out1 = acc(arg={'X' : T1})

    The accumulator then accumulates the average (or summed) gradient computed
    over all matrix slices of the inputs ``T0`` and ``T1``. It internally
    tracks the overall weight of gradients accumulated thus far (4 * 4 = 16
    after processing ``T0`` and 16 + 6 = 22 after also processing ``T1``).
    The accumulated gradient is accessible in the attribute ``acc``, and the
    gradient weight is accessible in the attribute ``acc_weight``. To reset
    the accumulated tensors, use the ``reset`` or ``zero_grad`` operations.

    .. note::
        The ``Accumulator`` only stores the accumulated local gradient in its
        attribute ``acc``. It is still the user's responsibility to apply that
        gradient in any backward pass. The additional classes
        :doc:`AccumulatingFunction <hypercoil.engine.accumulate.AccumulatingFunction>`
        and
        :doc:`Accumuline <hypercoil.engine.accumulate.Accumuline>`
        can facilitate this for many use cases.

    Parameters
    ----------
    model : callable
        Forward computation. This should be a callable that maps from some
        input arguments to some outputs.
    gradient : callable
        Backward computation. This should be a callable that takes as
        arguments all inputs to and outputs from the ``model`` callable and
        returns a dictionary of local gradients of outputs with respect to
        each parameter of interest, i.e. each parameter for which the local
        gradient is to be accumulated.
    retain_dims : iterable
        List of dimensions of each gradient tensor returned by ``gradient``
        over which averaging will not return a correct gradient.
    reduce_dims : ``'mean'`` or ``'sum'`` (default ``'mean'``)
        Reduction operation over all axes not protected by the ``retain_dims``
        argument.
    autouse_domain_gradient : bool
        Unused argument -- currently does nothing.
    model_outputs : iterable (default ``['out']``)
        Keys naming each output of the model.
    model_params : iterable
        Unused argument -- currently does nothing.

    Attributes
    ----------
    reduction : callable
        Callable applied to gradient tensors to reduce all dimensions not
        explicitly protected by ``retain_dim``. Defaults to ``torch.mean``.
    acc : dict
        Dictionary storing all accumulated gradients.
    acc_weight : dict
        Dictionary storing total weights of all accumulated gradients.
    """
    ##TODO: Should retain_dims optionally be a mapping over parameters of
    # interest?
    def __init__(self, model, gradient, retain_dims,
                 reduce_dims=None, autouse_domain_gradient=False,
                 model_outputs=None, model_params=None):
        super().__init__()
        if model_outputs is None:
            model_outputs = ['out']
        if model_params is None:
            model_params = ['weight']
        if reduce_dims is None:
            reduce_dims = 'mean'
        if reduce_dims == 'mean':
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

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        self.reset()

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
