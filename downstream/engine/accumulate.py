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
from functools import partial
from collections import OrderedDict
from .action import (
    SignalRelease,
    TerminateAccumuline,
    VerboseReceive
)
from .argument import (
    ModelArgument,
    UnpackingModelArgument
)
from .conveyance import (
    BaseConveyance,
    Conveyance,
    Conflux,
    DataPool
)
from .sentry import Sentry


class Accumuline(Conveyance):
    def __init__(
        self,
        model,
        accfn,
        gradient,
        origin,
        retain_dims,
        throughput,
        batch_size,
        reduction='mean',
        params=None,
        influx=None,
        efflux=None,
        lines=None,
        transmit_filters=None,
        receive_filters=None,
        skip_local=False,
        local_argmap=None,
        nonlocal_argmap=None,
    ):
        super().__init__(
            lines=lines,
            transmit_filters=transmit_filters,
            receive_filters=receive_filters
        )
        params = params or {}
        self.params = ModelArgument(**params)
        self.model = model
        self.gradient = gradient
        self.origin = origin
        self.retain_dims = retain_dims
        self.throughput = throughput
        self.batch_size = batch_size
        self.skip_local = skip_local
        self.local_argmap = local_argmap or (lambda _: ModelArgument())
        self.nonlocal_argmap = nonlocal_argmap or (lambda x: x)
        self.acc = Accumulator(
            model=self.model,
            gradient=self.gradient,
            retain_dims=self.retain_dims,
            model_params=[],
            reduce_dims=reduction
        )
        self.accfn = partial(
            accfn,
            acc=self.acc,
        )
        self.batched = 0
        self.influx = influx or (
            lambda arg: UnpackingModelArgument(**arg))
        self.efflux = efflux or (
            lambda output, arg: ModelArgument(out=output[0]))

        self.origin.add_line(('source', 'accumuline'))
        self.pool = DataPool(lines='accumuline')
        self.origin.connect_downstream(self.pool)
        self.add_line(('__nullsrc__', 'local'))

    def pool_and_sample(self, sample_size):
        while self.pool.batched < sample_size:
            self.origin(line='source')
        sample = self.pool.sample(sample_size, line='accumuline')
        return self._filter_received(sample, 'accumuline')

    def _accfn_call(self, arg, out, terminate, **params):
        accfn = partial(
            self.accfn,
            terminate=terminate,
            **params
        )
        if terminate:
            output = accfn(input=None, **out)
            return self.efflux(output=output, arg=arg)
        input = self.influx(arg)
        if isinstance(input, UnpackingModelArgument):
            output = accfn(**input, **out)
        else:
            raise TypeError('influx must map to an UnpackingModelArgument')
        output = ModelArgument(out=output)
        return output

    def _transmit_over_local(self, data):
        if self.skip_local:
            return
        local_arg = self.local_argmap(
            data, self.model
        )
        self.clear_transmission()
        self._update_transmission(local_arg, 'local')
        self._transmit()

    def _transmit_over_nonlocal(self, data):
        nonlocal_arg = self.nonlocal_argmap(data)
        self.clear_transmission()
        for line in self.transmit:
            if line in ('local', 'accumuline'):
                continue
            self._update_transmission(nonlocal_arg, line)
        self._transmit()

    def _step(self, out, **params):
        arg = ModelArgument(**self.params)
        arg.update(**params)
        if self.batched >= self.batch_size:
            terminate = True
            data = None
        else:
            terminate = False
            sample_size = min(self.throughput, self.batch_size - self.batched)
            data = self.pool_and_sample(sample_size)
            self._transmit_over_local(data)
            self.batched += sample_size
        out = self._accfn_call(
            arg=data,
            out=out,
            terminate=terminate,
            **arg
        )
        return out, terminate

    def forward(self, **params):
        out = ModelArgument(out=[])
        terminate = False
        while not terminate:
            out, terminate = self._step(out, **params)
        self._transmit_over_nonlocal(out)
        self.batched = 0
        return out


class _AccumulineRecursive(Conveyance):
    """
    There is no such thing as a recursive accumuline. If it is recursive, an
    accumuline it is not.

    In other words: do not use this class. It is not an accumuline. It remains
    in the unit tests as a template in case the future brings changes to the
    rules for traversing conveyances and purging their messages.
    """
    def __init__(
        self,
        model,
        params,
        origin,
        accfn,
        gradient,
        cfx_fields,
        retain_dims,
        throughput,
        batch_size,
        reduction='mean',
        influx=None,
        efflux=None,
        lines=None,
        transmit_filters=None,
        receive_filters=None,
        batch_key=None,
        verbose=False
    ):
        self.module = model
        self.gradient = gradient
        self.retain_dims = retain_dims
        self.batched = 0
        self.throughput = throughput
        self.batch_size = batch_size
        self.argbase = ModelArgument(**params)
        assert self.batch_size > self.throughput, (
            'Line throughput must be strictly less than batch size')
        acc = Accumulator(
            model=self.module,
            gradient=self.gradient,
            retain_dims=self.retain_dims,
            model_params=[],
            reduce_dims=reduction
        )
        accmodel = partial(
            accfn,
            acc=acc,
        )
        influx = influx or (lambda arg: UnpackingModelArgument(**arg))
        efflux = efflux or (lambda output, arg: ModelArgument(out=output))
        if lines is None:
            lines = [
                ('accumuline', None),
                ('accumuline', 'accumuline'),
                ('accumuline', 'accumuline_terminate'),
                ('accumuline', 'terminal'),
                ('accumuline', 'null')
            ]
        else:
            lines += [
                ('accumuline', 'accumuline'),
                ('accumuline', 'accumuline_terminate'),
                ('accumuline', 'terminal'),
                ('accumuline', 'null')
            ]
        super().__init__(
            influx=influx,
            efflux=efflux,
            model=accmodel,
            lines=lines,
            transmit_filters=transmit_filters,
            receive_filters=receive_filters
        )
        self.register_action(SignalRelease())
        if isinstance(cfx_fields, str):
            cfx_fields = [cfx_fields]
        cfx_fields = list(cfx_fields) + ['out']
        self.batch_key = batch_key or cfx_fields[0]
        conflux = Conflux(
            fields=cfx_fields,
            lines=[('accumuline', 'accumuline')]
        )
        pool = DataPool(
            release_size=throughput,
            lines=[('accumuline', 'accumuline')]
        )
        terminator = Sentry()
        terminator.register_action(TerminateAccumuline())
        origin.add_line(('source', 'accumuline'))
        origin.connect_downstream(pool)
        pool.connect_downstream(conflux)
        conflux.connect_downstream(self)
        self.connect_downstream(conflux)
        self.register_sentry(terminator)
        terminator.register_sentry(self)
        self.acc = acc
        self.origin = origin
        self.conflux = conflux
        self.pool = pool
        self.terminator = terminator
        self.released = False
        if verbose:
            self.register_action(VerboseReceive())
            self.origin.register_action(VerboseReceive())
            self.conflux.register_action(VerboseReceive())
            self.pool.register_action(VerboseReceive())
            self.terminator.register_action(VerboseReceive())

    def __repr__(self):
        return 'Accumuline()'

    def release(self):
        if self.released:
            return
        self.released = True
        self.clear_transmission()
        out = super().forward(
            arg=self.cache['arg'],
            line=self.cache['line'],
            transmit_lines={'null'}
        )
        self.cache = {}
        self.out = out.out
        for line in self.transmit:
            if line not in {'accumuline', 'accumuline_terminate', 'null'}:
                self._update_transmission(self.out, line)
        self._transmit()

    def _update_batched(self, arg):
        self.batched += arg[self.batch_key].shape[0]

    def _load_into_conflux(self):
        init = True
        while (self.pool.batched != 0) or (init):
            self.origin(line='source')
            init = False

    def _initialise_chain(self, line, arg=None):
        #print('initialising')
        self.released = False
        arg_out = ModelArgument(**self.argbase)
        self.out = []
        arg_out.update(out=self.out)
        self._update_transmission(arg_out, line=line)
        self._transmit()

    def _propagate_chain(self, line, arg=None):
        #print('propagating')
        arg_out = ModelArgument(**self.argbase)
        self._update_batched(arg)
        arg_out.update(**arg)
        out = super().forward(
            arg=arg_out,
            line=line,
            transmit_lines={'accumuline'}
        )
        #print(self.acc.acc)
        self.out = out.out

    def _terminate_chain(self, line, arg):
        #print(f'\n\n\ndurrrrp on {line}\n\n\n')
        arg_out = ModelArgument(**self.argbase)
        arg_out.update(**arg)
        arg_out.update(out=self.out, terminate=True)
        self._update_transmission(arg_out, line='accumuline_terminate')
        self.message.update(('DATA', self.data_transmission))
        self.cache = {}
        self.cache['arg'] = arg_out
        self.cache['line'] = line
        self.terminator._listen(self.message)

    def forward(self, arg=None, line='accumuline'):
        #print(self.batched)
        if arg is None:
            self._load_into_conflux()
            self._initialise_chain(line=line, arg=arg)
        elif self.released:
            return
        elif self.batched >= self.batch_size:
            self._terminate_chain(line=line, arg=arg)
        elif self.batched + self.throughput >= self.batch_size:
            self._propagate_chain(line=line, arg=arg)
        else:
            self._load_into_conflux()
            self._propagate_chain(line=line, arg=arg)


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
        out = accfwd(acc, None, argmap, passmap, T0, out, False, W)
        T1 = torch.random.rand(6, W.shape[-1], 100)
        out = accfwd(acc, None, argmap, passmap, T1, out, False, W)

    Each call to ``accfwd`` above will accumulate the local derivative in the
    provided ``acc`` object and will also append any outputs to the ``out``
    iterable that we provided in our call. When we've collected all of the
    outputs we need in ``out``, we make a terminal call to ``accfwd``::

        out = accfwd(acc, bwd, None, None, None, out, True, W)

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
        ctx.n_params = len(params)
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
            r_param = [None for _ in range(ctx.n_params)]
            ret = (None, None, None, None, None, None, *r_param)
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
