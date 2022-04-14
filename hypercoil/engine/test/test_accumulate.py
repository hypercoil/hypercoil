# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for accumulators.
"""
import pytest
import torch
from hypercoil.loss import LossArgument as ModelArgument
from hypercoil.engine.accumulate import (
    Accumuline,
    Accumulator,
    AccumulatingFunction
)


class TestAccumulator:

    def test_acc_class(self):
        torch.manual_seed(0)
        W = torch.rand(10, 50, dtype=torch.double)
        W.requires_grad = True
        T = torch.rand(20, 50, 45, dtype=torch.double)

        W2 = torch.rand(10, 10, dtype=torch.double)
        W3 = torch.rand(45, 5, dtype=torch.double)
        W2.requires_grad = True
        W3.requires_grad = True

        loss = torch.nn.MSELoss()

        def model_grad(x, *args, **kwargs):
            return ModelArgument(
                weight=x.transpose(-1, -2)
            )

        acc = Accumulator(
            model=lambda x: W @ x,
            gradient=model_grad,
            retain_dims=(-1, -2),
            model_params=[],
            reduce_dims='sum'
        )

        out = []
        for t in T:
            out += [acc.forward(ModelArgument(x=t))['out']]
        out = torch.stack(out)
        out.requires_grad = True

        Y = torch.tanh(out)
        Y = W2 @ out @ W3
        l = loss(Y, torch.zeros_like(Y))
        l.backward()

        mult = out.grad.detach()
        acc.acc['weight'].shape, out.grad.shape
        attempt = (mult.sum(0) @ (acc.acc['weight']))

        out = W @ T
        out.retain_grad()
        Y = torch.tanh(out)
        Y = W2 @ out @ W3
        l = loss(Y, torch.zeros_like(Y))
        l.backward()

        ref = W.grad.clone()


    def test_acc_fn(self):
        torch.manual_seed(0)
        W = torch.rand(10, 50, dtype=torch.double)
        W.requires_grad = True
        T = torch.rand(20, 50, 45, dtype=torch.double)

        W2 = torch.rand(10, 10, dtype=torch.double)
        W3 = torch.rand(45, 5, dtype=torch.double)
        W2.requires_grad = True
        W3.requires_grad = True

        loss = torch.nn.MSELoss()

        def bwd(grad_output, grad_local):
            return grad_output @ grad_local,

        def model_grad(x, *args, **kwargs):
            return ModelArgument(
                weight=x.transpose(-1, -2)
            )

        class Slicing0Source:
            def __init__(self, tensor):
                self.tensor = tensor
                self.idx = 0

            def sample(self, samples):
                s = slice(self.idx, self.idx + samples)
                sample = self.tensor[s]
                self.idx += samples
                return sample

        acc = Accumulator(
            model=lambda x: W @ x,
            gradient=model_grad,
            retain_dims=(-1, -2),
            model_params=[],
            reduce_dims='mean'
        )

        af = AccumulatingFunction.apply
        argmap = lambda t: ModelArgument(x=t)
        data_source = Slicing0Source(T)

        sampled = 0
        batch_size = 20
        throughput = 3
        out = []
        terminate = False
        while not terminate:
            if sampled > batch_size:
                terminate = True
                sample = None
            else:
                sample = data_source.sample(throughput)
                sampled += throughput
            out = af(
                acc,
                bwd,
                argmap,
                sample,
                out,
                terminate,
                W
            )
        out = out[0]
        Y = torch.tanh(out)
        Y = W2 @ out @ W3
        l = loss(Y, torch.zeros_like(Y))
        l.backward()

        attempt = W.grad.clone()

        W.grad.zero_()
        W2.grad.zero_()
        W3.grad.zero_()

        out = W @ T
        out.retain_grad()
        Y = torch.tanh(out)
        Y = W2 @ out @ W3
        l = loss(Y, torch.zeros_like(Y))
        l.backward()

        ref = W.grad.clone()

        gradchk = (
            (ref - attempt).abs() /
            torch.maximum(ref.abs(), attempt.abs())
        )
        assert gradchk.max() < 2e-3

    def test_acc_line(self):
        #TODO: we've seed some larger errors than the limit of tolerance with
        # other seeds. Include a warning about this utility.
        torch.manual_seed(0)
        W = torch.rand(10, 50, dtype=torch.double)
        W.requires_grad = True
        T = torch.rand(20, 50, 45, dtype=torch.double)

        W2 = torch.rand(10, 10, dtype=torch.double)
        W3 = torch.rand(45, 5, dtype=torch.double)
        W2.requires_grad = True
        W3.requires_grad = True

        loss = (lambda x, y: (x ** 2).sum())

        def bwd(grad_output, grad_local):
            return grad_output @ grad_local,

        def model_grad(x, *args, **kwargs):
            return ModelArgument(
                weight=x.transpose(-1, -2)
            )

        class Slicing0Source:
            def __init__(self, tensor):
                self.tensor = tensor
                self.idx = 0

            def sample(self, samples):
                s = slice(self.idx, self.idx + samples)
                sample = self.tensor[s]
                self.idx += samples
                return sample

        data_source = Slicing0Source(T)

        ##TODO: If we just change all the softmax axes to -1 instead of
        # -2, this fails the gradient check catastrophically! In further
        # learning tests, it additionally exxhibited extremely undesirable
        # behaviour, including rebounding losses. At some point, we should
        # figure out what is going on, and in the meantime we should
        # implement some kind of built-in test to the accumuline class and
        # furthermore include a major warning in the documentation.
        aline = Accumuline(
            model=lambda x: torch.softmax(W, -2) @ x,
            gradient=model_grad,
            backward=bwd,
            retain_dims=(-1, -2),
            argmap=(lambda t: ModelArgument(x=t)),
            throughput=3,
            batch_size=20
        )
        out = aline(
            data_source, torch.softmax(W, -2)
        )
        out = out[0]
        Y = torch.tanh(out)
        Y = W2 @ out @ W3
        l = loss(Y, torch.zeros_like(Y))
        l.backward()

        attempt = W.grad.clone()

        W.grad.zero_()
        W2.grad.zero_()
        W3.grad.zero_()

        out = torch.softmax(W, -2) @ T
        out.retain_grad()
        Y = torch.tanh(out)
        Y = W2 @ out @ W3
        l = loss(Y, torch.zeros_like(Y))
        l.backward()

        ref = W.grad.clone()

        gradchk = (
            (ref - attempt).abs() /
            torch.maximum(ref.abs(), attempt.abs())
        )
        """
        print(ref.ravel()[torch.argsort(gradchk.ravel(),
            descending=True)])
        print(attempt.ravel()[torch.argsort(gradchk.ravel(),
            descending=True)])
        print(gradchk.ravel()[torch.argsort(gradchk.ravel(),
            descending=True)])
        """
        assert gradchk.max() < 5e-3
