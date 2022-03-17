# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Support Vector Machine
~~~~~~~~~~~~~~~~~~~~~~
Differentiation through the support vector machine, a convex optimisation
problem.
"""
import torch
import cvxpy as cvx
from functools import partial
from ..functional.symmap import symsqrt
from ..functional.matrix import recondition_eigenspaces
from cvxpylayers.torch import CvxpyLayer


def hinge_loss(Y_hat, Y):
    return torch.maximum(
        1 - Y * Y_hat,
        torch.tensor([0])
    ).sum()


def polynomial_kernel(X, Z, order):
    return (X @ Z.transpose(-1, -2)) ** order


# TODO: extend so that we can use a covariance (orientation) instead
# of just a diagonal matrix in the multivariate case. Is this really
# necessary? Eh, probably not.
def gaussian_kernel(X, Z, sigma):
    return torch.exp(
        -((X.unsqueeze(-3) - Z.unsqueeze(-2)) ** 2).sum(-1) /
        (2 * sigma ** 2)
    )


def mv_gaussian_kernel(X, Z, sigma):
    raise NotImplementedError('This feature does not work.')
    diff = (X.unsqueeze(-3) - Z.unsqueeze(-2))
    # inv mahalanobis distance
    return torch.exp(
        -(diff.unsqueeze(-2) @ sigma @ diff.unsqueeze(-1)).squeeze() /
        (2 * torch.det(sigma) ** 2)
    )


class Kernel():
    def __init__(self, K):
        self.K = K

    def __call__(self, X, Z=None):
        if Z is None: Z = X
        return self.K(X, Z)


class LinearKernel(Kernel):
    def __init__(self):
        self.K = lambda X, Z: X @ Z.transpose(-1, -2)


class PolynomialKernel(Kernel):
    def __init__(self, order=2):
        self.K = partial(polynomial_kernel, order=order)


class GaussianKernel(Kernel):
    def __init__(self, sigma=1):
        self.K = partial(gaussian_kernel, sigma=sigma)


class MVGaussianKernel(Kernel):
    def __init__(self, sigma=None):
        default_dim = 2
        if sigma is None: sigma = torch.eye(default_dim)
        self.K = partial(mv_gaussian_kernel, sigma=sigma)


class SVM(CvxpyLayer):
    def __init__(self, n, K=GaussianKernel(), C=1.):
        """
        Initialise a kernelised support vector machine as a fully
        differentiable convex optimisation layer.

        Parameters
        ----------
        n : int
            Number of observations in the training dataset. Because we use
            the dual, this is a required parameter. We could automatically
            grab it in `fit` and this would probably be better in the long
            run (and more conformant with sklearn syntax), but we'll get to
            that later...
        K : callable (default Gaussian)
            Kernel function. The callable signature should accept either
            one or two :math:`n \times d` matrices and return a single
            positive semidefinite :math:`d \times d` matrix (the kernel
            matrix). If the callable receives a single argument, it
            should return the kernel matrix computed 
        C : float (default 1)
            Hyperparameter that trades off between maximising the margin
            and penalising soft-margin violations.

        Notes
        -----
            This problem is framed as the dual of a soft-margin SVM. See
            10:15 here: https://www.youtube.com/watch?v=zzn80wmclnw
        
            Note that we must introduce the new variable Z below to
            satisfy the `cxvpylayers` disciplined parametric programming
            requirement.
        """
        alpha = cvx.Variable((n, 1))
        Z_ = cvx.Variable((n, 1))

        self.C = C
        self.K = K
        self.n = n

        X_ = cvx.Parameter((n, n), PSD=True)
        Y_ = cvx.Parameter((n, 1))

        # For DPP compliance, we cannot use the quadratic form.
        # We formulate instead the following equivalent problem,
        # taking the matrix square root in the forward pass below.
        #
        # This is equivalent to the more 'canonical' quadratic form
        # objective = cvx.Maximize(
        #     cvx.sum(alpha) - 0.5 * cvx.quad_form(
        #         cvx.multiply(alpha, Y_),
        #         cvx.atoms.affine.wraps.psd_wrap(X_)
        #     )
        # )
        #
        # See https://www.cvxpy.org/tutorial/advanced/index.html ... 
        # ... #disciplined-parametrized-programming
        objective = cvx.Maximize(
            cvx.sum(alpha) - 0.5 * cvx.sum_squares(X_ @ Z_)
        )
        constraints = [
            alpha >= 0,
            alpha <= self.C,
            Y_.T @ alpha == 0,
            Z_ == cvx.multiply(alpha, Y_)
        ]
        problem = cvx.Problem(objective, constraints)
        super().__init__(problem, parameters=[X_, Y_], variables=[alpha])

    def fit(self, X, Y):
        """
        Fit the SVM to a training dataset.
        """
        #TODO: Figure out what the deal is with typing in cvxpy layers.
        # It appears it might sometimes cast to double precision.
        self.X, self.Y = X, Y
        ker = self.K(self.X)
        #TODO: the below could be a problem. Make this a parameter to the
        # module.
        eps = torch.finfo(ker.dtype).eps
        fac = 0.01 * torch.linalg.matrix_norm(ker, 2)
        ker = recondition_eigenspaces(ker, psi=fac, xi=fac)
        symsqker = symsqrt(ker)
        L, Q = torch.linalg.eigh(ker)
        self.alpha = super().forward(
            symsqker, Y
        )[0]

        # Recover the primal solution implicitly for prediction.
        # As here (11:31): https://www.youtube.com/watch?v=zzn80wmclnw
        # Also https://haipeng-luo.net/courses/CSCI567/2018_fall/lec6.pdf

        # Bias term: first get the support vectors
        sv_idx = self._support_vectors()
        Y_sv = self.Y[sv_idx]
        K_sv = ker[sv_idx]
        
        self.bias = self._estimate_bias(K_sv, Y_sv)

    def _support_vectors(self):
        """
        Identify the indices of support vectors in the training set.
        """
        e1 = torch.logical_and(
            self.alpha > 0,
            self.alpha < self.C
        )
        e2 = torch.logical_and(
            torch.logical_not(
                torch.isclose(
                    self.alpha,
                    torch.tensor(self.C),
                    atol=1e-3,
                    rtol=1e-3)
            ),
            torch.logical_not(
                torch.isclose(
                    self.alpha,
                    torch.tensor(0.),
                    atol=1e-3,
                    rtol=1e-3)
            ),
        )
        sv_idx = torch.logical_and(e1, e2).squeeze()
        return sv_idx

    def _estimate_bias(self, K_sv, Y_sv):
        """
        Use the support vectors to reconstitute the SVM bias term from the
        primal solution. Because the estimates of the bias might differ across
        support vectors, we use a consistency weighting approach.
        """
        b_est = (Y_sv - K_sv @ (self.alpha * self.Y)).squeeze()
        # Phantom code using consistency weighting.
        #wei = self._consistency_weighting(b_est)
        #b = ((b_est * wei).sum() / wei.sum()).item()
        # b = torch.sign(
        #     b_est[torch.abs(b_est).argmin()]
        # ) * torch.abs(b_est).min()
        b = self._modal_selection(b_est)
        #print(b_est, wei, b)
        return b
    
    def _consistency_weighting(self, estimates):
        """
        Come up with weights for each estimate based on its internal
        consistency.
        
        In the SVM, we can use this to reconstitute the bias term using
        support vectors. The estimates of the bias might not agree across the
        support vectors, so we use the consistency of different estimates to
        upweight or downweight them and produce a consensus.

        NOTE: In practice I haven't found that we need to use this, so it's
        never actually called right now. The functionality is retained here in
        case it becomes relevant at some point.

        I have no reference for this, and it is not guaranteed to work.
        I found that it worked reasonably well in practice on several examples
        that I tried.
        """
        wei = torch.abs((estimates.view(-1, 1) - estimates.view(1, -1)))
        wei = (wei).mean(0)
        wei = wei - wei.min()
        wei = torch.tanh(wei.median() / wei)
        return wei
    
    def _modal_selection(self, estimates, ratio=2):
        """
        Select a reasonable bias term by finding what passes as the mode.

        TODO: This is probably not necessary. Test the code using all the
        estimates and see if it suffers.
        """
        dist = torch.abs((estimates.view(-1, 1) - estimates.view(1, -1)))
        dist = (torch.sqrt(dist)).mean(0)
        _, modal_idx = torch.sort(dist)
        n_sv = len(modal_idx)
        modal = estimates[modal_idx[:(n_sv // ratio + 1)]]
        return modal.mean()

    def transform(self, X):
        """
        Differentiably classify new data using the fit SVM parameters.
        """
        return (self.alpha * self.Y).squeeze() @ (
            self.K(self.X, X)) + self.bias

    def fit_transform(self, X, Y):
        """
        Fit the SVM to a training dataset and return its predictions
        on that dataset.
        """
        self.fit(X, Y)
        return self.transform(X)

    def forward(self, X, Y=None):
        if self.training:
            if Y is None:
                raise ValueError(
                    'Labelled examples are required when training. Either '
                    'provide a label set as input `Y` or set the `training` '
                    'attribute to False to perform inference.'
                )
            return self.fit_transform(X, Y)
        else:
            return self.transform(X)
