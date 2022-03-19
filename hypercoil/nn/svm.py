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
from itertools import product
from ..functional.symmap import symsqrt
from ..functional.matrix import recondition_eigenspaces
from cvxpylayers.torch import CvxpyLayer


def hinge_loss(Y_hat, Y):
    return torch.maximum(
        1 - Y * Y_hat,
        torch.zeros_like(Y)
    ).sum()


def labels_binary(labels, dtype=torch.float):
    l = labels.unique()[0]
    return [2 * (labels == l).type(dtype) - 1]


def labels_one_vs_rest(labels, dtype=torch.float):
    uniq = labels.unique()
    return [
        2 * (labels == u).type(dtype) - 1
        for u in uniq
    ]


def labels_one_vs_one(labels, dtype=torch.float):
    uniq = labels.unique()
    uniqlist = uniq.tolist()
    label_combinations = product(uniqlist, uniqlist)
    return [
        (labels == label1).type(dtype) - (labels == label2).dtype(dtype)
        for label1, label2 in label_combinations
        if label1 != label2
    ]


def class_weight_vector(labels, class_weight):
    return class_weight[labels.long()]


def polynomial_kernel(X, Z, gamma, order, r):
    return (gamma * X @ Z.transpose(-1, -2) + r) ** order


def gaussian_kernel(X, Z, gamma):
    return torch.exp(
        -gamma * ((X.unsqueeze(-3) - Z.unsqueeze(-2)) ** 2).sum(-1)
    )


def sigmoid_kernel(X, Z, r):
    return torch.tanh(gamma * X @ Z.transpose(-1, -2) + r)


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
    def __init__(self, order=2, gamma=1, r=0):
        self.order = order
        self.gamma = gamma
        self.r = r

    def __call__(self, X, Z=None):
        self.K = partial(
            polynomial_kernel,
            order=self.order,
            gamma=self.gamma,
            r=self.r
        )
        return super(PolynomialKernel, self).__call__(X, Z)


class GaussianKernel(Kernel):
    def __init__(self, gamma=1, sigma=None):
        if sigma is not None:
            import math
            gamma = 1 / (2 * sigma ** 2)
        self.gamma = gamma

    def __call__(self, X, Z=None):
        self.K = partial(gaussian_kernel, gamma=self.gamma)
        return super(GaussianKernel, self).__call__(X, Z)


class SigmoidKernel(Kernel):
    def __init__(self, gamma=1, r=0):
        self.gamma = gamma
        self.r = r

    def __call__(self, X, Z=None):
        self.K = partial(
            sigmoid_kernel,
            gamma=self.gamma,
            r=self.r
        )
        return super(SigmoidKernel, self).__call__(X, Z)


class SVM(torch.nn.Module):
    def __init__(
        self,
        C=1.0,
        kernel='rbf',
        degree=3,
        gamma='scale',
        coef0=0.0,
        class_weight=None,
        verbose=False,
        decision_function_shape='ovr',
        recondition=0.01,
        formulate_on_forward_pass=False,
        solver=None,
        n_observations=None,
        n_classes=None,
        sample_weight=None
    ):
        """
        Initialise a kernelised support vector machine as a fully
        differentiable convex optimisation layer.

        Do not use this module unless you have a good reason; to enforce DPP
        compliance and ensure well-formed gradients, some performance
        sacrifices have been made, and the forward pass quickly becomes
        prohibitively slow for large problems. A long-term solution would
        involve finding the analytic derivative through the SVM and
        implementing a custom backward method while using the sklearn forward.
        Or, alternatively finding a way to use libSVM as the back end solver.

        Parameters
        ----------
        C : positive float (default 1)
            Regularisation hyperparameter; a smaller value of C increases the
            strength of the regularisation.
        kernel : 'rbf', 'poly', 'linear', 'sigmoid', or callable
            Kernel function. The callable signature should accept either
            one or two :math:`n \times d` matrices and return a single
            positive semidefinite :math:`n \times n` matrix (the kernel
            matrix). By default, a Gaussian kernel (RBF) is used. It is also
            possible to pass a pre-computed Gram matrix by setting this to
            'precomputed'.
        degree : int (default 3)
            If a polynomial kernel is selected, specifies the maximum order
            of the polynomial.
        gamma : float, 'auto', or 'scale' (default 'scale')
            Kernel coefficient for RBF, polynomial, and sigmoid kernels.
            'scale' and 'auto' automatically configure the kernel coefficient
            based on the input dataset, as in sklearn.
        coef0 : float (default 0)
            Bias coefficient for polynomial and sigmoid kernels.
        class_weight : list or None (default None)
            Sets the weight of C to C * `class_weight[i]` for class i. Can be
            used to adjust the fit if classes are unbalanced.
        verbose : bool (default False)
            Toggles the verbosity of the solver.
        decision_function_shape : 'ovr'
            Currently, only one-vs-rest classifiers are available in the
            multi-class setting.
        recondition : float (default 0.01)
            Reconditioning parameter to ensure positiveness and uniqueness of
            all eigenvalues of the Gram matrix, which is necessary for
            differentiation but likely sacrifices classification performance.
            Reconditioning adds nonnegative values to the main diagonal of the
            Gram matrix, approximately on the order of the largest singular
            value of the matrix scaled by the provided reconditioning
            factor.
        formulate_on_forward_pass : bool (default False)
            Indicates that the problem should be formulated on every forward
            pass through the network. This will make the forward pass
            extremely slow, but is necessary if training with a variable
            number of observations. If this is false, the model cannot be used
            unless either the `formulate` method is called or both the
            `n_observations` and `n_classes` parameters are passed to the
            constructor.
        solver : None, 'ECOS', or 'SCS' (default None)
            Convex solver to use for solving the SVM. By default the decision
            is passed to the `diffcp` package. Note that this is generally
            incredibly inefficient compared with using a dedicated SVM solver.
        n_observations : int or None (default None)
            Number of observations in the training dataset. If not specified,
            this must be set when calling `formulate`; otherwise it will be
            determined automatically during the forward pass.
        n_classes : int or None (default None)
            Number of classes in the dataset. If not specified, this must be
            set when calling `formulate`; otherwise it will be determined
            automatically during the forward pass.
        sample_weight : tensor or None (default None)
            Weight for each observation. If specified, the value of C for
            observation i is scaled to C * sample_weight[i].

        Notes
        -----
            This problem is framed as the dual of a soft-margin SVM. See
            10:15 here: https://www.youtube.com/watch?v=zzn80wmclnw

            Note that we must introduce the new variable Z below to
            satisfy the `cxvpylayers` disciplined parametric programming
            requirement.
        """
        self.update_params(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            class_weight=class_weight,
            verbose=verbose,
            decision_function_shape=decision_function_shape,
            recondition=recondition,
            formulate_on_forward_pass=formulate_on_forward_pass,
            solver=solver,
            n_observations=n_observations,
            n_classes=n_classes,
            sample_weight=sample_weight
        )
        super(SVM, self).__init__()

    def update_params(
        self,
        C=None,
        kernel=None,
        degree=None,
        gamma=None,
        coef0=None,
        class_weight=False,
        verbose=None,
        decision_function_shape=None,
        recondition=None,
        formulate_on_forward_pass=None,
        solver=False,
        n_observations=None,
        n_classes=None,
        sample_weight=None
    ):
        """
        Function for safely updating the SVM parameters. Particularly when
        changing the kernel, use this function instead of directly accessing
        the attributes.
        """
        if C is not None:
            self.C = C
        if kernel is not None:
            self.kernel = kernel
            if kernel == 'rbf':
                self.K = GaussianKernel(gamma=gamma)
            elif kernel == 'poly':
                self.K = PolynomialKernel(gamma=gamma, order=degree, r=coef0)
            elif kernel == 'sigmoid':
                self.K = SigmoidKernel(gamma=gamma, r=coef0)
            elif kernel == 'linear':
                self.K = LinearKernel()
            elif kernel == 'precomputed':
                self.K = 'precomputed'
            else:
                self.K = kernel
        if degree is not None:
            self.degree = degree
        if gamma is not None:
            self.gamma = gamma
        if coef0 is not None:
            self.coef0 = coef0
        if class_weight is not False:
            self.class_weight = class_weight
        if verbose is not None:
            self.verbose = verbose
        if decision_function_shape is not None:
            self.decision_function_shape = decision_function_shape
        if recondition is not None:
            self.recondition = recondition
        if formulate_on_forward_pass is not None:
            self.formulate_on_forward_pass = formulate_on_forward_pass
        if solver is not False:
            self.solver = solver
        if n_observations is not None and n_classes is not None:
            self.formulate(
                n_observations=n_observations,
                n_classes=n_classes,
                sample_weight=sample_weight
            )

    def formulate_problem(self, n, C=None):
        """
        Formulate a single binary SVM classifier's dual problem in CVX.

        Notes
        -----
            This problem is framed as the dual of a soft-margin SVM. See
            10:15 here: https://www.youtube.com/watch?v=zzn80wmclnw

            Note that we must introduce the new variable Z below to
            satisfy the `cxvpylayers` disciplined parametric programming
            requirement.
        """
        if C is None:
            C = self.C
        alpha = cvx.Variable((n, 1))
        Z_ = cvx.Variable((n, 1))

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
            alpha <= C,
            Y_.T @ alpha == 0,
            Z_ == cvx.multiply(alpha, Y_)
        ]
        problem = cvx.Problem(objective, constraints)
        return CvxpyLayer(
            problem,
            parameters=[X_, Y_],
            variables=[alpha]
        )

    def formulate(self, n_observations, n_classes, sample_weight=None):
        """
        Formulate all SVM problems to be solved during each forward pass.

        Parameters
        ----------
        n_observations : int
            Number of observations in the training dataset. If not specified,
            this must be set when calling `formulate`; otherwise it will be
            determined automatically during the forward pass.
        n_classes : int
            Number of classes in the dataset. If not specified, this must be
            set when calling `formulate`; otherwise it will be determined
            automatically during the forward pass.
        sample_weight : tensor or None (default None)
            Weight for each observation. If specified, the value of C for
            observation i is scaled to C * sample_weight[i].
        """
        C = self.C
        if self.class_weight is not None:
            C = class_weight_vector(Y, self.class_weight) * C
        if sample_weight is not None:
            C = sample_weight * C
        self.C_vector = C
        if n_classes == 2:
            self.problem = [self.formulate_problem(
                n=n_observations,
                C=C
            )]
        elif self.decision_function_shape == 'ovr':
            self.problem = [self.formulate_problem(
                n=n_observations,
                C=C
            ) for _ in range(n_classes)]
        elif self.decision_function_shape == 'ovo':
            raise NotImplementedError(
                'One-vs-one classifier not yet implemented.'
            )

    def fit(self, X, Y, sample_weight=None, formulate=False):
        """
        Fit the SVM to a training dataset.
        """
        n_observations, n_features = X.shape[-2:]
        n_classes = len(Y.unique())
        self.configure_gamma(X, n_features)

        self.X = X
        if self.kernel == 'precomputed':
            ker = X
        else:
            ker = self.K(self.X)
        try:
            recond_factor = (
                self.recondition * torch.linalg.matrix_norm(ker, 2)
            )
        except torch.linalg.LinAlgError:
            recond_factor = self.recondition
        ker = recondition_eigenspaces(
            ker,
            psi=recond_factor,
            xi=recond_factor
        )
        symsqker = symsqrt(ker)

        if formulate:
            self.formulate(
                n_observations=n_observations,
                n_classes=n_classes,
                sample_weight=sample_weight
            )

        self.alpha = torch.empty(
            (len(self.problem), n_observations),
            dtype=X.dtype,
            device=X.device
        )
        self.Y = torch.empty(
            (len(self.problem), n_observations),
            dtype=X.dtype,
            device=X.device
        )
        self.bias = torch.empty(
            (len(self.problem), 1),
            dtype=X.dtype,
            device=X.device
        )

        if n_classes == 2:
            labels = labels_binary(Y, dtype=X.dtype)
        elif self.decision_function_shape == 'ovr':
            labels = labels_one_vs_rest(Y, dtype=X.dtype)
        elif self.decision_function_shape == 'ovo':
            raise NotImplementedError(
                'One-vs-one classifier not yet implemented.'
            )

        for i, (p, y) in enumerate(zip(self.problem, labels)):
            alpha = p.forward(
                symsqker, y,
                solver_args={
                    'solve_method': self.solver,
                    'verbose': self.verbose
                }
            )[0]
            sv_idx = self._support_vectors(self.C_vector, alpha)
            Y_sv = y[sv_idx]
            K_sv = ker[sv_idx]
            bias = self._estimate_bias(K_sv, Y_sv, alpha, y)

            self.alpha[i, :] = alpha.squeeze()
            self.bias[i] = bias.squeeze()
            self.Y[i, :] = y.squeeze()

    def configure_gamma(self, X, n_features):
        """
        Helper method for automatically estimating the kernel coefficient
        parameter gamma based on properties of the input dataset.
        """
        if self.gamma == 'scale':
            gamma = 1 / (n_features * X.var())
            try:
                self.K.gamma = gamma
            except AttributeError:
                pass
        elif self.gamma == 'auto':
            gamma = 1 / n_features
            try:
                self.K.gamma = gamma
            except AttributeError:
                pass

    def _support_vectors(self, C, alpha):
        """
        Identify the indices of support vectors in the training set.
        """
        e1 = torch.logical_and(
            alpha > 0,
            alpha < C
        )
        e2 = torch.logical_and(
            torch.logical_not(
                torch.isclose(
                    alpha,
                    torch.tensor(C, dtype=alpha.dtype, device=alpha.device),
                    atol=1e-3,
                    rtol=1e-3)
            ),
            torch.logical_not(
                torch.isclose(
                    alpha,
                    torch.tensor(0, dtype=alpha.dtype, device=alpha.device),
                    atol=1e-3,
                    rtol=1e-3)
            ),
        )
        sv_idx = torch.logical_and(e1, e2).squeeze()
        return sv_idx

    def _estimate_bias(self, K_sv, Y_sv, alpha, Y):
        """
        Use the support vectors to reconstitute the SVM bias term from the
        primal solution. Because the estimates of the bias might differ across
        support vectors, we use a consistency weighting approach.
        """
        b_est = (Y_sv - K_sv @ (alpha * Y)).squeeze()
        # Phantom code using consistency weighting.
        #wei = self._consistency_weighting(b_est)
        #b = ((b_est * wei).sum() / wei.sum()).item()
        # b = torch.sign(
        #     b_est[torch.abs(b_est).argmin()]
        # ) * torch.abs(b_est).min()
        b = self._modal_selection(b_est)
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
        wei = wei - wei.amin()
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
        if self.kernel == 'precomputed':
            return (self.alpha * self.Y) @ X + self.bias
        return (self.alpha * self.Y) @ (self.K(self.X, X)) + self.bias

    def fit_transform(self, X, Y, sample_weight=None, formulate=False):
        """
        Fit the SVM to a training dataset and return its predictions
        on that dataset.
        """
        self.fit(X, Y, sample_weight=sample_weight, formulate=formulate)
        return self.transform(X)

    def forward(self, X, Y=None, sample_weight=None):
        if self.training:
            if Y is None:
                raise ValueError(
                    'Labelled examples are required when training. Either '
                    'provide a label set as input `Y` or set the `training` '
                    'attribute to False to perform inference.'
                )
            return self.fit_transform(
                X, Y,
                sample_weight=sample_weight,
                formulate=self.formulate_on_forward_pass
            )
        else:
            return self.transform(X)
