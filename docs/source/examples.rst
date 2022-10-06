A first (real) example
======================

This is a first example of a real application of the differentiable programming paradigm in the context of mapping the brain's functional connectivity. This example replicates the third experiment from our first preprint on this subject ("Covariance modelling"). The goal of this experiment is to create a simple, differentiable map of the dynamics of community structure in the brain's functional connectome.

The connectome is a graph of the brain's functional connectivity, where each node represents a brain region and each edge represents the strength of the connection between two regions. The community structure of the connectome is a partition of the nodes into groups that are more strongly connected to each other than to nodes in other groups. The dynamics of community structure is the evolution of the community structure over time. Here, we use a simple operationalisation of the dynamics that tracks only whether a community is present or absent at each time point.

This tutorial steps through the code for this experiment.

Loading the dataset
-------------------

In this tutorial, we'll perform the experiment using a subset of the Midnight Scan Club (MSC) dataset. We select a subset of the dataset to expedite the processing steps, and because we can find a rich community structure even using only this subset. The MSC dataset is a collection of fMRI scans of 10 subjects, each performing a number of in-scanner tasks across 10 scanning sessions. Here, we use the first 3 resting-state scans from each subject.

The MSC dataset is available from the OpenNeuro website. Below, we're using a utility function to retrieve a version of the dataset that has already been preprocessed. The preprocessing includes, among other standard steps, dimensionality reduction using a 400-region parcellation (brain atlas) and denoising using a 36-parameter model of motion estimates and nuisance signals.

.. code-block:: python

    import pathlib
    import jax.numpy as jnp
    import pandas as pd

    from hypercoil.engine.axisutil import extend_to_max_size
    from hypercoil.neuro.data_msc import minimal_msc_download

    dataset_root = f"{minimal_msc_download()}/data/ts/"
    paths = pathlib.Path(dataset_root).glob("*ses-func0[1-3]*task-rest*ts.1D")
    time_series = tuple(pd.read_csv(path, sep=" ", header=None) for path in paths)
    time_series = tuple(jnp.array(t.values.T) for t in time_series)
    time_series = jnp.stack(extend_to_max_size(time_series, 0.0))
    assert time_series.shape == (30, 400, 814)


Defining the model
------------------

Next, let's define the model that will be learning the community structure. We need to define the model's parameters, the model's forward pass, and the model's loss function. We begin here with the parameters. The model has two parameter tensors: one for the community structure and one for the dynamics of community structure. The community structure is a matrix of shape (n_nodes, n_communities), where each row represents a node and each column represents a community. The dynamics of community structure is a binary matrix of shape (n_time_points, n_communities), where each row represents a time point and each column represents a community.

We want to learn a community structure that is common to all subjects and all scans in the dataset. However, we also want the dynamics to be specific to each scan. To achieve this, we define the community structure as a parameter tensor that is shared across all scans, and we define the dynamics as a parameter tensor that is specific to each scan. We can achieve this by defining the community structure as a parameter tensor of shape (n_nodes, n_communities), and the dynamics as a parameter tensor of shape (n_scans, n_time_points, n_communities). The first dimension of the dynamics tensor will be used to index the dynamics of each scan.

We also want to impose some constraints on the values that the parameter tensors can take. For the community structure tensor, we's ideally want each node to belong to exactly one community. But this would give us a combinatorial optimisation problem instead of a differentiable one, so we'll relax this constraint to instead allow each node's community assignment to be a categorical probability distribution over communities. We can achieve this by projecting the community structure tensor onto the probability simplex, which is the set of vectors whose elements are nonnegative and sum to 1.0. Behind the scenes, ``hypercoil``'s ``Probability SimplexParameter`` implements this constraint using a softmax mapping.

For the dynamics tensor, we'd ideally want to impose the constraint that each community is either present or absent at each time point -- but again, we'll relax this constraint for the sake of differentiability. We'll instead allow the presence of each community at each time point to vary continuously in (0, 1) using a ``MappedLogits`` parameter. Later, we'll introduce some regularisations that encourage the dynamics to be close to binary.

The last model parameter is a scalar that sets the resolution of the community detection algorithm. This scalar, gamma, promotes discovery of more, smaller communities as it is increased. We won't learn this parameter, but we'll set it to a reasonable value for the purposes of this tutorial. In practice, we've found that a "default" value of 1 for gamma results in an unbalanced community structure that is dominated by a few large communities. We've found that a value of 5 for gamma results in a more balanced community structure.

With that said, let's implement the model:

.. code-block:: python

    import jax
    import equinox as eqx
    from hypercoil.engine import Tensor, PyTree
    from hypercoil.init.mapparam import (
        MappedLogits,
        ProbabilitySimplexParameter,
    )


    class DynamicCommunityModel(eqx.Module):
        n_nodes: int
        n_communities: int
        n_scans: int
        n_time_points: int
        gamma: float
        affiliation: Tensor
        dynamics: Tensor

        def __init__(
            self,
            n_nodes: int,
            n_scans: int,
            n_communities: int,
            n_time_points: int,
            gamma: float = 1.0,
            init_scale_affiliation: float = 0.01,
            init_scale_dynamics: float = 0.001,
            *,
            key: 'jax.random.PRNGKey',
        ):
            super().__init__()
            self.n_nodes = n_nodes
            self.n_communities = n_communities
            self.n_scans = n_scans
            self.n_time_points = n_time_points
            self.gamma = gamma

            self.affiliation = init_scale_affiliation * jax.random.normal(
                key, shape=(n_nodes, n_communities)) + 1.0
            self.dynamics = init_scale_dynamics * jax.random.normal(
                key, shape=(n_scans, n_communities, n_time_points)) + 0.5

        def __call__(self, time_series: Tensor) -> Tensor:
            return model_forward(
                time_series,
                self.affiliation,
                self.dynamics,
                self.gamma,
            )


    def parameterise_model(model):
        model = ProbabilitySimplexParameter.map(
            model, where="affiliation", axis=-1)
        model = MappedLogits.map(
            model, where="dynamics")
        return model


Defining the loss function
--------------------------

Next, 

.. code-block:: python

    from hypercoil.loss import (
        LossScheme,
        LossApply,
        Loss,
        LossArgument,
        UnpackingLossArgument,
        ModularityLoss,
        SmoothnessLoss,
        BimodalSymmetricLoss,
        identity,
        sum_scalarise,
        mean_scalarise,
        vnorm_scalarise,
    )

    def dynamic_community_loss(
        modularity_nu: float,
        smoothness_nu: float,
        dynamic_community_nu: float,
        bimodal_symmetric_nu: float,
        gamma: float,
    ) -> LossScheme:

        loss = LossScheme([
            LossApply(
                ModularityLoss(nu=modularity_nu, name='Modularity', gamma=gamma),
                apply=lambda arg: UnpackingLossArgument(
                    A=arg.corr_unparam,
                    Q=arg.affiliation,
                )),
            LossApply(
                Loss(
                    nu=dynamic_community_nu,
                    name='DynamicCommunities',
                    score=identity,
                    scalarisation=mean_scalarise(
                        axis=None,
                        inner=sum_scalarise(axis=(-1, -2), keepdims=True)
                    ),
                ),
                apply=lambda arg: -(arg.coaffiliation * arg.modularity)
            ),
            LossScheme([
                SmoothnessLoss(
                    nu=smoothness_nu,
                    scalarisation=mean_scalarise(
                        inner=vnorm_scalarise(axis=-1))
                ),
                BimodalSymmetricLoss(nu=bimodal_symmetric_nu, modes=(0, 1))
            ], apply=lambda arg: arg.dynamics)
        ])

        return loss


Defining the forward pass
-------------------------

.. code-block:: python

    from hypercoil.engine import _to_jax_array
    from hypercoil.functional import corr, modularity_matrix, coaffiliation

    def model_forward(
        time_series: Tensor,
        affiliation: Tensor,
        dynamics: Tensor,
        gamma: float,
    ) -> Tensor:
        # Ensure that all data tensors and parameters are JAX arrays.
        time_series = _to_jax_array(time_series)
        affiliation = _to_jax_array(affiliation)
        dynamics = _to_jax_array(dynamics)

        # Compute the correlation matrix for each scan.
        corr_unparam = corr(time_series)
        corr_param = corr(time_series[:, None, ...], weight=dynamics)

        # Compute the modularity matrix for each scan.
        B = modularity_matrix(
            corr_param,
            normalise_modularity=True,
            gamma=gamma,
        )
        # Compute the community co-affiliation matrix.
        H = coaffiliation(
            affiliation.T[..., None],
            normalise_coaffiliation=True,
        )

        # Build arguments for the loss function.
        args = LossArgument(
            corr_unparam=corr_unparam,
            corr_param=corr_param,
            affiliation=affiliation,
            dynamics=dynamics,
            modularity=B,
            coaffiliation=H,
        )

        return args


Defining the optimisation loop
------------------------------

.. code-block:: python

    from typing import Callable, Tuple
    import optax


    def init_optimiser(lr: float, model: PyTree) -> optax.GradientTransformation:
        optim = optax.adam(lr)
        optim_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        return optim, optim_state


    def update(
        model: PyTree,
        input: Tensor,
        loss_scheme: Callable,
        optim: optax.GradientTransformation,
        optim_state: PyTree,
        *,
        key: 'jax.random.PRNGKey',
    ) -> Tuple[PyTree, optax.OptState]:
        def loss_fn(model, input, key):
            args = model_forward(
                input, model.affiliation, model.dynamics, model.gamma
            )
            return loss_scheme(args, key=key)

        (loss, meta), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True)(model, input, key=key)
        updates, optim_state = optim.update(
            eqx.filter(grads, eqx.is_inexact_array),
            optim_state,
            eqx.filter(model, eqx.is_inexact_array),
        )
        model = eqx.apply_updates(model, updates)
        return model, optim_state, loss, meta


Train the model
---------------

.. code-block:: python

    # Configure the hyperparameters.
    n_communities = 10
    n_time_points = 814
    n_nodes = 400
    n_scans = 30
    lr = 0.05
    modularity_nu = 10
    dynamic_community_nu = 2e-3
    smoothness_nu = .2
    bimodal_symmetric_nu = 2
    max_epoch = 500
    gamma = 5
    key = jax.random.PRNGKey(0)

    key_model, key_train = jax.random.split(key)

    # Initialise the model.
    model = DynamicCommunityModel(
        n_nodes=n_nodes,
        n_scans=n_scans,
        n_communities=n_communities,
        n_time_points=n_time_points,
        gamma=gamma,
        key=key_model,
    )
    model = parameterise_model(model)

    # Initialise the loss function.
    loss_scheme = dynamic_community_loss(
        modularity_nu=modularity_nu,
        smoothness_nu=smoothness_nu,
        dynamic_community_nu=dynamic_community_nu,
        bimodal_symmetric_nu=bimodal_symmetric_nu,
        gamma=gamma,
    )

    # Initialise the optimiser.
    optim, optim_state = init_optimiser(lr, model)

    # Train the model.
    for epoch in range(max_epoch):
        key_epoch = jax.random.fold_in(key_train, epoch)
        model, optim_state, loss, meta = eqx.filter_jit(update)(
            model,
            time_series,
            loss_scheme,
            optim,
            optim_state,
            key=key_epoch,
        )

        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {loss}')
            for k, v in meta.items():
                print(f'{k}: {v.value:.4f}')
