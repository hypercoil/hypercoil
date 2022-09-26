Technical overview
------------------

This library is implemented using `JAX <https://jax.readthedocs.io/en/latest/>`_, which combines a NumPy-like API with automatic differentiation and support for GPU acceleration. The library is designed to be modular and extensible, and to be used in conjunction with existing tools in the Python ecosystem. The library is currently under active development, and is not yet ready for use outside of research and development.

``functional`` and ``nn``: composable differentiable functionals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``functional`` module provides a set of composable differentiable functionals, which can be used to construct differentiable programs. Common steps in a functional neuroimaging workflow are provided as pre-defined functionals, including (among others):

* ``cov`` : Covariance estimation: empirical covariance, Pearson correlation, partial correlation, conditional correlation, and others
* ``fourier`` : Frequency-domain operations, such as temporal filtering
* ``graph`` : Graph-theoretic operations, such as graph Laplacian estimation and community detection
* ``interpolate`` : Methods for temporal interpolation over artefact-contaminated time frames
* ``kernel`` : Similarity kernels and pairwise distance metrics
* ``resid`` : Residualisation and regression
* ``semidefinite`` : Projection between the positive semidefinite cone and tangent spaces
* ``sphere`` : Operations on spherical approximations to the cortex, such as geodesics and spherical convolution

The ``nn`` module provides a set of neural network layers that can be used to construct differentiable programs. These layers provide an alternative API to the ``functional`` module and also include more complicated parameterised functionals. They are implemented using the JAX-based `Equinox <https://docs.kidger.site/equinox/>`_ library.

``init``: functional parameterisation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``init`` module provides mechanisms for parameterising differentiable functionals without learning. This includes a set of pre-defined parameterisations that incorporate domain knowledge from functional neuroimaging. These components can be used to implement pre-existing workflows or to learn a new workflow starting from a pre-existing one.

Parameterisation includes both initialisation and mapping to a constrained space. For example, the ``init.atlas`` module provides a set of parameterisations that correspond to different types of brain atlases, such as surface atlases, volumetric atlases, discrete parcellations, and probabilistic functional modes. Complementarily, the ``init.mapparam`` module uses transformations to constrain parameters to a particular subspace or manifold, such as the positive semidefinite cone, the sphere, or the probability simplex.

``loss``: learning signals
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``loss`` module provides a set of learning signals (i.e., loss functions) that can be used to train differentiable programs. Loss functions are designed for use in combination with `Equinox's filters <https://docs.kidger.site/equinox/api/filtering/filter-functions/>`_ and the excellent `optax library <https://optax.readthedocs.io/en/latest/>`_ for optimisation.

Loss functions are implemented compositionally using a functional API, and comprise two components: a *score function* and a *scalarisation*. The score function maps tensors from a differentiable program to a tensor of scores, and the scalarisation maps the scores to a scalar loss. The ``loss`` module provides a set of pre-defined score functions and scalarisations with applications in functional neuroimaging and beyond.

``formula``: functional grammar
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This library also includes an extensible functional grammar for various purposes. Internally, we use it to implement confound model specification, an FSLmaths-like API for image manipulation, and a syntax for addressing and filtering neural network parameters.

``viz``: visualisation
^^^^^^^^^^^^^^^^^^^^^^

Visualisation utilities will include (*inter alia*) a PyVista-based 3D visualisation API for plotting brain surfaces, atlases, and networks, and a set of utilities for plotting brain connectivity matrices. These utilities will be designed to automatically read information from differentiable models using a functional reporting system. This framework remains under development.

A simple example
----------------

Here's a small example that shows how the above modules can be combined to construct a simple differentiable program for first filtering a time series, next estimating its correlation conditioned on a confound model, and finally projecting the estimated covariance out of the positive semidefinite cone and into a tangent space. The model is then trained using a simple loss function that promotes correlations with a large magnitude.

Note that this is not a particularly useful model, but it serves to illustrate the basic principles. (Astute readers will also remark several instances in the code of incorrect or oversimplified processing decisions. This is intentional, as this vignette is not intended to be instructional with regard to functional neuroimaging.)

.. code-block:: python

    import json
    from functools import partial
    from pkg_resources import resource_filename as pkgrf

    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import optax
    import pandas as pd

    from hypercoil.formula import ConfoundFormulaGrammar
    from hypercoil.functional import conditionalcorr
    from hypercoil.init import (
        FreqFilterSpec,
        DirichletInitialiser,
        MappedLogits,
        SPDGeometricMean,
    )
    from hypercoil.loss import (
        bimodal_symmetric,
        vnorm_scalarise,
    )
    from hypercoil.neuro.synth import (
        synthesise_matched,
    )
    from hypercoil.nn import (
        FrequencyDomainFilter,
        TangentProject,
        BinaryCovariance,
    )

    #-----------------------------------------------------------------------------#
    # 1. Generate some synthetic data: first, configure the dimensions.
    max_epoch = 10
    log_interval = 1
    n_subjects = 10
    n_voxels = 400
    n_time_points = 200
    n_channels = 4  # Data channels: These could be different connectivity
                    #                "states" captured by the covariance.
                    #                Or, if we made the weights fixed rather
                    #                than trainable, they could be different
                    #                pipeline configurations for multiverse
                    #                analysis.
    key = jax.random.PRNGKey(0)
    data_key, filter_key, cov_key, proj_key = jax.random.split(key, 4)

    #-----------------------------------------------------------------------------#
    # 2. Create a synthetic time series with spectrum and covariance matched to
    #    a parcellated human brain.
    ref_path = pkgrf(
        'hypercoil',
        'examples/synthetic/data/synth-regts/atlas-schaefer400_desc-synth_ts.tsv'
    )
    ref_data = pd.read_csv(ref_path, sep='\t', header=None).values.T
    reference = jnp.array(ref_data)

    X = synthesise_matched(
        reference=reference,
        key=key,
    )[..., :n_time_points]

    #-----------------------------------------------------------------------------#
    # 3. Define the confound model. Let's use a standard 36-parameter model with
    #    censoring.
    confounds = pkgrf('hypercoil', 'examples/data/desc-confounds_timeseries.tsv')
    metadata = pkgrf('hypercoil', 'examples/data/desc-confounds_timeseries.json')
    confounds = pd.read_csv(confounds, sep='\t')
    with open(metadata) as file:
        metadata = json.load(file)

    # Specify the confound model using a formula.
    model_36p = 'dd1((rps + wm + csf + gsr)^^2)'
    model_censor = '[SCATTER]([OR](1_[>0.5](fd) + 1_[>1.5](dv)))'
    model_formula = f'{model_36p} + {model_censor}'

    # Parse the formula into a function.
    f = ConfoundFormulaGrammar().compile(model_formula)
    confounds, metadata = f(confounds, metadata)
    confounds = confounds.fillna(0)
    confounds = jnp.array(confounds.values).T[..., :n_time_points]

    #-----------------------------------------------------------------------------#
    # 4. Create the differentiable program.

    # Define a parameterisation for the filter. Here, we're using an ideal
    # bandpass filter with a frequency range of 0.01-0.1 Hz.
    high_pass, low_pass = 0.01, 0.1
    filter_spec = FreqFilterSpec(Wn=(high_pass, low_pass), ftype='ideal')

    # Define a parameterisation for the tangent projection. Here, we're using
    # the geometric mean of the covariance matrices as the initial point of
    # tangency.
    proj_spec = SPDGeometricMean(psi=1e-3)

    # Instantiate the filter layer using the parameterisation we defined above.
    filter = FrequencyDomainFilter.from_specs(
        (filter_spec,),
        time_dim=n_time_points,
        key=filter_key,
    )
    # Using the `MappedLogits` parameter mapping, we can constrain the filter
    # weights within the range (0, 1). Each weight then represents the
    # attenuation of amplitude in a frequency band.
    filter = MappedLogits.map(filter, where='weight')

    # Instantiate the covariance estimator layer.
    cov = BinaryCovariance(
        estimator=conditionalcorr,
        dim=n_time_points,
        out_channels=n_channels,
        l2=0.1,
        key=cov_key,
    )
    # Let's initialise the covariance weights from a Dirichlet distribution.
    cov = DirichletInitialiser.init(
        cov,
        concentration=[1.0] * n_channels,
        where='weight',
        axis=0,
        key=cov_key,
    )
    # Note that the Dirichlet initialiser automatically transforms our
    # weight into a `ProbabilitySimplexParameter`! This way, the weights
    # are always guaranteed to be valid categorical probability distributions.

    # Instantiate the tangent projection layer using the parameterisation
    # we defined above.
    init_data = cov(filter(X), filter(confounds))
    proj = TangentProject.from_specs(
        mean_specs=(proj_spec,),
        init_data=init_data,
        recondition=1e-5,
        key=proj_key,
    )

    # Finally, let's create the program that combines the filter, covariance
    # estimator, and tangent projection layers.
    class Model(eqx.Module):
        filter: FrequencyDomainFilter
        cov: BinaryCovariance
        proj: TangentProject

        def __call__(self, x, confounds, *, key):
            x, confounds = self.filter(x), self.filter(confounds)
            x = self.cov(x, confounds)
            x = self.proj(x, key=key)
            return x

    model = Model(filter=filter, cov=cov, proj=proj)

    #-----------------------------------------------------------------------------#
    # 5. Define a learning signal. The "bimodal symmetric" score measures the
    #    distance from each element in the correlation matrix to the nearest
    #    of two modes. By setting the modes to -1 and 1, we assign large scores to
    #    weak correlations and small scores to strong correlations.
    #
    #    The "vnorm scalarise" function then takes the matrix of scores and
    #    converts it into a scalar by summing the absolute values of the scores.
    #    Later, we'll use an optimisation algorithm to minimise this scalar score,
    #    thereby promoting strong correlations.

    scalarisation = vnorm_scalarise(p=1, axis=None)
    score = partial(bimodal_symmetric, modes=(-1, 1))
    loss = scalarisation(score) # We are composing the two functions here to
                                # create a new function that takes a matrix
                                # and returns a scalar.

    #-----------------------------------------------------------------------------#
    # 6. Define the "forward pass" of the differentiable program. This is the
    #    function that maps from input data to the output score.
    def forward(model, X, confounds, *, key):
        return loss(model(X, confounds, key=key))

    #-----------------------------------------------------------------------------#
    # 7. Configure the optimisation algorithm. Here, we're using Adam with a
    #    learning rate of 5e-4.
    opt = optax.adam(5e-4)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    #-----------------------------------------------------------------------------#
    # 8. Define a function that updates the model parameters and returns the
    #    updated parameters and the loss.
    def update(model, opt_state, X, confounds, *, key):
        value, grad = eqx.filter_value_and_grad(forward)(
            model, X, confounds, key=key)
        updates, opt_state = opt.update(
            eqx.filter(grad, eqx.is_inexact_array),
            opt_state,
            eqx.filter(model, eqx.is_inexact_array),
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, value

    #-----------------------------------------------------------------------------#
    # 9. Run the optimisation loop.
    for i in range(max_epoch):
        model, opt_state, value = eqx.filter_jit(update)(
            model, opt_state, X, confounds, key=jax.random.fold_in(key, i))
        if i % log_interval == 0:
            print(f'Iteration {i}: loss = {value:.3f}')
