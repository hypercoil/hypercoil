# Differentiable program for brain mapping

_Documentation is actively undergoing reorganisation -- please check back in a few days for something more stable._

The `hypercoil` repository is the current home of this project. The overall objective is development of a brain mapping pipeline that learns its own parameters subject to a user-specified loss and regularisations, together with a reasonable initialisation. The current focus area is building a complete functional connectivity pipeline that ingests minimally preprocessed data and performs processing blocks from atlasing to connectome estimation. In the most recent phase of project development, we scaled up from synthetic datasets with known characteristics to deploying simple, minimal implementations of differentiable processing blocks on real data. This proof of concept is detailed in our preprint (coming soon).

To attain its objective, this software library implements neuroimage processing steps as differentiable PyTorch modules (neural network layers). This enables gradient to propagate back from a final model (or any set of differentiable loss functions) into the pipeline modules and reparameterise them.

**Public warning:** While numerous components are operational, the repository as a whole is not ready for deployment or use at this time. It should be considered as pre-alpha state. If you happen to find something of interest to your work in this repository, use extreme caution as many operations are fragile or incompletely documented. Edge cases might not be covered, errors might exist due to a combination of a small team / rapid pace of development, etc. Contributions or ideas for improvement, however, are always welcome.

### Status

Due to extremely rapid development in April and May 2022, substantial technical debt has accumulated. The next week or two will begin to mitigate this; if you happen across this, or come here from the preprint/poster, we'd suggest checking back in a few days. Operability of the `hypercoil` library varies by submodule.

* Please read docstrings: if a docstring indicates that a function isn't ready for use, or if the docstring is missing altogether, that's a good sign that it will be very important to extensively check that behaviour matches expectations.
* Even stable submodules are likely to undergo substantial API changes (which might be unannounced) during the coming refactor. "Stable" is relative to the experimental content of this software library.
* The `functional` submodule is mostly stable and usable. Tangent space implementations (and many operations on positive semidefinite matrices) lack numerical stability.
* The `nn` submodule (parameterised neural network modules) is likewise mostly stable and usable. The IIR filter lacks a reasonable initialisation and poorly conveys gradient; it is not currently ready for use. There might be a substantial refactor to create compatibility with ephemeral parameters (e.g., subject-specific weights).
* The `loss` submodule (loss functions and objectives) is mostly stable and usable.
* The `init` submodule (containing parameter initialisation schemes) is fairly stable/usable.
* The `data` and `formula` submodules will handle data I/O and data engineering workflows. In their current form, they should be considered as prototypes and will accordingly receive major overhauls.
* The `eval` submodule will contain standard performance benchmarks. It is currently in early prototype state.
* The `viz` submodule is the current location of visualisation utilities. The scope and generalisability of these utilities is very limited at this time, and they should likewise be considered as early prototypes.
* Code in the `workflows` and `synth` subdirectories should currently be understood as illustrative examples only (although the `synth` code is also used for the full unit test battery). (The `workflows` subdirectory, in particular, still has many hard-codes for our file systems.)
* The `engine` submodule contains `Sentry` functionality, which can be useful for clean training loops and rebalancing loss multipliers. Stay away from anything `Conveyance`-related, however. Tools for data and gradient flow control (e.g., `ReactiveTerminal` and `Accumuline`) should be used only with extreme caution. `Accumuline` in particular is known from test cases to distort the gradient; the reason for this is not currently known.

### Installation

Right now, just pip install from GitHub. Come back in a few weeks and ask again about PyPI.
