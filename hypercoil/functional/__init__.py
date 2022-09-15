# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Functional interfaces.

Functionals and neural network modules (in
:doc:`hypercoil.nn <nn>`)
constitute the elementary atoms of a differentiable program or
computational graph. All operations are composable and differentiable
unless explicitly specified.
"""
from .activation import (
    corrnorm,
    threshold,
    isochor
)
from .cmass import (
    cmass_coor
)
from .connectopy import (
    laplacian_eigenmaps,
    diffusion_mapping
)
from .cov import (
    cov, corr, partialcorr, pairedcov, conditionalcov, conditionalcorr,
    precision, pairedcorr, covariance, correlation, corrcoef, pcorr, ccov,
    ccorr, partialcov
)
from .crosssim import (
    crosshair_similarity,
    crosshair_l1_similarity,
    crosshair_l2_similarity,
    crosshair_cosine_similarity
)
from .fourier import (
    product_filter, product_filtfilt, unwrap,
    analytic_signal, hilbert_transform, envelope,
    instantaneous_phase, instantaneous_frequency,
    env_inst
)
from .graph import (
    girvan_newman_null, modularity_matrix, relaxed_modularity,
    graph_laplacian
)
from .interpolate import (
    hybrid_interpolate,
    linear_interpolate,
    spectral_interpolate,
    weighted_interpolate
)
from .kernel import (
    linear_kernel, param_norm, linear_distance,
    polynomial_kernel, sigmoid_kernel,
    gaussian_kernel, rbf_kernel, cosine_kernel,
    cov_kernel, corr_kernel,
)
from .matrix import (
    cholesky_invert, expand_outer, spd, symmetric,
    delete_diagonal, fill_diagonal, recondition_eigenspaces,
    toeplitz, sym2vec, vec2sym, squareform
)
from .resid import (
    residualise
)
from .sparse import (
    random_sparse, spdiagmm, dspdmm, spspmm_full, topk, as_topk, full_as_topk,
    sparse_astype, trace_spspmm, spspmm, spsp_pairdiff, select_indices, topkx,
    sp_block_serialise, splr_hadamard, spsp_innerpaired, topk_to_bcoo,
    topk_diagzero, topk_diagreplace,
    block_serialise,
)
from .sphere import (
    spherical_geodesic,
    sphere_to_normals,
    sphere_to_latlong,
    spatial_conv,
    spherical_conv,
    euclidean_conv,
)
from .sylo import (
    sylo
)
from .symmap import (
    symmap, symexp, symlog, symsqrt
)
from .semidefinite import (
    tangent_project_spd, cone_project_spd, mean_euc_spd, mean_harm_spd,
    mean_logeuc_spd, mean_geom_spd
)
from .tsconv import (
    conv, polychan, polyconv2d, basischan, basisconv2d, tsconv2d
)
from .utils import (
    complex_decompose,
    complex_recompose,
    conform_mask,
    apply_mask,
    mask_tensor,
    amplitude_apply
)
from .window import (
    sample_windows,
    sample_nonoverlapping_windows_existing_ax,
    sample_nonoverlapping_windows_new_ax,
    sample_overlapping_windows_new_ax,
    sample_overlapping_windows_existing_ax,
)
