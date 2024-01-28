# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
Functional interfaces.

Functionals and neural network modules (in
:doc:`hypercoil.nn <nn>`)
constitute the elementary atoms of a differentiable program or
computational graph. All operations are composable and differentiable
unless explicitly specified.
"""
from .activation import (
    amplitude_atanh,
    amplitude_tanh,
    corrnorm,
    isochor
)
from .cmass import (
    cmass_coor
)
from .connectopy import (
    diffusion_mapping,
    laplacian_eigenmaps
)
from .cov import (
    ccorr,
    ccov,
    conditionalcorr,
    conditionalcov,
    corr,
    corrcoef,
    correlation,
    cov,
    covariance,
    pairedcorr,
    pairedcov,
    partialcorr,
    partialcov,
    pcorr,
    precision,
)
from .crosssim import (
    crosshair_cosine_similarity,
    crosshair_l1_similarity,
    crosshair_l2_similarity,
    crosshair_similarity,
)
from .fourier import (
    analytic_signal,
    env_inst,
    envelope,
    hilbert_transform,
    instantaneous_frequency,
    instantaneous_phase,
    product_filter,
    product_filtfilt,
    unwrap,
)
from .graph import (
    coaffiliation,
    girvan_newman_null,
    graph_laplacian,
    modularity_matrix,
    relaxed_modularity,
)
from .interpolate import (
    hybrid_interpolate,
    linear_interpolate,
    spectral_interpolate,
    weighted_interpolate,
)
from .kernel import (
    corr_kernel,
    cosine_kernel,
    cov_kernel,
    gaussian_kernel,
    linear_distance,
    linear_kernel,
    param_norm,
    polynomial_kernel,
    rbf_kernel,
    sigmoid_kernel,
)
from .linear import (
    compartmentalised_linear,
)
from .matrix import (
    cholesky_invert,
    delete_diagonal,
    diag_embed,
    expand_outer,
    fill_diagonal,
    recondition_eigenspaces,
    spd,
    squareform,
    sym2vec,
    symmetric,
    toeplitz,
    toeplitz_2d,
    vec2sym,
)
from .metrictensor import (
    integrate_along_line_segment,
    metric_tensor_field_diag_plus_low_rank,
    quadratic_form_low_rank_plus_diag,
    sample_along_line_segment,
)
from .resid import (
    residualise,
)
from .semidefinite import (
    cone_project_spd,
    mean_euc_spd,
    mean_geom_spd,
    mean_harm_spd,
    mean_logeuc_spd,
    tangent_project_spd,
)
from .sparse import (
    as_topk,
    block_serialise,
    dspdmm,
    full_as_topk,
    random_sparse,
    select_indices,
    sp_block_serialise,
    sparse_astype,
    spdiagmm,
    splr_hadamard,
    spsp_innerpaired,
    spsp_pairdiff,
    spspmm,
    spspmm_full,
    topk,
    topk_diagreplace,
    topk_diagzero,
    topk_to_bcoo,
    topkx,
    trace_spspmm,
)
from .sphere import (
    euclidean_conv,
    icosphere,
    spatial_conv,
    sphere_to_latlong,
    sphere_to_normals,
    spherical_conv,
    spherical_geodesic,
)
from .sylo import (
    recombine,
    sylo,
    vertical_compression,
)
from .symmap import (
    symexp,
    symlog,
    symmap,
    symsqrt,
)
from .tsconv import (
    basischan,
    basisconv2d,
    conv,
    polychan,
    polyconv2d,
    tsconv2d,
)
from .utils import (
    amplitude_apply,
    apply_mask,
    complex_decompose,
    complex_recompose,
    conform_mask,
    mask_tensor,
)
from .window import (
    sample_nonoverlapping_windows_existing_ax,
    sample_nonoverlapping_windows_new_ax,
    sample_overlapping_windows_existing_ax,
    sample_overlapping_windows_new_ax,
    sample_windows,
)
