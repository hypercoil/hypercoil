# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Special matrix functions.
"""


def toeplitz(c, r=None, dim=None):
    if r is None:
        r = c.conj()
    clen, rlen = c.size(-1), r.size(-1)
    if dim is not None and dim is not (clen, rlen):
        r_ = torch.zeros(dim[1], dtype=c.dtype, device=c.device)
        c_ = torch.zeros(dim[0], dtype=c.dtype, device=c.device)
        r_[:rlen] = r
        c_[:clen] = c
        r, c = r_, c_
    out_shp = c.size(-1), r.size(-1)
    # return _strided_view_toeplitz(r, c, out_shp)
    X = torch.empty(*out_shp, dtype=c.dtype, device=c.device)
    for i, val in enumerate(c):
        m = min(i + out_shp[1], out_shp[0])
        for j in range(i, m):
            X[j, j - i] = val
    for i, val in list(enumerate(r))[1:]:
        m = min(i + out_shp[0], out_shp[1])
        for j in range(i, m):
            X[j - i, j] = val
    return X


def _strided_view_toeplitz(r, c, out_shp):
    """
    torch is *not* planning on implementing negative strides anytime soon, so
    this numpy-like code will likely not be usable for the foreseeable future.

    See el3ment's comments and the response here:
    https://github.com/pytorch/pytorch/issues/604

    It's not great practice, but we're keeping it here in case they have a
    change of heart.
    """
    raise NotImplementedError('This operation is not currently supported.')
    vals = torch.cat([c.flip(-1)[:-1], r])
    n = vals.stride(0)
    return torch.as_strided(vals[out_shp[0]:],
                            shape=out_shp,
                            strides=(-n, n))
